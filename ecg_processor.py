"""
ECG Processor Module for Heart Diary
Refactored from polar.py for use with FastAPI
"""
import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import os
import shutil
import base64
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 1. DATA LOADING & PREP
# ==========================================
def load_and_sync_data(ecg_path, marker_path=None):
    """
    Load ECG CSV (Polar H10 export) and optional marker file.
    """
    print("⏳ Loading ECG data...")
    try:
        df_ecg = pd.read_csv(ecg_path, sep=';', engine='python')
        # Find timestamp column (case insensitive)
        ts_col = [c for c in df_ecg.columns if 'phone' in c.lower()]
        if not ts_col:
            # Fallback if header is different
            ts_col = [df_ecg.columns[0]]
        ts_col = ts_col[0]

        start_time_str = df_ecg[ts_col].iloc[0]
        start_time = pd.to_datetime(start_time_str)
        if getattr(start_time, "tzinfo", None) is not None:
            start_time = start_time.tz_convert(None)

        # Calculate sampling rate
        duration_ms = df_ecg['timestamp [ms]'].iloc[-1] - df_ecg['timestamp [ms]'].iloc[0]
        duration_sec = max(duration_ms / 1000.0, 1.0)
        sampling_rate = int(len(df_ecg) / duration_sec)
        
        # Extract signal (assuming last column is ECG)
        ecg_values = df_ecg.iloc[:, -1].values.astype(float)
        
        print(f"   ✅ ECG Loaded: {len(df_ecg)} samples @ ~{sampling_rate} Hz over {duration_sec/60:.1f} min")
        return df_ecg, ecg_values, sampling_rate, start_time
    except Exception as e:
        print(f"   ❌ Error loading ECG: {e}")
        return None, None, None, None


def load_markers(marker_path, start_time, sampling_rate):
    df_markers = pd.DataFrame()
    if marker_path and os.path.exists(marker_path):
        try:
            df_markers = pd.read_csv(marker_path, sep=';', engine='python')
            print(f"   📍 Markers Loaded: {len(df_markers)} events found.")
        except Exception as e:
            print(f"   ⚠️ Could not load markers: {e}")
    return df_markers


def load_acc_motion(acc_path, ecg_start_time, sampling_rate, n_samples):
    """
    Load accelerometer CSV and produce motion flag.
    """
    if acc_path is None or not os.path.exists(acc_path):
        return None, {"rest_ratio": 100.0, "active_ratio": 0.0}

    print("📡 Loading accelerometer data...")
    try:
        df_acc = pd.read_csv(acc_path, sep=';', engine='python')
    except Exception as e:
        print(f"   ⚠️ Could not load ACC: {e}")
        return None, {"rest_ratio": 100.0, "active_ratio": 0.0}

    ts_candidates = [c for c in df_acc.columns if 'phone' in c.lower()]
    if not ts_candidates:
        return None, {"rest_ratio": 100.0, "active_ratio": 0.0}
    ts_col = ts_candidates[0]

    df_acc['phone_dt'] = pd.to_datetime(df_acc[ts_col])
    if getattr(ecg_start_time, "tzinfo", None) is not None:
        ecg_start_naive = ecg_start_time.tz_convert(None)
    else:
        ecg_start_naive = ecg_start_time

    acc_time_s = (df_acc['phone_dt'] - ecg_start_naive).dt.total_seconds().values

    if len(acc_time_s) < 5:
        return None, {"rest_ratio": 100.0, "active_ratio": 0.0}

    # Magnitude
    try:
        x = df_acc['X [mg]'].values.astype(float)
        y = df_acc['Y [mg]'].values.astype(float)
        z = df_acc['Z [mg]'].values.astype(float)
        acc_mag = np.sqrt(x**2 + y**2 + z**2)
    except KeyError:
        return None, {"rest_ratio": 100.0, "active_ratio": 0.0}

    # Interpolate to ECG
    ecg_time_s = np.arange(n_samples) / float(sampling_rate)
    acc_mag_interp = np.interp(ecg_time_s, acc_time_s, acc_mag)

    # Rolling std for motion intensity
    window = max(int(sampling_rate), 1)
    acc_std = pd.Series(acc_mag_interp).rolling(window, min_periods=1).std().fillna(0.0)

    MOTION_STD_THRESHOLD = 20.0  # mg (slightly higher to avoid minor shifts)
    motion_flag = (acc_std > MOTION_STD_THRESHOLD).values

    rest_ratio = float((~motion_flag).mean()) * 100.0
    active_ratio = 100.0 - rest_ratio
    print(f"   🚶 Motion: {rest_ratio:.1f}% Rest | {active_ratio:.1f}% Active")

    return motion_flag, {"rest_ratio": rest_ratio, "active_ratio": active_ratio}


# ==========================================
# 2. METRICS CALCULATOR
# ==========================================
def calculate_metrics_on_segment(segment, sampling_rate, description="Baseline"):
    """
    Calculate QRS, QTc, etc. on a SPECIFIC clean segment.
    """
    metrics = {
        "mean_qrs": "N/A", "mean_qtc": "N/A", 
        "mean_pr": "N/A", "t_pos": "N/A"
    }
    
    try:
        # Clean segment again to be sure
        ecg_clean = nk.ecg_clean(segment, sampling_rate=sampling_rate)
        _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=sampling_rate)
        
        if len(rpeaks["ECG_R_Peaks"]) < 3:
            return metrics

        # Delineate
        waves, _ = nk.ecg_delineate(ecg_clean, rpeaks["ECG_R_Peaks"], 
                                    sampling_rate=sampling_rate, method="dwt")

        # QRS
        ro = np.array(waves["ECG_R_Offsets"])
        rs = np.array(waves["ECG_R_Onsets"])
        mask = (~np.isnan(ro)) & (~np.isnan(rs))
        if mask.sum() > 0:
            qrs_vals = (ro[mask] - rs[mask]) / sampling_rate * 1000.0
            metrics["mean_qrs"] = int(np.nanmean(qrs_vals))

        # QTc (Bazett)
        to = np.array(waves["ECG_T_Offsets"])
        mask_qt = (~np.isnan(to)) & (~np.isnan(rs))
        if mask_qt.sum() > 0:
            qt_raw = (to[mask_qt] - rs[mask_qt]) / sampling_rate
            # Calculate RR for each beat for Bazett
            rr_intervals = np.diff(rpeaks["ECG_R_Peaks"]) / sampling_rate
            # We need to align QT with preceding RR. Rough alignment:
            avg_rr = np.mean(rr_intervals) # average RR for the segment
            
            qtc_vals = qt_raw * 1000.0 / np.sqrt(avg_rr)
            metrics["mean_qtc"] = int(np.nanmean(qtc_vals))
            
        # T-wave positivity
        t_peaks = waves["ECG_T_Peaks"]
        t_indices = [int(x) for x in t_peaks if not np.isnan(x) and int(x) < len(ecg_clean)]
        if t_indices:
            t_amps = ecg_clean[t_indices]
            metrics["t_pos"] = int((t_amps > 0).sum() / len(t_amps) * 100.0)

    except Exception as e:
        print(f"   ⚠️ Metrics calc failed on {description}: {e}")
        
    return metrics


def calculate_global_stats(signals, rpeaks, sampling_rate):
    """
    Calculate full comprehensive stats.
    Includes HR, HRV (SDNN, pNN50), and Frequency.
    """
    # --- A. Heart Rate Stats ---
    hr_series = signals["ECG_Rate"].astype(float)
    mean_hr = int(hr_series.mean())
    max_hr = int(hr_series.max())
    min_hr = int(hr_series.min())

    window_1min = max(int(sampling_rate * 60), 5)
    rolling_hr = hr_series.rolling(window=window_1min).mean()
    if rolling_hr.dropna().empty:
        highest_1min_hr = max_hr
        lowest_1min_hr = min_hr
    else:
        highest_1min_hr = int(rolling_hr.max())
        lowest_1min_hr = int(rolling_hr.min())

    perc_tachy = float((hr_series > 150).sum()) / len(hr_series) * 100.0
    perc_brady = float((hr_series < 50).sum()) / len(hr_series) * 100.0

    # --- B. HRV & Rhythm ---
    hrv_sdnn = 0.0
    pnn50 = 0.0
    pnn200 = 0.0
    if len(rpeaks) > 3:
        rr_intervals_ms = np.diff(rpeaks) / sampling_rate * 1000.0
        hrv_sdnn = float(np.std(rr_intervals_ms, ddof=1))
        diff_rr = np.abs(np.diff(rr_intervals_ms))
        pnn50 = float((diff_rr > 50).sum()) / len(diff_rr) * 100.0
        pnn200 = float((diff_rr > 200).sum()) / len(diff_rr) * 100.0

    # --- C. Frequency Domain (HF%) ---
    mean_hf = 0
    try:
        if len(rpeaks) > 10:
            analyze_peaks = rpeaks if len(rpeaks) < 5000 else rpeaks[:5000]
            hrv_freq = nk.hrv_frequency(analyze_peaks,
                                        sampling_rate=sampling_rate,
                                        show=False)
            if "HRV_HFnu" in hrv_freq.columns:
                hf_val = hrv_freq["HRV_HFnu"].values[0]
                mean_hf = int(hf_val) if not np.isnan(hf_val) else 0
    except Exception as e:
        print(f"   ⚠️ Frequency HRV skipped: {e}")

    return {
        "mean_hr": mean_hr,
        "max_hr": max_hr,
        "min_hr": min_hr,
        "high_1min": highest_1min_hr,
        "low_1min": lowest_1min_hr,
        "tachy_perc": perc_tachy,
        "brady_perc": perc_brady,
        "sdnn": int(hrv_sdnn),
        "pnn50": int(pnn50),
        "pnn200": int(pnn200),
        "hf": mean_hf
    }


# ==========================================
# 3. EVENT DETECTION
# ==========================================
def analyze_rhythm(rpeaks, sampling_rate):
    """
    Classify beats based on RR intervals.
    Returns beat_types array (N, S, L) and burden stats.
    """
    rr_intervals = np.diff(rpeaks) / sampling_rate # seconds
    if len(rr_intervals) == 0: 
        return [], [], 0.0, np.zeros(len(rpeaks), dtype=bool)
    
    # Rolling median for adaptive threshold
    rr_series = pd.Series(rr_intervals)
    rolling_med = rr_series.rolling(window=12, center=True).median().bfill().ffill()
    
    beat_types = []
    for i, rr in enumerate(rr_intervals):
        med = rolling_med.iloc[i]
        # Thresholds: <80% is premature, >120% is compensatory
        if rr < med * 0.80:
            beat_types.append("S") # Short
        elif rr > med * 1.20:
            beat_types.append("L") # Long
        else:
            beat_types.append("N") # Normal
            
    beat_types = np.array(beat_types)
    
    # Estimate Ectopic Burden (S followed by L is classic PVC/PAC sign)
    ectopic_count = 0
    ectopic_mask = np.zeros(len(rpeaks), dtype=bool)
    
    for i in range(len(beat_types)-1):
        if beat_types[i] == "S" and i+1 < len(beat_types) and beat_types[i+1] == "L":
            ectopic_count += 1
            ectopic_mask[i+1] = True # Mark the beat *after* the short interval
            
    burden = (ectopic_count / len(rpeaks)) * 100.0 if len(rpeaks) > 0 else 0.0
    
    return rr_intervals, beat_types, burden, ectopic_mask


def find_events_of_interest(rpeaks, beat_types, severity_map):
    events = []
    n = len(beat_types)
    i = 0
    while i < n - 1:
        idx = rpeaks[i+1]
        
        # Couplets (S-S)
        if beat_types[i] == "S" and i+1 < n and beat_types[i+1] == "S":
            events.append({"index": idx, "type": "Couplet (Premature x2)", "sev": 3})
            i += 2
        # Bigeminy (S-L-S-L)
        elif i+3 < n and "".join(beat_types[i:i+4]) in ["SLSL", "SNSN"]:
            events.append({"index": idx, "type": "Bigeminy Pattern", "sev": 2})
            i += 4
        # Trigeminy (N-N-S)
        elif i+2 < n and "".join(beat_types[i:i+3]) == "NNS":
            events.append({"index": idx, "type": "Trigeminy Pattern", "sev": 2})
            i += 3
        # Isolated
        elif beat_types[i] == "S" and i+1 < len(beat_types) and beat_types[i+1] == "L":
            events.append({"index": idx, "type": "Isolated Premature Beat", "sev": 1})
            i += 1
        else:
            i += 1
    return events


# ==========================================
# 4. PLOTTING & REPORTS
# ==========================================
def create_ecg_plot(segment, sr, title, filename, color_hex="#8D0A1E"):
    if len(segment) == 0: 
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 8), sharex=False, sharey=True)
    fig.suptitle(title, fontsize=16, color=color_hex, weight="bold")
    
    # Grid config
    major_ticks = np.arange(0, 10.1, 0.2)
    minor_ticks = np.arange(0, 10.1, 0.04)
    
    strip_seconds = 10
    samples_per_strip = int(sr * strip_seconds)
    
    for i in range(3):
        ax = axes[i]
        start = i * samples_per_strip
        end = start + samples_per_strip
        if start >= len(segment): 
            break
        
        data = segment[start:min(end, len(segment))]
        t = np.linspace(0, len(data)/sr, len(data))
        
        ax.plot(t, data, color="black", linewidth=1.0)
        
        # Medical Grid
        ax.set_xlim(0, 10)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.grid(which="major", linestyle="-", linewidth=0.8, color="red", alpha=0.5)
        ax.grid(which="minor", linestyle=":", linewidth=0.5, color="red", alpha=0.3)
        
        ax.set_ylabel(f"Strip {i+1}")
        if i < 2: 
            ax.set_xticklabels([])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def check_signal_quality(segment):
    # Simple heuristic: range too small (flatline) or too huge (artifact)
    r = np.max(segment) - np.min(segment)
    if r < 50 or r > 8000: 
        return False
    return True


# ==========================================
# 5. MAIN PROCESSING FUNCTION
# ==========================================
def process_ecg_data(ecg_file, marker_file=None, acc_file=None, output_dir="reports", report_id="report"):
    """
    Main processing function that returns a result dict with report data.
    """
    result = {
        "success": False,
        "error": None,
        "date": None,
        "duration_min": 0,
        "mean_hr": 0,
        "max_hr": 0,
        "ectopic_burden": 0
    }
    
    # Create temp directory for images
    temp_img_dir = Path(output_dir) / f"{report_id}_images"
    if temp_img_dir.exists():
        shutil.rmtree(temp_img_dir)
    temp_img_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Load
        _, ecg_raw, sr, start_time = load_and_sync_data(ecg_file, marker_file)
        if ecg_raw is None:
            result["error"] = "Failed to load ECG data"
            return result

        result["date"] = start_time.isoformat()
        result["duration_min"] = len(ecg_raw) / sr / 60.0

        # 2. Process (Whole Signal)
        print("🧠 Processing signal...")
        signals, info = nk.ecg_process(ecg_raw, sampling_rate=sr)
        rpeaks = info["ECG_R_Peaks"]
        ecg_clean = signals["ECG_Clean"].values

        # 3. Motion Gating
        motion_flag, motion_stats = load_acc_motion(acc_file, start_time, sr, len(ecg_clean))

        # 4. Rhythm & Events
        rr_ints, beat_types, global_burden, ectopic_mask = analyze_rhythm(rpeaks, sr)
        all_events = find_events_of_interest(rpeaks, beat_types, {})
        
        result["ectopic_burden"] = global_burden
        
        # 5. Baseline Search (Algorithmically find BEST 30s segment)
        print("🔍 Searching for clean baseline...")
        
        is_event_beat = np.zeros(len(ecg_clean), dtype=bool)
        for idx in [e['index'] for e in all_events]:
            is_event_beat[max(0, idx-sr):min(len(ecg_clean), idx+sr)] = True
            
        # Create a valid mask: Not Motion AND Not Near Event
        valid_mask = ~is_event_beat
        if motion_flag is not None:
            valid_mask = valid_mask & (~motion_flag)
            
        # Find longest contiguous True segment
        padded = np.concatenate(([False], valid_mask, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        lengths = ends - starts
        
        if len(lengths) > 0:
            best_seg_idx = np.argmax(lengths)
            best_start = starts[best_seg_idx]
            best_len = lengths[best_seg_idx]
        else:
            best_start = 0
            best_len = 0
            
        # Extract baseline segment (30s)
        baseline_center = best_start + best_len // 2
        half_window = 15 * sr
        b_start = max(0, baseline_center - half_window)
        b_end = min(len(ecg_clean), baseline_center + half_window)
        baseline_seg = ecg_clean[b_start:b_end]
        
        # 6. Calculate Advanced Metrics ON BASELINE
        baseline_metrics = calculate_metrics_on_segment(baseline_seg, sr, "Baseline")
        global_stats = calculate_global_stats(signals, rpeaks, sr)
        
        result["mean_hr"] = global_stats["mean_hr"]
        result["max_hr"] = global_stats["max_hr"]
        
        # 7. Process Markers
        markers_html = ""
        df_markers = load_markers(marker_file, start_time, sr)
        if df_markers is not None and not df_markers.empty:
            print(f"   📍 Generating {len(df_markers)} marker plots...")
            t_col = [c for c in df_markers.columns if "phone" in c.lower()]
            if t_col:
                t_col = t_col[0]
                l_col_candidates = [c for c in df_markers.columns if "marker" in c.lower()]
                l_col = l_col_candidates[0] if l_col_candidates else df_markers.columns[-1]

                for i, row in df_markers.iterrows():
                    try:
                        m_time = pd.to_datetime(row[t_col])
                        if getattr(start_time, "tzinfo", None) is not None:
                            start_naive = start_time.tz_convert(None)
                        else:
                            start_naive = start_time
                        
                        m_time_naive = m_time.tz_convert(None) if getattr(m_time, "tzinfo", None) is not None else m_time
                        
                        sec_diff = (m_time_naive - start_naive).total_seconds()
                        idx = int(sec_diff * sr)
                        
                        if 0 <= idx < len(ecg_clean):
                            s_idx = max(0, idx - 15 * sr)
                            e_idx = min(len(ecg_clean), idx + 15 * sr)
                            seg = ecg_clean[s_idx:e_idx]
                            
                            fn = str(temp_img_dir / f"Marker_{i}.png")
                            label_text = str(row[l_col])
                            create_ecg_plot(seg, sr, f"SYMPTOM MARKER: {label_text}", fn, "#673AB7")
                            
                            with open(fn, "rb") as f: 
                                b64 = base64.b64encode(f.read()).decode("utf-8")
                            
                            markers_html += f"""
                            <div class='event-box' style='border-left: 5px solid #673AB7;'>
                                <div class='event-header' style='background: #f3e5f5;'>
                                    <h3 style='color: #673AB7;'>📍 Symptom: {label_text}</h3>
                                    <p>Time: {m_time_naive.strftime('%H:%M:%S')}</p>
                                </div>
                                <div class='plot-container' style='margin: 0; border: none;'>
                                    <img src='data:image/png;base64,{b64}' alt='Marker Plot'>
                                </div>
                            </div>
                            """
                    except Exception as e:
                        print(f"   ⚠️ Error processing marker {i}: {e}")

        # 8. Generate Visuals
        
        # Baseline Plot
        baseline_img = str(temp_img_dir / "Baseline.png")
        create_ecg_plot(baseline_seg, sr, "BASELINE (Cleanest Resting Segment)", baseline_img, "#2E7D32")
        
        # Events Plot (Top 5)
        events_html = ""
        unique_events = []
        seen_times = []
        
        for e in sorted(all_events, key=lambda x: x['sev'], reverse=True):
            if len(unique_events) >= 5: 
                break
            if any(abs(e['index'] - t) < 10*sr for t in seen_times): 
                continue
            
            c = e['index']
            s = max(0, c - 15*sr)
            e_idx = min(len(ecg_clean), c + 15*sr)
            seg = ecg_clean[s:e_idx]
            
            if check_signal_quality(seg):
                unique_events.append(e)
                seen_times.append(c)
                fn = str(temp_img_dir / f"Event_{len(unique_events)}.png")
                time_str = (start_time + timedelta(seconds=c/sr)).strftime("%H:%M:%S")
                create_ecg_plot(seg, sr, f"{e['type']} @ {time_str}", fn)
                
                with open(fn, "rb") as f: 
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                events_html += f"""
                <div class='event-box'>
                    <div class='event-header'>
                        <h3>⚠️ {e['type']}</h3>
                        <p>Time: {time_str}</p>
                    </div>
                    <div class='plot-container' style='margin: 0; border: none;'>
                        <img src='data:image/png;base64,{b64}' alt='Event Plot'>
                    </div>
                </div>
                """

        # Count pattern occurrences for stats
        pattern_counts = {}
        for e in all_events:
            pattern_counts[e['type']] = pattern_counts.get(e['type'], 0) + 1
            
        pattern_rows = ""
        if pattern_counts:
            pattern_rows += "<tr><th colspan='2' style='background: #f5f5f5; color: #555; font-size: 0.85rem; padding-top: 10px;'>Detected Patterns (Count)</th></tr>"
            for p_type, p_count in pattern_counts.items():
                pattern_rows += f"<tr><td>{p_type}</td><td>{p_count}</td></tr>"
        else:
            pattern_rows = "<tr><td colspan='2' style='text-align:center; color:#999; font-size: 0.85rem; padding-top: 10px;'>No specific patterns detected</td></tr>"

        # 9. Responsive HTML Generation
        with open(baseline_img, "rb") as f: 
            b64_base = base64.b64encode(f.read()).decode("utf-8")
        
        html = generate_report_html(
            start_time=start_time,
            ecg_clean=ecg_clean,
            sr=sr,
            global_stats=global_stats,
            global_burden=global_burden,
            motion_stats=motion_stats,
            baseline_metrics=baseline_metrics,
            b64_base=b64_base,
            markers_html=markers_html,
            events_html=events_html,
            pattern_rows=pattern_rows
        )
        
        report_fn = Path(output_dir) / f"{report_id}.html"
        with open(report_fn, "w") as f: 
            f.write(html)
        print(f"\n✅ Report generated: {report_fn}")
        
        # Clean up temp images
        shutil.rmtree(temp_img_dir)
        
        result["success"] = True
        return result
        
    except Exception as e:
        import traceback
        result["error"] = str(e)
        print(f"❌ Processing error: {e}")
        traceback.print_exc()
        return result


def generate_report_html(start_time, ecg_clean, sr, global_stats, global_burden, 
                         motion_stats, baseline_metrics, b64_base, markers_html, 
                         events_html, pattern_rows):
    """Generate the HTML report content."""
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Clinical ECG Report - {start_time.strftime('%Y-%m-%d %H:%M')}</title>
        <style>
            :root {{
                --primary-color: #8D0A1E;
                --bg-color: #ffffff;
                --text-color: #333333;
                --light-bg: #f5f5f5;
                --border-color: #e0e0e0;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                color: var(--text-color);
                background: var(--bg-color);
                margin: 0;
                padding: 20px;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: #fff;
            }}
            .back-link {{
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                color: var(--primary-color);
                text-decoration: none;
                margin-bottom: 1rem;
                font-weight: 500;
            }}
            .back-link:hover {{
                text-decoration: underline;
            }}
            h1 {{
                font-size: 1.8rem;
                color: var(--text-color);
                margin-bottom: 0.5rem;
            }}
            h2 {{
                border-bottom: 2px solid var(--primary-color); 
                color: var(--primary-color); 
                font-size: 1.4rem;
                margin-top: 2rem;
                padding-bottom: 0.5rem;
            }}
            p {{ margin-bottom: 1rem; }}
            
            /* Metrics Grid */
            .metric-box {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                gap: 15px;
                background: var(--light-bg); 
                padding: 20px; 
                border-radius: 12px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            .metric {{
                text-align: center;
                padding: 10px;
                background: #fff;
                border-radius: 8px;
                border: 1px solid var(--border-color);
            }}
            .val {{
                font-size: 1.5rem; 
                font-weight: 700; 
                color: var(--primary-color); 
                margin-bottom: 5px;
            }}
            .lbl {{
                font-size: 0.85rem; 
                color: #666; 
                text-transform: uppercase; 
                letter-spacing: 0.5px;
                font-weight: 500;
            }}
            
            /* Extended Stats Table */
            .stats-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
                font-size: 0.9rem;
            }}
            .stats-table th {{ text-align: left; background: #eee; padding: 8px; }}
            .stats-table td {{ border-bottom: 1px solid #eee; padding: 8px; }}

            /* Images */
            .plot-container {{
                width: 100%;
                overflow-x: hidden;
                margin: 15px 0;
                border-radius: 8px;
                border: 1px solid var(--border-color);
            }}
            img {{
                width: 100%; 
                height: auto; 
                display: block;
            }}

            /* Events */
            .event-box {{
                margin-top: 20px; 
                border: 1px solid var(--border-color); 
                border-left: 5px solid var(--primary-color);
                border-radius: 8px;
                overflow: hidden;
                background: #fff;
            }}
            .event-header {{
                padding: 15px;
                background: var(--light-bg);
                border-bottom: 1px solid var(--border-color);
            }}
            .event-header h3 {{ margin: 0; font-size: 1.1rem; color: #333; }}
            .event-header p {{ margin: 5px 0 0 0; font-size: 0.9rem; color: #666; }}

            /* Disclaimer */
            .disclaimer {{
                font-size: 0.8rem; 
                color: #888; 
                margin-top: 40px; 
                padding-top: 20px; 
                border-top: 1px solid var(--border-color);
                text-align: center;
            }}

            /* Responsive Tweaks */
            @media (max-width: 600px) {{
                body {{ padding: 15px; }}
                h1 {{ font-size: 1.5rem; }}
                h2 {{ font-size: 1.2rem; }}
                .metric-box {{ grid-template-columns: 1fr 1fr; gap: 10px; padding: 15px; }}
                .val {{ font-size: 1.3rem; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/calendar" class="back-link">← Back to Calendar</a>
            
            <h1>🏥 Clinical ECG Report <span style="font-size: 1rem; font-weight: normal; color: #666;">(Heart Diary)</span></h1>
            <p style="color: #555;">
                <b>Date:</b> {start_time.strftime('%Y-%m-%d %H:%M')}<br>
                <b>Duration:</b> {len(ecg_clean)/sr/60:.1f} min
            </p>
            
            <h2>📊 Session Metrics</h2>
            <div class="metric-box">
                <div class="metric"><div class="val">{global_stats['mean_hr']}</div><div class="lbl">Avg HR (bpm)</div></div>
                <div class="metric"><div class="val">{global_stats['max_hr']}</div><div class="lbl">Max HR</div></div>
                <div class="metric"><div class="val">{global_burden:.2f}%</div><div class="lbl">Ectopic Burden</div></div>
                <div class="metric"><div class="val">{motion_stats['rest_ratio']:.1f}%</div><div class="lbl">Rest Time</div></div>
            </div>
            
            <h3>📈 Detailed Statistics</h3>
            <table class="stats-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>HRV (SDNN)</td><td>{global_stats['sdnn']} ms</td></tr>
                <tr><td>pNN50 (Var.)</td><td>{global_stats['pnn50']} %</td></tr>
                <tr><td>High 1-min HR</td><td>{global_stats['high_1min']} bpm</td></tr>
                <tr><td>Low 1-min HR</td><td>{global_stats['low_1min']} bpm</td></tr>
                <tr><td>HF Power (Norm)</td><td>{global_stats['hf']} %</td></tr>
                <tr><td>Tachycardia Time (>150)</td><td>{global_stats['tachy_perc']:.1f} %</td></tr>
                <tr><td>Bradycardia Time (<50)</td><td>{global_stats['brady_perc']:.1f} %</td></tr>
                {pattern_rows}
            </table>
            
            <h2>❤️ Baseline Morphology <span style="font-size: 1rem; font-weight: normal; color: #666;">(Resting)</span></h2>
            <p style="font-size: 0.9rem; color: #666; margin-bottom: 15px;">
                Metrics derived from the cleanest {baseline_metrics['mean_qrs'] != 'N/A' and '30s' or ''} resting segment.
            </p>
            <div class="metric-box" style="background: #e3f2fd; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));">
                <div class="metric" style="border-color: #90caf9;"><div class="val" style="color: #1565c0;">{baseline_metrics['mean_qrs']} ms</div><div class="lbl">QRS Duration</div></div>
                <div class="metric" style="border-color: #90caf9;"><div class="val" style="color: #1565c0;">{baseline_metrics['mean_qtc']} ms</div><div class="lbl">QTc (Bazett)</div></div>
            </div>
            <div class="plot-container">
                <img src="data:image/png;base64,{b64_base}" alt="Baseline ECG">
            </div>
            
            <h2>📍 Symptom Markers</h2>
            {markers_html if markers_html else "<p style='padding: 20px; background: #f9f9f9; border-radius: 8px; text-align: center; color: #666;'>No symptom markers recorded in this session.</p>"}

            <h2>⚠️ Notable Rhythm Events</h2>
            {events_html if events_html else "<p style='padding: 20px; background: #f9f9f9; border-radius: 8px; text-align: center; color: #666;'>✅ No significant ectopic patterns detected.</p>"}
            
            <div class="disclaimer">
                <b>Medical Disclaimer:</b> This report is generated by an automated Python script using Polar H10 data. 
                It identifies intervals consistent with premature beats but cannot definitively diagnose PVC vs PAC or ischemia. 
                QRS/QT values are estimates. Please review strips with a cardiologist.
            </div>
        </div>
    </body>
    </html>
    """


def get_report_html(report_id, output_dir="reports"):
    """Load and return report HTML content."""
    report_path = Path(output_dir) / f"{report_id}.html"
    if report_path.exists():
        with open(report_path, "r") as f:
            return f.read()
    return None
