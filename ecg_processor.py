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
from io import StringIO, BytesIO
import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 1. DATA LOADING & PREP
# ==========================================
def load_and_sync_data(ecg_content, marker_content=None):
    """
    Load ECG data from content string and optional marker content.
    """
    print("⏳ Loading ECG data...")
    try:
        df_ecg = pd.read_csv(StringIO(ecg_content), sep=';', engine='python')
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


def load_markers(marker_content, start_time, sampling_rate):
    df_markers = pd.DataFrame()
    if marker_content:
        try:
            df_markers = pd.read_csv(StringIO(marker_content), sep=';', engine='python')
            print(f"   📍 Markers Loaded: {len(df_markers)} events found.")
        except Exception as e:
            print(f"   ⚠️ Could not load markers: {e}")
    return df_markers


def load_hr_data(hr_content, ecg_start_time, sampling_rate, n_samples):
    """
    Load Polar HR file data and return HR series aligned to ECG samples.
    Polar HR file format: Phone timestamp;HR [bpm];HRV [ms];Breathing interval [rpm];
    """
    if hr_content is None:
        return None, None

    print("💓 Loading Polar HR data...")
    try:
        df_hr = pd.read_csv(StringIO(hr_content), sep=';', engine='python')
    except Exception as e:
        print(f"   ⚠️ Could not load HR file: {e}")
        return None, None

    # Find timestamp and HR columns
    ts_candidates = [c for c in df_hr.columns if 'phone' in c.lower()]
    hr_candidates = [c for c in df_hr.columns if 'hr' in c.lower() and 'bpm' in c.lower()]
    
    if not ts_candidates or not hr_candidates:
        print("   ⚠️ HR file missing required columns")
        return None, None
    
    ts_col = ts_candidates[0]
    hr_col = hr_candidates[0]

    # Parse timestamps and HR values
    df_hr['phone_dt'] = pd.to_datetime(df_hr[ts_col])
    df_hr['hr_bpm'] = pd.to_numeric(df_hr[hr_col], errors='coerce')
    
    # Drop rows with missing HR
    df_hr = df_hr.dropna(subset=['hr_bpm'])
    
    if len(df_hr) < 3:
        print("   ⚠️ Not enough HR data points")
        return None, None

    # Convert ECG start time to naive if needed
    if getattr(ecg_start_time, "tzinfo", None) is not None:
        ecg_start_naive = ecg_start_time.tz_convert(None)
    else:
        ecg_start_naive = ecg_start_time

    # Calculate time offset from ECG start
    hr_time_s = (df_hr['phone_dt'] - ecg_start_naive).dt.total_seconds().values
    hr_values = df_hr['hr_bpm'].values.astype(float)
    
    # Debug: Check time alignment
    print(f"   📊 HR time range: {hr_time_s[0]:.1f}s to {hr_time_s[-1]:.1f}s from ECG start")

    # Interpolate to ECG sample rate
    ecg_time_s = np.arange(n_samples) / float(sampling_rate)
    
    # Check if there's any overlap
    if hr_time_s[-1] < ecg_time_s[0] or hr_time_s[0] > ecg_time_s[-1]:
        print(f"   ⚠️ No time overlap between HR and ECG data!")
        return None, None
    
    hr_interp = np.interp(ecg_time_s, hr_time_s, hr_values)
    
    # Calculate stats from Polar HR
    polar_stats = {
        "mean_hr": int(np.mean(hr_values)),
        "max_hr": int(np.max(hr_values)),
        "min_hr": int(np.min(hr_values)),
        "hr_samples": len(hr_values)
    }
    
    print(f"   ✅ Polar HR Loaded: {len(hr_values)} samples | Avg: {polar_stats['mean_hr']} bpm | Range: {polar_stats['min_hr']}-{polar_stats['max_hr']} bpm")
    print(f"   ✅ Interpolated to {len(hr_interp)} samples for ECG alignment")
    
    return hr_interp, polar_stats


def load_acc_motion(acc_content, ecg_start_time, sampling_rate, n_samples):
    """
    Load accelerometer data from content string and produce motion flag.
    """
    if acc_content is None:
        return None, {"rest_ratio": 100.0, "active_ratio": 0.0}

    print("📡 Loading accelerometer data...")
    try:
        df_acc = pd.read_csv(StringIO(acc_content), sep=';', engine='python')
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


def calculate_global_stats(signals, rpeaks, sampling_rate, polar_hr_series=None, polar_hr_stats=None):
    """
    Calculate full comprehensive stats.
    Includes HR, HRV (SDNN, pNN50), and Frequency.
    Uses Polar HR data when available (more accurate), falls back to ECG-derived.
    """
    # --- A. Heart Rate Stats ---
    # Use Polar HR if available (more accurate), otherwise use ECG-derived
    if polar_hr_stats is not None:
        mean_hr = polar_hr_stats["mean_hr"]
        max_hr = polar_hr_stats["max_hr"]
        min_hr = polar_hr_stats["min_hr"]
        hr_source = "Polar"
        print(f"   📊 Using Polar HR data: {mean_hr} bpm (avg)")
    else:
        hr_source = "ECG-derived"
        print(f"   📊 Using ECG-derived HR (no Polar HR file)")
    
    # ECG-derived HR for comparison and detailed analysis
    ecg_hr_series = signals["ECG_Rate"].astype(float)
    ecg_mean_hr = int(ecg_hr_series.mean())
    ecg_max_hr = int(ecg_hr_series.max())
    ecg_min_hr = int(ecg_hr_series.min())
    
    # If no Polar data, use ECG-derived
    if polar_hr_stats is None:
        mean_hr = ecg_mean_hr
        max_hr = ecg_max_hr
        min_hr = ecg_min_hr
    
    # Use Polar HR series for rolling stats if available, else ECG
    hr_series = polar_hr_series if polar_hr_series is not None else ecg_hr_series
    hr_series = pd.Series(hr_series) if not isinstance(hr_series, pd.Series) else hr_series

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
        "hf": mean_hf,
        "hr_source": hr_source,
        "ecg_mean_hr": ecg_mean_hr,
        "ecg_max_hr": ecg_max_hr,
        "ecg_min_hr": ecg_min_hr
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


def detect_pattern_at_index(idx, rpeaks, beat_types, hr_series, sr, window_sec=10):
    """
    Detect rhythm patterns around a specific sample index.
    Returns pattern info including type, severity, and heart rate.
    """
    result = {
        "pattern": "Normal Sinus Rhythm",
        "severity": 0,
        "hr": None,
        "details": []
    }
    
    if len(rpeaks) == 0 or len(beat_types) == 0:
        return result
    
    # Find R-peaks within window around the index
    window_samples = window_sec * sr
    start_idx = max(0, idx - window_samples)
    end_idx = idx + window_samples
    
    # Find beat indices in window
    beats_in_window = []
    for i, rp in enumerate(rpeaks):
        if start_idx <= rp <= end_idx and i < len(beat_types):
            beats_in_window.append(i)
    
    if not beats_in_window:
        return result
    
    # Get beat types in window
    window_types = [beat_types[i] for i in beats_in_window if i < len(beat_types)]
    type_str = "".join(window_types)
    
    # Count pattern occurrences
    patterns_found = []
    
    # Check for various patterns
    if "SSS" in type_str:
        patterns_found.append(("Triplet (Premature x3)", 3))
    if "SS" in type_str:
        patterns_found.append(("Couplet (Premature x2)", 3))
    if "SLSL" in type_str or "SNSN" in type_str:
        patterns_found.append(("Bigeminy Pattern", 2))
    if "NNS" in type_str:
        patterns_found.append(("Trigeminy Pattern", 2))
    if "SL" in type_str and "SS" not in type_str:
        patterns_found.append(("Isolated Premature Beat", 1))
    
    # Count ectopic beats
    s_count = type_str.count("S")
    total_beats = len(type_str)
    ectopic_pct = (s_count / total_beats * 100) if total_beats > 0 else 0
    
    if patterns_found:
        # Get highest severity pattern
        patterns_found.sort(key=lambda x: x[1], reverse=True)
        result["pattern"] = patterns_found[0][0]
        result["severity"] = patterns_found[0][1]
        result["details"] = [p[0] for p in patterns_found]
    elif ectopic_pct > 10:
        result["pattern"] = "Frequent Ectopy"
        result["severity"] = 2
    elif s_count > 0:
        result["pattern"] = "Occasional Ectopy"
        result["severity"] = 1
    
    # Get heart rate at this point
    if hr_series is not None and idx < len(hr_series):
        hr_window = hr_series[max(0, idx-sr):min(len(hr_series), idx+sr)]
        if len(hr_window) > 0:
            result["hr"] = int(hr_window.mean())
    
    return result


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


def create_ecg_plot_base64(segment, sr, title, color_hex="#8D0A1E"):
    """
    Create an ECG plot and return it as base64 string (in-memory, no file I/O).
    """
    if len(segment) == 0: 
        return ""
    
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
    
    # Save to BytesIO instead of file
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return b64


def check_signal_quality(segment):
    # Simple heuristic: range too small (flatline) or too huge (artifact)
    r = np.max(segment) - np.min(segment)
    if r < 50 or r > 8000: 
        return False
    return True


def create_hr_trend_chart(hr_series, sr, start_time, duration_min, markers_data=None, 
                          ectopic_indices=None, polar_hr_series=None, motion_flag=None):
    """
    Create an HR trend chart showing heart rate over time with clinical annotations.
    
    Args:
        hr_series: ECG-derived HR (samples at ECG rate)
        sr: Sampling rate
        start_time: Recording start time
        duration_min: Duration in minutes
        markers_data: List of marker dicts with 'sample_index' and 'marker_label'
        ectopic_indices: Array of sample indices where ectopic beats occurred
        polar_hr_series: Polar HR data (if available) for comparison
        motion_flag: Boolean array indicating motion periods
    
    Returns:
        Base64 encoded PNG image
    """
    if hr_series is None or len(hr_series) == 0:
        return ""
    
    print("📈 Generating HR trend chart...")
    
    # Downsample for plotting (1 point per second is enough)
    downsample_factor = max(1, sr)
    hr_downsampled = hr_series[::downsample_factor]
    time_minutes = np.arange(len(hr_downsampled)) / 60.0  # Convert to minutes
    
    # Apply smoothing to ECG-derived HR to reduce noise from ectopic beats
    # Use rolling median (more robust to outliers than mean)
    window_size = min(15, len(hr_downsampled) // 4)  # ~15 second window
    if window_size > 1:
        hr_smoothed = pd.Series(hr_downsampled).rolling(window=window_size, center=True, min_periods=1).median().values
    else:
        hr_smoothed = hr_downsampled
    
    # Create figure with clinical styling
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # HR Zone backgrounds (clinical reference ranges)
    ax.axhspan(0, 50, alpha=0.15, color='#2196F3', label='Bradycardia (<50)')
    ax.axhspan(50, 60, alpha=0.08, color='#2196F3')
    ax.axhspan(60, 100, alpha=0.08, color='#4CAF50', label='Normal (60-100)')
    ax.axhspan(100, 150, alpha=0.08, color='#FF9800')
    ax.axhspan(150, 250, alpha=0.15, color='#F44336', label='Tachycardia (>150)')
    
    # Process Polar HR if available
    has_polar_hr = polar_hr_series is not None and len(polar_hr_series) > 0
    polar_smoothed = None
    min_len_polar = 0
    
    if has_polar_hr:
        polar_downsampled = polar_hr_series[::downsample_factor]
        # Apply light smoothing to Polar HR as well for cleaner visualization
        polar_window = min(5, len(polar_downsampled) // 4)
        if polar_window > 1:
            polar_smoothed = pd.Series(polar_downsampled).rolling(window=polar_window, center=True, min_periods=1).median().values
        else:
            polar_smoothed = polar_downsampled
        min_len_polar = min(len(polar_smoothed), len(time_minutes))
        print(f"   📊 Polar HR for chart: {len(polar_downsampled)} points, plotting {min_len_polar} points")
    
    # Calculate Y-axis limits FIRST to properly scale the chart
    min_len = min(len(hr_smoothed), len(time_minutes))
    all_hr_values = list(hr_smoothed[:min_len])
    if has_polar_hr and polar_smoothed is not None:
        all_hr_values.extend(list(polar_smoothed[:min_len_polar]))
    
    # Use combined data for Y-axis limits
    hr_min = max(30, np.nanpercentile(all_hr_values, 1) - 5)
    hr_max = min(200, np.nanpercentile(all_hr_values, 99) + 10)
    
    # Plot Polar HR if available (ground truth) - thicker, prominent
    if has_polar_hr and min_len_polar > 0:
        ax.plot(time_minutes[:min_len_polar], polar_smoothed[:min_len_polar], 
                color='#1565C0', linewidth=2.5, alpha=0.95, label='Polar HR (Validated)', zorder=5)
    
    # Plot smoothed ECG-derived HR (cleaner line)
    ax.plot(time_minutes[:min_len], hr_smoothed[:min_len], 
            color='#8D0A1E', linewidth=1.5, alpha=0.8, label='ECG HR (Smoothed)', zorder=4)
    
    # Shade motion periods if available
    if motion_flag is not None and len(motion_flag) > 0:
        motion_downsampled = motion_flag[::downsample_factor]
        min_len = min(len(motion_downsampled), len(time_minutes))
        for i in range(min_len - 1):
            if motion_downsampled[i]:
                ax.axvspan(time_minutes[i], time_minutes[i+1], 
                          alpha=0.1, color='#9E9E9E', linewidth=0)
    
    # Mark ectopic events as scatter points on the smoothed line
    if ectopic_indices is not None and len(ectopic_indices) > 0:
        ectopic_times = ectopic_indices / sr / 60.0  # Convert to minutes
        # Get smoothed HR at ectopic times (for better visual placement)
        ectopic_hr = []
        for idx in ectopic_indices:
            ds_idx = idx // downsample_factor
            if ds_idx < len(hr_smoothed):
                ectopic_hr.append(hr_smoothed[ds_idx])
            else:
                ectopic_hr.append(np.nan)
        
        valid_mask = ~np.isnan(ectopic_hr)
        if np.any(valid_mask):
            ax.scatter(np.array(ectopic_times)[valid_mask], np.array(ectopic_hr)[valid_mask],
                      color='#FF5722', s=30, alpha=0.7, marker='o', 
                      edgecolors='white', linewidths=0.5, label='Ectopic Events', zorder=6)
    
    # Add symptom markers as vertical lines with labels
    if markers_data:
        for i, marker in enumerate(markers_data):
            marker_time_min = marker.get('sample_index', 0) / sr / 60.0
            marker_label = marker.get('marker_label', f'Marker {i+1}')
            
            # Truncate long labels
            if len(marker_label) > 15:
                marker_label = marker_label[:12] + "..."
            
            ax.axvline(x=marker_time_min, color='#673AB7', linestyle='--', 
                      linewidth=2, alpha=0.8)
            
            # Add label at top
            y_pos = ax.get_ylim()[1] - 5
            ax.annotate(f'📍 {marker_label}', 
                       xy=(marker_time_min, y_pos),
                       fontsize=8, color='#673AB7', fontweight='bold',
                       rotation=90, ha='right', va='top')
    
    # Reference lines
    ax.axhline(y=60, color='#666', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=100, color='#666', linestyle=':', linewidth=1, alpha=0.5)
    
    # Styling
    ax.set_xlabel('Time (minutes)', fontsize=11, fontweight='500')
    ax.set_ylabel('Heart Rate (bpm)', fontsize=11, fontweight='500')
    ax.set_title('Heart Rate Trend', fontsize=14, fontweight='bold', color='#333')
    
    ax.set_xlim(0, duration_min)
    
    # Y limits already calculated above to encompass both HR series
    ax.set_ylim(hr_min, hr_max)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    # Add time annotations every 5 minutes
    for t in range(0, int(duration_min) + 1, 5):
        if t > 0 and t < duration_min:
            actual_time = start_time + timedelta(minutes=t)
            ax.annotate(actual_time.strftime('%H:%M'), 
                       xy=(t, hr_min + 2), fontsize=7, color='#999', ha='center')
    
    plt.tight_layout()
    
    # Save to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    print("   ✅ HR trend chart generated")
    return b64


# ==========================================
# 5. MAIN PROCESSING FUNCTION
# ==========================================
def process_ecg_data(ecg_content, marker_content=None, acc_content=None, hr_content=None, report_id="report"):
    """
    Main processing function that returns a result dict with report data.
    Works with file content strings instead of file paths.
    Returns structured data for database storage including marker patterns.
    
    Args:
        ecg_content: ECG file content string
        marker_content: Marker file content string (optional)
        acc_content: Accelerometer file content string (optional)
        hr_content: Polar HR file content string (optional) - provides validated HR data
        report_id: Unique report identifier
    """
    result = {
        "success": False,
        "error": None,
        "date": None,
        "duration_min": 0,
        "mean_hr": 0,
        "max_hr": 0,
        "min_hr": 0,
        "ectopic_burden": 0,
        "sdnn": 0,
        "pnn50": 0,
        "sampling_rate": 0,
        "sample_count": 0,
        "report_html": None,
        "baseline_plot_base64": None,
        "motion_stats": {"rest_ratio": 100.0, "active_ratio": 0.0},
        "markers": []  # List of marker data with patterns
    }

    try:
        # 1. Load from content strings
        _, ecg_raw, sr, start_time = load_and_sync_data(ecg_content, marker_content)
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

        # 3. Motion Gating (using content string now)
        motion_flag, motion_stats = load_acc_motion(acc_content, start_time, sr, len(ecg_clean))

        # 3b. Load Polar HR data if available
        polar_hr_series, polar_hr_stats = load_hr_data(hr_content, start_time, sr, len(ecg_clean))
        
        # Debug: Verify polar data was loaded
        if polar_hr_series is not None:
            print(f"   ✅ Polar HR interpolated: {len(polar_hr_series)} samples for chart")

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
        global_stats = calculate_global_stats(signals, rpeaks, sr, polar_hr_series, polar_hr_stats)
        
        result["mean_hr"] = global_stats["mean_hr"]
        result["max_hr"] = global_stats["max_hr"]
        result["min_hr"] = global_stats["min_hr"]
        result["sdnn"] = global_stats["sdnn"]
        result["pnn50"] = global_stats["pnn50"]
        result["sampling_rate"] = sr
        result["sample_count"] = len(ecg_raw)
        result["motion_stats"] = motion_stats
        result["hr_source"] = global_stats.get("hr_source", "ECG-derived")
        result["ecg_mean_hr"] = global_stats.get("ecg_mean_hr", 0)
        result["ecg_max_hr"] = global_stats.get("ecg_max_hr", 0)
        result["ecg_min_hr"] = global_stats.get("ecg_min_hr", 0)
        
        # Get heart rate series for marker pattern detection (use Polar if available)
        hr_series = polar_hr_series if polar_hr_series is not None else (signals["ECG_Rate"].astype(float) if "ECG_Rate" in signals else None)
        
        # 7. Process Markers with Pattern Detection
        markers_html = ""
        processed_markers = []
        df_markers = load_markers(marker_content, start_time, sr)
        if df_markers is not None and not df_markers.empty:
            print(f"   📍 Generating {len(df_markers)} marker plots with pattern detection...")
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
                            
                            # Detect pattern around this marker (now includes beat_markers for cursors)
                            pattern_info = detect_pattern_at_index(idx, rpeaks, beat_types, hr_series, sr)
                            
                            label_text = str(row[l_col])
                            
                            # Generate plot with pattern info in title
                            plot_title = f"SYMPTOM MARKER: {label_text}"
                            if pattern_info["pattern"] != "Normal Sinus Rhythm":
                                plot_title += f" | {pattern_info['pattern']}"
                            
                            b64 = create_ecg_plot_base64(seg, sr, plot_title, "#673AB7")
                            
                            # Store marker data for database
                            marker_data = {
                                "marker_time": m_time_naive.isoformat(),
                                "marker_label": label_text,
                                "sample_index": idx,
                                "detected_pattern": pattern_info["pattern"],
                                "pattern_severity": pattern_info["severity"],
                                "hr_at_marker": pattern_info["hr"],
                                "ecg_plot_base64": b64
                            }
                            processed_markers.append(marker_data)
                            
                            # Generate HTML with pattern info
                            severity_colors = {0: "#4CAF50", 1: "#FFC107", 2: "#FF9800", 3: "#F44336"}
                            severity_labels = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Significant"}
                            sev_color = severity_colors.get(pattern_info["severity"], "#673AB7")
                            sev_label = severity_labels.get(pattern_info["severity"], "Unknown")
                            
                            markers_html += f"""
                            <div class='event-box' style='border-left: 5px solid #673AB7;'>
                                <div class='event-header' style='background: #f3e5f5;'>
                                    <h3 style='color: #673AB7;'>📍 Symptom: {label_text}</h3>
                                    <p>Time: {m_time_naive.strftime('%H:%M:%S')}</p>
                                    <div style='margin-top: 10px; padding: 8px; background: white; border-radius: 6px;'>
                                        <p style='margin: 0;'><strong>🔍 Detected Pattern:</strong> 
                                            <span style='color: {sev_color}; font-weight: bold;'>{pattern_info["pattern"]}</span>
                                        </p>
                                        <p style='margin: 5px 0 0 0; font-size: 0.9rem; color: #666;'>
                                            Severity: <span style='color: {sev_color};'>{sev_label}</span>
                                            {f" | HR: {pattern_info['hr']} bpm" if pattern_info['hr'] else ""}
                                        </p>
                                    </div>
                                </div>
                                <div class='plot-container' style='margin: 0; border: none;'>
                                    <img src='data:image/png;base64,{b64}' alt='Marker Plot'>
                                </div>
                            </div>
                            """
                    except Exception as e:
                        print(f"   ⚠️ Error processing marker {i}: {e}")
        
        result["markers"] = processed_markers

        # 8. Generate Visuals (all in-memory now)
        
        # HR Trend Chart - generate with all clinical annotations
        ecg_hr_series = signals["ECG_Rate"].astype(float) if "ECG_Rate" in signals else None
        
        # Get ectopic beat indices for overlay
        ectopic_indices = np.where(ectopic_mask)[0] if ectopic_mask is not None else None
        # Convert R-peak indices to sample indices
        if ectopic_indices is not None and len(ectopic_indices) > 0:
            ectopic_sample_indices = rpeaks[ectopic_indices[ectopic_indices < len(rpeaks)]]
        else:
            ectopic_sample_indices = None
        
        # Debug: Check if polar_hr_series is available
        if polar_hr_series is not None:
            print(f"   📊 Polar HR series ready: {len(polar_hr_series)} samples, range {np.min(polar_hr_series):.0f}-{np.max(polar_hr_series):.0f} bpm")
        else:
            print("   ⚠️ No Polar HR series available for chart")
        
        b64_hr_trend = create_hr_trend_chart(
            hr_series=ecg_hr_series,
            sr=sr,
            start_time=start_time,
            duration_min=result["duration_min"],
            markers_data=processed_markers,
            ectopic_indices=ectopic_sample_indices,
            polar_hr_series=polar_hr_series,
            motion_flag=motion_flag
        )
        
        # Baseline Plot - generate to base64 directly
        b64_base = create_ecg_plot_base64(baseline_seg, sr, "BASELINE (Cleanest Resting Segment)", "#2E7D32")
        result["baseline_plot_base64"] = b64_base
        
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
                time_str = (start_time + timedelta(seconds=c/sr)).strftime("%H:%M:%S")
                
                # Generate plot to base64 directly
                b64 = create_ecg_plot_base64(seg, sr, f"{e['type']} @ {time_str}")
                
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

        # 9. Responsive HTML Generation (no file I/O)
        html = generate_report_html(
            report_id=report_id,
            start_time=start_time,
            ecg_clean=ecg_clean,
            sr=sr,
            global_stats=global_stats,
            global_burden=global_burden,
            motion_stats=motion_stats,
            baseline_metrics=baseline_metrics,
            b64_base=b64_base,
            b64_hr_trend=b64_hr_trend,
            markers_html=markers_html,
            events_html=events_html,
            pattern_rows=pattern_rows
        )
        
        # Return HTML in result instead of saving to file
        result["report_html"] = html
        print(f"\n✅ Report generated for {report_id}")
        
        result["success"] = True
        return result
        
    except Exception as e:
        import traceback
        result["error"] = str(e)
        print(f"❌ Processing error: {e}")
        traceback.print_exc()
        return result


def generate_report_html(report_id, start_time, ecg_clean, sr, global_stats, global_burden, 
                         motion_stats, baseline_metrics, b64_base, b64_hr_trend,
                         markers_html, events_html, pattern_rows):
    """Generate the HTML report content."""
    
    # Generate HR source indicator
    hr_source = global_stats.get("hr_source", "ECG-derived")
    hr_source_badge = ""
    if hr_source == "Polar HR File":
        hr_source_badge = "<span style='background: #e3f2fd; color: #1565c0; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; margin-left: 8px;'>✓ Polar Validated</span>"
    else:
        hr_source_badge = "<span style='background: #fff3e0; color: #e65100; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; margin-left: 8px;'>ECG-Derived</span>"
    
    # HR trend section
    hr_trend_section = ""
    if b64_hr_trend:
        hr_trend_section = f"""
            <h2>📈 Heart Rate Trend {hr_source_badge}</h2>
            <p style="font-size: 0.9rem; color: #666; margin-bottom: 15px;">
                Heart rate over time with symptom markers and ectopic events highlighted.
                {'''<br><strong style="color: #1565C0;">Blue line</strong>: Polar HR (validated) | 
                <strong style="color: #8D0A1E;">Red line</strong>: ECG-derived HR (smoothed)''' if global_stats.get("hr_source") == "Polar HR File" else 
                '<br><em>ECG-derived HR shown with smoothing to reduce noise from ectopic beats.</em>'}
            </p>
            <div class="plot-container">
                <img src="data:image/png;base64,{b64_hr_trend}" alt="HR Trend Chart">
            </div>
            <div class="hr-legend" style="display: flex; flex-wrap: wrap; gap: 15px; margin-top: 10px; font-size: 0.85rem; padding: 10px; background: #f9f9f9; border-radius: 8px;">
                <span><span style="display: inline-block; width: 12px; height: 12px; background: rgba(33,150,243,0.3); border: 1px solid #2196F3; margin-right: 5px;"></span>Bradycardia (&lt;50)</span>
                <span><span style="display: inline-block; width: 12px; height: 12px; background: rgba(76,175,80,0.15); border: 1px solid #4CAF50; margin-right: 5px;"></span>Normal (60-100)</span>
                <span><span style="display: inline-block; width: 12px; height: 12px; background: rgba(244,67,54,0.3); border: 1px solid #F44336; margin-right: 5px;"></span>Tachycardia (&gt;150)</span>
                <span><span style="display: inline-block; width: 12px; height: 12px; background: #673AB7; margin-right: 5px;"></span>Symptom Markers</span>
                <span><span style="display: inline-block; width: 10px; height: 10px; background: #FF5722; border-radius: 50%; margin-right: 5px;"></span>Ectopic Events</span>
            </div>
            <p style="font-size: 0.8rem; color: #888; margin-top: 10px; font-style: italic;">
                💡 <strong>Note:</strong> The faint red shading shows raw beat-to-beat HR variability caused by ectopic beats. 
                The solid red line is smoothed (15-second median) to show the underlying trend.
            </p>
        """
    
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
            
            /* Action buttons */
            .action-bar {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
                flex-wrap: wrap;
                gap: 0.5rem;
            }}
            .btn {{
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                font-weight: 500;
                text-decoration: none;
                cursor: pointer;
                border: none;
                font-size: 0.9rem;
                transition: all 0.2s;
            }}
            .btn-secondary {{
                background: var(--light-bg);
                color: var(--primary-color);
                border: 1px solid var(--border-color);
            }}
            .btn-secondary:hover {{
                background: #e0e0e0;
            }}
            .btn-primary {{
                background: var(--primary-color);
                color: white;
            }}
            .btn-primary:hover {{
                background: var(--primary-light);
            }}
            .btn:disabled {{
                opacity: 0.6;
                cursor: not-allowed;
            }}

            /* Responsive Tweaks */
            @media (max-width: 600px) {{
                body {{ padding: 15px; }}
                h1 {{ font-size: 1.5rem; }}
                h2 {{ font-size: 1.2rem; }}
                .metric-box {{ grid-template-columns: 1fr 1fr; gap: 10px; padding: 15px; }}
                .val {{ font-size: 1.3rem; }}
                .action-bar {{ flex-direction: column; align-items: stretch; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="action-bar">
                <a href="/calendar" class="btn btn-secondary">← Back to Calendar</a>
                <form action="/report/{report_id}/recalculate" method="post" style="margin: 0;" onsubmit="this.querySelector('button').disabled=true; this.querySelector('button').innerHTML='⏳ Recalculating...';">
                    <button type="submit" class="btn btn-primary">🔄 Recalculate Report</button>
                </form>
            </div>
            
            <h1>🏥 Clinical ECG Report <span style="font-size: 1rem; font-weight: normal; color: #666;">(Heart Diary)</span></h1>
            <p style="color: #555;">
                <b>Date:</b> {start_time.strftime('%Y-%m-%d %H:%M')}<br>
                <b>Duration:</b> {len(ecg_clean)/sr/60:.1f} min
            </p>
            
            <h2>📊 Session Metrics</h2>
            <div class="metric-box">
                <div class="metric"><div class="val">{global_stats['mean_hr']}</div><div class="lbl">Avg HR (bpm)<br><small style="color:#888;font-size:0.7rem;">{global_stats.get('hr_source', 'ECG')}</small></div></div>
                <div class="metric"><div class="val">{global_stats['max_hr']}</div><div class="lbl">Max HR</div></div>
                <div class="metric"><div class="val">{global_burden:.2f}%</div><div class="lbl">Ectopic Burden</div></div>
                <div class="metric"><div class="val">{motion_stats['rest_ratio']:.1f}%</div><div class="lbl">Rest Time</div></div>
            </div>
            
            <h3>📈 Detailed Statistics</h3>
            <table class="stats-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>HR Source</td><td><strong>{global_stats.get('hr_source', 'ECG-derived')}</strong></td></tr>
                <tr><td>HRV (SDNN)</td><td>{global_stats['sdnn']} ms</td></tr>
                <tr><td>pNN50 (Var.)</td><td>{global_stats['pnn50']} %</td></tr>
                <tr><td>High 1-min HR</td><td>{global_stats['high_1min']} bpm</td></tr>
                <tr><td>Low 1-min HR</td><td>{global_stats['low_1min']} bpm</td></tr>
                <tr><td>HF Power (Norm)</td><td>{global_stats['hf']} %</td></tr>
                <tr><td>Tachycardia Time (>150)</td><td>{global_stats['tachy_perc']:.1f} %</td></tr>
                <tr><td>Bradycardia Time (<50)</td><td>{global_stats['brady_perc']:.1f} %</td></tr>
                <tr><th colspan='2' style='background: #e3f2fd; color: #1565c0; font-size: 0.85rem; padding-top: 10px;'>ECG-Derived HR (for comparison)</th></tr>
                <tr><td>ECG Avg HR</td><td>{global_stats.get('ecg_mean_hr', 'N/A')} bpm</td></tr>
                <tr><td>ECG Max HR</td><td>{global_stats.get('ecg_max_hr', 'N/A')} bpm</td></tr>
                <tr><td>ECG Min HR</td><td>{global_stats.get('ecg_min_hr', 'N/A')} bpm</td></tr>
                {pattern_rows}
            </table>
            
            {hr_trend_section}
            
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
