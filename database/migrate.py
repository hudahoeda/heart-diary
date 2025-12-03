"""
Database Migration Runner
Handles applying SQL migrations to the database.
"""
import os
from pathlib import Path
from sqlalchemy import text
from .connection import engine, DATABASE_URL
from .models import Base


def get_migrations_dir():
    """Get the path to migrations directory."""
    return Path(__file__).parent / "migrations"


def run_migrations():
    """
    Run all pending migrations.
    For SQLite, uses the SQLite-specific migration files.
    For PostgreSQL, uses the standard migration files.
    """
    migrations_dir = get_migrations_dir()
    
    # Determine which migration file to use
    if DATABASE_URL.startswith("sqlite"):
        migration_file = migrations_dir / "001_initial_schema_sqlite.sql"
    else:
        migration_file = migrations_dir / "001_initial_schema.sql"
    
    if migration_file.exists():
        print(f"📦 Running migration: {migration_file.name}")
        with open(migration_file, "r") as f:
            sql_content = f.read()
        
        # Split by semicolon and execute each statement
        statements = [s.strip() for s in sql_content.split(';') if s.strip() and not s.strip().startswith('--')]
        
        with engine.connect() as conn:
            for statement in statements:
                if statement:
                    try:
                        conn.execute(text(statement))
                    except Exception as e:
                        # Skip errors for CREATE INDEX IF NOT EXISTS on SQLite
                        if "already exists" not in str(e).lower():
                            print(f"   ⚠️ Statement warning: {e}")
            conn.commit()
        
        print("✅ Migration completed successfully")
    else:
        print(f"❌ Migration file not found: {migration_file}")


def create_tables():
    """
    Create all tables using SQLAlchemy models.
    This is the preferred method for development.
    """
    print("📦 Creating database tables from models...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created successfully")


def drop_tables():
    """Drop all tables (use with caution!)."""
    print("⚠️ Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    print("✅ Tables dropped")


def reset_database():
    """Drop and recreate all tables."""
    drop_tables()
    create_tables()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m database.migrate [create|drop|reset|migrate]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create":
        create_tables()
    elif command == "drop":
        drop_tables()
    elif command == "reset":
        reset_database()
    elif command == "migrate":
        run_migrations()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
