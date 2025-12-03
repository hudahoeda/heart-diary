"""
Database Connection Management
Handles SQLAlchemy engine and session configuration.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Database URL from environment or default to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/heartdiary.db")

# Fix for SQLAlchemy 1.4+ compatibility: postgres:// -> postgresql://
# Some platforms (like Digital Ocean, Heroku) provide postgres:// which is deprecated
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


def get_engine():
    """Create and return database engine based on DATABASE_URL."""
    if DATABASE_URL.startswith("sqlite"):
        return create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    else:
        return create_engine(DATABASE_URL)


# Create engine and session factory
engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency for FastAPI to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
