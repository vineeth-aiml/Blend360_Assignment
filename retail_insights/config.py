"""
Configuration module for Retail Insights Assistant
"""
import os
from dotenv import load_dotenv

# Force .env to override stale Windows/session environment values.
load_dotenv(override=True)


def clean_env(value: str | None) -> str:
    return (value or "").strip().strip('"').strip("'")


# API Keys
GEMINI_API_KEY = clean_env(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

# Data Configuration
DATA_PATH = clean_env(os.getenv("DATA_PATH")) or "Sales Dataset/"

# Model Configuration
MODEL_NAME = clean_env(os.getenv("MODEL_NAME")) or "gemini-2.5-flash"
GEMINI_FALLBACK_MODEL = clean_env(os.getenv("GEMINI_FALLBACK_MODEL")) or "gemini-2.5-flash-lite"
TEMPERATURE = float(clean_env(os.getenv("TEMPERATURE")) or "0.1")

# Agent Configuration
MAX_ITERATIONS = 10
AGENT_TIMEOUT = 120  # seconds

# Database Configuration
DB_PATH = ":memory:"  # Use in-memory DuckDB for faster queries

# Logging Configuration
LOG_LEVEL = "INFO"
