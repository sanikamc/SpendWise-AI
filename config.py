import os
from pathlib import Path

# Base directory of the application
BASE_DIR = Path(__file__).resolve().parent

class Config:
    """Base application configuration."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "spendwise-secret-key-cse-project-2026")
    TEMPLATES_AUTO_RELOAD = True
    
    # SQLite Database configuration
    INSTANCE_PATH = BASE_DIR / "instance"
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{INSTANCE_PATH / 'expenses.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Application settings
    APP_NAME = "SpendWise AI"
    APP_SUBTITLE = "Smart Expense Anomaly Detection System"
    CURRENCY_SYMBOL = "₹"
    CURRENCY_CODE = "INR"
    
    # ML Model Settings
    MODEL_MIN_SAMPLES = 10  # Minimum transactions needed before running IsolationForest
    MODEL_CONTAMINATION = 0.08  # Estimated percentage of outliers (8%)
    MODEL_RANDOM_STATE = 42  # Seed for reproducible IsolationForest runs
    
    # Allowed Categories and Payment Methods
    EXPENSE_CATEGORIES = [
        "Food",
        "Transport",
        "Shopping",
        "Education",
        "Bills",
        "Healthcare",
        "Entertainment",
        "Travel",
        "Other",
    ]
    
    PAYMENT_METHODS = [
        "Cash",
        "UPI",
        "Debit Card",
        "Credit Card",
        "Net Banking",
        "Other",
    ]

class TestingConfig(Config):
    """Configuration for automated test suite."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False
