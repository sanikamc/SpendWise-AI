from datetime import datetime, date
import json
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model):
    """
    User account model supporting secure authentication with scrypt password hashing.
    """
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships - each user owns their expenses and budgets
    expenses = db.relationship("Expense", backref="user", lazy=True, cascade="all, delete-orphan")
    budgets = db.relationship("Budget", backref="user", lazy=True, cascade="all, delete-orphan")

    def set_password(self, password: str) -> None:
        """Securely hash and store the user's password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Verify the password against the stored scrypt hash."""
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
        }

    def __repr__(self):
        return f"<User #{self.id} {self.username}>"


class Expense(db.Model):
    """
    Represents an individual expense transaction.
    Money is stored in integer paise (1 Rupee = 100 Paise) to avoid floating-point errors.
    Nullable user_id allows safe coexistence of legacy unassigned records.
    """
    __tablename__ = "expenses"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    description = db.Column(db.String(255), nullable=False)
    amount_paise = db.Column(db.Integer, nullable=False)  # Stored as integer paise
    category = db.Column(db.String(50), nullable=False, index=True)
    date = db.Column(db.Date, nullable=False, index=True)
    payment_method = db.Column(db.String(50), nullable=False)
    notes = db.Column(db.Text, nullable=True)
    
    # Anomaly Detection results
    is_anomaly = db.Column(db.Boolean, default=False, index=True)
    anomaly_score = db.Column(db.Float, nullable=True)  # Normalized anomaly index (0.0 to 1.0)
    raw_score = db.Column(db.Float, nullable=True)      # Raw decision_function score
    anomaly_label = db.Column(db.String(100), default="Pending assessment")
    anomaly_reasons = db.Column(db.Text, nullable=True) # JSON structured reasons & comparison
    
    # Alert Review Workflow: 'pending', 'reviewed', 'dismissed', 'none'
    review_status = db.Column(db.String(20), default="none", index=True)
    
    # Demonstration data flag
    is_demo = db.Column(db.Boolean, default=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def amount(self) -> float:
        """Returns amount in Indian Rupees (float)."""
        return self.amount_paise / 100.0

    @amount.setter
    def amount(self, val: float):
        """Sets amount from Indian Rupees into paise."""
        self.amount_paise = int(round(float(val) * 100))

    def get_reasons_dict(self):
        """Parse anomaly reasons JSON safely."""
        if not self.anomaly_reasons:
            return {}
        try:
            return json.loads(self.anomaly_reasons)
        except Exception:
            return {"note": str(self.anomaly_reasons)}

    def to_dict(self):
        """Convert expense to dictionary for JSON APIs and reports."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "description": self.description,
            "amount": self.amount,
            "amount_formatted": f"₹{self.amount:,.2f}",
            "category": self.category,
            "date": self.date.strftime("%Y-%m-%d"),
            "payment_method": self.payment_method,
            "notes": self.notes or "",
            "is_anomaly": self.is_anomaly,
            "anomaly_score": round(self.anomaly_score, 3) if self.anomaly_score is not None else None,
            "raw_score": round(self.raw_score, 4) if self.raw_score is not None else None,
            "anomaly_label": self.anomaly_label,
            "reasons": self.get_reasons_dict(),
            "review_status": self.review_status,
            "is_demo": self.is_demo,
        }

    def __repr__(self):
        return f"<Expense #{self.id} {self.description} ₹{self.amount:.2f} ({self.category})>"


class Budget(db.Model):
    """
    Represents a monthly budget set by the user.
    Enforces user-scoped uniqueness: (user_id, year, month).
    """
    __tablename__ = "budgets"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    year = db.Column(db.Integer, nullable=False)
    month = db.Column(db.Integer, nullable=False)  # 1 - 12
    amount_paise = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("user_id", "year", "month", name="uq_user_budget_year_month"),
    )

    @property
    def amount(self) -> float:
        return self.amount_paise / 100.0

    @amount.setter
    def amount(self, val: float):
        self.amount_paise = int(round(float(val) * 100))

    def __repr__(self):
        return f"<Budget {self.year}-{self.month:02d}: ₹{self.amount:.2f}>"
