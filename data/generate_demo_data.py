"""
data/generate_demo_data.py - Synthetic Demonstration Data Generator
==================================================================
Generates realistic Indian personal expense data with normal daily spending
patterns and realistic anomalous transactions for college viva demonstration.

All generated records are tagged with `is_demo = True` so they can be
clearly separated, reviewed, or cleared at any time.
"""

from datetime import date, datetime, timedelta
import random
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from database import db, Expense, Budget
from model import detector


def generate_synthetic_expenses(target_date: date = None) -> list[dict]:
    """
    Creates a balanced synthetic dataset of typical expenses across 60 days
    with 4 deliberate, realistic outlier scenarios.
    """
    if target_date is None:
        target_date = date.today()

    random.seed(42)  # Reproducible seed for demo consistency

    normal_templates = [
        # (Description, Category, Min_INR, Max_INR, Methods)
        ("Grocery & Vegetables", "Food", 250, 950, ["UPI", "Cash", "Debit Card"]),
        ("College Canteen Lunch", "Food", 80, 220, ["UPI", "Cash"]),
        ("Coffee & Snacks with Friends", "Food", 120, 380, ["UPI", "Cash"]),
        ("Swiggy / Zomato Dinner", "Food", 280, 650, ["UPI", "Credit Card"]),
        ("Metro Card Recharge", "Transport", 200, 500, ["UPI", "Debit Card"]),
        ("Auto Rickshaw Fare", "Transport", 40, 150, ["UPI", "Cash"]),
        ("Petrol Refuel", "Transport", 300, 800, ["UPI", "Credit Card"]),
        ("Bus Ticket Fare", "Transport", 25, 80, ["Cash", "UPI"]),
        ("Mobile Postpaid Bill", "Bills", 499, 799, ["UPI", "Net Banking"]),
        ("Home WiFi Broadband", "Bills", 799, 1199, ["Net Banking", "UPI"]),
        ("Electricity Utility Bill", "Bills", 1100, 2400, ["Net Banking", "UPI"]),
        ("College Reference Books", "Education", 450, 1400, ["UPI", "Debit Card"]),
        ("Lab Manuals & Stationery", "Education", 120, 350, ["Cash", "UPI"]),
        ("Pharmacy Medicines", "Healthcare", 180, 650, ["UPI", "Cash"]),
        ("Routine Dental Checkup", "Healthcare", 500, 900, ["UPI", "Debit Card"]),
        ("Weekend Movie Ticket", "Entertainment", 250, 550, ["Credit Card", "UPI"]),
        ("OTT Subscription", "Entertainment", 199, 499, ["Credit Card", "UPI"]),
        ("Casual T-Shirt / Apparel", "Shopping", 600, 1600, ["Credit Card", "UPI"]),
        ("Supermarket Household Essentials", "Shopping", 350, 1200, ["Debit Card", "UPI"]),
        ("Train Ticket to Hometown", "Travel", 450, 1200, ["Net Banking", "UPI"]),
    ]

    records = []

    # Generate 50 normal transactions spread over the last 60 days
    for day_offset in range(58, 0, -1):
        # 1 or 2 expenses on some days
        if random.random() < 0.75:
            current_day = target_date - timedelta(days=day_offset)
            tmpl = random.choice(normal_templates)
            desc, cat, min_amt, max_amt, methods = tmpl
            amt = round(random.uniform(min_amt, max_amt), 2)
            method = random.choice(methods)
            notes = "Demonstration expense"
            records.append({
                "description": desc,
                "amount": amt,
                "category": cat,
                "date": current_day,
                "payment_method": method,
                "notes": notes,
                "is_demo": True,
            })

    # Add deliberate, realistic anomalous expenses
    anomalies = [
        {
            "description": "High-End Laptop for CSE Projects",
            "amount": 42500.00,
            "category": "Shopping",
            "date": target_date - timedelta(days=6),
            "payment_method": "Credit Card",
            "notes": "[DEMO ANOMALY] Major tech hardware purchase significantly exceeding typical shopping average.",
            "is_demo": True,
        },
        {
            "description": "Premium Luxury Five-Star Banquet",
            "amount": 8900.00,
            "category": "Food",
            "date": target_date - timedelta(days=12),
            "payment_method": "Credit Card",
            "notes": "[DEMO ANOMALY] Group celebratory dining bill 15x higher than typical food expense.",
            "is_demo": True,
        },
        {
            "description": "Emergency Diagnostic Hospitalization Scan",
            "amount": 16500.00,
            "category": "Healthcare",
            "date": target_date - timedelta(days=22),
            "payment_method": "Debit Card",
            "notes": "[DEMO ANOMALY] Sudden emergency medical diagnostics outside normal healthcare routine.",
            "is_demo": True,
        },
        {
            "description": "Last-Minute Flight Booking for Conference",
            "amount": 14200.00,
            "category": "Travel",
            "date": target_date - timedelta(days=34),
            "payment_method": "Net Banking",
            "notes": "[DEMO ANOMALY] Peak-fare flight booking substantially higher than standard travel spend.",
            "is_demo": True,
        },
    ]

    records.extend(anomalies)
    # Sort chronologically
    records.sort(key=lambda x: x["date"])
    return records


def seed_demo_data(app=None, user_id=None) -> dict:
    """
    Seeds synthetic demonstration data into the database and executes
    IsolationForest training, scoped to the specified user_id if provided.
    """
    from database import Expense, Budget, db
    import json

    today = date.today()
    cur_year = today.year
    cur_month = today.month

    # Set up budgets for current month and previous month
    # Previous month calculation
    prev_month = 12 if cur_month == 1 else cur_month - 1
    prev_year = cur_year - 1 if cur_month == 1 else cur_year

    # Check existing budgets or create while respecting uq_budget_year_month
    b_curr = Budget.query.filter_by(year=cur_year, month=cur_month).first()
    if not b_curr:
        b_curr = Budget(year=cur_year, month=cur_month, user_id=user_id, amount=40000.00)
        db.session.add(b_curr)
    else:
        if b_curr.user_id is None and user_id is not None:
            b_curr.user_id = user_id
        b_curr.amount = 40000.00

    b_prev = Budget.query.filter_by(year=prev_year, month=prev_month).first()
    if not b_prev:
        b_prev = Budget(year=prev_year, month=prev_month, user_id=user_id, amount=38000.00)
        db.session.add(b_prev)
    else:
        if b_prev.user_id is None and user_id is not None:
            b_prev.user_id = user_id
        b_prev.amount = 38000.00

    # Clean existing demo expenses for this user to allow clean re-seeding
    clean_query = Expense.query.filter_by(is_demo=True)
    if user_id is not None:
        clean_query = clean_query.filter_by(user_id=user_id)
    clean_query.delete(synchronize_session=False)
    db.session.commit()

    # Generate synthetic records
    raw_demo_records = generate_synthetic_expenses(today)

    # Save to database first to get IDs
    created_expenses = []
    for r in raw_demo_records:
        exp = Expense(
            user_id=user_id,
            description=r["description"],
            category=r["category"],
            date=r["date"],
            payment_method=r["payment_method"],
            notes=r["notes"],
            is_demo=True,
            review_status="none",
        )
        exp.amount = r["amount"]
        db.session.add(exp)
        created_expenses.append(exp)

    db.session.commit()

    # Run IsolationForest batch evaluation on user's expenses
    user_exp_query = Expense.query
    if user_id is not None:
        user_exp_query = user_exp_query.filter_by(user_id=user_id)
    all_expenses = user_exp_query.order_by(Expense.date.asc()).all()
    all_dicts = [e.to_dict() for e in all_expenses]

    metadata, assessments = detector.fit_and_predict_batch(all_dicts)

    # Update expenses with anomaly evaluations
    assessment_map = {a["id"]: a for a in assessments}
    for exp in created_expenses:
        eval_res = assessment_map.get(exp.id)
        if eval_res:
            exp.is_anomaly = eval_res["is_anomaly"]
            exp.anomaly_score = eval_res["anomaly_score"]
            exp.raw_score = eval_res["raw_score"]
            exp.anomaly_label = eval_res["anomaly_label"]
            exp.anomaly_reasons = json.dumps(eval_res["explanation"])
            exp.review_status = eval_res["review_status"]

    db.session.commit()

    return {
        "status": "success",
        "seeded_count": len(created_expenses),
        "anomalies_flagged": metadata.get("anomalies_detected", 0),
        "model_status": metadata.get("status"),
    }


def clear_demo_data(user_id=None) -> int:
    """Removes synthetic demonstration records from the database (user-scoped if provided)."""
    query = Expense.query.filter_by(is_demo=True)
    if user_id is not None:
        query = query.filter_by(user_id=user_id)
    count = query.delete(synchronize_session=False)
    db.session.commit()
    return count


if __name__ == "__main__":
    from app import create_app
    app = create_app()
    with app.app_context():
        res = seed_demo_data()
        print(f"Successfully generated demo data: {res}")
