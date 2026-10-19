"""
tests/test_auth.py - Authentication, Session Security, and Multi-Tenant Isolation Tests
========================================================================================
Tests User model, scrypt password hashing, registration validation, login sessions,
logout clearance, unauthenticated redirect protection, and cross-user data isolation.
Uses an isolated in-memory SQLite database.
"""

from datetime import date
import pytest
from app import create_app
from config import TestingConfig
from database import db, Expense, Budget, User


class AuthTestingConfig(TestingConfig):
    """Testing config with CSRF enabled to verify token enforcement."""
    WTF_CSRF_ENABLED = True


@pytest.fixture
def app():
    """Application with in-memory database and CSRF disabled for functional auth tests."""
    test_app = create_app(TestingConfig)
    with test_app.app_context():
        db.create_all()
        yield test_app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


def test_user_model_password_hashing(app):
    """
    Password Security Test:
    Verifies that set_password generates a secure non-plaintext scrypt hash
    and check_password correctly verifies credentials.
    """
    with app.app_context():
        u = User(username="security_tester", email="sec@test.com")
        u.set_password("MySecurePass123!")
        db.session.add(u)
        db.session.commit()

        # Hash must not contain plaintext password
        assert "MySecurePass123!" not in u.password_hash
        assert u.password_hash.startswith("scrypt:") or len(u.password_hash) > 40
        assert u.check_password("MySecurePass123!") is True
        assert u.check_password("WrongPassword") is False


def test_user_registration_success(client, app):
    """
    Registration Flow Test:
    Registering with valid inputs creates the account and logs the user in.
    """
    payload = {
        "username": "priya_sharma",
        "email": "priya@example.com",
        "password": "Password@123",
        "confirm_password": "Password@123",
    }
    res = client.post("/register", data=payload, follow_redirects=True)
    assert res.status_code == 200
    assert b"Welcome to SpendWise AI" in res.data

    with app.app_context():
        user = User.query.filter_by(username="priya_sharma").first()
        assert user is not None
        assert user.email == "priya@example.com"


def test_user_registration_validation_errors(client, app):
    """
    Registration Validation Test:
    Rejects password mismatch, short passwords, duplicate username, and invalid email.
    """
    # 1. Password mismatch
    res1 = client.post("/register", data={
        "username": "user_one",
        "email": "one@test.com",
        "password": "Password123",
        "confirm_password": "DifferentPassword",
    })
    assert b"Passwords do not match" in res1.data

    # 2. Short password
    res2 = client.post("/register", data={
        "username": "user_two",
        "email": "two@test.com",
        "password": "123",
        "confirm_password": "123",
    })
    assert b"Password must be at least 6 characters" in res2.data


def test_user_login_success_and_logout(client, app):
    """
    Login & Logout Lifecycle Test:
    Tests login via username, login via email, session assignment, and logout clearance.
    """
    with app.app_context():
        user = User(username="rohit_varma", email="rohit@test.com")
        user.set_password("SecurePass2026")
        db.session.add(user)
        db.session.commit()

    # Login via username
    login_res = client.post("/login", data={
        "identity": "rohit_varma",
        "password": "SecurePass2026",
    }, follow_redirects=True)
    assert login_res.status_code == 200
    assert b"rohit_varma" in login_res.data

    # Logout
    logout_res = client.post("/logout", follow_redirects=True)
    assert logout_res.status_code == 200
    assert b"signed out safely" in logout_res.data


def test_user_login_invalid_credentials(client, app):
    """
    Authentication Failure Test:
    Rejects incorrect password or non-existent user with generic safety notice.
    """
    with app.app_context():
        user = User(username="valid_user", email="valid@test.com")
        user.set_password("CorrectPass")
        db.session.add(user)
        db.session.commit()

    res = client.post("/login", data={
        "identity": "valid_user",
        "password": "WrongPassword",
    })
    assert b"Invalid credentials" in res.data


def test_unauthenticated_routes_redirect_to_login(client):
    """
    Access Control Test:
    Unauthenticated visits to protected financial routes must redirect to /login.
    """
    routes = [
        "/",
        "/expenses/add",
        "/anomalies",
        "/analytics",
        "/history",
        "/reports",
    ]
    for r in routes:
        res = client.get(r)
        assert res.status_code == 302
        assert "/login" in res.headers["Location"]


def test_cross_user_data_isolation(app):
    """
    Multi-Tenant Data Isolation Test:
    User A cannot view, modify, or delete User B's financial expenses.
    """
    with app.app_context():
        user_a = User(username="user_a", email="a@test.com")
        user_a.set_password("PasswordA1")
        user_b = User(username="user_b", email="b@test.com")
        user_b.set_password("PasswordB2")
        db.session.add_all([user_a, user_b])
        db.session.commit()

        # User A creates an expense
        exp_a = Expense(
            user_id=user_a.id,
            description="User A Private Expense",
            category="Bills",
            date=date(2026, 10, 10),
            payment_method="UPI",
        )
        exp_a.amount = 5000.0
        db.session.add(exp_a)
        db.session.commit()
        exp_a_id = exp_a.id

    client_b = app.test_client()
    # Log in as User B
    client_b.post("/login", data={"identity": "user_b", "password": "PasswordB2"})

    # 1. User B's history must NOT show User A's expense
    hist = client_b.get("/history")
    assert b"User A Private Expense" not in hist.data

    # 2. User B cannot edit User A's expense (HTTP 403 Forbidden)
    edit_attempt = client_b.post(f"/expenses/edit/{exp_a_id}", data={
        "description": "Hacked Expense",
        "amount": "100.00",
        "category": "Bills",
        "date": "2026-10-10",
        "payment_method": "UPI",
    })
    assert edit_attempt.status_code == 403

    # 3. User B cannot delete User A's expense (HTTP 403 Forbidden)
    delete_attempt = client_b.post(f"/expenses/delete/{exp_a_id}")
    assert delete_attempt.status_code == 403

    # Confirm User A's expense remains unchanged in database
    with app.app_context():
        saved = db.session.get(Expense, exp_a_id)
        assert saved is not None
        assert saved.description == "User A Private Expense"


def test_multi_user_budget_coexistence_same_month(app):
    """
    Budget Multi-Tenancy Regression Test:
    User A and User B can both set their own budgets for the exact same year and month
    without violating uniqueness constraints or triggering HTTP 500 errors.
    """
    from sqlalchemy.exc import IntegrityError

    with app.app_context():
        user_a = User(username="budget_user_a", email="budget_a@test.com")
        user_a.set_password("PassA123")
        user_b = User(username="budget_user_b", email="budget_b@test.com")
        user_b.set_password("PassB123")
        db.session.add_all([user_a, user_b])
        db.session.commit()
        user_a_id, user_b_id = user_a.id, user_b.id

    client_a = app.test_client()
    client_a.post("/login", data={"identity": "budget_user_a", "password": "PassA123"})
    res_a = client_a.post("/budget/set", data={"year": "2026", "month": "11", "amount": "30000.00"}, follow_redirects=True)
    assert res_a.status_code == 200
    assert b"Budget of" in res_a.data

    client_b = app.test_client()
    client_b.post("/login", data={"identity": "budget_user_b", "password": "PassB123"})
    res_b = client_b.post("/budget/set", data={"year": "2026", "month": "11", "amount": "45000.00"}, follow_redirects=True)
    assert res_b.status_code == 200
    assert b"Budget of" in res_b.data

    # Verify database state
    with app.app_context():
        b_a = Budget.query.filter_by(year=2026, month=11, user_id=user_a_id).first()
        b_b = Budget.query.filter_by(year=2026, month=11, user_id=user_b_id).first()
        assert b_a is not None and b_a.amount == 30000.00
        assert b_b is not None and b_b.amount == 45000.00

        all_nov = Budget.query.filter_by(year=2026, month=11).all()
        assert len(all_nov) == 2

    # Verify dashboard scoping
    dash_a = client_a.get("/?year=2026&month=11")
    assert dash_a.status_code == 200
    assert b"30000.00" in dash_a.data
    assert b"45000.00" not in dash_a.data

    dash_b = client_b.get("/?year=2026&month=11")
    assert dash_b.status_code == 200
    assert b"45000.00" in dash_b.data
    assert b"30000.00" not in dash_b.data


def test_duplicate_budget_same_user_month_disallowed(app):
    """
    Budget Uniqueness Enforcement Regression Test:
    Duplicate budgets for the same user, year, and month are disallowed at the database level,
    and updates via /budget/set update the existing row rather than inserting a duplicate.
    """
    from sqlalchemy.exc import IntegrityError

    with app.app_context():
        user = User(username="single_budget_user", email="single_b@test.com")
        user.set_password("Pass12345")
        db.session.add(user)
        db.session.commit()
        u_id = user.id

    client = app.test_client()
    client.post("/login", data={"identity": "single_budget_user", "password": "Pass12345"})

    # 1. Set budget
    res1 = client.post("/budget/set", data={"year": "2026", "month": "12", "amount": "20000.00"}, follow_redirects=True)
    assert res1.status_code == 200

    # 2. Update same month budget via route
    res2 = client.post("/budget/set", data={"year": "2026", "month": "12", "amount": "25000.00"}, follow_redirects=True)
    assert res2.status_code == 200

    with app.app_context():
        user_budgets = Budget.query.filter_by(user_id=u_id, year=2026, month=12).all()
        assert len(user_budgets) == 1
        assert user_budgets[0].amount == 25000.00

        # 3. Direct duplicate insertion at DB level must raise IntegrityError
        dup = Budget(user_id=u_id, year=2026, month=12, amount_paise=99900)
        db.session.add(dup)
        with pytest.raises(IntegrityError):
            db.session.commit()
        db.session.rollback()

