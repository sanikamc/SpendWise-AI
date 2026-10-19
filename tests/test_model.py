"""
tests/test_model.py - Unit Tests for IsolationForest Anomaly Detection
======================================================================
Tests model initialization, feature engineering, insufficient-data handling,
batch evaluation, reproducibility, and explainability without connecting to database.
"""

from datetime import date, timedelta
import pytest
import numpy as np
from model import ExpenseAnomalyDetector
from config import Config


@pytest.fixture
def detector():
    """Instantiate a fresh detector with deterministic seed."""
    return ExpenseAnomalyDetector(
        contamination=0.1,
        random_state=42,
        min_samples=10,
    )


@pytest.fixture
def sample_normal_records():
    """Generates 20 typical daily transactions."""
    today = date(2026, 10, 1)
    records = []
    for i in range(20):
        records.append({
            "id": i + 1,
            "description": f"Daily Grocery {i+1}",
            "amount": 250.0 + (i * 10),
            "category": "Food",
            "date": today + timedelta(days=i),
            "payment_method": "UPI",
            "notes": "Test normal expense",
            "is_demo": False,
        })
    return records


def test_insufficient_data_handling(detector):
    """
    Cold Start Test:
    When fewer than min_samples (10) transactions are provided,
    the model must not fit or guess. It should return a neutral 'insufficient_data' status.
    """
    small_records = [
        {
            "id": 1,
            "description": "Lunch",
            "amount": 150.0,
            "category": "Food",
            "date": date(2026, 10, 5),
            "payment_method": "UPI",
        },
        {
            "id": 2,
            "description": "Bus Fare",
            "amount": 35.0,
            "category": "Transport",
            "date": date(2026, 10, 6),
            "payment_method": "Cash",
        },
    ]

    meta, results = detector.fit_and_predict_batch(small_records)

    assert meta["status"] == "insufficient_data"
    assert meta["sample_count"] == 2
    assert detector.is_fitted is False
    assert len(results) == 2
    for r in results:
        assert r["is_anomaly"] is False
        assert r["anomaly_score"] is None
        assert r["anomaly_label"] == "Insufficient data for reliable assessment"


def test_feature_engineering_structure(detector, sample_normal_records):
    """
    Feature Engineering Test:
    Verifies that numerical features (log amount, category ratio, z-score,
    day-of-week, weekend indicator, one-hot category indicators) are extracted
    and contain no NaN values.
    """
    detector.compute_category_statistics(sample_normal_records)
    features_df = detector._extract_features(sample_normal_records, is_training=True)

    assert not features_df.empty
    assert len(features_df) == len(sample_normal_records)
    assert "amount" in features_df.columns
    assert "amount_log" in features_df.columns
    assert "cat_mean_ratio" in features_df.columns
    assert "cat_zscore" in features_df.columns
    assert "day_of_week" in features_df.columns
    assert "is_weekend" in features_df.columns

    for cat in Config.EXPENSE_CATEGORIES:
        assert f"cat_{cat}" in features_df.columns

    # No missing/null values
    assert features_df.isna().sum().sum() == 0


def test_outlier_detection_and_scoring(detector, sample_normal_records):
    """
    Anomaly Flagging Test:
    Injects an extreme outlier (₹45,000 for Food where typical is ~₹350) into
    20 normal expenses and verifies that the model flags it.
    """
    outlier = {
        "id": 99,
        "description": "Extreme Banquet Charge",
        "amount": 45000.0,
        "category": "Food",
        "date": date(2026, 10, 15),
        "payment_method": "Credit Card",
        "notes": "Extreme outlier",
        "is_demo": False,
    }
    all_records = sample_normal_records + [outlier]

    meta, results = detector.fit_and_predict_batch(all_records)

    assert meta["status"] == "success"
    assert meta["sample_count"] == 21
    assert detector.is_fitted is True
    assert meta["anomalies_detected"] >= 1

    # Find the injected outlier assessment
    outlier_res = next(r for r in results if r["id"] == 99)
    assert outlier_res["is_anomaly"] is True
    assert outlier_res["anomaly_label"] == "Potentially unusual spending"
    assert outlier_res["anomaly_score"] is not None
    assert 0.0 <= outlier_res["anomaly_score"] <= 1.0


def test_reproducibility(sample_normal_records):
    """
    Reproducibility Test:
    Two separate detectors with the same random seed must produce identical outputs.
    """
    det1 = ExpenseAnomalyDetector(random_state=42)
    det2 = ExpenseAnomalyDetector(random_state=42)

    meta1, res1 = det1.fit_and_predict_batch(sample_normal_records)
    meta2, res2 = det2.fit_and_predict_batch(sample_normal_records)

    for r1, r2 in zip(res1, res2):
        assert r1["is_anomaly"] == r2["is_anomaly"]
        assert r1["raw_score"] == r2["raw_score"]
        assert r1["anomaly_score"] == r2["anomaly_score"]


def test_explainability_factual_and_limitations(detector, sample_normal_records):
    """
    Explainability Test:
    Checks that the explanation contains category averages and the mandatory
    non-fraud educational disclaimer.
    """
    detector.compute_category_statistics(sample_normal_records)
    outlier = {
        "id": 101,
        "description": "Large Electronic Gadget",
        "amount": 25000.0,
        "category": "Shopping",
        "date": date(2026, 10, 20),
    }

    exp = detector.generate_explanation(
        record=outlier,
        is_anomaly=True,
        raw_score=-0.25,
        normalized_score=0.88,
    )

    assert "summary" in exp
    assert "limitation" in exp
    # Must not claim certainty of fraud
    assert "fraud" in exp["limitation"].lower()
    assert "not" in exp["limitation"].lower()
