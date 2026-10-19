"""
model.py - Machine Learning Anomaly Detection Module
===================================================
Uses Scikit-learn's IsolationForest algorithm to detect potentially unusual
expenses based on transaction amount, category baselines, and spending patterns.

Key Design Decisions & Educational Viva Notes:
---------------------------------------------
1. Unsupervised Learning:
   Real personal expense data has no ground-truth "fraud" labels.
   IsolationForest is an unsupervised algorithm that isolates anomalies by
   randomly selecting a feature and randomly selecting a split value between
   the maximum and minimum values of the selected feature.
   Anomalies (rare, extreme, or atypical points) require fewer splits to isolate,
   resulting in shorter paths in the decision trees.

2. No Hard-coded Fraud Claims:
   IsolationForest produces decision scores; it does NOT produce calibrated fraud
   probabilities. This module strictly outputs understandable labels:
   - "Typical spending pattern"
   - "Potentially unusual spending"
   - "Insufficient data for reliable assessment"

3. Context-Aware Feature Engineering:
   Rather than looking at raw amount alone, we engineer:
   - Log-transformed amount: log1p(amount) to stabilize high variance.
   - Category mean and standard deviation baselines.
   - Ratio of amount to category average: (amount / category_mean).
   - Z-Score deviation within category: (amount - category_mean) / category_std.
   - Day of week and weekend indicator.

4. Cold Start & Insufficient Data Handling:
   If fewer than MIN_TRAINING_SAMPLES (default 10) transactions exist, the model
   refuses to guess and transparently returns "Insufficient data for reliable assessment".
"""

from __future__ import annotations
import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import Config


class ExpenseAnomalyDetector:
    """
    Detector using Scikit-Learn's IsolationForest with context-aware
    feature engineering for personal finance anomaly detection.
    """

    def __init__(
        self,
        contamination: float = Config.MODEL_CONTAMINATION,
        random_state: int = Config.MODEL_RANDOM_STATE,
        min_samples: int = Config.MODEL_MIN_SAMPLES,
    ):
        self.contamination = contamination
        self.random_state = random_state
        self.min_samples = min_samples
        self.model: Optional[IsolationForest] = None
        self.is_fitted = False
        self.last_trained_at: Optional[datetime] = None
        self.training_sample_count: int = 0
        self.category_stats: Dict[str, Dict[str, float]] = {}
        self.overall_mean: float = 0.0

    def compute_category_statistics(self, records: List[Dict[str, Any]]) -> None:
        """
        Compute baseline historical stats (mean, std, max, count) per category
        to provide explainable context.
        """
        if not records:
            self.category_stats = {}
            self.overall_mean = 0.0
            return

        df = pd.DataFrame(records)
        self.overall_mean = float(df["amount"].mean())

        stats = {}
        for cat in Config.EXPENSE_CATEGORIES:
            cat_df = df[df["category"] == cat]
            if len(cat_df) > 0:
                mean_val = float(cat_df["amount"].mean())
                std_val = float(cat_df["amount"].std(ddof=0)) if len(cat_df) > 1 else 0.0
                max_val = float(cat_df["amount"].max())
                stats[cat] = {
                    "count": int(len(cat_df)),
                    "mean": round(mean_val, 2),
                    "std": round(std_val, 2),
                    "max": round(max_val, 2),
                }
            else:
                stats[cat] = {"count": 0, "mean": 0.0, "std": 0.0, "max": 0.0}
        self.category_stats = stats

    def _extract_features(
        self, records: List[Dict[str, Any]], is_training: bool = False
    ) -> pd.DataFrame:
        """
        Transforms transaction dictionaries into a rich numerical feature matrix.
        Features engineered:
        1. amount: raw rupees
        2. amount_log: log(1 + amount)
        3. cat_mean_ratio: amount / max(cat_mean, 10.0)
        4. cat_zscore: (amount - cat_mean) / max(cat_std, 50.0)
        5. day_of_week: 0 (Monday) to 6 (Sunday)
        6. is_weekend: 1 if Saturday or Sunday, else 0
        7. One-hot category indicators for standard categories.
        """
        df = pd.DataFrame(records)

        # Parse date safely across Pandas versions and formats (string, date, datetime)
        df["date_parsed"] = pd.to_datetime(df["date"])

        df["amount"] = df["amount"].astype(float)
        df["amount_log"] = np.log1p(df["amount"])
        df["day_of_week"] = df["date_parsed"].dt.weekday
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Category contextual features
        def get_cat_ratio(row):
            cat = row.get("category", "Other")
            cat_stat = self.category_stats.get(cat, {})
            cat_mean = cat_stat.get("mean", self.overall_mean or 100.0)
            return row["amount"] / max(cat_mean, 10.0)

        def get_cat_zscore(row):
            cat = row.get("category", "Other")
            cat_stat = self.category_stats.get(cat, {})
            cat_mean = cat_stat.get("mean", self.overall_mean or 100.0)
            cat_std = cat_stat.get("std", 0.0)
            return (row["amount"] - cat_mean) / max(cat_std, 50.0)

        df["cat_mean_ratio"] = df.apply(get_cat_ratio, axis=1)
        df["cat_zscore"] = df.apply(get_cat_zscore, axis=1)

        # One-hot encode categories safely with fixed columns
        for cat in Config.EXPENSE_CATEGORIES:
            df[f"cat_{cat}"] = (df["category"] == cat).astype(int)

        feature_cols = [
            "amount",
            "amount_log",
            "cat_mean_ratio",
            "cat_zscore",
            "day_of_week",
            "is_weekend",
        ] + [f"cat_{cat}" for cat in Config.EXPENSE_CATEGORIES]

        return df[feature_cols].fillna(0.0)

    def fit_and_predict_batch(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Fits the IsolationForest model on all available transactions if threshold is met,
        and generates explainable anomaly assessments for each record.
        """
        n_samples = len(records)
        metadata = {
            "status": "success",
            "sample_count": n_samples,
            "min_required": self.min_samples,
            "trained_at": None,
            "anomalies_detected": 0,
        }

        # Insufficient data handling
        if n_samples < self.min_samples:
            metadata["status"] = "insufficient_data"
            metadata["message"] = (
                f"You have recorded {n_samples} transaction(s). At least "
                f"{self.min_samples} transactions are required to build a reliable spending baseline."
            )
            # Produce neutral assessments
            results = []
            for r in records:
                results.append({
                    "id": r.get("id"),
                    "is_anomaly": False,
                    "anomaly_score": None,
                    "raw_score": None,
                    "anomaly_label": "Insufficient data for reliable assessment",
                    "review_status": "none",
                    "explanation": {
                        "summary": "Baseline profile requires more transactions.",
                        "category_comparison": "N/A",
                        "limitation": "Statistical model requires minimum 10 entries to establish baseline variance.",
                    },
                })
            return metadata, results

        # Compute baselines
        self.compute_category_statistics(records)

        # Extract features
        X = self._extract_features(records, is_training=True)

        # Fit IsolationForest
        # Contamination defines the proportion of outliers in the data set.
        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.random_state,
            bootstrap=False,
            n_jobs=-1,
        )
        self.model.fit(X)
        self.is_fitted = True
        self.last_trained_at = datetime.utcnow()
        self.training_sample_count = n_samples

        # In Scikit-learn:
        # decision_function: The anomaly score of the input samples. The lower, the more abnormal.
        # Negative scores represent outliers (flagged), positive represent inliers.
        # predict: -1 for anomalies, 1 for inliers.
        raw_scores = self.model.decision_function(X)
        predictions = self.model.predict(X)

        # Normalize raw scores to an intuitive 0.0 to 1.0 "Unusualness Index"
        # IsolationForest decision_function values typically lie in [-0.5, 0.5].
        # Negative means anomalous.
        # We map: normalized_score = 1.0 / (1.0 + exp(8 * raw_score)) or linear min-max.
        # Higher index => higher unusualness.
        min_s = float(np.min(raw_scores))
        max_s = float(np.max(raw_scores))
        score_range = max_s - min_s if max_s != min_s else 1.0

        results = []
        anomaly_count = 0

        for idx, r in enumerate(records):
            raw = float(raw_scores[idx])
            pred = int(predictions[idx])  # -1 or 1

            # Normalized anomaly index: 0.0 (very normal) to 1.0 (highly unusual)
            # Invert: lower raw score -> higher unusualness
            normalized_score = round(float(1.0 - (raw - min_s) / score_range), 3)

            is_anomaly = bool(pred == -1)
            if is_anomaly:
                anomaly_count += 1
                label = "Potentially unusual spending"
                review_status = "pending"
            else:
                label = "Typical spending pattern"
                review_status = "none"

            explanation = self.generate_explanation(
                record=r,
                is_anomaly=is_anomaly,
                raw_score=raw,
                normalized_score=normalized_score,
            )

            results.append({
                "id": r.get("id"),
                "is_anomaly": is_anomaly,
                "anomaly_score": normalized_score,
                "raw_score": round(raw, 4),
                "anomaly_label": label,
                "review_status": review_status,
                "explanation": explanation,
            })

        metadata["trained_at"] = self.last_trained_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        metadata["anomalies_detected"] = anomaly_count
        return metadata, results

    def assess_single(
        self, record: Dict[str, Any], historical_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Assesses a single newly added transaction against the existing historical corpus.
        Does not fit a model on 1 item. Reuses the trained model or retrains on historical data.
        """
        total_records = historical_records + [record]
        n_samples = len(total_records)

        if n_samples < self.min_samples:
            return {
                "is_anomaly": False,
                "anomaly_score": None,
                "raw_score": None,
                "anomaly_label": "Insufficient data for reliable assessment",
                "review_status": "none",
                "explanation": {
                    "summary": f"Historical profile currently has {n_samples} transaction(s). Minimum {self.min_samples} needed.",
                    "category_comparison": "Baseline requires more entries.",
                    "limitation": "IsolationForest requires sufficient historical volume to isolate statistical outliers.",
                },
            }

        # Train on full history + new transaction to ensure fresh context
        _, batch_results = self.fit_and_predict_batch(total_records)
        # Return the assessment of the last item (the target single record)
        return batch_results[-1]

    def generate_explanation(
        self,
        record: Dict[str, Any],
        is_anomaly: bool,
        raw_score: float,
        normalized_score: float,
    ) -> Dict[str, Any]:
        """
        Generates factual, beginner-friendly explanations comparing the expense
        against the category average and historical baseline.
        """
        cat = record.get("category", "Other")
        amount = float(record.get("amount", 0.0))
        cat_stat = self.category_stats.get(cat, {"count": 0, "mean": 0.0, "std": 0.0, "max": 0.0})

        cat_mean = cat_stat.get("mean", 0.0)
        cat_count = cat_stat.get("count", 0)
        cat_max = cat_stat.get("max", 0.0)

        # Factual comparisons
        comparison_parts = []
        if cat_count >= 2 and cat_mean > 0:
            ratio = amount / cat_mean
            diff = amount - cat_mean
            if ratio >= 1.5:
                comparison_parts.append(
                    f"This expense of ₹{amount:,.2f} is {ratio:.1f}x higher than your historical "
                    f"average of ₹{cat_mean:,.2f} for '{cat}'."
                )
            elif ratio <= 0.4:
                comparison_parts.append(
                    f"This expense of ₹{amount:,.2f} is significantly lower than your historical "
                    f"average of ₹{cat_mean:,.2f} for '{cat}'."
                )
            else:
                comparison_parts.append(
                    f"This expense of ₹{amount:,.2f} aligns closely with your historical '{cat}' "
                    f"average of ₹{cat_mean:,.2f}."
                )

            if amount > cat_max:
                comparison_parts.append(
                    f"It sets a new maximum recorded expense for '{cat}' (previous highest was ₹{cat_max:,.2f})."
                )
        else:
            comparison_parts.append(
                f"Limited historical records exist for '{cat}' ({cat_count} previous transactions). "
                f"Contextualized against overall spending baseline of ₹{self.overall_mean:,.2f}."
            )

        category_comparison_str = " ".join(comparison_parts)

        # Explanation summary
        if is_anomaly:
            summary = (
                f"Flagged as a potentially unusual expense. "
                f"IsolationForest decision score: {raw_score:.3f} (Unusualness index: {normalized_score}/1.0). "
                f"Its multi-feature profile deviates from historical spending patterns."
            )
        else:
            summary = (
                f"Consistent with typical spending patterns. "
                f"IsolationForest decision score: {raw_score:.3f} (Unusualness index: {normalized_score}/1.0)."
            )

        # Limitation disclaimer
        limitation = (
            "Statistical anomaly detection flags mathematical divergence from past spending patterns. "
            "It does NOT indicate fraud, unauthorized activity, or an error. Always verify with actual receipts."
        )

        return {
            "summary": summary,
            "category_comparison": category_comparison_str,
            "category_mean": cat_mean,
            "category_count": cat_count,
            "category_max": cat_max,
            "raw_score": round(raw_score, 4),
            "normalized_score": normalized_score,
            "limitation": limitation,
        }


# Global detector instance for runtime caching
detector = ExpenseAnomalyDetector()
