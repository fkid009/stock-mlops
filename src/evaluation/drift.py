"""Drift detection for model performance and features."""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.common import get_logger, get_pipeline_config
from src.data import DatabaseManager

logger = get_logger(__name__)


class DriftDetector:
    """Detect performance and feature drift."""

    def __init__(
        self,
        baseline_window: Optional[int] = None,
        recent_window: Optional[int] = None,
        threshold: Optional[float] = None,
    ):
        """Initialize drift detector.

        Args:
            baseline_window: Number of days for baseline performance (default from config)
            recent_window: Number of recent days to compare (default from config)
            threshold: Performance drop threshold to trigger drift (default from config)
        """
        config = get_pipeline_config()
        drift_config = config.get("drift_detection", {})

        self.baseline_window = (
            baseline_window
            if baseline_window is not None
            else drift_config.get("baseline_window", 30)
        )
        self.recent_window = (
            recent_window
            if recent_window is not None
            else drift_config.get("recent_window", 5)
        )
        self.threshold = (
            threshold
            if threshold is not None
            else drift_config.get("threshold", 0.05)
        )

        self.db = DatabaseManager()

    def check_performance_drift(self) -> dict:
        """Check for performance drift based on recent accuracy.

        Returns:
            Dictionary with drift status and details
        """
        # Get metrics from database
        metrics_df = self.db.get_metrics()

        min_required = self.baseline_window + self.recent_window
        if len(metrics_df) < min_required:
            logger.warning(
                f"Not enough data for drift detection: "
                f"{len(metrics_df)} < {min_required} (baseline {self.baseline_window} + recent {self.recent_window})"
            )
            return {
                "drift_detected": False,
                "reason": "insufficient_data",
                "baseline_accuracy": None,
                "recent_accuracy": None,
            }

        # Calculate baseline and recent accuracy
        metrics_df = metrics_df.sort_values("date", ascending=False)

        # Recent: most recent N days
        recent = metrics_df.head(self.recent_window)
        # Baseline: next M days after recent (excluding recent to avoid overlap)
        baseline = metrics_df.iloc[self.recent_window:self.baseline_window + self.recent_window]

        if len(baseline) == 0:
            logger.warning("Not enough baseline data after excluding recent period")
            return {
                "drift_detected": False,
                "reason": "insufficient_baseline_data",
                "baseline_accuracy": None,
                "recent_accuracy": None,
            }

        recent_accuracy = recent["accuracy"].mean()
        baseline_accuracy = baseline["accuracy"].mean()
        accuracy_drop = baseline_accuracy - recent_accuracy

        drift_detected = accuracy_drop > self.threshold

        result = {
            "drift_detected": drift_detected,
            "baseline_accuracy": baseline_accuracy,
            "recent_accuracy": recent_accuracy,
            "accuracy_drop": accuracy_drop,
            "threshold": self.threshold,
            "baseline_window": self.baseline_window,
            "recent_window": self.recent_window,
        }

        if drift_detected:
            logger.warning(
                f"Performance drift detected! "
                f"Baseline: {baseline_accuracy:.4f}, "
                f"Recent: {recent_accuracy:.4f}, "
                f"Drop: {accuracy_drop:.4f}"
            )
        else:
            logger.info(
                f"No drift detected. "
                f"Baseline: {baseline_accuracy:.4f}, "
                f"Recent: {recent_accuracy:.4f}"
            )

        return result

    def calculate_psi(
        self,
        baseline: pd.Series,
        current: pd.Series,
        bins: int = 10,
    ) -> float:
        """Calculate Population Stability Index (PSI).

        Args:
            baseline: Baseline distribution
            current: Current distribution
            bins: Number of bins for discretization

        Returns:
            PSI value
        """
        # Create bins based on baseline
        bin_edges = np.percentile(baseline, np.linspace(0, 100, bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Calculate proportions
        baseline_counts = np.histogram(baseline, bins=bin_edges)[0]
        current_counts = np.histogram(current, bins=bin_edges)[0]

        baseline_props = baseline_counts / len(baseline)
        current_props = current_counts / len(current)

        # Avoid division by zero
        baseline_props = np.clip(baseline_props, 1e-10, 1)
        current_props = np.clip(current_props, 1e-10, 1)

        # Calculate PSI
        psi = np.sum((current_props - baseline_props) * np.log(current_props / baseline_props))

        return psi

    def check_feature_drift(
        self,
        baseline_features: pd.DataFrame,
        current_features: pd.DataFrame,
        psi_threshold: float = 0.2,
    ) -> dict:
        """Check for feature distribution drift using PSI.

        Args:
            baseline_features: Baseline feature DataFrame
            current_features: Current feature DataFrame
            psi_threshold: PSI threshold for drift detection

        Returns:
            Dictionary with drift status per feature
        """
        results = {}
        drifted_features = []

        for col in baseline_features.columns:
            if col not in current_features.columns:
                continue

            baseline = baseline_features[col].dropna()
            current = current_features[col].dropna()

            if len(baseline) < 10 or len(current) < 10:
                continue

            psi = self.calculate_psi(baseline, current)
            is_drifted = psi > psi_threshold

            results[col] = {
                "psi": psi,
                "drifted": is_drifted,
                "threshold": psi_threshold,
            }

            if is_drifted:
                drifted_features.append(col)
                logger.warning(f"Feature drift detected in '{col}': PSI={psi:.4f}")

        return {
            "feature_results": results,
            "drifted_features": drifted_features,
            "any_drift": len(drifted_features) > 0,
        }

    def statistical_test(
        self,
        baseline: pd.Series,
        current: pd.Series,
        alpha: float = 0.05,
    ) -> dict:
        """Perform statistical test for distribution change.

        Uses Kolmogorov-Smirnov test.

        Args:
            baseline: Baseline values
            current: Current values
            alpha: Significance level

        Returns:
            Test results
        """
        statistic, p_value = stats.ks_2samp(baseline, current)

        return {
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < alpha,
            "alpha": alpha,
        }

    def get_drift_report(
        self,
        baseline_features: Optional[pd.DataFrame] = None,
        current_features: Optional[pd.DataFrame] = None,
    ) -> dict:
        """Generate a comprehensive drift report.

        Args:
            baseline_features: Optional baseline features for feature drift
            current_features: Optional current features for feature drift

        Returns:
            Comprehensive drift report
        """
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "performance_drift": self.check_performance_drift(),
        }

        if baseline_features is not None and current_features is not None:
            report["feature_drift"] = self.check_feature_drift(
                baseline_features, current_features
            )
        else:
            report["feature_drift"] = {"status": "not_checked"}

        # Overall drift status
        perf_drift = report["performance_drift"].get("drift_detected", False)
        feat_drift = report.get("feature_drift", {}).get("any_drift", False)

        report["overall_drift"] = perf_drift or feat_drift
        report["recommended_action"] = self._get_recommendation(report)

        return report

    def _get_recommendation(self, report: dict) -> str:
        """Get recommended action based on drift report.

        Args:
            report: Drift report

        Returns:
            Recommendation string
        """
        perf_drift = report["performance_drift"].get("drift_detected", False)
        feat_drift = report.get("feature_drift", {}).get("any_drift", False)

        if perf_drift and feat_drift:
            return "retrain_with_tuning"
        elif perf_drift:
            return "retrain_model"
        elif feat_drift:
            return "monitor_closely"
        else:
            return "no_action"

    def should_trigger_tuning(self) -> bool:
        """Check if hyperparameter tuning should be triggered.

        Returns:
            True if tuning should be triggered
        """
        result = self.check_performance_drift()
        return result.get("drift_detected", False)
