"""
Model and data monitoring utilities for production model performance tracking.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List
import mlflow
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)


class ModelMonitor:
    """Monitor model performance in production."""
    
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        """
        Initialize model monitor.
        
        Args:
            tracking_uri: MLflow tracking server URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.performance_history = []
        
    def log_prediction(self, 
                      prediction: int, 
                      actual: Optional[int] = None,
                      confidence: float = 0.0,
                      features: Optional[Dict] = None) -> Dict:
        """
        Log a single prediction for monitoring.
        
        Args:
            prediction: Model prediction
            actual: Actual label (if known)
            confidence: Model confidence score
            features: Input features
            
        Returns:
            Log entry with timestamp
        """
        log_entry = {
            "timestamp": datetime.now(),
            "prediction": prediction,
            "actual": actual,
            "confidence": confidence,
            "features": features or {}
        }
        
        self.performance_history.append(log_entry)
        return log_entry
    
    def calculate_batch_metrics(self, 
                               predictions: List[int], 
                               actuals: List[int]) -> Dict[str, float]:
        """
        Calculate performance metrics for a batch of predictions.
        
        Args:
            predictions: List of predictions
            actuals: List of actual labels
            
        Returns:
            Dictionary of metrics
        """
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have the same length")
        
        metrics = {
            "accuracy": accuracy_score(actuals, predictions),
            "f1": f1_score(actuals, predictions, average='weighted', zero_division=0),
            "precision": precision_score(actuals, predictions, average='weighted', zero_division=0),
            "recall": recall_score(actuals, predictions, average='weighted', zero_division=0),
            "n_samples": len(predictions)
        }
        
        return metrics
    
    def log_batch_metrics(self, 
                         experiment_name: str,
                         metrics: Dict[str, float],
                         tags: Optional[Dict] = None) -> str:
        """
        Log batch metrics to MLflow.
        
        Args:
            experiment_name: MLflow experiment name
            metrics: Dictionary of metrics
            tags: Optional tags for the run
            
        Returns:
            Run ID
        """
        try:
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run():
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)
                
                if tags:
                    for key, value in tags.items():
                        mlflow.set_tag(key, value)
                
                run_id = mlflow.active_run().info.run_id
                logger.info(f"Metrics logged to MLflow run: {run_id}")
                return run_id
        except Exception as e:
            logger.error(f"Error logging metrics to MLflow: {e}")
            raise


class DataDriftDetector:
    """Detect data drift in production data."""
    
    def __init__(self, baseline_data: pd.DataFrame, 
                 thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize drift detector with baseline data.
        
        Args:
            baseline_data: Reference training data
            thresholds: Drift detection thresholds (default: 0.1)
        """
        self.baseline_data = baseline_data
        self.baseline_stats = self._compute_stats(baseline_data)
        self.thresholds = thresholds or {col: 0.1 for col in baseline_data.columns}
        self.drift_history = []
    
    @staticmethod
    def _compute_stats(data: pd.DataFrame) -> Dict:
        """Compute statistics for a dataset."""
        stats = {}
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                stats[col] = {
                    "mean": data[col].mean(),
                    "std": data[col].std(),
                    "min": data[col].min(),
                    "max": data[col].max(),
                    "q25": data[col].quantile(0.25),
                    "q75": data[col].quantile(0.75)
                }
            else:
                stats[col] = {
                    "unique_values": data[col].nunique(),
                    "value_counts": data[col].value_counts().to_dict()
                }
        return stats
    
    def detect_drift(self, new_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Detect drift in new data compared to baseline.
        
        Args:
            new_data: New production data
            
        Returns:
            Drift detection results
        """
        new_stats = self._compute_stats(new_data)
        drift_results = {}
        
        for col in self.baseline_stats.keys():
            if col not in new_data.columns:
                drift_results[col] = {"status": "missing", "drift": True}
                continue
            
            if new_data[col].dtype in ['float64', 'int64']:
                mean_diff = abs(
                    (new_stats[col]["mean"] - self.baseline_stats[col]["mean"]) / 
                    (abs(self.baseline_stats[col]["mean"]) + 1e-10)
                )
                
                threshold = self.thresholds.get(col, 0.1)
                drift_detected = mean_diff > threshold
                
                drift_results[col] = {
                    "status": "drift" if drift_detected else "normal",
                    "drift": drift_detected,
                    "mean_diff_ratio": mean_diff,
                    "threshold": threshold
                }
            else:
                drift_results[col] = {
                    "status": "categorical",
                    "drift": False
                }
        
        self.drift_history.append({
            "timestamp": datetime.now(),
            "results": drift_results
        })
        
        return drift_results
    
    def get_drift_summary(self) -> Dict:
        """Get summary of drift detection history."""
        if not self.drift_history:
            return {"message": "No drift history yet"}
        
        total_drifts = sum(
            1 for col, result in self.drift_history[-1]["results"].items()
            if result.get("drift", False)
        )
        
        return {
            "total_columns_checked": len(self.drift_history[-1]["results"]),
            "columns_with_drift": total_drifts,
            "last_check": self.drift_history[-1]["timestamp"],
            "drift_ratio": total_drifts / len(self.drift_history[-1]["results"])
        }


class PerformanceDegrader:
    """Track performance degradation over time."""
    
    def __init__(self, baseline_metrics: Dict[str, float],
                 degradation_threshold: float = 0.05):
        """
        Initialize degradation tracker.
        
        Args:
            baseline_metrics: Baseline model metrics
            degradation_threshold: Threshold for alarm (default: 5%)
        """
        self.baseline_metrics = baseline_metrics
        self.degradation_threshold = degradation_threshold
        self.metrics_history = []
    
    def check_degradation(self, current_metrics: Dict[str, float]) -> Dict:
        """
        Check if model performance has degraded.
        
        Args:
            current_metrics: Current model metrics
            
        Returns:
            Degradation analysis
        """
        degradation_analysis = {
            "timestamp": datetime.now(),
            "degradation_detected": False,
            "metrics": {}
        }
        
        for metric, baseline_value in self.baseline_metrics.items():
            if metric in current_metrics:
                current_value = current_metrics[metric]
                degradation = (baseline_value - current_value) / baseline_value
                
                is_degraded = degradation > self.degradation_threshold
                
                degradation_analysis["metrics"][metric] = {
                    "baseline": baseline_value,
                    "current": current_value,
                    "degradation": degradation,
                    "degraded": is_degraded
                }
                
                if is_degraded:
                    degradation_analysis["degradation_detected"] = True
        
        self.metrics_history.append(degradation_analysis)
        return degradation_analysis
    
    def get_degradation_trend(self) -> List[Dict]:
        """Get trend of performance degradation."""
        return self.metrics_history[-10:]  # Last 10 checks
