import mlflow
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


"""
MLflow and FastAPI utilities for model training and serving.
"""

class MLflowConfig:
    """MLflow tracking configuration."""
    
    # Connection settings
    TRACKING_URI = "http://localhost:5000"
    REGISTRY_URI = "http://localhost:5000"
    
    # Experiment names
    TRAINING_EXPERIMENT = "incremental_model_training"
    VALIDATION_EXPERIMENT = "model_validation"
    INFERENCE_EXPERIMENT = "model_inference"
    
    # Backend store (file-based or database)
    BACKEND_STORE = "./mlflow_data"  # Local file store, can be: postgresql://user:pass@host/db
    ARTIFACT_STORE = "./mlflow_artifacts"


class MLflowTracker:
    """Utilities for MLflow experiment tracking."""
    
    def __init__(self, tracking_uri: str = MLflowConfig.TRACKING_URI):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow server URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
    
    def get_or_create_experiment(self, experiment_name: str) -> str:
        """
        Get existing experiment or create new one.
        
        Args:
            experiment_name: Name of experiment
            
        Returns:
            Experiment ID
        """
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                return experiment.experiment_id
        except Exception:
            pass
        
        experiment_id = mlflow.create_experiment(experiment_name)
        return experiment_id
    
    def log_training_metadata(self, run_id: str, metadata: Dict):
        """
        Log training metadata to MLflow.
        
        Args:
            run_id: MLflow run ID
            metadata: Dictionary of metadata to log
        """
        with mlflow.start_run(run_id=run_id, nested=True):
            for key, value in metadata.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        mlflow.log_param(f"{key}_{sub_key}", sub_value)
                else:
                    mlflow.log_param(key, value)
    
    def log_model_metadata(self, model_path: Path, model_name: str = "xgb_model"):
        """
        Log model artifact with MLflow.
        
        Args:
            model_path: Path to model file
            model_name: Name for the model
        """
        if model_path.exists():
            mlflow.log_artifact(str(model_path))
    
    def get_best_run(self, experiment_name: str, metric: str = "updated_f1") -> Optional[Dict]:
        """
        Get best run from experiment based on metric.
        
        Args:
            experiment_name: Name of experiment
            metric: Metric to sort by (default: updated_f1)
            
        Returns:
            Best run details or None
        """
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                return None
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric} DESC"],
                max_results=1
            )
            
            if runs.empty:
                return None
            
            return runs.iloc[0].to_dict()
        except Exception as e:
            print(f"Error retrieving best run: {e}")
            return None
    
    def get_run_history(self, experiment_name: str, max_results: int = 10) -> List[Dict]:
        """
        Get training run history.
        
        Args:
            experiment_name: Name of experiment
            max_results: Maximum number of runs to retrieve
            
        Returns:
            List of run details
        """
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                return []
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"]
            )
            
            return [run.to_dict() for _, run in runs.iterrows()]
        except Exception as e:
            print(f"Error retrieving run history: {e}")
            return []


class ModelRegistry:
    """Utilities for model registration and versioning."""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize model registry.
        
        Args:
            model_dir: Directory for local model storage
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.archive_dir = self.model_dir / "archive"
        self.archive_dir.mkdir(exist_ok=True)
    
    def save_model(self, model, name: str = "production_model") -> Path:
        """
        Save model to local registry.
        
        Args:
            model: Model object to save
            name: Model name (without .pkl extension)
            
        Returns:
            Path to saved model
        """
        model_path = self.model_dir / f"{name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return model_path
    
    def load_model(self, name: str = "production_model"):
        """
        Load model from local registry.
        
        Args:
            name: Model name (without .pkl extension)
            
        Returns:
            Loaded model object
        """
        model_path = self.model_dir / f"{name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def archive_model(self, source_name: str = "production_model"):
        """
        Archive current model version.
        
        Args:
            source_name: Model name to archive
        """
        source_path = self.model_dir / f"{source_name}.pkl"
        if source_path.exists():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_path = self.archive_dir / f"{source_name}_{timestamp}.pkl"
            with open(source_path, 'rb') as src:
                with open(archive_path, 'wb') as dst:
                    dst.write(src.read())
            return archive_path
        return None
    
    def list_archived_models(self) -> List[Path]:
        """
        List all archived model versions.
        
        Returns:
            List of archived model paths
        """
        return sorted(self.archive_dir.glob("*.pkl"), reverse=True)


# =====================================================================
# FASTAPI CONFIGURATION & UTILITIES
# =====================================================================

class FastAPIConfig:
    """FastAPI server configuration."""
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = 8000
    RELOAD = False
    WORKERS = 4
    
    # Model settings
    MODEL_NAME = "production_model"
    MODEL_DIR = "models"
    
    # API settings
    API_PREFIX = "/api/v1"
    TITLE = "Predictive Maintenance API"
    DESCRIPTION = "Industrial equipment failure prediction service"
    VERSION = "1.0.0"
    
    # CORS settings
    CORS_ORIGINS = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8080",
    ]
    CORS_CREDENTIALS = True
    CORS_METHODS = ["*"]
    CORS_HEADERS = ["*"]
    
    # Request/Response settings
    REQUEST_TIMEOUT = 60
    BATCH_SIZE_LIMIT = 1000


@dataclass
class PredictionRequest:
    """Request schema for predictions."""
    
    data: List[Dict]  # List of feature dictionaries
    return_probabilities: bool = False
    
    def __post_init__(self):
        """Validate request."""
        if not isinstance(self.data, list):
            raise ValueError("data must be a list")
        if len(self.data) == 0:
            raise ValueError("data cannot be empty")
        if len(self.data) > FastAPIConfig.BATCH_SIZE_LIMIT:
            raise ValueError(f"Batch size exceeds limit of {FastAPIConfig.BATCH_SIZE_LIMIT}")


@dataclass
class PredictionResponse:
    """Response schema for predictions."""
    
    predictions: List[int]
    probabilities: Optional[List[List[float]]] = None
    timestamp: str = ""
    model_version: str = ""
    
    def __post_init__(self):
        """Set default timestamp."""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ModelInferenceManager:
    """Utilities for model inference and serving."""
    
    def __init__(self, model_dir: str = FastAPIConfig.MODEL_DIR):
        """
        Initialize inference manager.
        
        Args:
            model_dir: Directory containing model
        """
        self.registry = ModelRegistry(model_dir)
        self.model = None
        self.model_name = FastAPIConfig.MODEL_NAME
        self._load_model()
    
    def _load_model(self):
        """Load model on initialization."""
        try:
            self.model = self.registry.load_model(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def predict(self, X_data, return_probabilities: bool = False) -> Tuple:
        """
        Make predictions on input data.
        
        Args:
            X_data: Feature data (DataFrame or array)
            return_probabilities: Whether to return probability estimates
            
        Returns:
            Tuple of (predictions, probabilities or None)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        predictions = self.model.predict(X_data)
        probabilities = None
        
        if return_probabilities:
            probabilities = self.model.predict_proba(X_data)
        
        return predictions, probabilities
    
    def health_check(self) -> Dict:
        """
        Check model health status.
        
        Returns:
            Health status dictionary
        """
        return {
            "status": "healthy" if self.model is not None else "unhealthy",
            "model_name": self.model_name,
            "model_loaded": self.model is not None,
            "timestamp": datetime.now().isoformat()
        }


# =====================================================================
# INITIALIZATION
# =====================================================================

def initialize_mlflow(experiment_name: str = MLflowConfig.TRAINING_EXPERIMENT):
    """
    Initialize MLflow for experiment tracking.
    
    Args:
        experiment_name: Name of experiment to use
    """
    mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
    mlflow.set_experiment(experiment_name)


def initialize_model_serving(model_dir: str = FastAPIConfig.MODEL_DIR) -> ModelInferenceManager:
    """
    Initialize model for serving.
    
    Args:
        model_dir: Directory containing model
        
    Returns:
        Initialized ModelInferenceManager
    """
    return ModelInferenceManager(model_dir)
