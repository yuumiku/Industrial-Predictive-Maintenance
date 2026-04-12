import pickle
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from preprocess import DataPreprocessor


class IncrementalModelTrainer:
    """Incremental training for existing deployed model."""
    
    def __init__(self, data_path: str, model_dir: str = "models"):
        """
        Initialize incremental trainer.
        
        Args:
            data_path: Path to processed CSV file
            model_dir: Directory for model storage
        """
        self.data_path = data_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.preprocessor = DataPreprocessor(data_path)
        
        self.production_model_path = self.model_dir / "production_model.pkl"
        self.archive_dir = self.model_dir / "archive"
        self.archive_dir.mkdir(exist_ok=True)
        
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.production_model = None
        self.updated_model = None
        self.metrics = {}
        
        self.xgb_params = {
            'objective': 'multi:softmax',
            'num_class': 5,
            'random_state': 42,
            'learning_rate': 0.05,
            'max_depth': 4,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'gamma': 0.1,
            'warm_start': True,  # Enable incremental learning
        }
        
        # SMOTE config
        self.smote_strategy = {1: 400}
        self.smote_k_neighbors = 3
        
        # Thresholds
        self.class_1_threshold = 0.7
        self.improvement_threshold = 0.01  # 1% improvement required (can be adjusted based on business needs)
        
        mlflow.set_experiment("incremental_model_training")
    
    def load_and_preprocess(self):
        """Load and preprocess new data."""
        print("Loading and preprocessing data...")
        self.preprocessor.load_data()
        self.X, self.y = self.preprocessor.preprocess_data()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"Data shape: {self.X.shape}")
        print(f"Train/Test split: {self.X_train.shape[0]}/{self.X_test.shape[0]}")
        print(f"Class distribution: {self.y.value_counts().to_dict()}")
    
    def load_production_model(self):
        """Load existing production model."""
        if self.production_model_path.exists():
            print(f"\nLoading production model from {self.production_model_path}...")
            with open(self.production_model_path, 'rb') as f:
                self.production_model = pickle.load(f)
            print("Production model loaded successfully")
        else:
            print("No production model found. This will be first training.")
            self.production_model = None
    
    def apply_smote_and_imputation(self, X, y):
        """Apply SMOTE oversampling and handle inf/nan values."""
        print("Applying SMOTE oversampling...")
        
        sm = SMOTE(
            sampling_strategy=self.smote_strategy,
            k_neighbors=self.smote_k_neighbors,
            random_state=42
        )
        X_resampled, y_resampled = sm.fit_resample(X, y)
        
        print(f"After SMOTE: {X_resampled.shape}")
        print(f"Class distribution after SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        # Handle inf and nan values
        X_resampled = X_resampled.replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy='median')
        X_resampled = pd.DataFrame(
            imputer.fit_transform(X_resampled),
            columns=X.columns
        )
        
        return X_resampled, y_resampled
    
    def train_updated_model(self):
        """Train new model using warm_start for incremental learning."""
        print("\n" + "="*70)
        print("TRAINING UPDATED MODEL")
        print("="*70)
        
        with mlflow.start_run(run_name=f"incremental_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            X_train_processed, y_train_processed = self.apply_smote_and_imputation(
                self.X_train, self.y_train
            )
            
            weights = compute_sample_weight(class_weight='balanced', y=self.y_train)
            
            # Create model with warm_start enabled
            self.updated_model = XGBClassifier(**self.xgb_params)
            
            print("\nTraining with sample weights...")
            self.updated_model.fit(
                X_train_processed, y_train_processed,
                sample_weight=weights,
                verbose=False
            )
            
            mlflow.log_params(self.xgb_params)
            
            self._evaluate_model(self.updated_model, self.X_test, self.y_test, "updated")
            
            return self.updated_model
    
    def _evaluate_model(self, model, X_test, y_test, model_type: str):
        """Evaluate model with custom threshold for class 1."""
        print(f"\nEvaluating {model_type} model...")
        
        # Get predictions with threshold adjustment
        y_probs = model.predict_proba(X_test)
        y_pred = np.argmax(y_probs, axis=1)
        
        # Apply custom threshold for class 1
        for i in range(len(y_probs)):
            if y_probs[i][1] < self.class_1_threshold and y_pred[i] == 1:
                y_pred[i] = 0
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        self.metrics[model_type] = metrics
        
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"{model_type}_{metric_name}", value)
        
        print(f"\n{model_type.upper()} Model Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        
        print(f"\nClassification Report ({model_type}):")
        print(classification_report(y_test, y_pred))
    
    def should_update_model(self) -> bool:
        """Determine if updated model should replace production."""
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        updated_f1 = self.metrics['updated']['f1']
        
        if self.production_model is None:
            print("First training - updated model becomes production.")
            return True
        
        self._evaluate_model(self.production_model, self.X_test, self.y_test, "production")
        
        production_f1 = self.metrics['production']['f1']
        improvement = updated_f1 - production_f1
        improvement_pct = (improvement / production_f1 * 100) if production_f1 > 0 else 0
        
        print("\n" + "-"*70)
        print(f"Production F1: {production_f1:.4f}")
        print(f"Updated F1:    {updated_f1:.4f}")
        print(f"Improvement:   {improvement:+.4f} ({improvement_pct:+.2f}%)")
        print("-"*70)
        
        if improvement >= self.improvement_threshold:
            print(f"Updated model improves F1 by {improvement_pct:.2f}%. Updating production.")
            return True
        elif improvement > 0:
            print(f"Slight improvement ({improvement_pct:+.2f}%) but below threshold (1%).")
            return False
        else:
            print(f"Updated model performs worse ({improvement_pct:+.2f}%). Keeping production.")
            return False
    
    def save_production_model(self):
        """Archive old model and save updated as production."""
        if self.production_model_path.exists():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_path = self.archive_dir / f"production_model_{timestamp}.pkl"
            print(f"Archiving old model to {archive_path}...")
            with open(self.production_model_path, 'rb') as src:
                with open(archive_path, 'wb') as dst:
                    dst.write(src.read())
        
        print(f"Saving updated model as production...")
        with open(self.production_model_path, 'wb') as f:
            pickle.dump(self.updated_model, f)
        print(f"Model saved to {self.production_model_path}")
    
    def train(self) -> bool:
        """
        Execute incremental training pipeline.
        
        Returns:
            True if production model was updated, False otherwise
        """
        try:
            self.load_and_preprocess()
            
            self.load_production_model()
            
            self.train_updated_model()
            
            should_update = self.should_update_model()
            
            if should_update:
                self.save_production_model()
                mlflow.log_param("update_decision", "yes")
                return True
            else:
                mlflow.log_param("update_decision", "no")
                print("\nTraining complete. Production model unchanged.")
                return False
        
        except Exception as e:
            print(f"Error during training: {e}")
            mlflow.log_param("training_status", "failed")
            raise
        finally:
            mlflow.end_run()


def main(data_path: str, model_dir: str = "models") -> int:
    """
    Main entry point for incremental training.
    
    Args:
        data_path: Path to processed CSV file
        model_dir: Directory for model storage
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    trainer = IncrementalModelTrainer(data_path, model_dir)
    
    try:
        trainer.train()
        return 0
    except Exception as e:
        print(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "Data/processed/ai4i2020.csv"
    model_dir = sys.argv[2] if len(sys.argv) > 2 else "models"
    exit_code = main(data_path, model_dir)
    sys.exit(exit_code)