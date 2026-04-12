# Industrial Predictive Maintenance

An MLOps project for predictive maintenance of industrial equipment using machine learning and great expectations for data validation.

## Project Overview

This project implements an incremental machine learning pipeline for predicting equipment failures in industrial settings. It uses the AI4I 2020 Predictive Maintenance dataset to train and continuously improve a model that predicts machine failures based on operational parameters.

### Key Features

- **Data Preprocessing**: Automated data cleaning, feature engineering, and class balancing
- **Incremental Model Training**: Support for continuous training with existing production models
- **Model Versioning**: Integration with MLflow for experiment tracking and model registry
- **Data Validation**: Great Expectations integration for data quality checks
- **Production API**: FastAPI-based REST service for real-time predictions
- **Model Monitoring**: Drift detection and performance degradation monitoring
- **Docker Support**: Containerized deployment ready

## Project Structure

```
.
├── Data/                          # Dataset storage
│   ├── raw/                      # Original raw data (AI4I 2020 dataset)
│   └── processed/                # Processed data after validation
├── src/                          # Source code
│   ├── train.py                 # Model training logic
│   ├── preprocess.py            # Data preprocessing
│   ├── monitor.py               # Monitoring utilities
│   ├── utils.py                 # MLflow and utility functions
│   └── __init__.py
├── tests/                        # Unit tests
│   ├── test_api.py              # API endpoint tests
│   └── test_data.py             # Data validation tests
├── Deployement/                  # Production deployment
│   ├── DockerFile               # Container configuration
│   ├── requirements.txt          # Deployment dependencies
│   └── src/
│       ├── main.py              # FastAPI application
│       └── __init__.py
├── gx/                           # Great Expectations configuration
│   ├── great_expectations.yml   # GX configuration
│   ├── expectations/            # Expectation suites
│   └── checkpoints/             # Validation checkpoints
├── models/                       # Model storage
│   └── Production/              # Production model versions
```

## Prerequisites

- Python 3.8+
- pip or conda for package management
- Docker (for containerized deployment)

## Installation

### 1. Clone or Setup the Project

```bash
cd "Industrial Predictive Maintenance"
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv .env
.\.env\Scripts\Activate.ps1  # On Windows PowerShell

# Or using conda
conda create -n predictive-maintenance python=3.8
conda activate predictive-maintenance
```

### 3. Install Dependencies

```bash
pip install -r Deployement/requirements.txt
pip install pytest pytest-cov  # For testing
```

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset** from Kaggle:
- 10,000 data points
- 14 features including temperature, rotational speed, torque, and tool wear
- Failure type classification
- Target variable: Machine failure (binary)

**Data Location**: `Data/raw/ai4i2020.csv`

### Data Features

- **Numeric Features**:
  - Air temperature [K]
  - Process temperature [K]
  - Rotational speed [rpm]
  - Torque [Nm]
  - Tool wear [min]

- **Categorical Features**:
  - Type: Product type (L, M, H)

- **Target**:
  - Machine failure: 0 = No failure, 1 = Failure

- **Failure Types** (when Machine failure = 1):
  - TWF: Tool wear failure
  - HDF: Heat dissipation failure
  - PWF: Power failure
  - OSF: Overstrain failure
  - RNF: Random failures

## Usage

### 1. Data Preprocessing

```python
from src.preprocess import DataPreprocessor

preprocessor = DataPreprocessor('Data/raw/ai4i2020.csv')
X, y = preprocessor.preprocess_data()
```

Features performed:
- Missing value imputation
- Feature engineering (power consumption, temperature ratios, etc.)
- Type variable encoding (L→1, M→2, H→3)
- Class balancing using SMOTE
- Column name normalization

### 2. Model Training

```python
from src.train import IncrementalModelTrainer

trainer = IncrementalModelTrainer('Data/processed/ai4i2020.csv')
trainer.load_and_preprocess()
trainer.load_production_model()
trainer.train_and_evaluate()
trainer.save_model()
```

### 3. Data Validation

```python
from tests.test_data import DataValidator

validator = DataValidator('Data/raw/ai4i2020.csv')
success = validator.validate('batch_1')
```

Validations include:
- Column presence and types
- Value ranges and distributions
- Class balance verification
- Failure rate verification (3-10%)

### 4. Model Monitoring

```python
from src.monitor import ModelMonitor, DataDriftDetector

monitor = ModelMonitor()
monitor.log_prediction(prediction=1, actual=1, confidence=0.95)

detector = DataDriftDetector(baseline_data=X_train)
drift_results = detector.detect_drift(X_new)
```

### 5. Production API

```bash
# Run FastAPI server
cd Deployement/src
python -m uvicorn main:app --reload --port 8000
```

**Endpoints**:
- `GET /health` - Health check
- `POST /predict` - Make predictions

**Example Request**:
```json
{
  "machine_id": "MACH_001",
  "Type": 1,
  "Air_temperature_K": 298.5,
  "Process_temperature_K": 308.7,
  "Rotational_speed_rpm": 1800,
  "Torque_Nm": 45.5,
  "Tool_wear_min": 100
}
```

**Example Response**:
```json
{
  "machine_id": "MACH_001",
  "predicted_class": 1,
  "class_probabilities": {
    "class_0": 0.1,
    "class_1": 0.8
  },
  "confidence": 0.8
}
```

## Testing

### Run All Tests

```bash
pytest tests/ -v --tb=short
```

### Run Specific Tests

```bash
# API tests
pytest tests/test_api.py -v

# Data validation tests
pytest tests/test_data.py -v
```

### Test Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

Tests cover:
- Data preprocessing and feature engineering
- Model training and evaluation
- API endpoints and responses
- Data validation rules
- Drift detection

## MLflow Integration

### Start MLflow Server

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow_artifacts
```

### Access MLflow UI

Open browser: `http://localhost:5000`

**Tracked Experiments**:
- `incremental_model_training` - Model training runs
- `model_validation` - Validation metrics
- `model_inference` - Production inference

## Docker Deployment

### Build Docker Image

```bash
docker build -t predictive-maintenance:latest -f Deployement/DockerFile .
```

### Run Container

```bash
docker run -p 8000:8000 predictive-maintenance:latest
```

## Model Performance

The model achieves:
- **Accuracy**: ~98%
- **F1-Score**: ~0.92 (weighted)
- **Recall**: ~0.95 (important for failure detection)

Metrics are tracked per failure type for detailed analysis.

## Key Technologies

- **Machine Learning**: XGBoost, scikit-learn
- **Data Processing**: pandas, numpy
- **Data Validation**: Great Expectations
- **API**: FastAPI, Uvicorn
- **MLOps**: MLflow
- **Testing**: pytest
- **Containerization**: Docker

## Project Workflow

1. **Raw Data** → Great Expectations Validation
2. **Validated Data** → Preprocessing & Feature Engineering
3. **Processed Data** → Model Training (Incremental)
4. **Trained Model** → MLflow Registry
5. **Production Model** → FastAPI Service
6. **Predictions** → Monitoring & Drift Detection
7. **Performance Degradation** → Triggers Retraining

## Configuration

### MLflow Configuration
- Tracking URI: `http://localhost:5000`
- Backend Store: `./mlflow_data`
- Artifact Store: `./mlflow_artifacts`

### Model Hyperparameters
- Algorithm: XGBoost Classifier
- Objective: Multi-class (5 classes)
- Learning Rate: 0.05
- Max Depth: 4
- Regularization (L1/L2): 1.0

## Troubleshooting

### Import Errors
- Ensure virtual environment is activated
- Run `pip install -r Deployement/requirements.txt`

### Database Connection Issues
- Check MLflow server is running: `http://localhost:5000`
- Verify database path permissions

### Data Validation Failures
- Check dataset format matches expected schema
- Verify column names exactly match (case-sensitive)
- Validate numeric ranges match expectations

## Future Enhancements

- [ ] Data version control with DVC
- [ ] Hyperparameter optimization with Optuna
- [ ] Advanced drift detection (Kolmogorov-Smirnov test)
- [ ] API authentication and rate limiting
- [ ] Advanced visualization dashboards
- [ ] Time-series forecasting models
- [ ] Explainability with SHAP values

## Contributing

1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes and commit: `git commit -am 'Add new feature'`
3. Push to branch: `git push origin feature/new-feature`
4. Submit pull request

## License

This project is part of an MLOps demonstration. Use and modify as needed.

## Contact & Support

For issues, questions, or contributions, please open an issue in the repository.

---

**Last Updated**: April 2026
**Project Status**: Active Development
