# Iris Species Classification Pipeline

This module implements a complete MLOps pipeline for training and evaluating a machine learning model for the Iris species classification task. The pipeline follows best practices for reproducibility, experiment tracking, and model deployment.

## Pipeline Components

1. **Data Processing**: Handles data loading, preparation, and preprocessing with support for:
   - Feature scaling (Standard, MinMax, or Robust)
   - Missing value imputation
   - Target encoding for categorical species labels
   - Feature engineering (polynomial features, PCA)

2. **Model Training**: Trains a classifier with support for:
   - Multiple model types (Random Forest, SVM, Logistic Regression)
   - Cross-validation
   - Hyperparameter configurations

3. **Model Evaluation**: Evaluates model performance with:
   - Accuracy, precision, recall, F1 score
   - Confusion matrix visualization with original class names
   - ROC curve generation
   - Feature importance analysis
   - Classification reports with human-readable class names

4. **MLflow Integration**: Provides experiment tracking with:
   - Parameter logging
   - Metric tracking
   - Artifact storage
   - Model registry

## Usage

Run the pipeline using the following command:

```bash
python -m baq.run --config configs/config.yaml
```

You can also import and use the pipeline in other Python scripts:

```python
from baq.pipelines.iris_classification_pipeline import IrisClassificationPipeline

pipeline = IrisClassificationPipeline("path/to/config.yaml")
results = pipeline.run()
```

## Configuration

The pipeline is configured using a YAML file with the following sections:

- `data`: Dataset information and features
- `processing`: Data preprocessing options
- `model`: Model type and hyperparameters
- `training`: Training options and metrics
- `mlops`: MLflow configuration
- `artifacts`: Output paths and file names

See `configs/config.yaml` for a complete example configuration.

## Artifacts

The pipeline generates several artifacts:

1. **Model Files**:
   - Trained model pickle file
   - Label encoder for categorical targets
   - Preprocessing pipeline

2. **Evaluation Reports**:
   - Classification report
   - Model card with performance metrics
   - Class mapping CSV (original species names to encoded values)

3. **Visualizations**:
   - Confusion matrix with original class names
   - ROC curves for each class
   - Feature importance plot 