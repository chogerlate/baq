# Bangkok Air Quality (BAQ) Forecasting

A comprehensive machine learning pipeline for PM2.5 air quality forecasting in Bangkok, Thailand. This project provides end-to-end capabilities for data processing, model training, evaluation, and deployment using multiple ML algorithms including LSTM, Random Forest, and XGBoost.


## MLOps Architecture
![MLOps Diagram drawio(1)](https://github.com/user-attachments/assets/ec84264a-0d7d-45fa-b3ca-aa128296dc18)




## 🔗 Related GitHub Repositories for the BAQ Project

Here are the main repositories that make up the **BAQ** project, covering everything from data pipelines to APIs, experiments, and frontend interfaces.

---

### 🏠 Main Repository  
- **Purpose**: Central codebase and project orchestration  
- **URL**: [chogerlate/baq](https://github.com/chogerlate/baq)

---

### ⛓️ DAG & Airflow Repository  
- **Purpose**: Airflow DAGs for ETL and scheduled workflows  
- **URL**: [Saranunt/baq-airflow](https://github.com/Saranunt/baq-airflow)

---

### ⚙️ FastAPI Backend  
- **Purpose**: API services for model inference and system integration  
- **URL**: [Saranunt/baq-api](https://github.com/Saranunt/baq-api)

---

### 🎨 Streamlit Frontend  
- **Purpose**: Interactive web UI for exploring model outputs and results  
- **URL**: [tawayahc/baq-frontend](https://github.com/tawayahc/baq-frontend)

---

### 🧪 Model Experimentation  
- **Purpose**: Notebooks, training scripts, and experimental ML workflows  
- **URL**: [tawayahc/baq-experiment](https://github.com/tawayahc/baq-experiment)

## 🌟 Features

### Core Capabilities
- **Multi-Model Support**: LSTM (deep learning), Random Forest, and XGBoost models
- **Advanced Data Processing**: Comprehensive preprocessing pipeline with feature engineering
- **Time Series Forecasting**: Single-step and multi-step PM2.5 predictions
- **Experiment Tracking**: Integration with Weights & Biases (W&B) for MLOps
- **Model Monitoring**: Automated performance monitoring and data drift detection
- **Configuration Management**: Hydra-based configuration with YAML files
- **Artifact Management**: Model and processor serialization with versioning

### Data Processing Features
- **Temporal Feature Engineering**: Cyclical time encoding, lag features, rolling statistics
- **Domain-Specific Features**: AQI tier classification, weekend/night indicators
- **Robust Data Cleaning**: Missing value imputation, outlier handling, seasonal median filling
- **Weather Code Encoding**: Categorical weather condition processing
- **Data Validation**: Comprehensive quality checks and drift detection

### Model Training & Evaluation
- **Cross-Validation**: Time series aware validation strategies
- **Performance Metrics**: MAE, RMSE, MAPE, R², accuracy calculations
- **Visualization**: Prediction plots, performance comparisons, monitoring dashboards
- **Hyperparameter Optimization**: Configurable model parameters
- **Early Stopping**: Intelligent training termination for deep learning models

## 📁 Repository Structure

```
baq/
├── 📄 README.md                           # Project documentation
├── 📄 pyproject.toml                      # Project configuration and dependencies
├── 📄 requirements.txt                    # Python dependencies
├── 📄 .env-example                        # Environment variables template
├── 📄 PERFORMANCE_RESTORATION_SUMMARY.md  # Performance analysis documentation
│
├── 📁 configs/                            # Configuration files
│   └── 📄 config.yaml                     # Main configuration file
│
├── 📁 src/baq/                            # Main source code package
│   ├── 📄 __init__.py                     # Package initialization
│   ├── 📄 py.typed                        # Type checking marker
│   ├── 📄 run.py                          # Main entry point
│   │
│   ├── 📁 core/                           # Core functionality
│   │   ├── 📄 evaluation.py               # Model evaluation metrics
│   │   └── 📄 inference.py                # Prediction and forecasting logic
│   │
│   ├── 📁 data/                           # Data processing modules
│   │   ├── 📄 processing.py               # Main data preprocessing pipeline
│   │   └── 📄 utils.py                    # Data utility functions
│   │
│   ├── 📁 models/                         # Model implementations
│   │   └── 📄 lstm.py                     # LSTM model architecture
│   │
│   ├── 📁 steps/                          # Pipeline steps
│   │   ├── 📄 load_data.py                # Data loading step
│   │   ├── 📄 process.py                  # Data processing step
│   │   ├── 📄 train.py                    # Model training step
│   │   ├── 📄 evaluate.py                 # Model evaluation step
│   │   ├── 📄 monitoring_report.py        # Performance monitoring
│   │   └── 📄 save_artifacts.py           # Artifact saving step
│   │
│   ├── 📁 pipelines/                      # ML pipelines
│   ├── 📁 utils/                          # Utility functions
│   ├── 📁 scripts/                        # Automation scripts
│   └── 📁 action_files/                   # Action configurations
│
├── 📁 data/                               # Data storage
├── 📁 notebooks/                          # Jupyter notebooks
│   ├── 📄 experiment.ipynb                # Experimentation notebook
│   ├── 📄 api_call.ipynb                  # API testing notebook
│   ├── 📄 wandb.ipynb                     # W&B integration examples
│   └── 📄 test_module.ipynb               # Module testing
│
├── 📁 outputs/                            # Pipeline outputs
├── 📁 wandb/                              # Weights & Biases artifacts
├── 📁 docs/                               # Documentation
└── 📁 .github/                            # GitHub workflows
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip or uv package manager
- Optional: AWS S3 access for data storage
- Optional: Weights & Biases account for experiment tracking

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd baq
```

2. **Install dependencies**:
```bash
# Using pip
pip install -r requirements.txt

# Using uv (recommended)
uv sync
```

3. **Set up environment variables**:
```bash
cp .env-example .env
# Edit .env with your configurations
```

4. **Configure Weights & Biases** (optional):
```bash
wandb login
```

### Basic Usage

**Run the complete training pipeline**:
```bash
python src/baq/run.py
```

**Run with custom configuration**:
```bash
python src/baq/run.py model.model_type=lstm training.epochs=100
```

## ⚙️ Configuration

The project uses Hydra for configuration management. Main configuration file: `configs/config.yaml`

### Key Configuration Sections

#### Model Configuration
```yaml
model:
  model_type: "random_forest"  # Options: "random_forest", "xgboost", "lstm"
  random_forest:
    model_params:
      n_estimators: 50
      max_depth: 10
  lstm:
    model_params:
      n_layers: 2
      hidden_size: 512
      dropout: 0.2
    training_params:
      learning_rate: 0.001
      batch_size: 64
      epochs: 100
```

#### Training Configuration
```yaml
training:
  forecast_horizon: 24
  sequence_length: 24
  target_column: "pm2_5_(μg/m³)"
  test_size: 0.2
  random_state: 42
```

#### Experiment Tracking
```yaml
wandb:
  tags: ["pm2.5", "forecasting", "air-quality"]
  log_model: true
  register_model: false
```

## 🔧 Data Processing Pipeline

### Input Data Format
The pipeline expects weather and air quality data with temporal features:
- **Meteorological**: Temperature, humidity, pressure, wind speed, precipitation
- **Environmental**: Soil conditions, UV index, visibility
- **Air Quality**: PM2.5 historical values and derived features
- **Temporal**: Timestamps for time series analysis

### Feature Engineering
The `TimeSeriesDataProcessor` creates comprehensive features:

1. **Temporal Features**:
   - Hour, day, month, day of week
   - Weekend/night indicators
   - Cyclical encoding (sin/cos transformations)

2. **Lag Features**:
   - PM2.5 values from 1, 3, 6, 12, 24 hours ago
   - Rolling means and standard deviations

3. **Domain-Specific Features**:
   - AQI tier classification (0-5 based on PM2.5 levels)
   - Weather code encoding

4. **Data Quality**:
   - Missing value imputation
   - Outlier detection and handling
   - Seasonal median filling

## 🤖 Model Training

### Supported Models

#### 1. LSTM (Long Short-Term Memory)
- **Use Case**: Complex temporal patterns, long-term dependencies
- **Architecture**: Dual-layer LSTM with dropout regularization
- **Features**: Early stopping, learning rate scheduling, model checkpointing

#### 2. Random Forest
- **Use Case**: Robust baseline, feature importance analysis
- **Features**: Ensemble learning, handles non-linear relationships

#### 3. XGBoost
- **Use Case**: High performance, gradient boosting
- **Features**: Advanced regularization, efficient training

### Training Process

1. **Data Loading**: Load raw weather and air quality data
2. **Preprocessing**: Apply feature engineering and scaling
3. **Model Training**: Train selected model with configured parameters
4. **Evaluation**: Calculate performance metrics on test set
5. **Artifact Saving**: Save trained model and preprocessors
6. **Monitoring**: Generate performance and drift reports

## 📊 Evaluation & Monitoring

### Performance Metrics
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Square Error): Penalizes large errors
- **MAPE** (Mean Absolute Percentage Error): Relative error percentage
- **R²** (Coefficient of Determination): Explained variance
- **Accuracy**: 1 - MAPE

### Forecasting Types
- **Single-Step**: Predict next time step
- **Multi-Step**: Predict multiple future time steps
- **Iterative Forecasting**: Use predictions as inputs for future steps

### Monitoring Features
- **Data Drift Detection**: Statistical tests for distribution changes
- **Performance Tracking**: Metric trends over time
- **Feature Importance**: Model interpretability analysis
- **Visualization**: Prediction plots, residual analysis

## 🔬 Experiment Tracking

### Weights & Biases Integration
- **Experiment Logging**: Automatic metric and parameter tracking
- **Model Versioning**: Artifact management and model registry
- **Visualization**: Interactive plots and dashboards
- **Collaboration**: Team experiment sharing

### Logged Information
- Model hyperparameters and architecture
- Training and validation metrics
- Feature importance scores
- Prediction visualizations
- Data quality reports

## 🛠️ Development

### Project Structure Principles
- **Modular Design**: Separate concerns into focused modules
- **Configuration-Driven**: Hydra-based parameter management
- **Type Safety**: Type hints and py.typed marker
- **Testing**: Comprehensive test coverage (notebooks for experimentation)
- **Documentation**: Detailed docstrings and examples

### Key Modules

#### `src/baq/data/processing.py`
- **TimeSeriesDataProcessor**: Main preprocessing pipeline
- **Features**: Data cleaning, feature engineering, scaling, validation
- **Methods**: `fit_transform()`, `transform()`, `inverse_transform_target()`

#### `src/baq/models/lstm.py`
- **LSTMForecaster**: Deep learning model implementation
- **Features**: Configurable architecture, callbacks, early stopping
- **Methods**: `fit()`, `predict()`, model checkpointing

#### `src/baq/core/inference.py`
- **Forecasting Functions**: Single-step and multi-step prediction
- **Features**: Model-agnostic interface, sequence handling
- **Methods**: `single_step_forecasting()`, `multi_step_forecasting()`

### Adding New Models
1. Implement model class in `src/baq/models/`
2. Add configuration section in `config.yaml`
3. Update training logic in `src/baq/steps/train.py`
4. Add evaluation support in `src/baq/steps/evaluate.py`

## 🚀 Deployment

### Model Artifacts
- **Model Files**: Serialized trained models (.h5, .joblib)
- **Preprocessors**: Fitted scalers and encoders (.joblib)
- **Metadata**: Training configuration and metrics (.json)

### Integration Options
- **Batch Prediction**: Process historical data in batches
- **Real-time API**: Deploy models as REST APIs
- **Scheduled Jobs**: Automated retraining and prediction
- **Cloud Deployment**: AWS, GCP, Azure integration

## 🔍 Troubleshooting

### Common Issues

1. **Data Loading Errors**
   - Check file paths in `config.yaml`
   - Verify data format and column names
   - Ensure proper datetime indexing

2. **Memory Issues**
   - Reduce batch size for LSTM training
   - Use data chunking for large datasets
   - Monitor memory usage during processing

3. **Model Performance**
   - Check feature engineering pipeline
   - Verify target column name format
   - Review hyperparameter settings

4. **W&B Connection Issues**
   - Verify API key: `wandb login`
   - Check internet connectivity
   - Review project permissions

### Performance Optimization
- **Feature Selection**: Use domain knowledge for feature engineering
- **Hyperparameter Tuning**: Grid search or Bayesian optimization
- **Data Quality**: Ensure clean, consistent input data
- **Model Selection**: Choose appropriate algorithm for data characteristics

## 📈 Performance Improvements

Recent performance restoration includes:
- **Enhanced Feature Engineering**: AQI tiers, cyclical encoding, weekend/night indicators
- **Robust Data Processing**: Better column handling, weather code encoding
- **Improved Target Handling**: Multiple column name format support
- **Extended Rolling Windows**: Additional temporal feature scales

See `PERFORMANCE_RESTORATION_SUMMARY.md` for detailed analysis.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make changes and add tests
4. Update documentation as needed
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints to new functions
- Include docstrings with examples
- Test changes with different model types
- Update configuration documentation

**Note**: This project is designed for educational and research purposes in air quality forecasting. For production use, additional validation and testing are recommended.
