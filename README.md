# Weather Forecasting API

A FastAPI-based weather forecasting application that provides machine learning-powered weather predictions using LSTM, sklearn, or XGBoost models.

## Features

- **Startup Model Loading**: Models and processors are loaded once during application startup for optimal performance
- **Multi-step forecasting**: Generate predictions for multiple time horizons
- **Model flexibility**: Supports LSTM (Keras/TensorFlow), sklearn, and XGBoost models
- **Preprocessing Pipeline**: Automatic data standardization and preprocessing with validation
- **Cloud integration**: Integrates with AWS S3 and Weights & Biases (W&B) for model artifacts
- **Caching**: Supports prediction caching to S3 for improved performance
- **Health monitoring**: Built-in health checks and model information endpoints
- **Robust Error Handling**: Comprehensive error handling with graceful fallbacks

## API Endpoints

### Core Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check endpoint with model/processor status
- `GET /model/info` - Get detailed information about the loaded model and processor

### Prediction Endpoints

- `POST /predict/onetime` - Single-step prediction
- `POST /predict/next` - Multi-step forecasting with configurable horizon
- `POST /predict/cache` - Generate 96-step predictions and cache them to S3

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up AWS credentials (for S3 access):
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=your_region
```

3. Set up Weights & Biases (optional, for model artifacts):
```bash
wandb login
```

## Usage

### Starting the Server

#### Using the startup script (recommended):
```bash
python run_app.py
```

#### Using uvicorn directly:
```bash
uvicorn baq.app.main:app --host 0.0.0.0 --port 8000
```

#### With development reload:
```bash
RELOAD=true python run_app.py
```

### Environment Variables

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `WORKERS`: Number of worker processes (default: 1)
- `LOG_LEVEL`: Logging level (default: info)
- `RELOAD`: Enable auto-reload for development (default: false)

### Startup Process

The application follows this startup sequence:

1. **Model Loading**: Downloads or loads models from W&B artifacts or local cache
2. **Processor Loading**: Loads preprocessing artifacts (scalers, encoders, etc.)
3. **Validation**: Validates that loaded artifacts have required methods
4. **Ready**: Application is ready to serve predictions

```
🚀 Loading model and processor during startup...
Loading Keras model from: /path/to/model.h5
Loading processor from: /path/to/processor.joblib
✅ Model and processor loaded successfully!
📊 Model type: Sequential
🔧 Processor type: StandardScaler
```

### API Usage Examples

#### Health Check with Status
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "service": "Weather Forecasting API",
  "model_status": "loaded",
  "processor_status": "loaded"
}
```

#### Model Information
```bash
curl http://localhost:8000/model/info
```

Response:
```json
{
  "model_type": "Sequential",
  "timestamp": "2024-01-15T10:30:00",
  "processor_available": true,
  "processor_type": "StandardScaler",
  "trainable_params": 12345
}
```

#### Single Prediction
```bash
curl -X POST http://localhost:8000/predict/onetime
```

#### Multi-step Forecasting
```bash
curl -X POST "http://localhost:8000/predict/next" \
     -H "Content-Type: application/json" \
     -d '{"forecast_horizon": 24}'
```

#### Cache 96-Hour Predictions
```bash
curl -X POST http://localhost:8000/predict/cache
```

## Model Loading Strategy

The application uses a robust model loading strategy with fallbacks:

### Primary: W&B Artifacts
1. Downloads models from Weights & Biases registry
2. Loads both model and preprocessing artifacts
3. Validates artifact integrity

### Fallback: Local Files
If W&B loading fails, searches for local files in:

**Model Files:**
- `models/best_lstm.h5`
- `models/model.joblib`
- `artifacts/model.h5`
- `artifacts/model.joblib`

**Processor Files:**
- `models/processor.joblib`
- `artifacts/processor.joblib`
- `models/scaler.joblib`
- `artifacts/scaler.joblib`

### Supported Formats

- **Models**: Keras/TensorFlow (`.h5`, `.keras`), scikit-learn/XGBoost (`.joblib`, `.pkl`)
- **Processors**: Any scikit-learn transformer (`.joblib`, `.pkl`)

## Data Processing Pipeline

### 1. Data Loading
- Loads weather data from S3
- Handles time column conversion and indexing

### 2. Column Standardization
- Ensures all required 52 features are present
- Fills missing columns with zeros
- Maintains consistent column order

### 3. Preprocessing
- Applies loaded preprocessing transformations (scaling, encoding)
- Validates processor before application
- Graceful fallback if preprocessing fails

### 4. Prediction
- Supports both single-step and multi-step forecasting
- Handles LSTM sequence creation automatically
- Returns formatted prediction responses

## Input Data Format

The application expects weather data with 52 standardized features:
- **Meteorological**: Temperature, humidity, pressure, wind, precipitation
- **Environmental**: Soil conditions, air quality, UV index
- **Temporal**: Hour, day of week, month, cyclical encodings
- **Lag Features**: PM2.5 historical values and rolling means
- **Categorical**: Weather codes, PM2.5 tier classifications

See `baq.app.utils.standardize_input_columns()` for the complete feature list.

## Error Handling

### Startup Errors
- Model loading failures log errors but allow app to start
- Validation failures are caught and reported
- Graceful degradation for missing processors

### Runtime Errors
- **503 Service Unavailable**: Model not loaded
- **500 Internal Server Error**: Data processing or prediction failures
- **Detailed error messages**: Help with debugging

### Logging
- Startup progress with emoji indicators
- Warning messages for missing processors
- Detailed error reporting for troubleshooting

## Performance Optimizations

- **Single Model Load**: Models loaded once at startup, not per request
- **Preprocessing Validation**: Validates processors to avoid runtime failures
- **Connection Pooling**: Reuses S3 client connections
- **Memory Efficiency**: Proper cleanup on shutdown

## Development

### Project Structure
```
src/baq/
├── app/
│   ├── main.py          # FastAPI application with startup events
│   ├── utils.py         # Preprocessing and validation utilities
│   └── artifacts.py     # W&B artifact handling
├── core/
│   └── inference.py     # Forecasting algorithms
├── models/
│   └── lstm.py          # LSTM model class
└── data/
    └── utils.py         # Data processing utilities
```

### Testing

Monitor startup logs to ensure proper loading:
```bash
python run_app.py
```

Test endpoints using curl or API testing tools.

## Troubleshooting

### Common Issues

1. **Model not loading**
   - Check W&B credentials: `wandb login`
   - Verify local model files exist
   - Check file permissions

2. **Processor warnings**
   - Normal if no preprocessing artifacts available
   - Predictions will use raw standardized features

3. **S3 access errors**
   - Verify AWS credentials and bucket permissions
   - Check network connectivity

4. **Memory issues**
   - Reduce forecast horizon for large predictions
   - Use single worker in development

### Monitoring

- Check `/health` endpoint for service status
- Monitor startup logs for loading issues
- Use `/model/info` for artifact details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure startup process works correctly
5. Submit a pull request

## License

[Add your license information here]
