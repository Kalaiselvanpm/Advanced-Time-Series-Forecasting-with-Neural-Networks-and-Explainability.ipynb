# Advanced Time Series Forecasting with Neural Networks
## Comprehensive Analysis Report

---

## Executive Summary

This project implements a production-ready time series forecasting system combining deep learning (LSTM/Transformer) with statistical baselines, rigorous hyperparameter optimization via Bayesian methods (Optuna), and explainability analysis using SHAP. The goal is to demonstrate mastery over advanced forecasting pipelines while maintaining interpretability.

---

## 1. DATA GENERATION & CHARACTERIZATION

### 1.1 Dataset Overview
- **Size**: 2,000 hourly observations (approximately 83 days)
- **Features**: 3 multivariate time series (feature_0, feature_1, feature_2)
- **Temporal Range**: 2020-01-01 to 2020-03-23

### 1.2 Synthetic Data Components

The generated dataset combines realistic time series components:

**Trend Component**: Linear trend with annual cyclical modulation
- Formula: `0.01*t + 50*sin(2π*t/365)`
- Captures long-term drift and yearly seasonality
- Amplitude: ~50 units

**Primary Seasonality (Daily Pattern)**: 7-step cycle
- Formula: `10*sin(2π*t/7)`
- Represents 7-hour recurring patterns (e.g., workday cycles)
- Amplitude: 10 units

**Secondary Seasonality (Weekly Pattern)**: 30-step cycle
- Formula: `8*sin(2π*t/30)`
- Represents monthly business cycles
- Amplitude: 8 units

**Tertiary Seasonality (Yearly Pattern)**: 365-step cycle
- Formula: `5*cos(2π*t/365)`
- Represents annual seasonality (e.g., heating/cooling demand)
- Amplitude: 5 units

**Autoregressive Component**: AR(1) process
- Maintains temporal dependency: `AR[t] = 0.7*AR[t-1] + 0.3*AR[t-1]*noise`
- Coefficient 0.7 ensures persistence without explosive growth
- Models self-reinforcing patterns

**Noise**: Gaussian white noise
- Distribution: N(0, 2)
- Standard deviation: 2 units
- Represents measurement error and unmeasured influences

### 1.3 Justification for Synthetic Data

Synthetic data provides several advantages:
1. **Ground Truth Knowledge**: We know exact data-generating process
2. **Reproducibility**: Consistent results across experiments
3. **Complexity Control**: Progressively add complexity
4. **Baseline Establishment**: Compare against known patterns
5. **No Privacy Concerns**: Suitable for public repositories

---

## 2. DATA PREPROCESSING & FEATURE ENGINEERING

### 2.1 Scaling Strategy

**StandardScaler Applied Per-Feature**:
- Fit on training data only (prevents data leakage)
- Transform: `X_scaled = (X - mean) / std`
- Maintains Gaussian distribution assumption in neural networks
- Inverse transformation for final predictions

**Rationale**:
- Neural networks train faster with normalized inputs
- Prevents gradient explosion/vanishing
- Ensures fair feature contribution during optimization

### 2.2 Sequence Windowing

**Temporal Window Configuration**:
- **Lookback Window**: 24 time steps (1 day of historical data)
- **Forecast Horizon**: 6 time steps (6-hour ahead predictions)
- **Total Context**: 24 hours to predict next 6 hours

**Sequence Creation Logic**:
```
For each position i in dataset:
  X[i] = data[i:i+24, :]  # All 3 features, 24 time steps
  y[i] = data[i+24:i+30, feature_0]  # Next 6 hours, target feature
```

**Dataset Splits** (Temporal Order Preservation):
- Training: 70% (1,400 sequences)
- Validation: 15% (300 sequences)
- Testing: 15% (300 sequences)

**Critical**: No shuffling between train/test to maintain temporal integrity

---

## 3. DEEP LEARNING ARCHITECTURE

### 3.1 LSTM Model (Primary)

```
Input (batch_size, 24, 3)
    ↓
LSTM Layer (bidirectional=False)
  - Hidden Size: [32, 64, 128, 256] (optimized)
  - Num Layers: [1, 2, 3, 4] (optimized)
  - Dropout: [0.1-0.5] (optimized)
    ↓
Last Hidden State (batch_size, hidden_size)
    ↓
Dense Layer (128 units) + ReLU
    ↓
Dropout Layer
    ↓
Output Layer (6 units) → 6-step forecast
```

**LSTM Advantages**:
- Captures long-range dependencies via cell states
- Mitigates vanishing gradient problem
- Interpretable hidden states and gates
- Well-established for time series

### 3.2 Transformer Model (Alternative)

```
Input (batch_size, 24, 3)
    ↓
Linear Projection → d_model dimensions
    ↓
Positional Encoding (sinusoidal, max_len=500)
    ↓
Transformer Encoder
  - Multi-Head Attention: [2, 4, 8] heads (optimized)
  - Feed-Forward: 256 hidden units
  - Num Layers: [1, 2, 3] (optimized)
  - Dropout: [0.1-0.5] (optimized)
    ↓
Last Encoder Output
    ↓
Dense Layer (128 units) + ReLU
    ↓
Dropout Layer
    ↓
Output Layer (6 units)
```

**Transformer Advantages**:
- Parallel processing (faster training)
- Attention weights provide interpretability
- Better at capturing complex temporal relationships
- Scalable to longer sequences

### 3.3 Loss Function & Optimization

**Loss Function**: Mean Squared Error (MSE)
- Chosen for regression task
- Sensitive to outliers (encourages accuracy)
- Differentiable for gradient-based optimization

**Optimizer**: Adam
- Adaptive learning rates per parameter
- Momentum and RMSprop combination
- Typically converges faster than SGD

**Learning Rate Scheduler**: ReduceLROnPlateau
- Reduces LR by factor of 0.5 if validation loss plateaus
- Patience: 5 epochs before reduction
- Enables fine-tuning in later training stages

**Gradient Clipping**: max_norm=1.0
- Prevents gradient explosion in RNNs
- Stabilizes training dynamics

---

## 4. HYPERPARAMETER OPTIMIZATION

### 4.1 Optuna Configuration

**Optimization Strategy**: Tree-structured Parzen Estimator (TPE)
- Probabilistic model: generates promising hyperparameter combinations
- Balances exploration vs exploitation
- More efficient than random search or grid search

**Search Space**:

| Parameter | Range | Type | Rationale |
|-----------|-------|------|-----------|
| Hidden Size | [32, 256] | Categorical (step=32) | Balance capacity vs overfitting |
| Num Layers | [1, 4] | Categorical | Depth vs training stability |
| Dropout | [0.1, 0.5] | Continuous | Regularization strength |
| Learning Rate | [1e-4, 1e-2] | Log-uniform | Convergence speed vs stability |
| Batch Size | [16, 128] | Categorical (step=16) | Gradient estimates quality |

**Pruning Strategy**: MedianPruner
- Prunes unpromising trials early
- Compares current trial to historical median
- Saves computational resources
- Startup trials: 5 (collect baseline)
- Warmup steps: 10 (allow trials to develop)

**Number of Trials**: 15-20
- Typical hyperparameter space exploration
- Balance between thoroughness and computational cost
- TPE becomes effective after ~5 trials

### 4.2 Optimization Workflow

1. **Trial Generation**: TPE samples from search space
2. **Model Instantiation**: Create model with suggested hyperparameters
3. **Training Loop**: 
   - Train for max 100 epochs
   - Early stopping if validation loss doesn't improve (patience=10)
   - Track learning curves
4. **Validation Evaluation**: Compute MSE on validation set
5. **Trial Report**: Report metric to Optuna
6. **Pruning Decision**: Optuna may prune poor-performing trials
7. **Iteration**: Repeat for n_trials

### 4.3 Expected Best Hyperparameters

Based on typical convergence patterns:

```
LSTM Configuration:
  Hidden Size: ~128-192
  Num Layers: 2-3
  Dropout: ~0.3-0.4
  Learning Rate: ~1e-3 to 5e-3
  Batch Size: 32-64

Reasoning:
  - Single layer too shallow; 4+ layers prone to vanishing gradients
  - Medium hidden size provides good expressiveness
  - Moderate dropout prevents overfitting without underfitting
  - Learning rates in 1e-3 range typical for Adam + time series
  - Batch sizes 32-64 provide stable gradient estimates
```

---

## 5. STATISTICAL BASELINE: SARIMAX

### 5.1 Model Specification

**SARIMAX(p,d,q)(P,D,Q,s) = (1,1,1)(1,1,1,24)**

- **AR(p=1)**: Autoregressive order 1 (use previous value)
- **I(d=1)**: Differencing order 1 (stationary transformation)
- **MA(q=1)**: Moving average order 1 (error term)
- **Seasonal AR(P=1)**: Seasonal autoregressive
- **Seasonal I(D=1)**: Seasonal differencing
- **Seasonal MA(Q=1)**: Seasonal moving average
- **Seasonal Period(s=24)**: 24-hour seasonality

### 5.2 SARIMAX Rationale

SARIMAX chosen because:
1. **Simplicity**: Non-neural baseline for comparison
2. **Interpretability**: Explicit statistical assumptions
3. **Seasonality Handling**: Explicitly captures 24-hour patterns
4. **Speed**: Fast training and inference
5. **Industry Standard**: Established forecasting method

### 5.3 Limitations of SARIMAX

- **Linear Relationships Only**: Cannot capture nonlinear temporal patterns
- **Stationarity Assumption**: Requires stationary data after differencing
- **Parameter Sensitivity**: Small changes in (p,d,q) yield different results
- **Scalability**: Difficult with many exogenous variables
- **No Learned Features**: Uses fixed statistical principles

### 5.4 Performance Expectations

Typical SARIMAX performance on synthetic data:
- **RMSE**: Often 15-30% higher than optimized LSTM
- **Bias**: May systematically over/underestimate trends
- **Advantage**: More stable for simple patterns
- **Disadvantage**: Misses complex nonlinear dynamics

---

## 6. MODEL INTERPRETABILITY & EXPLAINABILITY

### 6.1 SHAP (SHapley Additive exPlanations)

**Mathematical Foundation**: Game theory-based explanation
- Each feature is a "player" contributing to prediction
- SHAP value = marginal contribution of feature to prediction
- Satisfies efficiency, symmetry, and consistency axioms

**Implementation Strategy**:

```python
1. Flatten sequences: (N, 24, 3) → (N, 72)
2. Create background data: random subset of training sequences
3. Define prediction function: maps flattened → model output
4. KernelExplainer: model-agnostic explanation
5. Compute SHAP values for test samples
```

**Interpretation**:

- **High SHAP Value**: Feature strongly influences prediction
- **Positive SHAP**: Feature pushes prediction up
- **Negative SHAP**: Feature pushes prediction down
- **Base Value**: Model's average prediction without features

### 6.2 Key Questions Answered by SHAP

1. **Which historical lags matter most?**
   - Analyze SHAP values across time lags (0-23 hours)
   - Identify critical lookback windows
   - Example: If 6-12 hour lags have high SHAP, model relies on daily patterns

2. **Which features drive predictions?**
   - Compare SHAP contributions: feature_0, feature_1, feature_2
   - Identify which variables are most predictive
   - Example: If feature_0 has 70% of SHAP mass, focus on that variable

3. **When does the model fail?**
   - Compute SHAP on high-error predictions
   - Identify systematic weaknesses
   - Example: Large positive SHAP on feature indicating prediction up, but actual goes down

4. **Are predictions stable?**
   - SHAP consistency: similar inputs → similar explanations
   - Absence of random SHAP patterns = stable model
   - Presence of noise suggests overfitting

### 6.3 Attention Weight Visualization (Transformer Only)

If using Transformer architecture:

```python
# Extract attention weights from transformer layer
attention_weights = transformer_encoder[layer_idx].self_attn.forward()
# Shape: (batch_size, num_heads, seq_len, seq_len)

# Visualize which timesteps attend to which
import matplotlib.pyplot as plt
plt.imshow(attention_weights[0, 0, :, :], cmap='hot')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.title('Attention Pattern (Head 0)')
```

**Interpretation**:
- Diagonal pattern = model focuses on recent history
- Vertical stripes = certain timesteps critical for all future steps
- Sparse pattern = model selectively attends to specific lags

---

## 7. COMPARATIVE ANALYSIS: LSTM vs SARIMAX

### 7.1 Performance Metrics

**RMSE (Root Mean Squared Error)**:
- Measures average prediction error
- LSTM typically achieves 15-30% lower RMSE
- Lower is better

**MAE (Mean Absolute Error)**:
- Average absolute deviation
- More robust to outliers than RMSE
- LSTM advantage: 10-25%

**R² Score (Coefficient of Determination)**:
- Proportion of variance explained
- LSTM typically: 0.75-0.92
- SARIMAX: 0.60-0.85
- Range: [-∞, 1.0], higher is better

### 7.2 Performance by Forecast Horizon

| Step | Description | LSTM RMSE | SARIMAX RMSE | Advantage |
|------|-------------|-----------|--------------|-----------|
| 1 | 1 hour ahead | Best | Good | LSTM 20-30% |
| 2-3 | 2-3 hours ahead | Good | Moderate | LSTM 15-25% |
| 4-5 | 4-5 hours ahead | Moderate | Moderate | LSTM 10-20% |
| 6 | 6 hours ahead | Worst | Poor | LSTM 25-40% |

**Pattern**: LSTM maintains accuracy across horizons; SARIMAX degrades faster

### 7.3 Qualitative Comparison

**LSTM Strengths**:
- Captures nonlinear temporal dependencies
- Learns hierarchical representations
- Adapts to complex seasonality
- Better long-horizon forecasts

**LSTM Weaknesses**:
- Black-box nature (less interpretable)
- Requires more data for training
- Computationally expensive
- Risk of overfitting with small datasets

**SARIMAX Strengths**:
- Interpretable parameters
- Explicit statistical framework
- Works with limited data
- Fast inference

**SARIMAX Weaknesses**:
- Limited to linear relationships
- Requires manual parameter tuning
- Struggles with complex patterns
- Assumption violations lead to errors

### 7.4 Recommendation

**Choose LSTM if**:
- Complex, nonlinear patterns present
- Large dataset available (>1,000 observations)
- Long forecast horizons needed
- Interpretability via SHAP acceptable

**Choose SARIMAX if**:
- Statistical rigor required
- Limited training data
- Simple, linear patterns
- Traditional forecasting framework needed

**Hybrid Approach**:
- Ensemble: Average LSTM and SARIMAX predictions
- Typically improves robustness
- Combines strengths of both methods

---

## 8. MODEL FEATURE IMPORTANCE (FROM SHAP)

### 8.1 Lag-wise Importance

Analysis of which historical timesteps drive predictions:

```
High Importance Lags (Typical Pattern):
  - Lag 1-2: Previous 1-2 hours (immediate past)
  - Lag 7: 7 hours ago (daily cycle)
  - Lag 24: 24 hours ago (yesterday same hour)

Interpretation:
  Lag 1-2 High → Model uses recent momentum
  Lag 7 High → Strong 7-hour cycle captured
  Lag 24 High → Daily seasonality dominates
```

### 8.2 Feature-wise Importance

Multivariate contributions:

```
Typical Distribution (feature_0, feature_1, feature_2):
  feature_0: 50% (primary target)
  feature_1: 30% (secondary influence)
  feature_2: 20% (tertiary influence)

Interpretation:
  Feature_0 contains most predictive information
  Feature_1 provides weak additional signal
  Feature_2 may be noise or redundant
```

### 8.3 Error Analysis via SHAP

Predictions where LSTM fails:

```
High Error Characteristics:
  - Sudden value jumps (model unprepared)
  - Contradictory SHAP signals (model uncertainty)
  - Out-of-distribution lags (unseen patterns)
  - Feature anomalies (unmeasured shocks)

Mitigation:
  - Add anomaly detection layer
  - Increase model capacity
  - Include exogenous variables
  - Implement uncertainty quantification
```

---

## 9. DOCUMENTATION & REPRODUCIBILITY

### 9.1 Code Organization

```
time_series_forecasting/
├── data_generation.py       # Synthetic data creation
├── preprocessing.py         # Scaling, windowing
├── models.py               # LSTM, Transformer architectures
├── optimization.py         # Optuna hyperparameter tuning
├── training.py             # Training loops, early stopping
├── evaluation.py           # Metrics, comparisons
├── explainability.py       # SHAP, attention visualization
├── baseline_models.py      # SARIMAX, Prophet
├── visualization.py        # Plots, dashboards
├── main.py                 # Main execution script
├── requirements.txt        # Dependencies
└── README.md              # Full documentation
```

### 9.2 Key Assumptions & Dependencies

**Dependencies**:
- torch, torchvision (deep learning)
- numpy, pandas (data manipulation)
- scikit-learn (preprocessing, metrics)
- optuna (hyperparameter optimization)
- statsmodels (SARIMAX)
- shap (model explainability)
- matplotlib, seaborn (visualization)

**Assumptions**:
1. Time series is univariate or multivariate with clear structure
2. Training data contains sufficient observations (>500)
3. Stationarity or differencing makes data stationary
4. No extreme outliers or missing values
5. Temporal dependencies captured within lookback window

### 9.3 Reproducibility Measures

- **Fixed Random Seeds**: `np.random.seed(42)`, `torch.manual_seed(42)`
- **Deterministic Operations**: GPU operations may not be fully deterministic
- **Version Pinning**: Specific versions of key libraries
- **Documented Hyperparameters**: All defaults clearly specified
- **Sample Data**: Synthetic generation reproducible from code

---

## 10. PRODUCTION CONSIDERATIONS

### 10.1 Model Deployment

**Checkpoints**:
```python
torch.save(lstm_model.state_dict(), 'lstm_best_model.pt')
# Later: lstm_model.load_state_dict(torch.load('lstm_best_model.pt'))
```

**Serialization**:
- Use ONNX for framework-agnostic deployment
- Include scaler objects (pickle serialization)
- Version models with metadata

### 10.2 Inference Pipeline

```
Real-time data → Preprocessing (scaling, windowing)
    ↓
Sequence (24, 3) → LSTM Model → Raw predictions (6,)
    ↓
Inverse transform → Original scale predictions (6,)
    ↓
Uncertainty quantification → Prediction intervals
    ↓
Output dashboard with confidence bounds
```

### 10.3 Monitoring & Maintenance

- **Concept Drift**: Monitor prediction accuracy over time
- **Data Distribution Shift**: Retrain monthly with recent data
- **Model Degradation**: Alert if metrics fall below thresholds
- **Explainability Monitoring**: Track SHAP values for stability

---

## 11. CONCLUSION

This comprehensive framework demonstrates:

1. **Data Engineering**: Realistic synthetic data with complex seasonalities
2. **Deep Learning**: State-of-the-art LSTM/Transformer architectures
3. **Optimization**: Bayesian hyperparameter tuning via Optuna
4. **Interpretability**: SHAP-based model explanation
5. **Baselines**: Statistical comparison with SARIMAX
6. **Production Readiness**: Scalable, documented, monitored

**Expected Outcomes**:
- LSTM outperforms SARIMAX by 15-30% on RMSE
- Hyperparameter optimization improves baseline by 10-20%
- SHAP reveals lags 1-2 and 24 are most important
- Model suitable for production forecasting tasks

**Future Extensions**:
- Multivariate output forecasting (forecast all features)
- Uncertainty quantification (prediction intervals)
- Ensemble methods combining LSTM + SARIMAX
- Autoencoder-based anomaly detection
- Attention visualization for Transformer models
- Online learning for concept drift adaptation
- # Installation & Usage Guide
## Advanced Time Series Forecasting Project

---

## 1. ENVIRONMENT SETUP

### 1.1 Create Virtual Environment

```bash
# Using conda
conda create -n timeseries-forecast python=3.9
conda activate timeseries-forecast

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 1.2 Install Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Data & ML
pip install numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0

# Hyperparameter Optimization
pip install optuna==3.12.0

# Time Series & Statistics
pip install statsmodels==0.14.0 pmdarima==2.0.3

# Explainability
pip install shap==0.42.1

# Visualization
pip install matplotlib==3.7.2 seaborn==0.12.2

# Optional: Prophet for advanced baseline
pip install pystan==2.19.1.1 prophet==1.1.5

# Optional: Jupyter for interactive exploration
pip install jupyter==1.0.0 ipython==8.14.0
```

### 1.3 Verify Installation

```python
import torch
import pandas as pd
import numpy as np
import optuna
import statsmodels
import shap

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("All imports successful!")
```

---

## 2. QUICK START

### 2.1 Run Complete Pipeline

```bash
python main.py
```

This executes all steps:
1. Generate synthetic data
2. Preprocess and create sequences
3. Run Optuna hyperparameter optimization
4. Train optimized LSTM model
5. Train SARIMAX baseline
6. Generate SHAP explanations
7. Create visualizations
8. Print analysis report

**Expected Runtime**: 
- Without GPU: 15-30 minutes
- With GPU: 5-10 minutes

### 2.2 Run Specific Components

```python
# Only generate data
from data_generation import generate_multivariate_timeseries
df = generate_multivariate_timeseries(n_steps=2000)

# Only train LSTM without optimization
from models import LSTMForecaster
from training import train_model
model = LSTMForecaster(input_size=3, hidden_size=128, 
                       num_layers=2, dropout=0.3, 
                       forecast_horizon=6)

# Only run SHAP analysis
from explainability import explain_predictions
explainer, shap_vals, X_test = explain_predictions(
    model, X_test, feature_names, device)
```

---

## 3. CONFIGURATION & CUSTOMIZATION

### 3.1 Modify Data Generation

```python
# Generate larger dataset with different characteristics
df = generate_multivariate_timeseries(
    n_steps=5000,        # Increase to 5000 samples
    n_features=5,        # Add more features
    seed=123             # Different random seed
)
```

### 3.2 Adjust Model Architecture

```python
# Custom LSTM configuration
model = LSTMForecaster(
    input_size=3,
    hidden_size=256,         # Increase capacity
    num_layers=3,            # Add depth
    dropout=0.35,            # Adjust regularization
    forecast_horizon=12,     # Predict 12 steps ahead
    bidirectional=True       # Add bidirectional processing
)
```

### 3.3 Customize Preprocessing

```python
preprocessor = TimeSeriesPreprocessor(
    lookback=48,             # Use 2 days history
    forecast_horizon=24      # Predict 1 day ahead
)
```

### 3.4 Control Hyperparameter Optimization

```python
best_params = optimize_hyperparameters(
    X_train, y_train, X_val, y_val,
    model_type='lstm',       # Or 'transformer'
    n_trials=50,             # More trials for better results
    device='cuda'            # Force GPU usage
)
```

---

## 4. INTERPRETING RESULTS

### 4.1 Understanding Output Metrics

**RMSE (Root Mean Squared Error)**
```
RMSE = sqrt(mean((y_actual - y_pred)^2))
- Lower is better
- Same units as target variable
- Penalizes large errors heavily
```

**MAE (Mean Absolute Error)**
```
MAE = mean(|y_actual - y_pred|)
- More interpretable than RMSE
- Represents average prediction error
- More robust to outliers
```

**R² Score**
```
R² = 1 - (SS_res / SS_tot)
- Range: (-∞, 1.0]
- 0.8-1.0: Excellent fit
- 0.6-0.8: Good fit
- <0.6: Poor fit
```

### 4.2 Interpreting SHAP Values

```python
import shap

# Bar plot of mean absolute SHAP values
shap.summary_plot(shap_vals, X_test_flat, plot_type="bar")
# Shows which features most important overall

# Beeswarm plot
shap.summary_plot(shap_vals, X_test_flat)
# Shows distribution of feature impacts

# Force plot for single prediction
shap.force_plot(explainer.expected_value, 
                shap_vals[0], X_test_flat[0])
# Visualizes individual prediction breakdown
```

### 4.3 Analyzing Forecast Performance

```python
# Performance by time horizon
import matplotlib.pyplot as plt

horizons = [1, 2, 3, 4, 5, 6]
rmses = [0.85, 0.92, 1.05, 1.18, 1.35, 1.52]

plt.plot(horizons, rmses, marker='o')
plt.xlabel('Forecast Horizon (hours)')
plt.ylabel('RMSE')
plt.title('Error Growth Across Horizons')
plt.grid(True, alpha=0.3)
plt.show()

# Analysis: Error grows with horizon
# Typical: ~20% increase per step
```

---

## 5. TROUBLESHOOTING

### 5.1 Common Issues

**Issue**: "CUDA out of memory"
```python
# Solution 1: Reduce batch size
batch_size = 16  # Instead of 128

# Solution 2: Use CPU
device = torch.device('cpu')

# Solution 3: Clear cache
torch.cuda.empty_cache()
```

**Issue**: "NaN values in predictions"
```python
# Causes: Gradient explosion, bad learning rate
# Solutions:
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Lower LR
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
model = model.float()  # Ensure float32, not float16
```

**Issue**: "Model overfitting"
```python
# Increase dropout
dropout = 0.5

# Add L2 regularization
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Use early stopping (already implemented)
patience = 20  # Increase from 15
```

**Issue**: "Optuna optimization slow"
```python
# Reduce n_trials
n_trials = 10  # Instead of 20

# Use fewer epochs per trial
epochs = 50  # Instead of 100

# Prune more aggressively
pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=5)
```

**Issue**: "SHAP computation slow"
```python
# Reduce n_samples
explainer, shap_vals = explain_predictions(
    model, X_test, feature_names, device, n_samples=20)

# Use sampling-based explainer
explainer = shap.SamplingExplainer(predict_fn, X_background)
```

### 5.2 Performance Optimization

**For CPU-only systems**:
```python
# Reduce model size
hidden_size = 64
num_layers = 1
batch_size = 32

# Use simpler preprocessing
lookback = 12
forecast_horizon = 3
```

**For GPU systems**:
```python
# Increase batch size for efficiency
batch_size = 256

# Use larger models
hidden_size = 256
num_layers = 3

# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

---

## 6. EXTENDING THE PROJECT

### 6.1 Add Uncertainty Quantification

```python
class LSTMForecasterWithUncertainty(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, 
                 dropout, forecast_horizon):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           dropout=dropout, batch_first=True)
        self.fc_mean = nn.Linear(hidden_size, forecast_horizon)
        self.fc_std = nn.Linear(hidden_size, forecast_horizon)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        mean = self.fc_mean(last_hidden)
        std = torch.exp(self.fc_std(last_hidden))  # Ensure positive
        return mean, std
```

### 6.2 Implement Multivariate Output

```python
# Predict all features simultaneously
y_train_multivariate = np.concatenate([
    y_train[:, :, np.newaxis],  # Current target
    X_train[:, -1, 1:].reshape(len(X_train), 1, -1)  # Other features
], axis=-1)  # Shape: (N, 6, 3)

# Update model output layer
self.fc2 = nn.Linear(128, forecast_horizon * num_features)
output = self.fc2(x).reshape(-1, forecast_horizon, num_features)
```

### 6.3 Add Anomaly Detection

```python
from sklearn.ensemble import IsolationForest

# Detect anomalies in training data
iso_forest = IsolationForest(contamination=0.05)
anomalies = iso_forest.fit_predict(X_train.reshape(len(X_train), -1))

# Filter out anomalies
X_train_clean = X_train[anomalies == 1]
y_train_clean = y_train[anomalies == 1]
```

### 6.4 Implement Ensemble Methods

```python
# Average predictions from LSTM and Transformer
def ensemble_predict(X, lstm_model, transformer_model, device):
    lstm_model.eval()
    transformer_model.eval()
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        lstm_preds = lstm_model(X_tensor).cpu().numpy()
        transformer_preds = transformer_model(X_tensor).cpu().numpy()
    
    ensemble_preds = (lstm_preds + transformer_preds) / 2
    return ensemble_preds
```

---

## 7. ADVANCED TOPICS

### 7.1 Distributed Training (Multi-GPU)

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model = model.to(device)
```

### 7.2 Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        with autocast():
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
```

### 7.3 Quantization for Deployment

```python
# Quantize model for faster inference
model_quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8)

# Save quantized model
torch.save(model_quantized.state_dict(), 
          'lstm_quantized.pt')
```

---

## 8. PROJECT SUBMISSION CHECKLIST

- [ ] All code runs without errors
- [ ] Generated synthetic dataset documented
- [ ] LSTM/Transformer implemented with architecture diagram
- [ ] Optuna optimization completed with 15+ trials
- [ ] Best hyperparameters identified and documented
- [ ] SHAP explanations generated for test samples
- [ ] SARIMAX baseline trained and evaluated
- [ ] Comparative analysis written (500+ words)
- [ ] Visualizations created (performance plots, residuals, SHAP)
- [ ] Model checkpoints saved
- [ ] README.md with full documentation
- [ ] requirements.txt with all dependencies
- [ ] Code follows PEP 8 style guide
- [ ] Comments explain non-obvious logic
- [ ] Test script demonstrates reproducibility

---

## 9. REFERENCES & RESOURCES

### Foundational Papers
- Hochreiter & Schmidhuber (1997): LSTM original paper
- Vaswani et al. (2017): Attention is All You Need
- Lundberg & Lee (2017): SHAP framework

### Libraries Documentation
- PyTorch: https://pytorch.org/docs/stable/index.html
- Optuna: https://optuna.readthedocs.io/
- SHAP: https://shap.readthedocs.io/
- Statsmodels: https://www.statsmodels.org/

### Time Series Courses
- Fast.ai: Practical Deep Learning for Coders
- Andrew Ng: Sequence Models (Coursera)
- Forecasting: Principles and Practice (otexts.com)

---

## 10. CONTACT & SUPPORT

For issues or questions:
1. Check troubleshooting section above
2. Review error messages carefully
3. Consult library documentation
4. Test with smaller dataset first
5. Enable debug mode: `logging.basicConfig(level=logging.DEBUG)`
