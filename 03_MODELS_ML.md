# Spec 03 — Machine Learning & Deep Learning Models

All models live in `quantflow/models/ml/`.

## 1. Gradient Boosted Trees (XGBoost, LightGBM, CatBoost)
**File:** `gradient_boosting.py`

```python
class GBTSignalModel(BaseQuantModel):
    """
    Predict 21-day forward return quintile (classification) or
    raw return (regression), using all engineered features.
    
    Library: xgboost, lightgbm, catboost (ensemble of all three)
    
    Feature set: all features from spec 01
    Target: sign(ret_21d_forward)  [classification] or ret_21d_forward [regression]
    
    Training protocol:
      - Walk-forward cross-validation (no lookahead)
      - TimeSeriesSplit with gap=21 (skip forward return period)
      - Hyperparameter tuning: Optuna (100 trials, TPE sampler)
    
    Output:
      - SHAP values for every prediction (stored for explainability API)
      - Feature importance ranking
      - Calibrated probability via Platt scaling
    """
    framework: Literal["xgboost","lightgbm","catboost","ensemble"]
```

### Required Hyperparameter Search Space
```python
# LightGBM example
space = {
    "num_leaves": optuna.IntTrial(16, 256),
    "learning_rate": optuna.FloatTrial(1e-3, 0.3, log=True),
    "feature_fraction": optuna.FloatTrial(0.4, 1.0),
    "bagging_fraction": optuna.FloatTrial(0.4, 1.0),
    "min_child_samples": optuna.IntTrial(5, 100),
    "lambda_l1": optuna.FloatTrial(1e-8, 10.0, log=True),
    "lambda_l2": optuna.FloatTrial(1e-8, 10.0, log=True),
}
```

## 2. Random Forest & Regularized Linear Models
**File:** `classic_ml.py`
```python
class RandomForestModel(BaseQuantModel): ...
class LASSOModel(BaseQuantModel): ...
class RidgeModel(BaseQuantModel): ...
class ElasticNetModel(BaseQuantModel): ...
```
- Random Forest: `n_estimators=500`, `max_features="sqrt"`, OOB score as validation
- LASSO/Ridge: rolling standardization required before fit; `cv=TimeSeriesSplit(5)`
- Elastic Net: `l1_ratio` searched via CV

## 3. LSTM / GRU
**File:** `recurrent.py`
```python
class LSTMSignalModel(BaseQuantModel):
    """
    Sequence model for return prediction.
    
    Input: sliding window of T=63 days × F features
    Target: ret_21d_forward (regression head) + direction (classification head)
    Architecture:
      - 2-layer LSTM (hidden=128) or GRU
      - Dropout=0.3 between layers
      - LayerNorm after each recurrent layer
      - Multi-task output: [return_forecast, direction_prob]
    
    Training:
      - AdamW optimizer, lr=1e-3, weight_decay=1e-4
      - CosineAnnealingLR scheduler
      - Early stopping (patience=10 epochs on val loss)
      - Gradient clipping: max_norm=1.0
      - Batch size=64, sequence length=63
    
    Data split: MUST respect time ordering.
      - Train: first 70% of data
      - Val: next 15% (gap=21 days from train end)
      - Test: final 15% (never used for tuning)
    """
```

## 4. Transformer for Time Series
**File:** `transformer_ts.py`
```python
class TimeSeriesTransformerModel(BaseQuantModel):
    """
    Attention-based model for financial time series.
    
    Architecture options (implement all, select best via val loss):
      A) Vanilla Transformer encoder
      B) Informer (sparse attention — efficient for long sequences)
      C) PatchTST (patch-based input — SOTA on benchmarks)
    
    Positional encoding: learnable + sinusoidal hybrid
    Multi-head attention: 8 heads, d_model=128
    Feed-forward dim: 512
    Dropout: 0.1
    
    Additional innovations for finance:
      - Temporal attention mask (prevent attending to future)
      - Regime-conditioning: concatenate Markov regime probs as context
      - Uncertainty head: predict mean + variance (NLL loss)
    """
```
- Use HuggingFace `transformers` if importing pretrained; else pure PyTorch
- Log attention weights to understand which historical periods influence prediction

## 5. CNN for Financial Sequences
**File:** `cnn_ts.py`
```python
class CNNSignalModel(BaseQuantModel):
    """
    1D Convolutional model treating feature matrix as "image".
    
    Architecture:
      - 3x Conv1D blocks (filters: 64→128→256, kernel=3)
      - BatchNorm + ReLU + MaxPool after each block
      - Global Average Pooling
      - Dense head: 256→64→1
    
    Alternative: 2D CNN treating (time × feature) as image.
    Apply to: candlestick pattern recognition on OHLCV.
    """
```

## 6. Deep Reinforcement Learning
**File:** `deep_rl.py`
```python
class DRLPortfolioAgent(BaseQuantModel):
    """
    RL agent that learns portfolio allocation policy.
    
    Environment:
      - State: feature vector at time t (normalized)
      - Action: portfolio weights vector (continuous, softmax constrained)
      - Reward: Sharpe-penalized return = ret_t - 0.5 * lambda * vol_t
                minus transaction cost penalty
    
    Algorithms (implement all three):
      1. PPO (Proximal Policy Optimization) — stable, good for finance
      2. SAC (Soft Actor-Critic) — off-policy, sample efficient
      3. DQN (Deep Q-Network) — discretized action space version
    
    Library: stable-baselines3 + gymnasium
    
    Custom Environment must implement:
      - gym.Env interface
      - realistic transaction costs
      - position limits (max 20% per asset)
      - no shorting constraint (toggle)
    
    Evaluation: compare vs equal-weight and S&P 500 Sharpe ratio
    """
```

## 7. Autoencoders for Regime Discovery
**File:** `autoencoder.py`
```python
class VariationalAutoencoderModel(BaseQuantModel):
    """
    VAE to learn latent market regimes unsupervised.
    
    Encoder: 5 assets × 252 days → latent z (dim=8)
    Decoder: z → reconstructed returns
    Loss: ELBO = reconstruction_loss + KL_divergence
    
    Latent space clustering (KMeans on z):
      - Cluster = regime label
      - Track cluster assignment over time → regime timeline
    
    Use for: discovering new regimes not captured by Markov model
    """
```

## 8. SVM
**File:** `svm.py`  
- RBF kernel, `C` and `gamma` tuned via GridSearchCV
- Use only top-20 SHAP features from GBT to avoid curse of dimensionality
- SVC for direction classification; SVR for return magnitude

## Universal Training Requirements
```python
# quantflow/models/ml/base_trainer.py

class BaseMLTrainer:
    def walk_forward_evaluate(
        self,
        model: BaseQuantModel,
        data: pd.DataFrame,
        n_splits: int = 5,
        gap: int = 21,  # days gap between train and test to avoid leakage
    ) -> EvaluationReport:
        """
        Metrics to compute per fold and aggregate:
          - IC (Information Coefficient): Spearman rank corr of signal vs realized return
          - ICIR: IC / std(IC) → annualized
          - Hit rate: % predictions correct direction
          - Return-weighted accuracy
          - Sharpe ratio of long-short signal portfolio
          - Max drawdown of signal portfolio
        """
```

## MLflow Experiment Tracking
- Every model training run logged to MLflow:
  - Parameters, metrics, artifacts (model pickle, SHAP plots)
  - Tags: `symbol`, `model_type`, `training_date`, `data_version`
- Best model per symbol per category auto-registered in Model Registry
- Automated retraining triggered when IC drops below threshold (0.03)
