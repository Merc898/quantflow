"""Deep Reinforcement Learning portfolio agent.

Custom gymnasium environment with Sharpe-penalised reward, transaction costs,
and position limits.  Supports PPO, SAC, and DQN via stable-baselines3.

The agent's final portfolio weight for the target asset is converted to a
ModelOutput signal for use by the signal fusion engine.
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from quantflow.config.constants import (
    MAX_POSITION_SIZE,
    TRADING_DAYS_PER_YEAR,
)
from quantflow.config.logging import get_logger
from quantflow.models.base import BaseQuantModel, ModelOutput

logger = get_logger(__name__)

DRLAlgorithm = Literal["PPO", "SAC", "DQN"]

_N_TRAINING_STEPS = 50_000  # total env steps (reduce for speed; scale up in production)
_TRANSACTION_COST_BPS = 10  # 10 basis points per trade
_MAX_WEIGHT = MAX_POSITION_SIZE
_REWARD_LAMBDA = 0.5  # Sharpe penalty coefficient


# ---------------------------------------------------------------------------
# Custom gymnasium environment
# ---------------------------------------------------------------------------

try:
    import gymnasium as _gymnasium_base

    _GYMNASIUM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _gymnasium_base = None  # type: ignore[assignment]
    _GYMNASIUM_AVAILABLE = False

if TYPE_CHECKING:
    import pandas as pd

# When gymnasium is unavailable, use object as a stand-in base so the module
# still imports (tests guard with importorskip and skip gracefully).
_EnvBase: type = _gymnasium_base.Env if _GYMNASIUM_AVAILABLE else object  # type: ignore[union-attr]


class PortfolioEnv(_EnvBase):  # type: ignore[misc]
    """Single-asset portfolio environment following the gymnasium Env interface.

    State: Normalised feature vector at time ``t``.
    Action (continuous): Portfolio weight in [-1, +1] (negative = short).
    Reward: Risk-adjusted return minus transaction costs.

    Args:
        features: Feature matrix (N, F) — pre-computed, pre-scaled.
        returns: Daily return vector (N,) — aligned to features.
        allow_shorting: If False, clip action to [0, max_weight].
        max_weight: Maximum absolute position size.
        tc_bps: Transaction cost in basis points per trade (round-trip half).
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        allow_shorting: bool = False,
        max_weight: float = _MAX_WEIGHT,
        tc_bps: float = _TRANSACTION_COST_BPS,
    ) -> None:
        """Initialise the portfolio environment.

        Args:
            features: Pre-scaled feature matrix.
            returns: Aligned daily return vector.
            allow_shorting: Allow negative weights.
            max_weight: Maximum absolute position size.
            tc_bps: Transaction cost in basis points.
        """
        super().__init__()

        self.features = features.astype(np.float32)
        self.returns = returns.astype(np.float32)
        self.allow_shorting = allow_shorting
        self.max_weight = max_weight
        self.tc = tc_bps / 10_000.0

        self.n_steps = len(features)
        self.n_features = features.shape[1]

        # gymnasium spaces
        obs_low = np.full(self.n_features, -5.0, dtype=np.float32)
        obs_high = np.full(self.n_features, 5.0, dtype=np.float32)
        self.observation_space = _gymnasium_base.spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = _gymnasium_base.spaces.Box(
            low=np.array([-1.0] if allow_shorting else [0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._t: int = 0
        self._prev_weight: float = 0.0
        self._returns_history: list[float] = []

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to time step 0.

        Args:
            seed: Random seed (unused — deterministic env).
            options: Unused.

        Returns:
            Initial observation and info dict.
        """
        self._t = 0
        self._prev_weight = 0.0
        self._returns_history = []
        return self.features[0], {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take one environment step.

        Args:
            action: Scalar weight in the asset.

        Returns:
            (next_obs, reward, terminated, truncated, info)
        """
        weight = float(np.clip(action[0], -1.0 if self.allow_shorting else 0.0, self.max_weight))

        # Realised return for this step (weight applied to next-bar return)
        ret = float(self.returns[self._t])
        portfolio_ret = weight * ret

        # Transaction cost (one-way, proportional to change in weight)
        tc_cost = abs(weight - self._prev_weight) * self.tc
        net_ret = portfolio_ret - tc_cost
        self._prev_weight = weight
        self._returns_history.append(net_ret)

        # Reward: Sharpe-penalised return
        if len(self._returns_history) > 20:
            hist = np.array(self._returns_history[-20:])
            vol = float(hist.std()) + 1e-8
            reward = net_ret - _REWARD_LAMBDA * vol
        else:
            reward = net_ret

        self._t += 1
        done = self._t >= self.n_steps - 1
        next_obs = self.features[min(self._t, self.n_steps - 1)]
        return next_obs, reward, done, False, {"weight": weight, "ret": ret}

    def render(self) -> None:
        """No-op render (not used)."""


# ---------------------------------------------------------------------------
# DQN discrete wrapper (discretise the action space)
# ---------------------------------------------------------------------------


class DiscretePortfolioEnv(PortfolioEnv):
    """Discrete action space variant for DQN.

    Actions: 0 = cash (weight=0), 1 = 25%, 2 = 50%, 3 = 75%, 4 = 100%.
    """

    _WEIGHTS = [0.0, 0.25, 0.50, 0.75, 1.0]

    def __init__(self, features: np.ndarray, returns: np.ndarray, **kwargs: Any) -> None:
        super().__init__(features, returns, **kwargs)
        self.action_space = _gymnasium_base.spaces.Discrete(len(self._WEIGHTS))

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        idx = int(action) if not isinstance(action, np.ndarray) else int(action.item())
        weight_arr = np.array([self._WEIGHTS[idx]], dtype=np.float32)
        return super().step(weight_arr)


# ---------------------------------------------------------------------------
# Main DRL model class
# ---------------------------------------------------------------------------


class DRLPortfolioAgent(BaseQuantModel):
    """Deep RL portfolio agent (PPO / SAC / DQN).

    Trains a stable-baselines3 agent in the custom PortfolioEnv.
    The trained policy's recommended weight is converted to a signal.

    Args:
        symbol: Ticker symbol.
        algorithm: "PPO", "SAC", or "DQN".
        n_training_steps: Total training env steps.
        allow_shorting: Whether to allow short positions.
        mlflow_experiment: MLflow experiment name.
    """

    def __init__(
        self,
        symbol: str,
        algorithm: DRLAlgorithm = "PPO",
        n_training_steps: int = _N_TRAINING_STEPS,
        allow_shorting: bool = False,
        mlflow_experiment: str | None = "quantflow_drl",
    ) -> None:
        """Initialise the DRL portfolio agent.

        Args:
            symbol: Ticker symbol.
            algorithm: RL algorithm to use.
            n_training_steps: Total environment interactions.
            allow_shorting: Allow negative weights.
            mlflow_experiment: MLflow experiment name.
        """
        super().__init__(f"DRL_{algorithm}", symbol)
        self.algorithm: DRLAlgorithm = algorithm
        self.n_training_steps = n_training_steps
        self.allow_shorting = allow_shorting
        self.mlflow_experiment = mlflow_experiment

        self._agent: Any = None
        self._env: PortfolioEnv | DiscretePortfolioEnv | None = None
        self._scaler: Any = None
        self._vol_estimate: float = 0.02
        self._train_sharpe: float = 0.0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> DRLPortfolioAgent:
        """Train the RL agent on historical feature/return data.

        Args:
            data: DataFrame with ``close`` and feature columns.

        Returns:
            Self (fitted model).
        """
        from sklearn.preprocessing import StandardScaler as SKScaler

        from quantflow.data.features import compute_all_features
        from quantflow.models.ml.gradient_boosting import _build_xy

        close = data["close"].astype(np.float64)
        required = {"open", "high", "low", "close", "volume"}
        features = (
            compute_all_features(data)
            if required.issubset(set(data.columns))
            else data.drop(columns=["close"], errors="ignore")
        )

        # Use 1-day return as the immediate reward return
        X_raw, y_raw, _valid_idx = _build_xy(features, close, horizon=1)
        if len(X_raw) < 100:
            raise ValueError("Insufficient data for DRL training.")

        self._vol_estimate = float(np.std(y_raw))

        # Scale features
        self._scaler = SKScaler()
        X_scaled = self._scaler.fit_transform(X_raw).astype(np.float32)
        returns = y_raw.astype(np.float32)

        # Use first 80% for training
        n_train = int(len(X_scaled) * 0.80)
        X_train = X_scaled[:n_train]
        ret_train = returns[:n_train]

        if self.algorithm == "DQN":
            env = DiscretePortfolioEnv(X_train, ret_train, allow_shorting=False)
        else:
            env = PortfolioEnv(X_train, ret_train, allow_shorting=self.allow_shorting)  # type: ignore[assignment]
        self._env = env

        agent = self._build_agent(env)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent.learn(total_timesteps=self.n_training_steps, progress_bar=False)
        self._agent = agent

        # Evaluate on training data
        self._train_sharpe = self._eval_sharpe(X_train, ret_train)

        if self.mlflow_experiment:
            self._log_mlflow()

        self._is_fitted = True
        self._log_fit_complete(
            algorithm=self.algorithm,
            n_steps=self.n_training_steps,
            train_sharpe=round(self._train_sharpe, 4),
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, data: pd.DataFrame | None = None) -> ModelOutput:
        """Generate signal from the agent's recommended portfolio weight.

        Args:
            data: Optional new data — last row of features used for obs.

        Returns:
            :class:`ModelOutput` with signal = 2 × weight − 1 (centred at 0).
        """
        self._require_fitted()
        assert self._agent is not None
        assert self._scaler is not None

        if data is not None:
            from quantflow.data.features import compute_all_features

            required = {"open", "high", "low", "close", "volume"}
            features = (
                compute_all_features(data)
                if required.issubset(set(data.columns))
                else data.drop(columns=["close"], errors="ignore")
            )
            X_raw = (
                features.replace([np.inf, -np.inf], np.nan)
                .dropna()
                .tail(1)
                .values.astype(np.float32)
            )
        else:
            n = self._scaler.n_features_in_
            X_raw = np.zeros((1, n), dtype=np.float32)

        if X_raw.shape[0] == 0:
            return self._neutral_output()

        obs = self._scaler.transform(X_raw).astype(np.float32)[0]
        action, _ = self._agent.predict(obs, deterministic=True)

        if self.algorithm == "DQN":
            weight = DiscretePortfolioEnv._WEIGHTS[int(action)]
        else:
            weight = float(np.clip(action[0], 0.0, _MAX_WEIGHT))

        # Map weight [0, max_weight] → signal [-1, +1]
        signal = self.normalise_signal((weight / _MAX_WEIGHT - 0.5) * 4.0)

        # Confidence based on how decisive the weight is
        confidence = min(0.70, max(0.20, abs(weight / _MAX_WEIGHT - 0.5) * 2.0 + 0.30))

        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=UTC),
            signal=signal,
            confidence=confidence,
            forecast_return=weight * self._vol_estimate,
            forecast_std=self._vol_estimate,
            metadata={
                "algorithm": self.algorithm,
                "recommended_weight": round(weight, 4),
                "train_sharpe": round(self._train_sharpe, 4),
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_agent(self, env: Any) -> Any:
        """Build the stable-baselines3 agent.

        Args:
            env: Gymnasium environment.

        Returns:
            Initialised SB3 agent.
        """
        policy = "MlpPolicy"
        if self.algorithm == "PPO":
            from stable_baselines3 import PPO

            return PPO(
                policy, env, verbose=0, seed=42, learning_rate=3e-4, n_steps=512, batch_size=64
            )
        elif self.algorithm == "SAC":
            from stable_baselines3 import SAC

            return SAC(policy, env, verbose=0, seed=42, learning_rate=3e-4)
        else:  # DQN
            from stable_baselines3 import DQN

            return DQN(policy, env, verbose=0, seed=42, learning_rate=1e-4)

    def _eval_sharpe(self, X: np.ndarray, returns: np.ndarray) -> float:
        """Evaluate the agent's Sharpe ratio on a data slice.

        Args:
            X: Feature matrix.
            returns: Daily return vector.

        Returns:
            Annualised Sharpe ratio.
        """
        assert self._agent is not None
        portfolio_rets: list[float] = []
        for i in range(len(X)):
            obs = X[i]
            action, _ = self._agent.predict(obs, deterministic=True)
            if self.algorithm == "DQN":
                weight = DiscretePortfolioEnv._WEIGHTS[int(action)]
            else:
                weight = float(np.clip(action[0], 0.0, _MAX_WEIGHT))
            portfolio_rets.append(weight * float(returns[i]))

        pr = np.array(portfolio_rets)
        std = float(pr.std()) + 1e-8
        return float(pr.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))

    def _log_mlflow(self) -> None:
        """Log training metrics to MLflow."""
        try:
            import mlflow

            mlflow.set_experiment(self.mlflow_experiment)
            with mlflow.start_run(run_name=f"DRL_{self.algorithm}_{self.symbol}"):
                mlflow.log_params(
                    {
                        "algorithm": self.algorithm,
                        "symbol": self.symbol,
                        "n_training_steps": self.n_training_steps,
                        "allow_shorting": self.allow_shorting,
                    }
                )
                mlflow.log_metrics({"train_sharpe": self._train_sharpe})
        except Exception as exc:
            logger.warning("MLflow logging failed", error=str(exc))

    def _neutral_output(self) -> ModelOutput:
        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=UTC),
            signal=0.0,
            confidence=0.0,
            forecast_return=0.0,
            forecast_std=self._vol_estimate,
            metadata={"error": "prediction_failed"},
        )
