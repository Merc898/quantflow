"""Neural Ordinary Differential Equation model for price dynamics.

Models asset price dynamics as a latent continuous-time ODE:

  dh/dt = f_theta(h, t)

where h is the latent state vector and f_theta is a neural network.

Architecture:
  - Encoder:   LSTM processes return sequence → initial hidden state h0
  - ODE func:  MLP(h, t) → dh/dt  (hidden=64, 2 layers, Tanh activation)
  - Decoder:   Linear layer h_T → return forecast

If ``torchdiffeq`` is available, uses the adaptive-step Dormand-Prince
(dopri5) solver with the adjoint sensitivity method (memory-efficient).
Otherwise falls back to a fixed-step Euler integrator (slightly less
accurate but numerically equivalent for short integration windows).

References:
  Chen, R. T. Q. et al. (2018). Neural Ordinary Differential Equations.
  NeurIPS 2018.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from quantflow.config.constants import TRADING_DAYS_PER_YEAR
from quantflow.config.logging import get_logger
from quantflow.models.base import BaseQuantModel, ModelOutput

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

try:
    from torchdiffeq import odeint_adjoint as _odeint

    _TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    _TORCHDIFFEQ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Network components (only defined when torch is available)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _ODEFunc(nn.Module):
        """Neural network parametrising dh/dt.

        Args:
            hidden_dim: Latent state dimension.
        """

        def __init__(self, hidden_dim: int = 64) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for time
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            # Append time as an extra feature
            t_expand = t.expand(h.shape[0], 1)
            return self.net(torch.cat([h, t_expand], dim=-1))

    class _NeuralODENet(nn.Module):
        """Full encoder-ODE-decoder architecture.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Latent state dimension.
            n_steps: Fixed Euler steps (fallback only).
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            n_steps: int = 10,
        ) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.n_steps = n_steps
            # Encoder: maps last return → initial ODE state
            self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
            self.h0_proj = nn.Linear(hidden_dim, hidden_dim)
            # ODE function
            self.ode_func = _ODEFunc(hidden_dim)
            # Decoder
            self.decoder = nn.Linear(hidden_dim, 1)

        def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x_seq: Shape (batch, seq_len, input_dim).

            Returns:
                Predicted returns, shape (batch,).
            """
            _, (hn, _) = self.encoder(x_seq)
            h0 = self.h0_proj(hn[-1])  # (batch, hidden_dim)

            # Integrate ODE over t=[0, 1]
            t_span = torch.linspace(0.0, 1.0, self.n_steps + 1, dtype=torch.float32)

            if _TORCHDIFFEQ_AVAILABLE:
                hT = _odeint(
                    self.ode_func,
                    h0,
                    t_span,
                    method="dopri5",
                    rtol=1e-3,
                    atol=1e-4,
                )[
                    -1
                ]  # (batch, hidden_dim)
            else:
                # Euler integrator (fallback)
                hT = h0
                dt = 1.0 / self.n_steps
                for i in range(self.n_steps):
                    t_i = torch.tensor(i * dt, dtype=torch.float32)
                    hT = hT + self.ode_func(t_i, hT) * dt

            return self.decoder(hT).squeeze(-1)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def _build_sequences(
    returns: pd.Series,
    seq_len: int = 21,
    horizon: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X_seq, y) arrays for supervised ODE training.

    Each row: X = returns[t-seq_len:t], y = sum_return[t:t+horizon].

    Args:
        returns: Daily log-return series.
        seq_len: Input sequence length.
        horizon: Forecast horizon (days).

    Returns:
        Tuple of arrays (X_seq, y) with shapes
        (n_samples, seq_len, 1) and (n_samples,).
    """
    r = returns.to_numpy(dtype=np.float32)
    X_list, y_list = [], []
    for i in range(seq_len, len(r) - horizon):
        X_list.append(r[i - seq_len : i].reshape(seq_len, 1))
        y_list.append(float(r[i : i + horizon].sum()))
    if not X_list:
        return np.empty((0, seq_len, 1), dtype=np.float32), np.empty(0, dtype=np.float32)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class NeuralODEModel(BaseQuantModel):
    """Neural ODE for continuous-time asset dynamics modelling.

    Args:
        symbol: Ticker symbol.
        hidden_dim: Latent state dimension for the ODE.
        seq_len: Input look-back window (days).
        n_ode_steps: Number of Euler steps in the fallback integrator.
        n_epochs: Training epochs.
        lr: Learning rate.
        horizon: Forecast horizon (days).
        seed: RNG seed.
    """

    def __init__(
        self,
        symbol: str,
        hidden_dim: int = 64,
        seq_len: int = 21,
        n_ode_steps: int = 10,
        n_epochs: int = 100,
        lr: float = 1e-3,
        horizon: int = 21,
        seed: int | None = 42,
    ) -> None:
        super().__init__("NeuralODEModel", symbol)
        self._hidden_dim = hidden_dim
        self._seq_len = seq_len
        self._n_steps = n_ode_steps
        self._n_epochs = n_epochs
        self._lr = lr
        self._horizon = horizon
        self._seed = seed
        self._net: Any = None
        self._ret_mean: float = 0.0
        self._ret_std: float = 1.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> NeuralODEModel:
        """Train Neural ODE on historical return sequences.

        Args:
            data: OHLCV DataFrame (UTC DatetimeIndex).

        Returns:
            Self.
        """
        if not _TORCH_AVAILABLE:
            self._fallback_fit(data)
            return self

        log_returns = np.log(data["close"] / data["close"].shift(1)).dropna()
        if len(log_returns) < self._seq_len + self._horizon + 10:
            raise ValueError(
                f"NeuralODEModel requires ≥{self._seq_len + self._horizon + 10} "
                f"observations; got {len(log_returns)}"
            )

        X, y = _build_sequences(log_returns, self._seq_len, self._horizon)
        if len(X) < 20:
            raise ValueError("Too few sequences; increase data length.")

        # Normalise targets
        self._ret_mean = float(y.mean())
        self._ret_std = max(float(y.std()), 1e-8)
        y_norm = (y - self._ret_mean) / self._ret_std

        if self._seed is not None:
            torch.manual_seed(self._seed)

        self._net = _NeuralODENet(
            input_dim=1,
            hidden_dim=self._hidden_dim,
            n_steps=self._n_steps,
        )
        opt = optim.Adam(self._net.parameters(), lr=self._lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_norm, dtype=torch.float32)

        self._net.train()
        for _ in range(self._n_epochs):
            opt.zero_grad()
            pred = self._net(X_t)
            loss = criterion(pred, y_t)
            loss.backward()
            nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
            opt.step()

        self._is_fitted = True
        solver = "dopri5 (adjoint)" if _TORCHDIFFEQ_AVAILABLE else "euler (fallback)"
        self._log_fit_complete(
            n_train_samples=len(X),
            hidden_dim=self._hidden_dim,
            solver=solver,
        )
        return self

    def predict(self, data: pd.DataFrame) -> ModelOutput:
        """Generate a Neural ODE trading signal.

        Args:
            data: OHLCV DataFrame.

        Returns:
            :class:`ModelOutput`.
        """
        self._require_fitted()

        if not _TORCH_AVAILABLE or self._net is None:
            return self._fallback_predict(data)

        log_returns = np.log(data["close"] / data["close"].shift(1)).dropna()
        if len(log_returns) < self._seq_len:
            raise ValueError(f"Need ≥{self._seq_len} observations for prediction.")

        recent = log_returns.iloc[-self._seq_len :].to_numpy(dtype=np.float32)
        X_t = torch.tensor(recent.reshape(1, self._seq_len, 1), dtype=torch.float32)

        self._net.eval()
        with torch.no_grad():
            pred_norm = self._net(X_t).item()

        # De-normalise
        pred_ret = pred_norm * self._ret_std + self._ret_mean

        ann_vol = float(log_returns.iloc[-21:].std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        horizon_vol = ann_vol * np.sqrt(self._horizon / TRADING_DAYS_PER_YEAR)
        signal = self.normalise_signal(pred_ret / max(horizon_vol, 1e-6))

        # Confidence: moderate baseline (single deterministic prediction)
        confidence = 0.55

        regime = "HIGH_VOL" if ann_vol > 0.25 else ("LOW_VOL" if ann_vol < 0.12 else "MEDIUM_VOL")

        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=UTC),
            signal=round(signal, 6),
            confidence=confidence,
            forecast_return=round(pred_ret * (TRADING_DAYS_PER_YEAR / self._horizon), 6),
            forecast_std=round(ann_vol, 6),
            regime=regime,
            metadata={
                "pred_ret_21d": round(pred_ret, 6),
                "hidden_dim": self._hidden_dim,
                "seq_len": self._seq_len,
                "solver": "dopri5" if _TORCHDIFFEQ_AVAILABLE else "euler",
            },
        )

    # ------------------------------------------------------------------
    # Fallbacks (no PyTorch)
    # ------------------------------------------------------------------

    def _fallback_fit(self, data: pd.DataFrame) -> None:
        """Simple momentum fallback when PyTorch is unavailable."""
        log_returns = np.log(data["close"] / data["close"].shift(1)).dropna()
        self._ret_mean = float(log_returns.mean())
        self._ret_std = max(float(log_returns.std()), 1e-8)
        self._is_fitted = True
        logger.warning("PyTorch not available; using momentum fallback for NeuralODEModel")

    def _fallback_predict(self, data: pd.DataFrame) -> ModelOutput:
        log_returns = np.log(data["close"] / data["close"].shift(1)).dropna()
        recent_ret = float(log_returns.iloc[-self._horizon :].sum())
        ann_vol = float(log_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        signal = self.normalise_signal(
            recent_ret / max(ann_vol / np.sqrt(TRADING_DAYS_PER_YEAR / self._horizon), 1e-6)
        )
        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=UTC),
            signal=round(signal, 6),
            confidence=0.30,
            forecast_return=round(recent_ret * (TRADING_DAYS_PER_YEAR / self._horizon), 6),
            forecast_std=round(ann_vol, 6),
            regime="MEDIUM_VOL",
            metadata={"fallback": True, "reason": "torch_unavailable"},
        )
