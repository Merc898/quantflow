"""Transformer model for financial time-series prediction.

Implements a PatchTST-inspired Transformer encoder with:
- Patch-based input tokenisation (non-overlapping patches of features)
- Learnable + sinusoidal positional encoding
- Multi-head self-attention (8 heads, d_model=128)
- Uncertainty head outputting mean + log-variance (NLL loss)
- Causal masking to prevent attending to future tokens

Architecture option: vanilla Transformer encoder (no decoder needed for prediction).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from quantflow.config.constants import WALK_FORWARD_GAP
from quantflow.config.logging import get_logger
from quantflow.models.base import BaseQuantModel, ModelOutput

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

# Architecture constants
_D_MODEL = 128
_N_HEADS = 8
_FF_DIM = 512
_N_LAYERS = 3
_DROPOUT = 0.1
_PATCH_LEN = 16  # patch length in time steps
_SEQ_LEN = 64  # input sequence length (must be divisible by PATCH_LEN)
_MAX_EPOCHS = 80
_PATIENCE = 10
_LR = 5e-4
_WEIGHT_DECAY = 1e-4
_BATCH_SIZE = 32
_HORIZON = 21


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


def _sinusoidal_pe(max_len: int, d_model: int) -> Any:
    """Create a sinusoidal positional encoding matrix.

    Args:
        max_len: Maximum sequence length.
        d_model: Model dimension.

    Returns:
        Tensor of shape (max_len, d_model).
    """
    import torch

    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (max_len, d_model)


# ---------------------------------------------------------------------------
# PyTorch Transformer model
# ---------------------------------------------------------------------------


def _build_transformer(n_features: int, n_patches: int) -> Any:
    """Build the PatchTST-style Transformer model.

    Args:
        n_features: Number of input features.
        n_patches: Number of patches in the input sequence.

    Returns:
        Initialised PyTorch nn.Module.
    """
    import torch
    import torch.nn as nn

    class PatchEmbedding(nn.Module):
        """Projects a patch of raw features into d_model space."""

        def __init__(self, patch_len: int, n_features: int, d_model: int) -> None:
            super().__init__()
            self.proj = nn.Linear(patch_len * n_features, d_model)
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x: Any) -> Any:
            # x: (batch, n_patches, patch_len, n_features)
            b, np_, pl, nf = x.shape
            x = x.reshape(b, np_, pl * nf)
            return self.norm(self.proj(x))  # (batch, n_patches, d_model)

    class FinanceTransformer(nn.Module):
        """PatchTST-style Transformer for financial signal generation."""

        def __init__(self) -> None:
            super().__init__()
            self.patch_embed = PatchEmbedding(_PATCH_LEN, n_features, _D_MODEL)

            # Learnable positional embedding
            self.pos_embed = nn.Parameter(torch.randn(1, n_patches, _D_MODEL) * 0.02)

            # Register sinusoidal PE as a buffer
            sin_pe = _sinusoidal_pe(n_patches, _D_MODEL)
            self.register_buffer("sin_pe", sin_pe.unsqueeze(0))  # (1, n_patches, D)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=_D_MODEL,
                nhead=_N_HEADS,
                dim_feedforward=_FF_DIM,
                dropout=_DROPOUT,
                batch_first=True,
                norm_first=True,  # pre-norm for training stability
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=_N_LAYERS)

            # Uncertainty head: predict mean and log-variance
            self.mean_head = nn.Linear(_D_MODEL, 1)
            self.logvar_head = nn.Linear(_D_MODEL, 1)

        def forward(self, x: Any) -> tuple[Any, Any]:
            # x: (batch, seq_len, n_features)
            b, t, f = x.shape
            n_p = t // _PATCH_LEN
            x_patches = x[:, : n_p * _PATCH_LEN, :].reshape(b, n_p, _PATCH_LEN, f)

            tokens = self.patch_embed(x_patches)  # (b, n_patches, D)
            tokens = tokens + self.pos_embed + self.sin_pe[:, :n_p, :]  # type: ignore[index]

            # Causal mask: prevent attending to future patches
            mask = torch.triu(torch.ones(n_p, n_p, device=x.device) * float("-inf"), diagonal=1)
            enc = self.encoder(tokens, mask=mask)  # (b, n_patches, D)

            last = enc[:, -1, :]  # use last patch token for prediction
            mean = self.mean_head(last).squeeze(-1)
            logvar = self.logvar_head(last).squeeze(-1)
            return mean, logvar

    return FinanceTransformer()


# ---------------------------------------------------------------------------
# NLL loss for uncertainty prediction
# ---------------------------------------------------------------------------


def _gaussian_nll(mean: Any, logvar: Any, target: Any) -> Any:
    """Gaussian negative log-likelihood loss.

    Args:
        mean: Predicted mean tensor.
        logvar: Predicted log-variance tensor.
        target: True target tensor.

    Returns:
        Scalar loss tensor.
    """
    return 0.5 * ((target - mean) ** 2 * (-logvar).exp() + logvar).mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _train_transformer(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = _MAX_EPOCHS,
    patience: int = _PATIENCE,
) -> dict[str, Any]:
    """Train the Transformer with NLL loss and early stopping.

    Args:
        model: PyTorch FinanceTransformer module.
        X_train: (N, seq_len, F) train sequences.
        y_train: (N,) train targets.
        X_val: Validation sequences.
        y_val: Validation targets.
        max_epochs: Maximum training epochs.
        patience: Early-stopping patience.

    Returns:
        Training history dict.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    Xt = torch.tensor(X_train, dtype=torch.float32).to(device)
    yt = torch.tensor(y_train, dtype=torch.float32).to(device)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=_BATCH_SIZE, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=_LR, weight_decay=_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val = float("inf")
    best_state: dict = {}
    no_improve = 0
    history: dict[str, list[float]] = {"train": [], "val": []}

    for _epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in loader:
            optimizer.zero_grad()
            mean, logvar = model(Xb)
            loss = _gaussian_nll(mean, logvar, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vm, vl = model(Xv)
            val_loss = float(_gaussian_nll(vm, vl, yv).item())

        history["train"].append(epoch_loss / max(len(loader), 1))
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    return {"best_val_nll": best_val, "epochs": len(history["train"])}


# ---------------------------------------------------------------------------
# Main Transformer model class
# ---------------------------------------------------------------------------


class TimeSeriesTransformerModel(BaseQuantModel):
    """PatchTST-style Transformer for financial signal generation.

    Predicts 21-day forward return as a Gaussian distribution
    (mean + variance) — enabling proper uncertainty quantification.

    Args:
        symbol: Ticker symbol.
        seq_len: Input sequence length in trading days.
        horizon: Forecast horizon in trading days.
    """

    def __init__(
        self,
        symbol: str,
        seq_len: int = _SEQ_LEN,
        horizon: int = _HORIZON,
    ) -> None:
        """Initialise the Transformer model.

        Args:
            symbol: Ticker symbol.
            seq_len: Input sequence length.
            horizon: Forecast horizon.
        """
        super().__init__("PatchTST_Transformer", symbol)
        self.seq_len = seq_len
        self.horizon = horizon

        self._model: Any = None
        self._scaler_X: Any = None
        self._scaler_y: Any = None
        self._train_history: dict[str, Any] = {}
        self._n_features: int = 1
        self._vol_estimate: float = 0.02

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> TimeSeriesTransformerModel:
        """Fit the Transformer on feature sequences.

        Args:
            data: DataFrame with ``close`` and feature columns.

        Returns:
            Self (fitted model).
        """
        from sklearn.preprocessing import StandardScaler as SKScaler

        from quantflow.data.features import compute_all_features
        from quantflow.models.ml.gradient_boosting import _build_xy
        from quantflow.models.ml.recurrent import _build_sequences

        close = data["close"].astype(np.float64)
        required = {"open", "high", "low", "close", "volume"}
        features = (
            compute_all_features(data)
            if required.issubset(set(data.columns))
            else data.drop(columns=["close"], errors="ignore")
        )

        X_raw, y_raw, _ = _build_xy(features, close, self.horizon)
        if len(X_raw) < self.seq_len + 50:
            raise ValueError(f"Insufficient data for Transformer: {len(X_raw)} rows.")

        self._n_features = X_raw.shape[1]
        self._vol_estimate = float(np.std(y_raw))

        self._scaler_X = SKScaler()
        self._scaler_y = SKScaler()
        X_sc = self._scaler_X.fit_transform(X_raw).astype(np.float32)
        y_sc = self._scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel().astype(np.float32)

        # Build patch-length-aligned sequences
        patch_seq_len = (self.seq_len // _PATCH_LEN) * _PATCH_LEN
        n_patches = patch_seq_len // _PATCH_LEN

        X_seqs, y_seqs = _build_sequences(X_sc, y_sc, seq_len=patch_seq_len)
        n = len(X_seqs)
        split = int(n * 0.80)
        val_start = split + WALK_FORWARD_GAP
        X_tr, y_tr = X_seqs[:split], y_seqs[:split]
        X_v = X_seqs[min(val_start, n - 10) :]
        y_v = y_seqs[min(val_start, n - 10) :]

        self._model = _build_transformer(self._n_features, n_patches)
        self._train_history = _train_transformer(self._model, X_tr, y_tr, X_v, y_v)

        self._is_fitted = True
        self._log_fit_complete(
            n_patches=n_patches,
            epochs=self._train_history.get("epochs"),
            best_val_nll=round(self._train_history.get("best_val_nll", 0.0), 6),
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, data: pd.DataFrame | None = None) -> ModelOutput:
        """Generate signal from the last ``seq_len`` feature rows.

        Args:
            data: Optional new data.

        Returns:
            :class:`ModelOutput` with Gaussian uncertainty quantification.
        """
        self._require_fitted()
        import torch

        patch_seq_len = (self.seq_len // _PATCH_LEN) * _PATCH_LEN

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
                .tail(patch_seq_len)
                .values.astype(np.float32)
            )
        else:
            X_raw = np.zeros((patch_seq_len, self._n_features), dtype=np.float32)

        if len(X_raw) < patch_seq_len:
            return self._neutral_output()

        X_sc = self._scaler_X.transform(X_raw[-patch_seq_len:]).astype(np.float32)
        tensor = torch.tensor(X_sc[np.newaxis, :, :])

        self._model.eval()
        with torch.no_grad():
            mean_sc, logvar = self._model(tensor)

        mean_raw = float(
            self._scaler_y.inverse_transform(mean_sc.cpu().numpy().reshape(-1, 1))[0, 0]
        )
        pred_std = float(np.exp(0.5 * logvar.item()))
        # Scale std back to original units
        y_std_factor = float(self._scaler_y.scale_[0]) if hasattr(self._scaler_y, "scale_") else 1.0
        forecast_std = pred_std * y_std_factor * np.sqrt(self.horizon)

        raw_signal = mean_raw / (self._vol_estimate * np.sqrt(self.horizon) + 1e-8)
        signal = self.normalise_signal(raw_signal)

        # Confidence: higher when predicted variance is low
        rel_uncertainty = pred_std / (abs(float(mean_sc.item())) + 1e-8)
        confidence = min(0.80, max(0.15, 1.0 / (1.0 + rel_uncertainty)))

        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=UTC),
            signal=signal,
            confidence=confidence,
            forecast_return=round(mean_raw, 6),
            forecast_std=round(max(forecast_std, 1e-6), 6),
            metadata={
                "pred_logvar": round(float(logvar.item()), 4),
                "best_val_nll": round(self._train_history.get("best_val_nll", 0.0), 6),
                "epochs_trained": self._train_history.get("epochs", 0),
                "n_patches": (self.seq_len // _PATCH_LEN),
            },
        )

    def _neutral_output(self) -> ModelOutput:
        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=UTC),
            signal=0.0,
            confidence=0.0,
            forecast_return=0.0,
            forecast_std=self._vol_estimate,
            metadata={"error": "insufficient_data"},
        )
