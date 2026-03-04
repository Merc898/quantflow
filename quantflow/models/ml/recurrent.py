"""LSTM / GRU sequence model for financial return prediction.

Multi-task architecture predicting both return magnitude (regression)
and direction (classification) from a 63-day sliding window of features.

Training follows strict temporal ordering: 70/15/15 train/val/test split
with a 21-day gap between train end and val start.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from quantflow.config.constants import WALK_FORWARD_GAP
from quantflow.config.logging import get_logger
from quantflow.models.base import BaseQuantModel, ModelOutput

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

_SEQ_LEN = 63  # 63-day sliding window
_HIDDEN = 128  # LSTM hidden units
_N_LAYERS = 2  # stacked layers
_DROPOUT = 0.3
_BATCH_SIZE = 64
_MAX_EPOCHS = 100
_PATIENCE = 10  # early stopping patience
_LR = 1e-3
_WEIGHT_DECAY = 1e-4
_GRAD_CLIP = 1.0
_HORIZON = 21


# ---------------------------------------------------------------------------
# Dataset and sequence builder
# ---------------------------------------------------------------------------


def _build_sequences(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    seq_len: int = _SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """Build overlapping sliding-window sequences.

    Each sequence ``seq[i]`` is ``X[i : i+seq_len]`` and the target is
    ``y[i+seq_len-1]``.  This guarantees no look-ahead: the target at
    index ``i+seq_len-1`` is the forward return known after the last
    bar in the window.

    Args:
        X_arr: Feature matrix (N, F).
        y_arr: Target vector (N,).
        seq_len: Sliding window length.

    Returns:
        Tuple of (sequences, targets) with shapes (M, seq_len, F) and (M,).
    """
    n = len(X_arr)
    if n < seq_len + 1:
        return np.empty((0, seq_len, X_arr.shape[1])), np.empty(0)

    seqs, targets = [], []
    for i in range(n - seq_len):
        seqs.append(X_arr[i : i + seq_len])
        targets.append(y_arr[i + seq_len - 1])

    return np.array(seqs, dtype=np.float32), np.array(targets, dtype=np.float32)


# ---------------------------------------------------------------------------
# PyTorch model definition
# ---------------------------------------------------------------------------


def _build_lstm_model(
    n_features: int,
    hidden_size: int = _HIDDEN,
    n_layers: int = _N_LAYERS,
    dropout: float = _DROPOUT,
    cell_type: str = "LSTM",
) -> Any:
    """Build a PyTorch LSTM or GRU multi-task model.

    Args:
        n_features: Number of input features.
        hidden_size: Hidden state dimension.
        n_layers: Number of stacked recurrent layers.
        dropout: Dropout probability between layers.
        cell_type: "LSTM" or "GRU".

    Returns:
        Initialised PyTorch nn.Module.
    """
    import torch.nn as nn

    class MultiTaskRNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            rnn_cls = nn.LSTM if cell_type == "LSTM" else nn.GRU
            self.rnn = rnn_cls(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0.0,
                batch_first=True,
            )
            self.layer_norm = nn.LayerNorm(hidden_size)
            self.dropout = nn.Dropout(dropout)
            # Regression head: predict forward return
            self.reg_head = nn.Linear(hidden_size, 1)
            # Classification head: predict direction probability
            self.cls_head = nn.Linear(hidden_size, 1)

        def forward(self, x: Any) -> tuple[Any, Any]:
            out, _ = self.rnn(x)
            last = out[:, -1, :]  # last time step
            last = self.layer_norm(last)
            last = self.dropout(last)
            ret_pred = self.reg_head(last).squeeze(-1)
            dir_logit = self.cls_head(last).squeeze(-1)
            return ret_pred, dir_logit

    return MultiTaskRNN()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _train_rnn(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = _MAX_EPOCHS,
    patience: int = _PATIENCE,
    lr: float = _LR,
    weight_decay: float = _WEIGHT_DECAY,
    grad_clip: float = _GRAD_CLIP,
    batch_size: int = _BATCH_SIZE,
) -> dict[str, Any]:
    """Run training loop with early stopping and CosineAnnealingLR.

    Args:
        model: PyTorch nn.Module (MultiTaskRNN).
        X_train: Training sequences (N, seq_len, F).
        y_train: Training targets (N,).
        X_val: Validation sequences.
        y_val: Validation targets.
        max_epochs: Maximum epochs.
        patience: Early-stopping patience.
        lr: Initial learning rate.
        weight_decay: AdamW weight decay.
        grad_clip: Gradient clipping max-norm.
        batch_size: Mini-batch size.

    Returns:
        Dict with training history and best validation loss.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state: dict = {}
    no_improve = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            ret_pred, dir_logit = model(X_batch)
            reg_loss = mse_loss(ret_pred, y_batch)
            dir_target = (y_batch > 0).float()
            cls_loss = bce_loss(dir_logit, dir_target)
            loss = reg_loss + 0.3 * cls_loss  # multi-task loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            vr, vd = model(X_v)
            v_reg = mse_loss(vr, y_v)
            v_cls = bce_loss(vd, (y_v > 0).float())
            val_loss = float((v_reg + 0.3 * v_cls).item())

        train_losses.append(epoch_loss / max(len(loader), 1))
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.debug("Early stopping", epoch=epoch, val_loss=round(val_loss, 6))
                break

    if best_state:
        model.load_state_dict(best_state)

    return {
        "best_val_loss": best_val_loss,
        "epochs_trained": len(train_losses),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


# ---------------------------------------------------------------------------
# Main LSTM model class
# ---------------------------------------------------------------------------


class LSTMSignalModel(BaseQuantModel):
    """Multi-task LSTM (or GRU) for return + direction prediction.

    Args:
        symbol: Ticker symbol.
        cell_type: "LSTM" or "GRU".
        seq_len: Sliding window length in trading days.
        hidden_size: RNN hidden dimension.
        horizon: Forward-return horizon.
        device: Torch device string ("cpu", "cuda", or "auto").
    """

    def __init__(
        self,
        symbol: str,
        cell_type: str = "LSTM",
        seq_len: int = _SEQ_LEN,
        hidden_size: int = _HIDDEN,
        horizon: int = _HORIZON,
        device: str = "auto",
    ) -> None:
        """Initialise the LSTM signal model.

        Args:
            symbol: Ticker symbol.
            cell_type: "LSTM" or "GRU".
            seq_len: Sliding window length.
            hidden_size: RNN hidden units.
            horizon: Forecast horizon in days.
            device: Torch device.
        """
        super().__init__(f"{cell_type}_Signal", symbol)
        self.cell_type = cell_type
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.horizon = horizon
        self._device_str = device

        self._model: Any = None
        self._train_history: dict[str, Any] = {}
        self._vol_estimate: float = 0.02
        self._n_features: int = 1
        self._scaler_X: Any = None  # sklearn StandardScaler
        self._scaler_y: Any = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> LSTMSignalModel:
        """Fit the LSTM on sequenced feature data.

        Temporal split: 70% train, 15% val (with 21-day gap), 15% test (unused).

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
        X_raw, y_raw, _ = _build_xy(features, close, self.horizon)

        if len(X_raw) < self.seq_len + 50:
            raise ValueError(
                f"Insufficient data for LSTM: need ≥{self.seq_len + 50} rows, got {len(X_raw)}"
            )

        # Feature scaling
        self._scaler_X = SKScaler()
        self._scaler_y = SKScaler()
        X_scaled = self._scaler_X.fit_transform(X_raw).astype(np.float32)
        y_scaled = self._scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel().astype(np.float32)
        self._vol_estimate = float(np.std(y_raw))
        self._n_features = X_scaled.shape[1]

        # Temporal train/val split (70/15 with gap)
        n = len(X_scaled)
        train_end = int(n * 0.70)
        val_start = train_end + WALK_FORWARD_GAP
        val_end = int(n * 0.85)

        if val_start >= val_end:
            val_start = train_end
            val_end = n

        X_seqs, y_seqs = _build_sequences(X_scaled, y_scaled, self.seq_len)
        n_seq = len(X_seqs)
        # Adjust split indices for sequence reduction
        seq_train_end = max(1, train_end - self.seq_len)
        seq_val_start = max(seq_train_end, val_start - self.seq_len)
        seq_val_end = min(n_seq, val_end - self.seq_len)

        X_tr = X_seqs[:seq_train_end]
        y_tr = y_seqs[:seq_train_end]
        X_v = X_seqs[seq_val_start:seq_val_end]
        y_v = y_seqs[seq_val_start:seq_val_end]

        if len(X_tr) < 10 or len(X_v) < 5:
            raise ValueError("Insufficient sequences for train/val split.")

        self._model = _build_lstm_model(
            n_features=self._n_features,
            hidden_size=self.hidden_size,
            n_layers=_N_LAYERS,
            dropout=_DROPOUT,
            cell_type=self.cell_type,
        )

        self._train_history = _train_rnn(
            self._model,
            X_tr,
            y_tr,
            X_v,
            y_v,
            max_epochs=_MAX_EPOCHS,
            patience=_PATIENCE,
        )

        self._is_fitted = True
        self._log_fit_complete(
            cell_type=self.cell_type,
            epochs=self._train_history.get("epochs_trained"),
            best_val_loss=round(self._train_history.get("best_val_loss", 0.0), 6),
            n_sequences=n_seq,
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, data: pd.DataFrame | None = None) -> ModelOutput:
        """Generate signal from the last ``seq_len`` feature rows.

        Args:
            data: Optional new data.  Must have ≥ ``seq_len`` rows.

        Returns:
            :class:`ModelOutput` with LSTM signal and direction probability.
        """
        self._require_fitted()
        import torch

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
                .tail(self.seq_len)
                .values.astype(np.float32)
            )
        else:
            X_raw = np.zeros((self.seq_len, self._n_features), dtype=np.float32)

        if len(X_raw) < self.seq_len:
            return self._neutral_output()

        X_scaled = self._scaler_X.transform(X_raw[-self.seq_len :]).astype(np.float32)
        tensor = torch.tensor(X_scaled[np.newaxis, :, :])  # (1, seq_len, F)

        self._model.eval()
        with torch.no_grad():
            ret_scaled, dir_logit = self._model(tensor)

        # Inverse-transform return prediction
        ret_scaled_np = ret_scaled.cpu().numpy().reshape(-1, 1)
        forecast_return = float(self._scaler_y.inverse_transform(ret_scaled_np)[0, 0])
        direction_prob = float(torch.sigmoid(dir_logit).item())

        forecast_std = self._vol_estimate * np.sqrt(self.horizon)
        raw_signal = forecast_return / (forecast_std + 1e-8)
        signal = self.normalise_signal(raw_signal)

        # Blend regression signal with direction probability
        direction_signal = (direction_prob - 0.5) * 2.0  # map [0,1] → [-1,1]
        blended = 0.7 * signal + 0.3 * direction_signal
        blended = float(np.clip(blended, -1.0, 1.0))

        confidence = min(0.75, max(0.20, abs(direction_prob - 0.5) * 2.0))

        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=UTC),
            signal=blended,
            confidence=confidence,
            forecast_return=round(forecast_return, 6),
            forecast_std=round(forecast_std, 6),
            metadata={
                "cell_type": self.cell_type,
                "direction_prob": round(direction_prob, 4),
                "best_val_loss": round(self._train_history.get("best_val_loss", 0.0), 6),
                "epochs_trained": self._train_history.get("epochs_trained", 0),
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
            metadata={"error": "insufficient_sequence_data"},
        )
