"""Small LSTM over the last `seq_len` bars of features, same labels and splits as LightGBM."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import optuna
import torch
from torch import nn


class LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # chỉ dùng hidden state ở bước cuối (bar t) để dự đoán
        return self.head(self.drop(out[:, -1])).squeeze(-1)


def windows(x: np.ndarray, rows: np.ndarray, seq_len: int) -> np.ndarray:
    """(len(rows), seq_len, features): for each row t the features of bars t-seq_len+1 .. t."""
    if rows.min() < seq_len - 1:
        raise ValueError("rows too early for a full window")
    idx = rows[:, None] + np.arange(-seq_len + 1, 1)[None, :]
    return x[idx]


@dataclass
class LSTMModel:
    seq_len: int
    mean: np.ndarray
    std: np.ndarray
    net: LSTMNet | None = None
    params: dict = field(default_factory=dict)
    val_loss: float = float("nan")

    def prepare(self, x: np.ndarray) -> np.ndarray:
        # chuẩn hóa theo mean/std của tập train; NaN (warm-up) -> 0 = giá trị trung bình
        z = (x - self.mean) / self.std
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray, rows: np.ndarray) -> np.ndarray:
        self.net.eval()
        seq = torch.from_numpy(windows(self.prepare(x), rows, self.seq_len))
        out = [torch.sigmoid(self.net(seq[i : i + 2048])) for i in range(0, len(seq), 2048)]
        return torch.cat(out).numpy().astype(np.float64)


def _train(
    xs: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    val: np.ndarray,
    seq_len: int,
    hidden: int,
    lr: float,
    dropout: float,
    seed: int,
    max_epochs: int,
    patience: int = 3,
) -> tuple[LSTMNet, float]:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = LSTMNet(xs.shape[1], hidden, dropout)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    x_tr = torch.from_numpy(windows(xs, tr, seq_len))
    y_tr = torch.from_numpy(y[tr].astype(np.float32))
    x_val = torch.from_numpy(windows(xs, val, seq_len))
    y_val = torch.from_numpy(y[val].astype(np.float32))

    best_loss, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        net.train()
        perm = torch.randperm(len(x_tr), generator=gen)
        for i in range(0, len(perm), 256):
            b = perm[i : i + 256]
            opt.zero_grad()
            loss_fn(net(x_tr[b]), y_tr[b]).backward()
            # TODO: thử gradient clipping nếu loss nhảy
            opt.step()
        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net(x_val), y_val).item()
        if val_loss < best_loss - 1e-5:
            best_loss, bad = val_loss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, best_loss


def fit_lstm(
    x: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    val: np.ndarray,
    seq_len: int = 48,
    trials: int = 6,
    max_epochs: int = 15,
    seed: int = 0,
) -> LSTMModel:
    """Tune hidden size / lr / dropout with Optuna, keep the best net by validation loss.

    `x` is the full feature matrix (windows reach back before the first train row), `y` the
    0/1 labels aligned with it.
    """
    mean = np.nanmean(x[tr], axis=0)
    std = np.nanstd(x[tr], axis=0)
    std[~(std > 0)] = 1.0
    model = LSTMModel(seq_len=seq_len, mean=mean, std=std)
    xs = model.prepare(x)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "hidden": trial.suggest_categorical("hidden", [16, 32, 64]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        }
        net, loss = _train(xs, y, tr, val, seq_len, seed=seed, max_epochs=max_epochs, **params)
        if np.isnan(model.val_loss) or loss < model.val_loss:
            model.net, model.params, model.val_loss = net, params, loss
        return loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    return model
