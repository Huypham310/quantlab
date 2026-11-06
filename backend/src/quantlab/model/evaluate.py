"""Walk-forward training of the ML signals and how good their predictions are."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from quantlab.model.lgbm import fit_lgbm
from quantlab.model.lstm import fit_lstm
from quantlab.model.splits import Split


@dataclass
class ModelRun:
    proba: np.ndarray  # P(up) for every out-of-sample row, NaN elsewhere
    folds: list[dict] = field(default_factory=list)
    importance: dict[str, float] = field(default_factory=dict)


def prediction_metrics(proba: np.ndarray, fwd_ret: np.ndarray) -> dict:
    """AUC / accuracy of the direction call and the rank IC against the realized return."""
    ok = np.isfinite(proba) & np.isfinite(fwd_ret)
    p, r = proba[ok], fwd_ret[ok]
    up = (r > 0).astype(int)
    if len(p) < 2 or up.min() == up.max():
        return {"rows": int(ok.sum()), "auc": float("nan"), "accuracy": float("nan"), "ic": 0.0}
    # a model that early-stopped at its first tree predicts one constant: no ranking at all,
    # so IC is 0 rather than spearman's NaN
    ic = 0.0 if np.ptp(p) == 0 else float(spearmanr(p, r).statistic)
    return {
        "rows": int(ok.sum()),
        "auc": float(roc_auc_score(up, p)),
        "accuracy": float(((p > 0.5) == (up == 1)).mean()),
        "ic": ic,
        "base_rate": float(up.mean()),
    }


def run_walk_forward(
    x: np.ndarray,
    fwd_ret: np.ndarray,
    splits: list[Split],
    kind: str = "lgbm",
    trials: int = 20,
    seed: int = 0,
    names: list[str] | None = None,
    seq_len: int = 48,
    verbose: bool = False,
) -> ModelRun:
    """Fit one model per fold on its train rows (tuned on its val rows), predict its test rows."""
    if kind not in ("lgbm", "lstm"):
        raise ValueError(f"unknown model kind {kind!r}")
    y = (fwd_ret > 0).astype(np.float64)
    labelled = np.isfinite(fwd_ret)
    proba = np.full(len(x), np.nan)
    run = ModelRun(proba=proba)
    importance = np.zeros(x.shape[1])

    for k, s in enumerate(splits):
        start = time.perf_counter()
        tr, val = s.train[labelled[s.train]], s.val[labelled[s.val]]
        if kind == "lgbm":
            model = fit_lgbm(x[tr], y[tr], x[val], y[val], trials=trials, seed=seed)
            proba[s.test] = model.predict_proba(x[s.test])
            if names:
                importance += np.array(list(model.importance(names).values()))
        else:
            model = fit_lstm(x, y, tr, val, seq_len=seq_len, trials=trials, seed=seed)
            proba[s.test] = model.predict_proba(x, s.test)
        fold = {
            "fold": k,
            "test": list(s.test_range),
            "params": model.params,
            "val_loss": round(model.val_loss, 5),
            **prediction_metrics(proba[s.test], fwd_ret[s.test]),
            "seconds": round(time.perf_counter() - start, 1),
        }
        run.folds.append(fold)
        if verbose:
            print(
                f"  fold {k:2d} test {s.test_range}  val_loss {fold['val_loss']:.4f}  "
                f"auc {fold['auc']:.3f}  ic {fold['ic']:+.3f}  ({fold['seconds']}s)",
                flush=True,
            )
    if names and kind == "lgbm":
        importance /= max(len(splits), 1)
        run.importance = dict(
            sorted(zip(names, importance.tolist(), strict=True), key=lambda kv: -kv[1])
        )
    return run


def proba_to_weights(
    proba: np.ndarray, mode: str = "long_only", threshold: float = 0.0, smooth: int = 1
) -> np.ndarray:
    """Turn P(up) into target weights.

    The label spans `horizon` bars but we rebalance every bar, so a raw hourly prediction flips
    the position far too often; `smooth` averages the last few predictions first (trailing,
    so still no look-ahead). Then: long_only -> 1 if p > 0.5 + threshold else 0; long_short ->
    +1 / -1 outside the [0.5 - threshold, 0.5 + threshold] dead zone. NaN -> flat.
    """
    p = np.asarray(proba, dtype=np.float64)
    if smooth > 1:
        filled = np.where(np.isfinite(p), p, 0.0)
        valid = np.isfinite(p).astype(np.float64)
        c, cv = np.cumsum(np.insert(filled, 0, 0.0)), np.cumsum(np.insert(valid, 0, 0.0))
        lo = np.maximum(np.arange(len(p)) + 1 - smooth, 0)
        hi = np.arange(len(p)) + 1
        cnt = cv[hi] - cv[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            p = np.where(np.isfinite(p) & (cnt > 0), (c[hi] - c[lo]) / cnt, np.nan)
    w = np.zeros(len(p))
    if mode == "long_only":
        w[p > 0.5 + threshold] = 1.0
    elif mode == "long_short":
        w[p > 0.5 + threshold] = 1.0
        w[p < 0.5 - threshold] = -1.0
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return w
