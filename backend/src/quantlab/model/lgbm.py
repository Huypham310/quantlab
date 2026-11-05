"""LightGBM direction classifier, tuned with Optuna on the validation tail of each window."""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import log_loss


@dataclass
class LGBMModel:
    params: dict = field(default_factory=dict)
    booster: lgb.Booster | None = None
    val_loss: float = float("nan")

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        return self.booster.predict(x, num_iteration=self.booster.best_iteration)

    def importance(self, names: list[str]) -> dict[str, float]:
        gain = self.booster.feature_importance("gain")
        total = gain.sum() or 1.0
        return {n: float(g / total) for n, g in zip(names, gain, strict=True)}


BASE = {
    "objective": "binary",
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
}


def _fit(params, x_tr, y_tr, x_val, y_val, seed) -> lgb.Booster:
    train = lgb.Dataset(x_tr, y_tr, free_raw_data=False)
    val = lgb.Dataset(x_val, y_val, reference=train)
    return lgb.train(
        BASE | params | {"seed": seed},
        train,
        num_boost_round=1000,
        valid_sets=[val],
        # stop once the validation log loss hasn't improved for 50 rounds
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )


def fit_lgbm(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    trials: int = 20,
    seed: int = 0,
) -> LGBMModel:
    """Search a small space with TPE, keep the booster with the best validation log loss.

    The space leans towards strong regularization on purpose: hourly returns are mostly noise,
    and a deep, lightly regularized forest just memorizes it.
    """

    best = LGBMModel()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 32, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 800, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100.0, log=True),
        }
        booster = _fit(params, x_tr, y_tr, x_val, y_val, seed)
        p = booster.predict(x_val, num_iteration=booster.best_iteration)
        loss = log_loss(y_val, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])
        if loss < best.val_loss or np.isnan(best.val_loss):
            best.params, best.booster, best.val_loss = params, booster, loss
        return loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    return best
