"""Walk-forward splits with purging for overlapping labels.

A label at row t covers bars t .. t + horizon. If a training row's label window reaches into
the test window, the model has effectively seen test prices during training (the classic
leak with overlapping returns, see Lopez de Prado, *Advances in Financial ML*, ch. 7). So the
last `horizon` rows before every boundary are dropped from the earlier side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    train: np.ndarray  # row indexes used to fit
    val: np.ndarray  # tail of the train window, for early stopping / tuning
    test: np.ndarray  # out-of-sample rows traded with this model

    @property
    def test_range(self) -> tuple[int, int]:
        return int(self.test[0]), int(self.test[-1]) + 1


def walk_forward_splits(
    n: int,
    train_size: int,
    test_size: int,
    horizon: int,
    val_frac: float = 0.2,
    start: int = 0,
) -> list[Split]:
    """Rolling windows: fit on `train_size` rows, test on the next `test_size`.

    Inside each train window the last `val_frac` is held out for validation, with the same
    purge in front of it. `start` skips warm-up rows whose features are still NaN.
    """
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be in (0, 1)")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    splits = []
    t0 = start
    while t0 + train_size < n:
        tr_end = t0 + train_size
        te_end = min(tr_end + test_size, n)
        val_start = tr_end - int(train_size * val_frac)
        # purge: labels of the kept rows must end before the next block starts
        train = np.arange(t0, val_start - horizon)
        val = np.arange(val_start, tr_end - horizon)
        if len(train) == 0 or len(val) == 0:
            raise ValueError("train window too small for this horizon / val_frac")
        splits.append(Split(train=train, val=val, test=np.arange(tr_end, te_end)))
        t0 += test_size
    return splits
