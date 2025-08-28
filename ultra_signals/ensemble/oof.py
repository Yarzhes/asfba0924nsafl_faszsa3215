"""Purged K-Fold with embargo generator and OOF stacking harness."""
from typing import Iterator, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def purged_kfold_indexes(n_samples: int, n_splits: int = 5, purge: int = 0, embargo: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield train_idx, test_idx pairs with purge and embargo applied.

    purge: number of samples to remove around each test sample from the train set
    embargo: proportion of data to embargo after each test fold (as integer samples)
    Note: This is a lightweight implementation for time-series-safe splitting.
    """
    kf = KFold(n_splits=n_splits, shuffle=False)
    idx = np.arange(n_samples)
    for train_idx, test_idx in kf.split(idx):
        if purge > 0:
            # remove neighbors from train_idx that are within purge distance of any test index
            mask = np.ones(len(train_idx), dtype=bool)
            for t in test_idx:
                dist = np.abs(train_idx - t)
                mask = mask & (dist > purge)
            train_idx = train_idx[mask]
        if embargo > 0:
            # apply embargo after the test block: remove train samples whose index is within embargo after test max
            max_test = test_idx.max()
            embargo_mask = train_idx > (max_test + embargo)
            train_idx = train_idx[embargo_mask]
        yield train_idx, test_idx
