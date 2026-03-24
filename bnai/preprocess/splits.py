from __future__ import annotations

import numpy as np
import pandas as pd


def make_split_masks(df: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg['project']['random_seed'])
    n = len(df)
    indices = np.arange(n)
    rng.shuffle(indices)

    train_end = int(n * cfg['split']['train_ratio'])
    val_end = train_end + int(n * cfg['split']['val_ratio'])

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    masks = {}
    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        masks[name] = mask
    return masks
