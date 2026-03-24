from __future__ import annotations

import re
import pandas as pd

ATOMIC_NUMBERS = {
    'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'Al': 13, 'P': 15,
    'Ga': 31, 'As': 33, 'In': 49, 'Sb': 51, 'Tl': 81, 'Bi': 83,
}


def _elements(formula: str) -> list[str]:
    return re.findall(r'[A-Z][a-z]?', formula or '')


def _basic_features(formula: str) -> dict:
    elements = _elements(formula)
    z = [ATOMIC_NUMBERS.get(e, 0) for e in elements]
    return {
        'n_elements': len(elements),
        'sum_z': sum(z),
        'max_z': max(z) if z else 0,
        'min_z': min(z) if z else 0,
        'mean_z': sum(z) / len(z) if z else 0,
        'contains_B': int('B' in elements),
        'contains_N': int('N' in elements),
    }


def build_feature_table(df: pd.DataFrame, formula_col: str = 'formula') -> pd.DataFrame:
    base = df.copy()
    feature_rows = base[formula_col].astype(str).apply(_basic_features).apply(pd.Series)
    out = pd.concat([base.reset_index(drop=True), feature_rows.reset_index(drop=True)], axis=1)
    return out
