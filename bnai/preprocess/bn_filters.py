from __future__ import annotations

import re
import pandas as pd


def extract_elements(formula: str) -> list[str]:
    return re.findall(r'[A-Z][a-z]?', formula or '')


def filter_bn(df: pd.DataFrame, formula_col: str = 'formula') -> pd.DataFrame:
    mask = df[formula_col].astype(str).apply(lambda x: {'B', 'N'}.issubset(set(extract_elements(x))))
    out = df.loc[mask].copy()
    out['elements'] = out[formula_col].astype(str).apply(extract_elements)
    return out


def generate_bn_candidates() -> pd.DataFrame:
    group13 = ['B', 'Al', 'Ga', 'In', 'Tl']
    group15 = ['N', 'P', 'As', 'Sb', 'Bi']
    rows = []
    for left in group13:
        for right in group15:
            rows.append({'formula': f'{left}{right}', 'candidate_source': 'simple_bn_substitutions'})
    return pd.DataFrame(rows)
