from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd

from core.schema import DatasetManifest


def _basic_formula_from_entry(entry: dict) -> str:
    for key in ('formula', 'full_formula', 'reduced_formula', 'jid'):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    atoms = entry.get('atoms')
    if isinstance(atoms, dict):
        for key in ('formula', 'composition', 'reduced_formula'):
            value = atoms.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        elements = atoms.get('elements')
        if isinstance(elements, list) and elements:
            try:
                from collections import Counter
                from pymatgen.core import Composition

                counts = Counter(str(x) for x in elements)
                formula_dict = {k: int(v) for k, v in counts.items()}
                return Composition(formula_dict).reduced_formula
            except Exception:
                return ''.join(str(x) for x in elements)

    return 'UNKNOWN'


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _extract_target(entry: dict, target_col: str) -> float | None:
    value = entry.get(target_col)
    if value is not None:
        return _safe_float(value)

    nested = entry.get('atoms')
    if isinstance(nested, dict):
        nested_val = nested.get(target_col)
        if nested_val is not None:
            return _safe_float(nested_val)

    for key in ('bandgap', 'gap pbe', 'gap tbmbj', 'optb88vdw_bandgap'):
        if target_col == 'band_gap' and key in entry:
            return _safe_float(entry.get(key))
    return None


def _normalize(raw: list[dict], target_col: str) -> pd.DataFrame:
    rows = []
    for idx, entry in enumerate(raw):
        formula = _basic_formula_from_entry(entry)
        rows.append({
            'record_id': str(entry.get('jid', idx)),
            'source': 'twod_matpd',
            'formula': formula,
            'target': _extract_target(entry, target_col),
        })
    df = pd.DataFrame(rows)
    df = df[df['formula'].notna()].copy()
    df['formula'] = df['formula'].astype(str)
    return df


def load_or_build_dataset(cfg: dict) -> tuple[pd.DataFrame, dict]:
    raw_dir = Path(cfg['data']['raw_dir'])
    processed_dir = Path(cfg['data']['processed_dir'])
    target_col = cfg['data']['target_column']
    raw_path = raw_dir / 'twod_matpd.json'
    processed_path = processed_dir / 'twod_matpd.parquet'
    manifest_path = processed_dir / 'manifest.json'

    if processed_path.exists() and manifest_path.exists():
        return pd.read_parquet(processed_path), json.loads(manifest_path.read_text())

    from jarvis.db.figshare import data as jarvis_data

    raw = jarvis_data(cfg['data']['dataset'])
    if cfg['data'].get('cache_raw_json', True):
        raw_path.write_text(json.dumps(raw))

    df = _normalize(raw, target_col)
    df.to_parquet(processed_path, index=False)

    manifest = DatasetManifest(
        name=cfg['data']['dataset'],
        source='jarvis-tools/figshare',
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        version_hint='runtime download',
    ).model_dump()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return df, manifest
