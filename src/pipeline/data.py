from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd

from core.schema import DatasetManifest


DATASET_NAME = 'twod_matpd'
STRUCTURE_SUMMARY_COLUMNS = (
    'structure_n_sites',
    'structure_lattice_a',
    'structure_lattice_b',
    'structure_lattice_c',
    'structure_lattice_gamma',
    'structure_inplane_area',
    'structure_cell_height',
    'structure_thickness',
    'structure_vacuum',
    'structure_areal_number_density',
    'structure_thickness_fraction',
)
REFERENCE_PROPERTY_COLUMNS = (
    'energy_per_atom',
    'exfoliation_energy_per_atom',
    'total_magnetization',
    'abs_total_magnetization',
)
REQUIRED_NORMALIZED_COLUMNS = (
    'record_id',
    'source',
    'formula',
    'target',
    *STRUCTURE_SUMMARY_COLUMNS,
    *REFERENCE_PROPERTY_COLUMNS,
)


def _basic_formula_from_entry(entry: dict) -> str:
    for key in ('formula', 'full_formula', 'reduced_formula'):
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

    jid = entry.get('jid')
    if isinstance(jid, str) and jid.strip():
        return jid.strip()

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


def _empty_structure_summary() -> dict[str, float | None]:
    return {column: None for column in STRUCTURE_SUMMARY_COLUMNS}


def _structure_summary_from_atoms(atoms: dict | None) -> dict[str, float | None]:
    summary = _empty_structure_summary()
    if not isinstance(atoms, dict):
        return summary

    try:
        lattice_mat = np.asarray(atoms.get('lattice_mat'), dtype=float)
        abc = np.asarray(atoms.get('abc'), dtype=float)
        angles = np.asarray(atoms.get('angles'), dtype=float)
        coords = np.asarray(atoms.get('coords'), dtype=float)
    except Exception:
        return summary

    if lattice_mat.shape != (3, 3) or abc.shape != (3,) or angles.shape != (3,):
        return summary
    if coords.ndim != 2 or coords.shape[1] != 3:
        return summary

    try:
        cart_coords = coords if bool(atoms.get('cartesian', False)) else np.matmul(coords, lattice_mat)
        inplane_normal = np.cross(lattice_mat[0], lattice_mat[1])
        inplane_area = float(np.linalg.norm(inplane_normal))
        if not np.isfinite(inplane_area) or inplane_area <= 0:
            return summary

        cell_height = float(abs(np.linalg.det(lattice_mat)) / inplane_area)
        if not np.isfinite(cell_height) or cell_height < 0:
            return summary

        unit_normal = inplane_normal / inplane_area
        projections = np.matmul(cart_coords, unit_normal)
        thickness = float(projections.max() - projections.min()) if len(projections) else 0.0
        vacuum = float(max(cell_height - thickness, 0.0))

        elements = atoms.get('elements')
        n_sites = int(len(elements)) if isinstance(elements, list) else int(len(coords))
        areal_number_density = float(n_sites / inplane_area)
        thickness_fraction = float(thickness / cell_height) if cell_height > 0 else None
    except Exception:
        return summary

    raw_values = {
        'structure_n_sites': float(n_sites),
        'structure_lattice_a': float(abc[0]),
        'structure_lattice_b': float(abc[1]),
        'structure_lattice_c': float(abc[2]),
        'structure_lattice_gamma': float(angles[2]),
        'structure_inplane_area': inplane_area,
        'structure_cell_height': cell_height,
        'structure_thickness': thickness,
        'structure_vacuum': vacuum,
        'structure_areal_number_density': areal_number_density,
        'structure_thickness_fraction': thickness_fraction,
    }
    for key, value in raw_values.items():
        if value is None or not np.isfinite(value):
            continue
        summary[key] = float(value)
    return summary


def _normalize(raw: list[dict], target_col: str) -> pd.DataFrame:
    rows = []
    for idx, entry in enumerate(raw):
        formula = _basic_formula_from_entry(entry)
        total_magnetization = _safe_float(entry.get('total_magnetization'))
        rows.append({
            'record_id': str(entry.get('jid', idx)),
            'source': DATASET_NAME,
            'formula': formula,
            'target': _extract_target(entry, target_col),
            'energy_per_atom': _safe_float(entry.get('energy_per_atom')),
            'exfoliation_energy_per_atom': _safe_float(entry.get('exfoliation_energy_per_atom')),
            'total_magnetization': total_magnetization,
            'abs_total_magnetization': abs(total_magnetization) if total_magnetization is not None else None,
            **_structure_summary_from_atoms(entry.get('atoms')),
        })

    df = pd.DataFrame(rows)
    df = df[df['formula'].notna()].copy()
    df['formula'] = df['formula'].astype(str)
    return df


def _build_manifest(dataset_name: str, version_hint: str) -> dict:
    return DatasetManifest(
        name=dataset_name,
        source='jarvis-tools/figshare',
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        version_hint=version_hint,
    ).model_dump()


def _has_required_normalized_columns(df: pd.DataFrame) -> bool:
    return all(column in df.columns for column in REQUIRED_NORMALIZED_COLUMNS)


def _load_cached_raw(raw_path: Path) -> list[dict] | None:
    if not raw_path.exists():
        return None
    payload = json.loads(raw_path.read_text())
    return payload if isinstance(payload, list) else None


def _write_dataset_artifacts(
    raw: list[dict],
    target_col: str,
    processed_path: Path,
    manifest_path: Path,
    dataset_name: str,
    version_hint: str,
) -> tuple[pd.DataFrame, dict]:
    df = _normalize(raw, target_col)
    df.to_parquet(processed_path, index=False)
    manifest = _build_manifest(dataset_name, version_hint=version_hint)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return df, manifest


def load_or_build_dataset(cfg: dict) -> tuple[pd.DataFrame, dict]:
    raw_dir = Path(cfg['data']['raw_dir'])
    processed_dir = Path(cfg['data']['processed_dir'])
    target_col = cfg['data']['target_column']
    dataset_name = cfg['data']['dataset']
    raw_path = raw_dir / f'{dataset_name}.json'
    processed_path = processed_dir / f'{dataset_name}.parquet'
    manifest_path = processed_dir / 'manifest.json'

    if processed_path.exists() and manifest_path.exists():
        cached_df = pd.read_parquet(processed_path)
        cached_manifest = json.loads(manifest_path.read_text())
        if _has_required_normalized_columns(cached_df):
            return cached_df, cached_manifest

        cached_raw = _load_cached_raw(raw_path)
        if cached_raw is not None:
            return _write_dataset_artifacts(
                raw=cached_raw,
                target_col=target_col,
                processed_path=processed_path,
                manifest_path=manifest_path,
                dataset_name=dataset_name,
                version_hint='rebuilt from cached raw json',
            )

    cached_raw = _load_cached_raw(raw_path)
    if cached_raw is not None:
        return _write_dataset_artifacts(
            raw=cached_raw,
            target_col=target_col,
            processed_path=processed_path,
            manifest_path=manifest_path,
            dataset_name=dataset_name,
            version_hint='loaded from cached raw json',
        )

    from jarvis.db.figshare import data as jarvis_data

    raw = jarvis_data(dataset_name)
    if cfg['data'].get('cache_raw_json', True):
        raw_path.write_text(json.dumps(raw))

    return _write_dataset_artifacts(
        raw=raw,
        target_col=target_col,
        processed_path=processed_path,
        manifest_path=manifest_path,
        dataset_name=dataset_name,
        version_hint='runtime download',
    )
