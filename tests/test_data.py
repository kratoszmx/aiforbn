from __future__ import annotations

import json
import sys
import types

import pandas as pd
import pytest

from pipeline.data import REFERENCE_PROPERTY_COLUMNS, STRUCTURE_SUMMARY_COLUMNS, load_or_build_dataset


def _raw_entry(jid: str, formula: str | None, target: float, *, composition: str | None = None) -> dict:
    atoms = {
        'lattice_mat': [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 20.0]],
        'coords': [[0.0, 0.0, 0.45], [0.5, 0.5, 0.55]],
        'elements': ['B', 'N'] if (formula or composition) == 'BN' else ['Al', 'N'],
        'abc': [2.0, 2.0, 20.0],
        'angles': [90.0, 90.0, 90.0],
        'cartesian': False,
        'props': ['', ''],
    }
    if composition is not None:
        atoms['composition'] = composition
        atoms['band_gap'] = target

    entry = {
        'jid': jid,
        'atoms': atoms,
        'energy_per_atom': -8.0 if (formula or composition) == 'BN' else -6.0,
        'exfoliation_energy_per_atom': 0.06 if (formula or composition) == 'BN' else 0.12,
        'total_magnetization': 0.0,
    }
    if formula is not None:
        entry['formula'] = formula
        entry['bandgap'] = target
    return entry


def test_load_or_build_dataset_builds_normalized_cache_and_reuses_it(tmp_path, monkeypatch):
    raw_dir = tmp_path / 'raw'
    processed_dir = tmp_path / 'processed'
    raw_dir.mkdir()
    processed_dir.mkdir()

    payload = [
        _raw_entry('1', 'BN', 5.5),
        _raw_entry('2', None, 2.0, composition='AlN'),
    ]

    fake_figshare = types.ModuleType('jarvis.db.figshare')
    fake_figshare.data = lambda dataset: payload
    fake_db = types.ModuleType('jarvis.db')
    fake_db.figshare = fake_figshare
    fake_jarvis = types.ModuleType('jarvis')
    fake_jarvis.db = fake_db

    monkeypatch.setitem(sys.modules, 'jarvis', fake_jarvis)
    monkeypatch.setitem(sys.modules, 'jarvis.db', fake_db)
    monkeypatch.setitem(sys.modules, 'jarvis.db.figshare', fake_figshare)

    cfg = {
        'data': {
            'dataset': 'twod_matpd',
            'raw_dir': str(raw_dir),
            'processed_dir': str(processed_dir),
            'target_column': 'band_gap',
            'cache_raw_json': True,
        }
    }

    df1, manifest1 = load_or_build_dataset(cfg)

    assert df1[['record_id', 'formula', 'target']].to_dict('records') == [
        {'record_id': '1', 'formula': 'BN', 'target': 5.5},
        {'record_id': '2', 'formula': 'AlN', 'target': 2.0},
    ]
    assert set(STRUCTURE_SUMMARY_COLUMNS).issubset(df1.columns)
    assert set(REFERENCE_PROPERTY_COLUMNS).issubset(df1.columns)
    assert df1.loc[0, 'structure_n_sites'] == 2.0
    assert df1.loc[0, 'structure_cell_height'] == pytest.approx(20.0)
    assert df1.loc[0, 'structure_thickness'] == pytest.approx(2.0)
    assert df1.loc[0, 'structure_vacuum'] == pytest.approx(18.0)
    assert json.loads((raw_dir / 'twod_matpd.json').read_text())[0]['formula'] == 'BN'
    assert (processed_dir / 'twod_matpd.parquet').exists()
    assert manifest1['name'] == 'twod_matpd'

    def should_not_be_called(_dataset):
        raise AssertionError('jarvis download should not run when cache exists')

    fake_figshare.data = should_not_be_called
    df2, manifest2 = load_or_build_dataset(cfg)

    pd.testing.assert_frame_equal(df2, df1)
    assert manifest2 == json.loads((processed_dir / 'manifest.json').read_text())


def test_load_or_build_dataset_rebuilds_stale_processed_cache_from_cached_raw_json(tmp_path, monkeypatch):
    raw_dir = tmp_path / 'raw'
    processed_dir = tmp_path / 'processed'
    raw_dir.mkdir()
    processed_dir.mkdir()

    payload = [
        _raw_entry('1', 'BN', 5.5),
        _raw_entry('2', None, 2.0, composition='AlN'),
    ]
    (raw_dir / 'twod_matpd.json').write_text(json.dumps(payload))
    pd.DataFrame([
        {'record_id': '1', 'source': 'twod_matpd', 'formula': 'BN', 'target': 5.5},
    ]).to_parquet(processed_dir / 'twod_matpd.parquet', index=False)
    (processed_dir / 'manifest.json').write_text(json.dumps({'name': 'twod_matpd'}))

    fake_figshare = types.ModuleType('jarvis.db.figshare')
    fake_figshare.data = lambda dataset: (_ for _ in ()).throw(
        AssertionError('jarvis download should not run when cached raw JSON is available')
    )
    fake_db = types.ModuleType('jarvis.db')
    fake_db.figshare = fake_figshare
    fake_jarvis = types.ModuleType('jarvis')
    fake_jarvis.db = fake_db

    monkeypatch.setitem(sys.modules, 'jarvis', fake_jarvis)
    monkeypatch.setitem(sys.modules, 'jarvis.db', fake_db)
    monkeypatch.setitem(sys.modules, 'jarvis.db.figshare', fake_figshare)

    cfg = {
        'data': {
            'dataset': 'twod_matpd',
            'raw_dir': str(raw_dir),
            'processed_dir': str(processed_dir),
            'target_column': 'band_gap',
            'cache_raw_json': True,
        }
    }

    rebuilt_df, rebuilt_manifest = load_or_build_dataset(cfg)

    assert set(STRUCTURE_SUMMARY_COLUMNS).issubset(rebuilt_df.columns)
    assert set(REFERENCE_PROPERTY_COLUMNS).issubset(rebuilt_df.columns)
    assert len(rebuilt_df) == 2
    assert rebuilt_df.loc[0, 'structure_thickness_fraction'] == pytest.approx(0.1)
    assert rebuilt_manifest['version_hint'] == 'rebuilt from cached raw json'
