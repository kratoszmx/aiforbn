from __future__ import annotations

import json
import sys
import types

import pandas as pd

from pipeline.data import load_or_build_dataset


def test_load_or_build_dataset_builds_normalized_cache_and_reuses_it(tmp_path, monkeypatch):
    raw_dir = tmp_path / 'raw'
    processed_dir = tmp_path / 'processed'
    raw_dir.mkdir()
    processed_dir.mkdir()

    payload = [
        {'jid': '1', 'formula': 'BN', 'bandgap': 5.5},
        {'jid': '2', 'atoms': {'composition': 'AlN', 'band_gap': 2.0}},
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
    assert json.loads((raw_dir / 'twod_matpd.json').read_text())[0]['formula'] == 'BN'
    assert (processed_dir / 'twod_matpd.parquet').exists()
    assert manifest1['name'] == 'twod_matpd'

    def should_not_be_called(_dataset):
        raise AssertionError('jarvis download should not run when cache exists')

    fake_figshare.data = should_not_be_called
    df2, manifest2 = load_or_build_dataset(cfg)

    pd.testing.assert_frame_equal(df2, df1)
    assert manifest2 == json.loads((processed_dir / 'manifest.json').read_text())
