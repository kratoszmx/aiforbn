from __future__ import annotations

import builtins
from io import BytesIO
import json
import sys
import types
import zipfile

import pandas as pd
import pytest

from runtime import io_utils
from materials.data import (
    REFERENCE_PROPERTY_COLUMNS,
    STRUCTURE_SUMMARY_COLUMNS,
    load_cached_raw_record_lookup,
    load_or_build_dataset,
)


def test_load_or_build_dataset_rejects_human_docs_cache_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    cfg = {
        'data': {
            'raw_dir': str(tmp_path / 'human_docs' / 'raw'),
            'processed_dir': str(tmp_path / 'processed'),
            'target_column': 'band_gap',
            'dataset': 'twod_matpd',
        },
    }

    with pytest.raises(ValueError, match='user-owned human_docs'):
        load_or_build_dataset(cfg)

    assert not (tmp_path / 'human_docs').exists()


@pytest.mark.parametrize('alias_kind', ['broken_symlink', 'hardlink'])
def test_load_or_build_dataset_rejects_processed_leaf_aliases_into_human_docs(
    tmp_path,
    monkeypatch,
    alias_kind,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    processed_dir = tmp_path / 'processed'
    processed_dir.mkdir()
    human_docs_file = tmp_path / 'human_docs' / 'twod_matpd.parquet'
    human_docs_file.parent.mkdir()
    processed_alias = processed_dir / 'twod_matpd.parquet'
    if alias_kind == 'broken_symlink':
        processed_alias.symlink_to(human_docs_file)
    else:
        human_docs_file.write_text('user-owned', encoding='utf-8')
        processed_alias.hardlink_to(human_docs_file)
    cfg = {
        'data': {
            'raw_dir': str(tmp_path / 'raw'),
            'processed_dir': str(processed_dir),
            'target_column': 'band_gap',
            'dataset': 'twod_matpd',
        },
    }

    with pytest.raises(ValueError, match='human_docs|multiple hard links'):
        load_or_build_dataset(cfg)

    if alias_kind == 'broken_symlink':
        assert not human_docs_file.exists()
    else:
        assert human_docs_file.read_text(encoding='utf-8') == 'user-owned'


@pytest.mark.parametrize(
    'invalid_leaf_kind',
    ['external_symlink', 'in_root_symlink', 'directory'],
)
def test_load_or_build_dataset_rejects_invalid_processed_file_leaves_before_effects(
    tmp_path,
    monkeypatch,
    invalid_leaf_kind,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    processed_dir = tmp_path / 'processed'
    processed_dir.mkdir()
    processed_path = processed_dir / 'twod_matpd.parquet'
    external_target = tmp_path / 'outside' / 'twod_matpd.parquet'
    if invalid_leaf_kind == 'external_symlink':
        processed_path.symlink_to(external_target)
    elif invalid_leaf_kind == 'in_root_symlink':
        processed_path.symlink_to(processed_dir / 'manifest.json')
    else:
        processed_path.mkdir()
    cfg = {
        'data': {
            'raw_dir': str(tmp_path / 'raw'),
            'processed_dir': str(processed_dir),
            'target_column': 'band_gap',
            'dataset': 'twod_matpd',
        },
    }

    with pytest.raises(ValueError, match='configured output root|symbolic-link|regular-file'):
        load_or_build_dataset(cfg)

    assert not external_target.exists()
    assert not (processed_dir / 'manifest.json').exists()


def test_load_or_build_dataset_rejects_non_directory_output_roots_before_effects(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    processed_file = tmp_path / 'processed-file'
    processed_file.write_text('keep', encoding='utf-8')
    cfg = {
        'data': {
            'raw_dir': str(tmp_path / 'new-raw'),
            'processed_dir': str(processed_file),
            'target_column': 'band_gap',
            'dataset': 'twod_matpd',
        },
    }

    with pytest.raises(ValueError, match='directory'):
        load_or_build_dataset(cfg)

    assert not (tmp_path / 'new-raw').exists()
    assert processed_file.read_text(encoding='utf-8') == 'keep'


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


def _install_fake_jarvis(
    monkeypatch,
    data_func,
    *,
    get_db_info_func=None,
    get_request_data_func=None,
):
    fake_figshare = types.ModuleType('jarvis.db.figshare')
    fake_figshare.data = data_func
    fake_figshare.get_db_info = get_db_info_func or (lambda: {
        'twod_matpd': ('unused-url', 'twodmatpd.json', 'message', 'reference'),
    })
    fake_figshare.get_request_data = get_request_data_func or (
        lambda *, js_tag, url, store_dir: data_func(
            'twod_matpd',
            store_dir=store_dir,
        )
    )
    fake_db = types.ModuleType('jarvis.db')
    fake_db.figshare = fake_figshare
    fake_jarvis = types.ModuleType('jarvis')
    fake_jarvis.db = fake_db
    monkeypatch.setitem(sys.modules, 'jarvis', fake_jarvis)
    monkeypatch.setitem(sys.modules, 'jarvis.db', fake_db)
    monkeypatch.setitem(sys.modules, 'jarvis.db.figshare', fake_figshare)
    return fake_figshare


class _JarvisResponse:
    def __init__(self, body: bytes, status_code: int = 200):
        self.body = body
        self.status_code = status_code
        self.headers = {'content-length': str(len(body))}

    def iter_content(self, block_size: int):
        for start in range(0, len(self.body), block_size):
            yield self.body[start:start + block_size]


def _jarvis_archive_bytes(
    member_name: str,
    member_payload: str | None,
    *,
    extra_members: tuple[tuple[str, str], ...] = (),
) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, 'w') as archive:
        if member_payload is not None:
            archive.writestr(member_name, member_payload)
        for name, payload in extra_members:
            archive.writestr(name, payload)
    return output.getvalue()


def _installed_jarvis_case(tmp_path, monkeypatch):
    import jarvis.db.figshare as installed_figshare

    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'mpl-cache'))
    raw_dir = tmp_path / 'raw'
    processed_dir = tmp_path / 'processed'
    processed_dir.mkdir()
    metadata = installed_figshare.get_db_info()['twod_matpd']
    cfg = {
        'data': {
            'dataset': 'twod_matpd',
            'raw_dir': str(raw_dir),
            'processed_dir': str(processed_dir),
            'target_column': 'band_gap',
            'cache_raw_json': True,
        }
    }
    return installed_figshare, cfg, raw_dir, processed_dir, metadata


def test_load_or_build_dataset_uses_one_guarded_jarvis_metadata_snapshot(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'mpl-cache'))
    raw_dir = tmp_path / 'raw'
    processed_dir = tmp_path / 'processed'
    processed_dir.mkdir()
    safe_metadata = ('unused-url', 'twodmatpd.json', 'message', 'reference')
    changed_metadata = (
        'unused-url',
        '../human_docs/escaped.json',
        'message',
        'reference',
    )
    metadata_calls = []
    request_calls = []
    payload = [_raw_entry('1', 'BN', 5.5)]
    fake_figshare = None

    def changing_db_info():
        metadata_calls.append(len(metadata_calls) + 1)
        metadata = safe_metadata if len(metadata_calls) == 1 else changed_metadata
        return {'twod_matpd': metadata}

    def fake_request_data(*, js_tag, url, store_dir):
        request_calls.append((js_tag, url, store_dir))
        assert js_tag == safe_metadata[1], 'dependency consumed unguarded metadata'
        return payload

    def fake_data(dataset, store_dir=None):
        metadata = fake_figshare.get_db_info()[dataset]
        return fake_figshare.get_request_data(
            js_tag=metadata[1],
            url=metadata[0],
            store_dir=store_dir,
        )

    fake_figshare = _install_fake_jarvis(
        monkeypatch,
        fake_data,
        get_db_info_func=changing_db_info,
        get_request_data_func=fake_request_data,
    )
    cfg = {
        'data': {
            'dataset': 'twod_matpd',
            'raw_dir': str(raw_dir),
            'processed_dir': str(processed_dir),
            'target_column': 'band_gap',
            'cache_raw_json': False,
        }
    }

    load_or_build_dataset(cfg)

    assert metadata_calls == [1]
    assert request_calls == [(
        safe_metadata[1],
        safe_metadata[0],
        str(raw_dir.resolve()),
    )]
    assert not (tmp_path / 'human_docs').exists()


def _jarvis_db_info(metadata):
    return {'twod_matpd': metadata}


@pytest.mark.parametrize(
    'db_info',
    [
        _jarvis_db_info(('unused-url', '/absolute.json', 'message', 'reference')),
        _jarvis_db_info(('unused-url', '../escape.json', 'message', 'reference')),
        _jarvis_db_info(('unused-url', 'nested/name.json', 'message', 'reference')),
        _jarvis_db_info(('unused-url', '~/escape.json', 'message', 'reference')),
        _jarvis_db_info((
            'unused-url',
            '../HUMAN_DOCS/escape.json',
            'message',
            'reference',
        )),
        _jarvis_db_info(('unused-url', '', 'message', 'reference')),
        _jarvis_db_info(('unused-url', '.', 'message', 'reference')),
        _jarvis_db_info(('unused-url', '..', 'message', 'reference')),
        _jarvis_db_info(('unused-url', 'directory/', 'message', 'reference')),
        _jarvis_db_info(('unused-url', 'not-json.txt', 'message', 'reference')),
        _jarvis_db_info(('unused-url', 'nested\\name.json', 'message', 'reference')),
        _jarvis_db_info(('unused-url', '\x00.json', 'message', 'reference')),
        _jarvis_db_info((' unused-url', 'safe.json', 'message', 'reference')),
        _jarvis_db_info(('', 'safe.json', 'message', 'reference')),
        _jarvis_db_info(('unused-url', ' safe.json', 'message', 'reference')),
        None,
        [],
        {},
        _jarvis_db_info(None),
        _jarvis_db_info(('url', 'safe.json', 'message')),
        _jarvis_db_info(('url', None, 'message', 'reference')),
    ],
    ids=[
        'absolute',
        'parent-traversal',
        'nested-separator',
        'tilde',
        'case-equivalent-human-docs-traversal',
        'empty',
        'dot',
        'dot-dot',
        'directory-valued',
        'wrong-extension',
        'windows-separator',
        'nul',
        'url-leading-whitespace',
        'empty-url',
        'tag-leading-whitespace',
        'non-mapping',
        'sequence-root',
        'missing-dataset',
        'null-entry',
        'short-entry',
        'non-string-field',
    ],
)
def test_load_or_build_dataset_rejects_invalid_jarvis_metadata_before_effects(
    tmp_path,
    monkeypatch,
    db_info,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'mpl-cache'))
    request_calls = []

    def reject_request(**kwargs):
        request_calls.append(kwargs)
        raise AssertionError('JARVIS request must not run for invalid metadata')

    _install_fake_jarvis(
        monkeypatch,
        lambda *_args, **_kwargs: None,
        get_db_info_func=lambda: db_info,
        get_request_data_func=reject_request,
    )
    cfg = {
        'data': {
            'dataset': 'twod_matpd',
            'raw_dir': str(tmp_path / 'raw'),
            'processed_dir': str(tmp_path / 'processed'),
            'target_column': 'band_gap',
            'cache_raw_json': False,
        }
    }

    with pytest.raises(ValueError, match='Check DB name options|JARVIS'):
        load_or_build_dataset(cfg)

    assert request_calls == []
    assert not (tmp_path / 'raw').exists()
    assert not (tmp_path / 'processed').exists()
    assert not (tmp_path / 'human_docs').exists()


def test_load_or_build_dataset_binds_jarvis_cache_to_guarded_raw_dir(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    human_docs_cache = tmp_path / 'human_docs'
    monkeypatch.setenv('ATOMGPTLAB_CACHE', str(human_docs_cache))
    raw_dir = tmp_path / 'raw'
    processed_dir = tmp_path / 'processed'
    processed_dir.mkdir()
    captured_store_dirs = []
    payload = [_raw_entry('1', 'BN', 5.5)]

    def fake_data(dataset, store_dir=None):
        captured_store_dirs.append(store_dir)
        if store_dir is None:
            escaped_cache = human_docs_cache / 'jarvis_data'
            escaped_cache.mkdir(parents=True)
            (escaped_cache / 'twodmatpd.json.zip').write_bytes(b'escaped')
        return payload

    _install_fake_jarvis(monkeypatch, fake_data)
    cfg = {
        'data': {
            'dataset': 'twod_matpd',
            'raw_dir': str(raw_dir),
            'processed_dir': str(processed_dir),
            'target_column': 'band_gap',
            'cache_raw_json': False,
        }
    }

    load_or_build_dataset(cfg)

    assert captured_store_dirs == [str(raw_dir.resolve())]
    assert not human_docs_cache.exists()


def test_load_or_build_dataset_rejects_mpl_cache_before_jarvis_import(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    human_docs_cache = tmp_path / 'human_docs' / 'mpl'
    monkeypatch.setenv('MPLCONFIGDIR', str(human_docs_cache))
    real_import = builtins.__import__

    def reject_jarvis_import(name, *args, **kwargs):
        if name == 'jarvis.db.figshare':
            raise AssertionError('JARVIS import must happen after MPL cache preflight')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', reject_jarvis_import)
    cfg = {
        'data': {
            'dataset': 'twod_matpd',
            'raw_dir': str(tmp_path / 'raw'),
            'processed_dir': str(tmp_path / 'processed'),
            'target_column': 'band_gap',
            'cache_raw_json': False,
        }
    }

    with pytest.raises(ValueError, match='user-owned human_docs'):
        load_or_build_dataset(cfg)

    assert not (tmp_path / 'human_docs').exists()
    assert not (tmp_path / 'raw').exists()
    assert not (tmp_path / 'processed').exists()


@pytest.mark.parametrize(
    'leaf_kind',
    ['human-docs-symlink', 'dangling-symlink', 'hardlink', 'directory'],
)
def test_load_or_build_dataset_preflights_jarvis_archive_leaf(
    tmp_path,
    monkeypatch,
    leaf_kind,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    raw_dir = tmp_path / 'raw'
    processed_dir = tmp_path / 'processed'
    raw_dir.mkdir()
    archive_path = raw_dir / 'twodmatpd.json.zip'
    archive_target = tmp_path / 'human_docs' / archive_path.name
    if leaf_kind == 'human-docs-symlink':
        archive_path.symlink_to(archive_target)
    elif leaf_kind == 'dangling-symlink':
        archive_path.symlink_to(raw_dir / 'missing-archive.zip')
    elif leaf_kind == 'hardlink':
        archive_target.parent.mkdir()
        archive_target.write_text('user-owned', encoding='utf-8')
        archive_path.hardlink_to(archive_target)
    else:
        archive_path.mkdir()

    _install_fake_jarvis(
        monkeypatch,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('JARVIS download must not run before archive preflight')
        ),
    )
    cfg = {
        'data': {
            'dataset': 'twod_matpd',
            'raw_dir': str(raw_dir),
            'processed_dir': str(processed_dir),
            'target_column': 'band_gap',
            'cache_raw_json': False,
        }
    }

    with pytest.raises(
        ValueError,
        match='user-owned human_docs|symbolic-link|multiple hard links|regular-file',
    ):
        load_or_build_dataset(cfg)

    if leaf_kind in {'human-docs-symlink', 'dangling-symlink'}:
        assert archive_path.is_symlink()
    elif leaf_kind == 'hardlink':
        assert archive_target.read_text(encoding='utf-8') == 'user-owned'
    else:
        assert archive_path.is_dir()
    assert not processed_dir.exists()


def test_load_or_build_dataset_uses_installed_jarvis_request_implementation(
    tmp_path,
    monkeypatch,
):
    installed_figshare, cfg, raw_dir, processed_dir, metadata = (
        _installed_jarvis_case(tmp_path, monkeypatch)
    )
    expected_url, json_tag = metadata[:2]
    payload = [_raw_entry('installed-1', 'BN', 5.5)]
    response_body = _jarvis_archive_bytes(
        json_tag,
        json.dumps(payload),
        extra_members=(
            ('../ignored-traversal.json', '{}'),
            ('/ignored-absolute.json', '{}'),
        ),
    )
    request_calls = []

    def fake_get(url, stream=True):
        request_calls.append((url, stream))
        return _JarvisResponse(response_body)

    monkeypatch.setattr(installed_figshare.requests, 'get', fake_get)

    dataset_df, manifest = load_or_build_dataset(cfg)

    assert request_calls == [(expected_url, True)]
    assert dataset_df[['record_id', 'formula', 'target']].to_dict('records') == [
        {'record_id': 'installed-1', 'formula': 'BN', 'target': 5.5},
    ]
    assert manifest['name'] == 'twod_matpd'
    assert (raw_dir / f'{json_tag}.zip').is_file()
    assert (raw_dir / 'twod_matpd.json').is_file()
    assert (processed_dir / 'twod_matpd.parquet').is_file()
    assert (processed_dir / 'manifest.json').is_file()
    assert not (tmp_path / 'ignored-traversal.json').exists()
    assert not (tmp_path / 'ignored-absolute.json').exists()
    assert not (tmp_path / 'human_docs').exists()


@pytest.mark.parametrize(
    ('failure_kind', 'expected_error'),
    [
        ('http-error', zipfile.BadZipFile),
        ('empty-response', zipfile.BadZipFile),
        ('non-archive', zipfile.BadZipFile),
        ('missing-member', KeyError),
        ('malformed-json', json.JSONDecodeError),
        ('wrong-top-level', ValueError),
        ('empty-payload', ValueError),
        ('non-object-record', ValueError),
    ],
)
def test_load_or_build_dataset_cleans_new_invalid_installed_jarvis_archive(
    tmp_path,
    monkeypatch,
    failure_kind,
    expected_error,
):
    installed_figshare, cfg, raw_dir, processed_dir, metadata = (
        _installed_jarvis_case(tmp_path, monkeypatch)
    )
    json_tag = metadata[1]
    if failure_kind == 'http-error':
        invalid_response = _JarvisResponse(b'service unavailable', status_code=503)
    elif failure_kind == 'empty-response':
        invalid_response = _JarvisResponse(b'')
    elif failure_kind == 'non-archive':
        invalid_response = _JarvisResponse(b'not a zip archive')
    elif failure_kind == 'missing-member':
        invalid_response = _JarvisResponse(
            _jarvis_archive_bytes('other.json', '[]')
        )
    elif failure_kind == 'malformed-json':
        invalid_response = _JarvisResponse(
            _jarvis_archive_bytes(json_tag, '{broken')
        )
    elif failure_kind == 'empty-payload':
        invalid_response = _JarvisResponse(
            _jarvis_archive_bytes(json_tag, '[]')
        )
    elif failure_kind == 'non-object-record':
        invalid_response = _JarvisResponse(
            _jarvis_archive_bytes(json_tag, '[1]')
        )
    else:
        invalid_response = _JarvisResponse(
            _jarvis_archive_bytes(json_tag, json.dumps({'unexpected': 'mapping'}))
        )
    valid_response = _JarvisResponse(
        _jarvis_archive_bytes(
            json_tag,
            json.dumps([_raw_entry('retry-1', 'BN', 5.5)]),
        )
    )
    responses = iter((invalid_response, valid_response))
    request_calls = []

    def fake_get(url, stream=True):
        request_calls.append((url, stream))
        return next(responses)

    monkeypatch.setattr(installed_figshare.requests, 'get', fake_get)
    archive_path = raw_dir / f'{json_tag}.zip'
    raw_json_path = raw_dir / 'twod_matpd.json'
    processed_path = processed_dir / 'twod_matpd.parquet'
    manifest_path = processed_dir / 'manifest.json'

    with pytest.raises(expected_error):
        load_or_build_dataset(cfg)

    assert not archive_path.exists()
    assert not raw_json_path.exists()
    assert not processed_path.exists()
    assert not manifest_path.exists()
    assert not (tmp_path / 'human_docs').exists()

    dataset_df, manifest = load_or_build_dataset(cfg)

    assert len(request_calls) == 2
    assert dataset_df['record_id'].tolist() == ['retry-1']
    assert manifest['name'] == 'twod_matpd'


def test_load_or_build_dataset_preserves_preexisting_invalid_jarvis_archive(
    tmp_path,
    monkeypatch,
):
    installed_figshare, cfg, raw_dir, processed_dir, metadata = (
        _installed_jarvis_case(tmp_path, monkeypatch)
    )
    raw_dir.mkdir()
    json_tag = metadata[1]
    archive_path = raw_dir / f'{json_tag}.zip'
    archive_path.write_bytes(b'preexisting invalid cache')

    def reject_request(*_args, **_kwargs):
        raise AssertionError('An existing archive must not trigger a request')

    monkeypatch.setattr(installed_figshare.requests, 'get', reject_request)

    with pytest.raises(zipfile.BadZipFile):
        load_or_build_dataset(cfg)

    assert archive_path.read_bytes() == b'preexisting invalid cache'
    assert not (raw_dir / 'twod_matpd.json').exists()
    assert not (processed_dir / 'twod_matpd.parquet').exists()
    assert not (processed_dir / 'manifest.json').exists()


def test_load_or_build_dataset_builds_normalized_cache_and_reuses_it(tmp_path, monkeypatch):
    raw_dir = tmp_path / 'raw'
    processed_dir = tmp_path / 'processed'
    raw_dir.mkdir()
    processed_dir.mkdir()

    payload = [
        _raw_entry('1', 'BN', 5.5),
        _raw_entry('2', None, 2.0, composition='AlN'),
    ]

    fake_figshare = _install_fake_jarvis(
        monkeypatch,
        lambda dataset, store_dir=None: payload,
    )

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
    assert manifest1['target_column'] == 'band_gap'

    def should_not_be_called(**_kwargs):
        raise AssertionError('jarvis download should not run when cache exists')

    fake_figshare.get_request_data = should_not_be_called
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

    _install_fake_jarvis(
        monkeypatch,
        lambda dataset: (_ for _ in ()).throw(
            AssertionError(
                'jarvis download should not run when cached raw JSON is available'
            )
        ),
    )

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


def test_load_or_build_dataset_rebuilds_cache_with_mismatched_manifest(tmp_path, monkeypatch):
    raw_dir = tmp_path / 'raw'
    processed_dir = tmp_path / 'processed'
    raw_dir.mkdir()
    processed_dir.mkdir()
    raw_payload = [_raw_entry('fresh', 'BN', 5.5)]
    (raw_dir / 'twod_matpd.json').write_text(json.dumps(raw_payload), encoding='utf-8')
    stale_df = pd.DataFrame([{column: 0.0 for column in STRUCTURE_SUMMARY_COLUMNS}])
    stale_df['record_id'] = 'stale'
    stale_df['source'] = 'other_dataset'
    stale_df['formula'] = 'AlN'
    stale_df['target'] = 2.0
    for column in REFERENCE_PROPERTY_COLUMNS:
        stale_df[column] = 0.0
    stale_df.to_parquet(processed_dir / 'twod_matpd.parquet', index=False)
    (processed_dir / 'manifest.json').write_text(
        json.dumps({'name': 'other_dataset', 'source': 'jarvis-tools/figshare'}),
        encoding='utf-8',
    )

    cfg = {
        'data': {
            'dataset': 'twod_matpd',
            'raw_dir': str(raw_dir),
            'processed_dir': str(processed_dir),
            'target_column': 'band_gap',
        }
    }

    rebuilt_df, rebuilt_manifest = load_or_build_dataset(cfg)

    assert rebuilt_df['record_id'].tolist() == ['fresh']
    assert rebuilt_manifest['name'] == 'twod_matpd'
    assert rebuilt_manifest['version_hint'] == 'rebuilt from cached raw json'


def test_load_or_build_dataset_rebuilds_processed_cache_when_target_column_changes(tmp_path):
    raw_dir = tmp_path / 'raw'
    processed_dir = tmp_path / 'processed'
    raw_dir.mkdir()
    processed_dir.mkdir()
    raw_payload = [_raw_entry('target-switch', 'BN', 5.5)]
    (raw_dir / 'twod_matpd.json').write_text(
        json.dumps(raw_payload),
        encoding='utf-8',
    )
    cfg = {
        'data': {
            'dataset': 'twod_matpd',
            'raw_dir': str(raw_dir),
            'processed_dir': str(processed_dir),
            'target_column': 'band_gap',
        }
    }

    band_gap_df, band_gap_manifest = load_or_build_dataset(cfg)
    cfg['data']['target_column'] = 'energy_per_atom'
    energy_df, energy_manifest = load_or_build_dataset(cfg)

    assert band_gap_df['target'].tolist() == [5.5]
    assert band_gap_manifest['target_column'] == 'band_gap'
    assert energy_df['target'].tolist() == [-8.0]
    assert energy_manifest['target_column'] == 'energy_per_atom'
    assert energy_manifest['version_hint'] == 'rebuilt from cached raw json'


def test_load_cached_raw_record_lookup_handles_missing_non_list_and_duplicate_ids(tmp_path):
    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    cfg = {'data': {'dataset': 'twod_matpd', 'raw_dir': str(raw_dir)}}

    assert load_cached_raw_record_lookup(cfg) == {}

    raw_path = raw_dir / 'twod_matpd.json'
    raw_path.write_text(json.dumps({'not': 'a list'}), encoding='utf-8')
    assert load_cached_raw_record_lookup(cfg) == {}

    raw_path.write_text(
        json.dumps([
            {'jid': 'dup', 'formula': 'BN', 'value': 1},
            'ignored',
            {'jid': 'dup', 'formula': 'BN', 'value': 2},
            {'formula': 'AlN'},
        ]),
        encoding='utf-8',
    )

    assert load_cached_raw_record_lookup(cfg) == {
        'dup': {'jid': 'dup', 'formula': 'BN', 'value': 2},
        '3': {'formula': 'AlN'},
    }
