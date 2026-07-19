import pytest
from pydantic import ValidationError

from runtime.schema import DatasetManifest, MaterialRecord


def test_manifest():
    obj = DatasetManifest(
        name='x',
        source='y',
        retrieved_at='z',
        target_column='band_gap',
    )
    assert obj.name == 'x'
    assert obj.version_hint is None
    assert obj.model_dump() == {
        'name': 'x',
        'source': 'y',
        'retrieved_at': 'z',
        'target_column': 'band_gap',
        'version_hint': None,
    }


def test_material_record_defaults_and_serialization():
    record = MaterialRecord(source='jarvis', formula='BN')

    assert record.model_dump() == {
        'record_id': None,
        'source': 'jarvis',
        'formula': 'BN',
        'elements': [],
        'targets': {},
        'provenance': {},
    }


@pytest.mark.parametrize(
    ('model_type', 'payload'),
    [
        (
            DatasetManifest,
            {'source': 'jarvis', 'retrieved_at': 'now', 'target_column': 'band_gap'},
        ),
        (
            DatasetManifest,
            {'name': 'dataset', 'source': 'jarvis', 'retrieved_at': 'now'},
        ),
        (MaterialRecord, {'formula': 'BN'}),
    ],
)
def test_schema_required_fields_are_enforced(model_type, payload):
    with pytest.raises(ValidationError):
        model_type(**payload)
