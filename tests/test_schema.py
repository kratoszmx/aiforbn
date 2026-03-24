from bnai.schema import DatasetManifest


def test_manifest():
    obj = DatasetManifest(name='x', source='y', retrieved_at='z')
    assert obj.name == 'x'
