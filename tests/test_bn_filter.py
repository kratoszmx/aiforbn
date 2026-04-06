import pandas as pd

from pipeline.features import extract_elements, filter_bn


def test_extract_elements():
    assert extract_elements('BN') == ['B', 'N']


def test_filter_bn():
    df = pd.DataFrame({'formula': ['BN', 'AlN', 'C']})
    out = filter_bn(df)
    assert out['formula'].tolist() == ['BN']
