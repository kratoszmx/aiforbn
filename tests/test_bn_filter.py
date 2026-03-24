from bnai.preprocess.bn_filters import extract_elements, filter_bn
import pandas as pd


def test_extract_elements():
    assert extract_elements('BN') == ['B', 'N']


def test_filter_bn():
    df = pd.DataFrame({'formula': ['BN', 'AlN', 'C']})
    out = filter_bn(df)
    assert out['formula'].tolist() == ['BN']
