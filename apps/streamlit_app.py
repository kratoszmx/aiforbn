import json
from pathlib import Path

import pandas as pd
import streamlit as st

from bnai.utils.io import delete_cache

delete_cache('.')

st.set_page_config(page_title='BN Explorer', layout='wide')
st.title('BN Explorer')

artifact_dir = Path('artifacts')
metrics_path = artifact_dir / 'metrics.json'
pred_path = artifact_dir / 'predictions.csv'
screen_path = artifact_dir / 'screened_candidates.csv'

st.write('Minimal PoC UI for BN property prediction and candidate screening.')

if metrics_path.exists():
    st.subheader('Metrics')
    st.json(json.loads(metrics_path.read_text()))
else:
    st.info('Run `python main.py` first to generate artifacts.')

if pred_path.exists():
    st.subheader('Prediction samples')
    st.dataframe(pd.read_csv(pred_path).head(30), use_container_width=True)

if screen_path.exists():
    st.subheader('Top screened BN-related candidates')
    st.dataframe(pd.read_csv(screen_path).head(30), use_container_width=True)
