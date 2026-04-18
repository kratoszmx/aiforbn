import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.io_utils import clear_project_cache

clear_project_cache('.')

st.set_page_config(page_title='BN Explorer', layout='wide')
st.title('BN Explorer')

artifact_dir = Path('artifacts')
metrics_path = artifact_dir / 'metrics.json'
summary_path = artifact_dir / 'experiment_summary.json'
benchmark_path = artifact_dir / 'benchmark_results.csv'
robustness_path = artifact_dir / 'robustness_results.csv'
pred_path = artifact_dir / 'predictions.csv'
screen_path = artifact_dir / 'demo_candidate_ranking.csv'

st.write('Minimal PoC UI for BN property prediction, grouped evaluation, and demo candidate ranking.')

if metrics_path.exists():
    st.subheader('Metrics')
    st.json(json.loads(metrics_path.read_text()))
else:
    st.info('Run `python main.py` first to generate artifacts.')

if summary_path.exists():
    st.subheader('Experiment summary')
    st.json(json.loads(summary_path.read_text()))

if benchmark_path.exists():
    st.subheader('Benchmark results')
    st.dataframe(pd.read_csv(benchmark_path), use_container_width=True)

if robustness_path.exists():
    st.subheader('Grouped robustness results')
    st.dataframe(pd.read_csv(robustness_path), use_container_width=True)

if pred_path.exists():
    st.subheader('Prediction samples')
    st.dataframe(pd.read_csv(pred_path).head(30), use_container_width=True)

if screen_path.exists():
    st.subheader('Top demo candidate ranking')
    st.dataframe(pd.read_csv(screen_path).head(30), use_container_width=True)
