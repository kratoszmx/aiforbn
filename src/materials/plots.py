from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from runtime.io_utils import (
    configure_matplotlib_cache,
    make_json_safe,
    validate_runtime_output_path,
    write_json_file,
)

configure_matplotlib_cache()
import matplotlib.pyplot as plt

from materials.data import load_cached_raw_record_lookup
from materials.constants import *
from materials.candidate_space import *
from materials.feature_building import *
from materials.benchmarking import *
from materials.common import *

def save_basic_plots(prediction_df, cfg):
    artifact_dir = Path(cfg['project']['artifact_dir'])
    artifact_dir = validate_runtime_output_path(
        artifact_dir,
        expected_output_kind='directory',
    )
    parity_plot_path = artifact_dir / 'parity_plot.png'
    validate_runtime_output_path(
        parity_plot_path,
        required_parent_path=artifact_dir,
        expected_output_kind='file',
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(prediction_df['target'], prediction_df['prediction'], alpha=0.7)
    ax.set_xlabel('True target')
    ax.set_ylabel('Predicted target')
    ax.set_title('Parity plot')
    fig.tight_layout()
    fig.savefig(parity_plot_path, dpi=160)
    plt.close(fig)
