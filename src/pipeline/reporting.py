from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt


def save_metrics_and_predictions(metrics, prediction_df, bn_df, screened_df, manifest, cfg):
    artifact_dir = Path(cfg['project']['artifact_dir'])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    prediction_df.to_csv(artifact_dir / 'predictions.csv', index=False)
    bn_df.to_csv(artifact_dir / 'bn_slice.csv', index=False)
    screened_df.to_csv(artifact_dir / 'screened_candidates.csv', index=False)
    (artifact_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def save_basic_plots(prediction_df, cfg):
    artifact_dir = Path(cfg['project']['artifact_dir'])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(prediction_df['target'], prediction_df['prediction'], alpha=0.7)
    ax.set_xlabel('True target')
    ax.set_ylabel('Predicted target')
    ax.set_title('Parity plot')
    fig.tight_layout()
    fig.savefig(artifact_dir / 'parity_plot.png', dpi=160)
    plt.close(fig)
