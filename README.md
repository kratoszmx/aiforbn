# AI for BN PoC

Minimal AI-first PoC for boron nitride (BN) materials exploration.

## MVP
- Dataset: 2DMatPedia (via JARVIS-Tools)
- Task: band gap prediction + BN candidate screening
- Baseline: sklearn + composition features
- UI: Streamlit
- LLM: optional explanation layer only

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Main entry
- `main.py` is intentionally simple and linear so it can be moved into a notebook easily.
