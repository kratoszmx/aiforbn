# tests module public surface

`src/tests` is a pytest coverage directory, not a production API module.

## Public callable surface

- No stable external callable functions are exposed for production use.

## What lives here

- `test_config.py`
  - Validates the real `src/config.py` defaults.
- `test_main.py`
  - Covers the top-level pipeline entrypoint and the fast `--dry-run` smoke path.
- `utils.py`
  - Currently exposes no public helper functions.

## Notes

- Production code must not import from `src/tests`.
- Keep shared test helpers private unless a clearly reusable test-only helper emerges.
