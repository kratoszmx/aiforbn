# ui module public surface

This file lists the stable public functions that external code may call from `ui`.
Anything underscore-prefixed or omitted here should be treated as internal.

## streamlit_app.py

- `render_streamlit_app()`
  - Render the Streamlit artifact viewer for the generated project outputs.

## utils.py

- No public functions are currently exposed.

## tests/

- `test_streamlit_app.py`
  - Covers every declared artifact section through a fake renderer and verifies the supported `width='stretch'` dataframe contract.
  - Runs the app through Streamlit's real `AppTest` renderer so import and render failures remain text-verifiable.
