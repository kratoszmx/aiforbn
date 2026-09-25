# Services and external interfaces

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`.

This repository defines no managed background service, scheduler, launchd/systemd unit, or MCP server. Its primary operation is a finite `main.py` batch run. No separate MCP usage document is applicable. Overleaf is an external proposal-delivery target; an available session connector is not a service maintained by this repository.

## Optional artifact viewer

The viewer is [src/ui/streamlit_app.py](src/ui/streamlit_app.py). It reads the configured artifact root and displays report content only when provenance and committed output bytes are current. A running server with suppressed/stale content is not a successful scientific run.

From the repository root, use the verified `quant` interpreter described in [TESTING.md](TESTING.md):

```sh
conda run --no-capture-output -n quant python -m streamlit run src/ui/streamlit_app.py --server.headless true --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

This stays in the foreground; stop the owned process with Ctrl-C. If that port is occupied, choose a free loopback port and use it consistently rather than stopping an unrelated listener. In a separate terminal or process tool, use bounded checks:

```sh
curl --fail --max-time 5 http://127.0.0.1:8501/_stcore/health
curl --fail --max-time 5 --output /dev/null http://127.0.0.1:8501/
```

HTTP 200 proves listener/HTTP startup only. Use the emitted `ui_render_smoke` validation command for application rendering and artifact validation. A startup smoke should stop its process after checking and confirm that its listener has exited. This viewer is optional; routine docs work does not need a live listener.
