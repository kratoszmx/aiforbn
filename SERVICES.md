# Services and external interfaces

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`.

The historical band-gap pipeline is a finite `main.py` batch run. The separator partner prototype adds the bounded public web service below; there is no project-owned MCP server. Overleaf is an external proposal-delivery target; an available session connector is not a service maintained by this repository.

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

## Separator partner prototype

Entry: `src/ui/separator_app.py`; static frontend: `src/ui/separator_web/`; scientific/data policy: documented `materials.separator_data` and `materials.separator_model` APIs. It listens only on `127.0.0.1:8766`. Cloudflare Quick Tunnel supplies a temporary public HTTPS hostname using outbound connections; no firewall port is opened. The prototype exposes only curated public records and named downloads, not the repository or private meeting material.

Local deployment state is ignored under `.runtime/separator/`: `deployment.json` contains public URL, seven-day application expiry and the rolling model-call budget; `access_token.txt` is the private invitation (0600); `partner_invite.txt` is the copyable partner link; `public.sqlite` contains only public evidence. Separate `usage.sqlite` tracks model requests and real cached outputs. `model_verified.json` records a completed real provider test; it is historical evidence, not continuous provider health.

Two per-user launchd jobs are `com.zmx.aiforbn-separator-web` and `com.zmx.aiforbn-separator-tunnel`. Their plists and logs are in `.runtime/separator/`. Program arguments call the quant interpreter and installed cloudflared directly. Each runs without a terminal and restarts on failure. A Quick Tunnel hostname can change when its process is recreated, and Quick Tunnels carry no uptime guarantee. Keep the host powered and online. Application expiry returns HTTP 410, including for model calls; it does not itself unload launchd jobs.

Health: `curl --fail --max-time 5 http://127.0.0.1:8766/health`. Check the saved public URL through an independent HTTP path as well. The response includes the dataset digest and expiry. HTML 200 alone does not prove model inference; use a strict recipe plus `X-Demo-Token` for a bounded actual call, and retain its provider usage/latency separately. Public evidence/CSV/SQLite require no token. Live model use requires the invitation; default limit is 20 new calls per rolling 24 hours, one active call, 120-second provider timeout, with real-result caching.

Lifecycle commands from the repository root:

```sh
launchctl print gui/$(id -u)/com.zmx.aiforbn-separator-web
launchctl print gui/$(id -u)/com.zmx.aiforbn-separator-tunnel
launchctl bootout gui/$(id -u)/com.zmx.aiforbn-separator-tunnel
launchctl bootout gui/$(id -u)/com.zmx.aiforbn-separator-web
```

To restart after an intentional stop, bootstrap each existing plist with `launchctl bootstrap gui/$(id -u) /absolute/path/to/plist`. Read the new tunnel hostname from its log, update the saved public URL and partner link, and reverify externally before claiming availability. Keep invitations and logs out of Git. No global Codex settings or network routing are modified.
