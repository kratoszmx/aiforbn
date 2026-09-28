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

## AI for Science partner prototype

Entry: `src/ui/separator_app.py`; static frontend: `src/ui/separator_web/`; scientific/data policy: documented `materials.separator_data`, `materials.separator_model` and `materials.experiment_planning` APIs. It listens only on `127.0.0.1:8766`. Cloudflare Quick Tunnel supplies a temporary public HTTPS hostname using outbound connections; no firewall port is opened. The prototype exposes only curated public records and named downloads, not the repository or private meeting material.

Local deployment state is ignored under `.runtime/separator/`: `deployment.json` contains public URL, application expiry, `daily_model_limit` (100) and trusted `model_name` (default `gpt-6-astra`). The public website does not expose that provider identity. `partner_invite.txt` now contains the plain public URL; invitations are not required. Old invitation files are backed up privately. `public.sqlite` contains only public BN evidence; `electrolytes.csv` is a human-labelled normalized export. Separate `usage.sqlite` tracks all new numerical/planning analyses and real cached outputs. The cache identity includes task/input, source digests, provider selection and prompt bytes; private provider records never become public response metadata. `model_verified.json` records a completed real provider test; it is historical evidence, not continuous provider health.

Two per-user launchd jobs are `com.zmx.aiforbn-separator-web` and `com.zmx.aiforbn-separator-tunnel`. Their plists and logs are in `.runtime/separator/`. Program arguments call the quant interpreter and installed cloudflared directly. Each runs without a terminal and restarts on failure. A Quick Tunnel hostname can change when its process is recreated, and Quick Tunnels carry no uptime guarantee. Keep the host powered and online. Application expiry returns HTTP 410, including for model calls; it does not itself unload launchd jobs.

Health: `curl --fail --max-time 5 http://127.0.0.1:8766/health`. Check the saved public URL through an independent HTTP path as well. The response includes the dataset digest and expiry. HTML 200 alone does not prove model inference; use an anonymous strict recipe for a bounded actual call, and retain its provider usage/latency privately. The default shared limit is 100 new analyses per rolling 24 hours, one active analysis and a 120-second remote-provider timeout. Local numerical/planning tasks also count; cached results, browsing and downloads do not. Failed provider attempts count toward the budget. The trusted `model_name` can change without exposing provider controls to partners. Public input cannot select a model, run tools or submit arbitrary prompts.

Lifecycle commands from the repository root:

```sh
launchctl print gui/$(id -u)/com.zmx.aiforbn-separator-web
launchctl print gui/$(id -u)/com.zmx.aiforbn-separator-tunnel
launchctl bootout gui/$(id -u)/com.zmx.aiforbn-separator-tunnel
launchctl bootout gui/$(id -u)/com.zmx.aiforbn-separator-web
```

To restart after an intentional stop, bootstrap each existing plist with `launchctl bootstrap gui/$(id -u) /absolute/path/to/plist`. Read the new tunnel hostname from its log, update the saved public URL and partner link, and reverify externally before claiming availability. Keep old invitation backups and logs out of Git. Restart only the web job with `launchctl kickstart -k gui/$(id -u)/com.zmx.aiforbn-separator-web` for code updates; this preserves the running tunnel hostname. No global Codex settings or network routing are modified.

## Scheduled response monitor

`src/ui/separator_monitor.py` is installed as the per-user job
`com.zmx.aiforbn-separator-monitor`, with its plist in
`~/Library/LaunchAgents/`. It checks loopback health, public health/data identity
and the actual page every 300 seconds. Every six hours it calls the public
`/api/predict` with a fresh bounded recipe and requires uncached model completion.
Normal use consumes four of the shared 100 daily slots. HTTP 429 defers; failures
and stale evidence never pass. At planned expiry it verifies HTTP 410 without
calling the model. It does not restart services or send notifications itself.

Private, atomic evidence is `.runtime/separator/monitor.json`; logs are
`monitor.log` / `monitor-error.log`. Supervisor's
`service.ai-for-science-response` runs the following receipt-only command and
includes failures in the existing daily report, not an immediate push alert:

```sh
conda run -n quant python src/ui/separator_monitor.py --status
launchctl print gui/$(id -u)/com.zmx.aiforbn-separator-monitor
```

Evidence expires after 15 minutes for HTTP and 6 hours plus 15 minutes for model
completion; changing deployment configuration invalidates the old receipt.
To stop: `launchctl bootout gui/$(id -u)/com.zmx.aiforbn-separator-monitor`.
The daemon can then be restored by bootstrapping its existing plist. Do not
delete receipts to force repeat paid calls. A monitor success is response proof,
not scientific validation. Full behavior: [revision 3](docs/research/separator_prototype/revision_3_report.md).
