# Initial delivery verification — 2026-09-28 (historical)

This is evidence for the initial version, including its invitation gate, quota and exploratory recipes. Current operation and later delivery evidence are routed through the [prototype index](INDEX.md).

The prototype is functional; numerical reliability and experimental savings remain
unproven. See the [partner walkthrough](partner_demo_zh.md) and
[phase 2 findings](phase_2_report.md).

## Exact submitted code

The validation copy was exported from the Git index at base commit
`302103363e8047811c5207a2d299ce236f48a149`, tree
`e1ec93e24f76cdd79dcd717e7daaafe3ea3821ba`. It contains this delivery's changes
and excludes the five pre-existing dirty files/hunks from the previous coding
chat. Those unrelated edits remain in the original worktree. This verification
report is added after the tested code snapshot; no code or data changes are
permitted between that snapshot and the recorded delivery without revalidation.

Commands use Conda `quant` and the existing local `myutils` checkout. No packages
were installed. All production sources, test targets and external dependencies
remain under the project's manifest contracts.

| Check | Result |
| --- | --- |
| `python main.py --verify-agent-contract` on the submission copy | Passed; zero errors and warnings |
| `python main.py --dry-run` on the submission copy | Passed; no training or historical artifact refresh |
| `python -m pytest -q -ra src` on the submission copy | 1,193 passed, one archive-environment failure, four upstream warnings; 1,600.18 seconds |
| Sole failed test rerun with read-only Git context | Passed; 7.83 seconds; source bytes unchanged |
| Targeted dependency-discovery and real Chrome invitation regression checks | Passed |
| `git diff --cached --check` | Passed before snapshot export |

An earlier worktree-wide run stopped after 925 passing tests because descendant
dependency discovery executed FastAPI's parent initializer under a deliberately
broken Pydantic fixture. Discovery now resolves package paths without executing
those initializers; active imports remain in bounded subprocesses. The original
failing case and a direct side-effect/cache-invalidation regression passed. A
subsequent validation run was intentionally interrupted to incorporate a browser
fix: selecting an example must retain the invitation code. Neither interrupted
nor failed run is counted as a successful full-suite result.

The final full-scope invocation completed all 1,194 cases. Its only failure was
`test_build_agent_state_returns_json_serializable_status`: the plain archive had
no `.git`, so its branch was `None`. The same test passed on the unchanged
submission copy with `GIT_DIR` pointing to the original repository, `GIT_WORK_TREE`
pointing to the copy, and `GIT_OPTIONAL_LOCKS=0`. Only this read-only metadata test
received that Git context. Thus every selected case has a passing result across
the full run and the one-case rerun; this is not described as a single all-green
invocation. The four warnings concern Starlette/HTTPX, legacy WebSocket APIs and
PyTorch nested tensors. Private complete logs remain under `.runtime/separator/`.

## Historical public-operation evidence

Verified URL: <https://digital-plus-craps-chambers.trycloudflare.com>.
Application expiry: **2026-10-05 01:38 Asia/Shanghai / Hong Kong**
(`2026-10-04T17:38:32.914189+00:00`). The host must stay powered and online.
The temporary hostname can change if the tunnel process is recreated; this is
not a guaranteed-uptime hosting service. Lifecycle instructions are in
[SERVICES.md](../../../SERVICES.md).

Real checks through public HTTPS confirmed 20 records, four comparison methods,
matching dataset hashes, source evidence retrieval, rejection of unauthenticated
model requests (401), and no private-file route (404). A fresh Chrome session at
390 × 844 used DOM/text checks only: all three demonstration cases worked, the
invitation survived example selection, the invitation fragment was removed from
the address bar, there were no JavaScript errors or horizontal overflow.
Latest browser check: `2026-09-27T18:14:14.176943+00:00`.

A real, uncached public model request for raw BNNT at 0.4 mg/cm² returned
**0.80 mS/cm**, in 18.761 seconds, with valid training-record citations and explicit
limitations. This proves provider integration, not that the property is correct.
It used dataset hash `beb2204abb58b9640683e11e501952f9e745ecf712003bcb097788ae889cb8fa`.
A later update added an evidence paragraph for a different sample; the current
dataset hash is `b180de6a7411aa7bee18afa723aef08825e43f48313f48d8555d61cd76f58b49`.
Both the frozen evaluation prompt and the live-test prompt were checked unchanged;
the live-test prompt SHA-256 is
`c7c8ed3a63934e16c0d3f17db7d11578c3c8182951838eb87c8bc191dcb7ae63`.
The provider call was not repeated simply for the added source anchor. Current
public data and the final frontend were verified separately after that update.

Public data access is unrestricted; inference requires the invitation. The whole
deployment permits at most 20 new calls per rolling 24 hours, one concurrent call,
and a 120-second provider timeout. Previously successful results may be cached and
are labelled accordingly. Credentials and raw local service logs are excluded
from Git. The invitation link is in `.runtime/separator/partner_invite.txt` (0600).

Direct HTTP and Chrome reached the public service. A configured proxy route had
a TLS failure; the successful direct checks do not establish every network route.
Source collection used official full-text XML where publisher/PMC HTML access was
blocked. No image reading, PDF vision, screenshots or computer-use automation was
used. Source hashes, text anchors and original CC BY notices are retained.

## Scientific scope

The corpus has 12 source leads, four extracted studies, 20 formulation/sample
records and 18 numeric or categorical observations. These are different material
systems, not 20 compatible training labels. The numeric PP task has two training
labels and one withheld label from one study. Its model error is 0.0600 mS/cm;
loading-only linear regression error is 0.0567 mS/cm. No cross-study accuracy,
calibrated uncertainty, preparation success, or laboratory time/cost reduction is
established. Public-paper exposure during model pretraining cannot be excluded.
