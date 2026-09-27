# Phase 1 delivery — 2026-09-28

Delivered: 12-source inventory, four licensed primary-study full texts, twenty
source/sample formulations including controls and one failure, eighteen reported
observations, CSV and SQLite exports, a dictionary, three evidence cards and three
recalculation checks. Independent study count is four; the primary PP numerical
cohort has only one study and three explicit conductivity labels. The collection
target is met at the formulation inventory level, not at statistical sufficiency.

The complete source → paragraph/table → normalized record → check chain is
demonstrable. `src/materials/separator_ingest.py` rebuilds the package and fails
when selected text anchors disappear. Startup additionally verifies source-file
hashes, paragraph text, observation membership and split identities.

Known-case checks: raw-BNNT PP conductivity improvement recalculates to 65.1163%;
CA@BN-100 vs PP at 65°C to 118.75%. A published PP sulfur-utilization statement
does not reproduce: `1197/1675 × 100 = 71.4627%`, versus the stated 72.6%.
It is reported as a discrepancy, not silently corrected in source evidence.

Access and extraction gaps: PMC HTML delivered challenge pages; the permitted
Europe PMC XML route retrieved the articles. MDPI HTML returned 403, but the
publisher's XML attachment supplied the CNF study. Full methods for five relevant
abstract-only leads remain unextracted; three background leads are not training
data. Figure-only values, uncertain sample assignments and incompatible systems
were not converted into additional labels.

Phase 1 acceptance is met for an inspectable public evidence package. The first
model task and its limits are in `model_task.md`. Partner feedback has not been
received; the immediate useful feedback is which substrate, electrolyte and
adjustable preparation parameters match their actual laboratory workflow.
