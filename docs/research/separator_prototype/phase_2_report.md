# Phase 2 delivery — 2026-09-28

Delivered: a text-verifiable web prototype, public evidence database, structured
recipe applicability checks, an invitation-protected real GPT-6 Astra integration,
three reproducible demonstration types and a frozen comparison. No extra Python
packages were installed. Existing quant packages and Codex authentication are used.

Model call: the configured model is exactly `gpt-6-astra`, without substitution or
provider fallback. Each request is a fresh ephemeral CLI session; shell execution,
multi-agent delegation, images, web search and user configuration are disabled for
this application invocation. Requests contain a strict recipe and selected public
training records. Record citations are validated against that input. Local
subprocess errors never become a fabricated answer. No chat is created in the
user's sidebar and no messages are sent to collaborators.

The real frozen test result is in `evaluation.json` and
`prediction_vs_measurement.csv`. Measured conductivity was 0.84 mS/cm. Mean,
similar-formulation and Astra predictions were 0.57, 0.71 and 0.90; absolute
errors were 0.27, 0.13 and 0.06 mS/cm. The one-example error reduction versus
nearest formulation is 53.85%, and versus mean is 77.78%. A loading-only linear regression, added during review using the same frozen training
values, predicts 0.8967 with error 0.0567 mS/cm, slightly better than Astra. There
is therefore no demonstrated advantage over this simple regression. Astra was not
rerun or tuned after adding this baseline. These percentages
describe this single retrospective example only. They do not estimate general
accuracy, superior selection, or saved laboratory work.

The test answer was withheld from the actual prompt and retrieval used by the
model; `frozen_prompt.txt` makes this checkable. The source paper may have appeared
in the language model's pretraining. All three numeric labels are from one study,
with only one reserved formulation. Source-held-out accuracy is not estimable,
uncertainty is uncalibrated, and no statistical significance is claimed.

Fixed cases:

- Known: raw BNNT, 0.3 mg/cm², PP/PVDF/NMP, ratio 4:1, 50°C/24 h. Display the
  measured record and evidence, alongside clearly identified baseline/model output.
- Unused formulation: purified BNNT, 0.5 mg/cm², same protocol. The frozen result
  is compared with the withheld 0.84 mS/cm value in the evaluation panel.
- Unsupported: calcium alginate, DMF, ratio 3:1. Numeric inference is refused before
  a provider call, while its related published failure record remains inspectable.

The local/mobile DOM flow and an actual new-formulation inference through the
public HTTPS route passed. Required full-suite validation is recorded separately
in the delivery verification record. Numerical reliability and experimental
acceleration are not established. There are no fabricated manual-time, lab-time or
cost savings. Actual operator timing and a prospective control workflow remain
future evaluation inputs.

Next research work: acquire compatible independent PP studies, improve missing
conditions/labels, freeze a study-level test set, then compare candidate selection
with a measured laboratory baseline. Reports R1–R4 in the original plan are not
backfilled as four elapsed weeks; these phase reports describe the current delivery.
