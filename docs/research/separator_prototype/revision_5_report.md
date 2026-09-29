# AI for Science — language support and architecture update, 2026-09-29

The partner website defaults to Simplified Chinese, with persistent English and
Traditional Chinese choices. Forms, estimates, statuses, catalogue fields,
section locations and accessible labels share a presentation catalogue.
Language changes preserve entered values and never start model calls. An
existing language-model explanation is cleared when its language differs;
the next explicit estimate requests the chosen language. Original source
passages and bibliographic titles remain in their source language.

The three allowed response languages are validated before inference and enter
the prompt/cache identity. Language-independent local numerical results share
their cache across languages. Public records no longer include internal
curation notes; original records, source hashes and factual failure outcomes
remain intact. No model, numerical method or training label was changed.

## Meeting and architecture check

The existing private reviewed transcript was read as text, without processing
video or audio. Around 02:13–02:15, the meeting describes scattered public data,
an initial model, a simple user interface and iterative improvement. The
[Simplified Chinese weekly draft](weekly_update_draft_zh.md) therefore describes
data foundations, usable task flows and extension points instead of presenting
paper counts as the main deliverable. It has not been sent.

The implemented layers are source-checked data, bounded numerical methods,
configurable model explanation, a public web interface and service monitoring.
The electrolyte input/recommendation example demonstrates feedback, but does
not persist partner experiment history, import private lab datasets or
automatically retrain models. These are future extensions, not completed work.

## Scientific self-check

- Aqueous viscosity uses interpolation inside one fixed preparation series.
  This is a defensible reference estimate, not a physical law or validated
  mapping for arbitrary particle sizes, solids and dispersants.
- Coating thickness uses same-process observations; applicator gap remains
  distinct from final thickness. PP uses its simple numerical reference with
  a separate material explanation. No advantage over that reference is claimed.
- The next-formulation score balances forest mean and tree disagreement using
  observed labels only. Disagreement is a heuristic, not a calibrated confidence
  interval. Replay methods receive the same initial data; estimates apply to
  that electrolyte pool and objective. This update generates no new laboratory
  saving or scientific result.
- Practical value today is source retrieval, comparisons under specified
  conditions and a recommendation demonstration. Partner improvement requires
  frozen recommendations and prospective comparison with their usual workflow
  under an equal experiment budget.
- The phase 1/2 prototype is usable and inspectable. Bimodal particle blending,
  cross-system dispersant selection, matched rheology and 180°C / 30 min MD/TD
  shrinkage prediction remain research tasks. See [revision 4](revision_4_report.md)
  for the detailed meeting coverage matrix.

## Requested source retry

Google Patents CN122338356A was opened in visible Chrome through an isolated
ATT proxy: HTTP 404, Google error page. Visible Chrome with `--no-proxy-server`
timed out after 25 seconds. The first direct-browser setup using `direct://`
was invalid and is not counted as a direct-network result. The isolated proxy
and profile were removed; live Clash configuration stayed unchanged. No source
text was recovered or added to numerical data. Possible next sources are the
CNIPA publication document, an applicant copy or a later Google Patents index
update; none is claimed successful here.

## Verification

Tests cover language persistence despite an English browser locale, unchanged
inputs across switches, translated generated results and catalogue searches,
source-text preservation, internal-note removal, model language/cache isolation,
unchanged local values, mobile/desktop layout and theme interaction. Real
provider/public-host checks are separate from fixtures. Scoped candidate, test
and deployment receipts are under ignored `.runtime/separator/revision5/`;
unrelated runtime work is excluded. No historical band-gap output or human-owned
document was rewritten.
