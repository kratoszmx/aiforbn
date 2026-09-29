# AI for Science — aqueous data and partner usability, 2026-09-29

Dated delivery record; current usage and subsequent changes are in the [prototype index](INDEX.md).

The demo now starts with aqueous BN slurry viscosity, supports persistent
light/dark preference, shows task-specific estimated waits, and locates evidence
by paper title and section. The standalone sources section, database link and
generic model notices are removed. Conditions, estimate/observation labels and
reported preparation failures remain useful user-facing facts.

## Meeting coverage

The private reviewed machine transcript under `official_docs/meetings/2026-09-09_bn_research/transcripts/`
was checked as text. Names/attribution are not human-approved; no audio/images were processed.

| Meeting requirement | Delivered | Still missing |
| --- | --- | --- |
| 00:48:07–48:48: aqueous BN, particle-size mixtures, higher solids | 12 new recipes/controls with particle size and explicit mass/volume bases | Validated mapping from bimodal BN distribution to maximum solids |
| 00:49:36–50:07: ~2 μm coating, viscosity “30–40”, 180°C/30 min shrinkage ≤3% | Within-series viscosity; 1.5 μm coating outcomes and viscosity-related failure | Viscosity unit/temperature/shear protocol; aligned 180°C/30 min MD/TD labels |
| 02:13–15: dispersed public data, KH550/KH570, simple interface | Cellulose/silane/PDA/Pluronic/mixed-filler reading, readable tables and direct examples | A defensible universal dispersant selector across substrates/binders |
| Iteratively reduce unnecessary experiments | One next electrolyte recipe from supplied observations; measured-pool replay | Prospective same-budget partner comparison and BN sequential validation |

This is a usable/checkable prototype increment, not completion of the larger
scientific objective. At 01:13:52 the colleague was described as a company
employee, not a confirmed student. The [weekly draft](weekly_update_draft_zh.md)
therefore uses “實驗室負責這部分的同事” and describes formulations/dispersion.

## Literature reading and stopping decision

Ten additional sources were reviewed at the depths below, with Bouville–Deville
and Chen rechecked from the existing archive. Full-text access does not mean
every figure was interpreted; only text methods/results were used.

| Source | Relevant reading and inclusion decision |
| --- | --- |
| [US20260121222A1](https://patents.google.com/patent/US20260121222A1/en) | Full patent examples, methods and table 1: 8 cases, 6 in the fixed BaTiO3/BN/SBA viscosity series. Alumina control and unmeasured paste excluded from fit |
| [CN105206783A](https://patents.google.com/patent/CN105206783A/zh) | Original Chinese paragraphs 0042/0048/0054/0060: 4 aqueous BN/polyacrylate recipes, 27/18/24/20 mPa·s. Several factors change together; references remain separate |
| [WO2014087375A1](https://patents.google.com/patent/WO2014087375A1/en) | Full text cellulose-ether examples/rheology; related Bouville/Deville work, not an independent replication. Graph-only viscosity curves not digitized |
| [JP2023080287A](https://patents.google.com/patent/JP2023080287A/en) | Full description around mixed BN/boehmite and figure 13; useful packing hypothesis, no usable exact text-readable viscosity series |
| [Pluronic/BNNT, 2019](https://doi.org/10.3390/polym11040582) | Full XML: 0.06 g BNNT + 0.6 g polymer + 30 g water; sonication affects tube integrity. ~5 wt% TGA fraction refers to dried polymer/BNNT, not aqueous solids; conclusion wording alone is misleading |
| [BNNT liquid crystals, 2024](https://doi.org/10.1002/sstr.202400281) | Full article text, rheology sections 2.2–2.3: concentration/viscosity can be nonmonotonic across phase transition. No universal monotonicity assumption or graph extraction |
| [Si3N4–BN gelcasting, 2016](https://doi.org/10.13005/msri/130105) | Full methods/results: PEI/pH effects, but sintered mixed ceramics and inconsistent particle-size/solids/pH descriptions. No clean matched separator labels |
| [BN–silica suspensions, 2007](https://doi.org/10.4028/www.scientific.net/KEM.336-338.988) | Publisher abstract only; pH/dispersant/high-solids background, no numeric training series extracted |
| [Modified BN/waterborne epoxy, 2023](https://doi.org/10.13416/j.ca.2023.10.010) | Publisher abstract compares KH550/560/570; an epoxy-specific result cannot establish the separator dispersant winner |
| [CN117177938A](https://patents.google.com/patent/CN117177938A/en) | Methods/examples: 25°C/3 rpm measurement, only broad 1–60 mPa·s range; powder-production/resin application, no per-formulation target labels |

Search batches covered water/cellulose/silane, separator slurry viscosity,
particle blending, thermal coating and data availability. After finding the two
extractable patents, two further Chinese/English checks of the target
viscosity/dataset questions added no compatible numerical series: repeats,
other systems, ranges or graph-only evidence dominated. For this prototype,
implementing/checking the new data has higher expected value than another broad
pass. This is a bounded low-marginal-return decision, not an exhaustive review.
New author data, partner conditions or a different target can reopen it.

CN122338356A remained an inaccessible search lead and was not used. Reusable
downloads are in ignored `official_docs/dispersion/local_cache/`; committed
patent excerpts preserve original paragraphs/table cells with pinned hashes.
The Pluronic CC BY XML retains its licence. No images were inspected.

## Scientific checks

The new catalogue has 12 cases and 11 numeric viscosities across two patents;
only six cases form the numerical task. Existing 20 BN records and Clio's
125 readings/38 compositions remain separate. Patent-reported outcomes are not
newly reproduced experiments. Missing viscosity is not zero; unknown measurement
temperature/shear rate remains null. LG's 25°C is preparation temperature, not
an assumed rheometer setting. cP and mPa·s have identical numerical values.

The task varies BN 0–12 vol% of solids, replacing BaTiO3; binder 15 vol%, total
solids 30 wt%, particle sizes and preparation stay fixed. Neighbouring measured
points define straight-line interpolation, with no extrapolation. At 4 vol%
BN the estimate is 7.45 mPa·s. The published 12 vol% case is 31.5 mPa·s but fails
the 1.5 μm coating target: “30–40” cannot be treated as a universal success range.

CN recipe masses are not converted to dry solids because supplier solution
concentrations are missing. Its third example says 50 nm then calls the
suspension micron-scale; the explicit diameter is retained, with this conflict
recorded. Its shrinkage claim lacks an aligned duration and is not a training
label for the partner's timed shrinkage target.

The [development evaluation](aqueous_development_evaluation.json) removes each
interior composition's label before fitting. Endpoints are excluded because
their removal requires extrapolation. All four cases/methods remain: MAE is
**1.948 mPa·s interpolation, 1.550 nearest-reference, 4.383 global linear**.
Interpolation improves over a single straight line here but does not beat
nearest-reference overall. No tuning on these results is sold as independent
validation; neither experimental savings nor confidence intervals are invented.

Existing principles and claims were also rechecked:

- PP: the historical one-case Astra result still did not beat the simple linear
  reference. The displayed number uses that reference plus a language explanation.
- Clio: forest mean plus tree spread is a heuristic, not calibrated uncertainty.
  Only observed labels enter the selector. Shared starts/candidate pool make the
  replay comparison fair within that retrospectively measured pool.
- At 13 mS/cm, random 9.32 versus forest 7.69 experiments is a 17.5% conditional
  reduction. Nearest-reference 7.75 is close; the paired difference interval
  crosses zero. There is no established significant win over that stronger
  baseline or the partner's actual process.
- Present usefulness is evidence gathering, avoiding known condition mismatches
  and failed recipes, bounded numerical exploration and iterative planning.
  Prospective same-budget measurements are still needed for actual lab savings.

## Delivery evidence

Only the BNNT/PP explanation makes a fresh remote language-model call; other
numerical tasks and planning run locally. Estimated waits are 20–60 and 1–3
seconds respectively; elapsed-time text handles longer calls. A fresh public
request returned HTTP 200, uncached and `model_called=true`, in **21.625 seconds**.
The HTTP/model monitor and Supervisor registration remain active. No new port,
dependency, subproject, tunnel restart or partner message was introduced.

Focused data/API/browser checks passed **75 tests**. Final candidate-tree full
suite, deployed DOM and model receipts live under ignored
`.runtime/separator/revision4/`; the delivery receipt records exact tested tree
and synchronized commit. Software cases are not independent research samples.
Previously dirty runtime/documentation work is preserved; only the owned appended
API-index section is staged from the mixed root index. Temporary public access
is scheduled to end 2026-10-05 01:38 Hong Kong/Beijing time.
