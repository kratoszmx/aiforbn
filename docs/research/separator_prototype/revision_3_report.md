# September 28: partner alignment and response monitoring

## Meeting fit

The reviewed machine transcript supports starting with separators (01:13:38),
aqueous BN dispersion and particle-size mixtures (00:48:07), and collecting public
formulation evidence (02:12:38–02:16:46). Mentioned targets include approximately
2 μm coatings, viscosity “30–40” without a reliable unit/shear-rate definition,
and 180°C/30 min shrinkage ≤3%. These are meeting leads, not an accepted protocol.
Do not equate the mentioned 18% solids with vol% without confirming its basis.

The contact is transcribed as 小项/小向. At 01:13:52–01:14:18 she is described as a
company employee placed in a ceramics laboratory, not a university student. Her
surname spelling is not established; the weekly draft uses “負責隔膜實驗的同事”.
No partner communication was sent.

## Scientific audit and changes

- PP conductivity does not answer water-slurry viscosity or thermal shrinkage.
  Its two training labels cannot identify a purification effect. The frozen Astra
  result remains 0.900 mS/cm, error 0.060; linear gives 0.896667, error 0.056667,
  against the already known 0.840 observation.
- API v3 publishes the training-only linear reference and uses the language model
  for formulation explanation. Its raw numerical guess stays in the private
  receipt. No correction was fitted to 0.840, no best-of-many call was selected,
  and no historical report was overwritten. This matches linear; **it does not
  establish superiority**. See [development audit](pp_development_audit.json).
  Purified predictions still have an unlearned formulation-change effect.
- The electrolyte selector uses a 64-tree random forest fitted only to currently
  supplied measurements. Its score is predicted conductivity plus half the
  tree-to-tree standard deviation: an exploration heuristic, not calibrated
  uncertainty. Candidates exclude future answers. Repeated measurements are
  grouped by composition; all experiments, including shared initial ones, count.
- A mismatch was fixed: replay updates after each individual experiment, but the
  website offered three simultaneous recommendations. The website/API now return
  **one next formulation**, matching the tested sequential policy. Batch selection
  needs its own diversity policy and evaluation.
- The unchanged 13 mS/cm replay gives 7.69 experiments versus random 9.32 (17.5%
  fewer), nearest-formulation 7.75 (0.8% fewer; interval includes zero), and linear
  12.61. Results vary by target. This is one previously selected, measured liquid-
  electrolyte pool, not an independent BN dataset, an Astra result, or the partner's
  workflow. There is **no demonstrated advantage over the strongest simple
  selection comparator and no measured laboratory saving**.

## Literature coverage

The catalogue retains four extracted BN studies / 20 formulations and separate
Clio data (38 formulations / 125 measurements). Twelve BN source leads are not
twelve extracted training studies. Two previous context-only leads were now read
in full; reviews do not silently become compatible numerical training labels.

| Source | Use and limitation |
| --- | --- |
| [Kim 2022](https://doi.org/10.3390/nano12010011) | BNNT/PP, PVDF/NMP coating; PP conductivity reference. Not aqueous-slurry viscosity. |
| [Tian 2024](https://doi.org/10.3390/molecules29225311) | CA/BN thickness, DMF process. Different from PP water coating. |
| [Yin 2020](https://doi.org/10.1002/advs.202001303) | BN solid PEO/PVDF electrolyte; separate cohort. |
| [Hong 2026](https://doi.org/10.3390/en19071600) | Cellulose/BNNT separator; different substrate/process. |
| [Bouville and Deville 2014](https://arxiv.org/abs/1710.04239) | BN aqueous dispersion: HEC/MC, particle size, dispersant amount, rheology. Author manuscript read by text extraction; graphs not digitized. |
| [Chen 2017](https://doi.org/10.1371/journal.pone.0170523) | PDA increases BN hydrophilicity but can worsen dispersion in hydrophobic PP without compatibilizer. Not slurry-viscosity training data. |
| [Dave 2022 / Clio](https://doi.org/10.1038/s41467-022-32938-1) | Open liquid-electrolyte measurements; not BN separator outcomes. |

Bouville reports HEC/BN mass-ratio optima 0.0025 and 0.02 for 8 μm and 1 μm
powders in the described 19 vol% comparison. Rheology depends on conditioning,
shear rate and ageing. Its fitted Krieger–Dougherty parameters are not fresh
held-out measurements. Do not mix volume/mass fractions or invent graph labels.
Chen XML and source hashes are archived in `official_docs/dispersion/`; the author
PDF/text remain local research files excluded from Git and public downloads.

Coverage is **not sufficient** for validated partner-specific viscosity,
dispersion stability or shrinkage predictions. Further matching evidence needs
aqueous BN, particle size/blend, solids basis, dispersant identity/grade/dose,
pH, mixing, viscosity at defined temperature/shear rate, ageing, adhesion and
thermal shrinkage. KH550/KH570 comparisons remain a gap; an emulsion thermal-paste
abstract was screened, not counted as a matched training study.

To measure value, freeze candidate recipes and the partner's selection baseline;
give both policies identical starting results/materials; record selections before
measurement, failures, repeats, person-hours and elapsed days. Compare experiments
to reach an agreed viscosity/coating/shrinkage goal, reserving a new batch or whole
study for validation. Reduced useless experimentation remains a hypothesis.

## Website and operations

Removed the requested debug/privacy paragraph, hours calculator, detailed replay
table/intervals and glossary. The title is “下一個配方，先試哪一個？”; the retained
summary identifies the liquid-electrolyte demonstration. Provider identity stays
private. See [partner instructions](partner_demo_zh.md).

`separator_monitor.py` checks loopback/public `/health` and the actual public
page every 300 seconds. Every six hours it posts a bounded fresh recipe through
public `/api/predict`, requiring a real model call, no cache hit, numeric output
and explanation. Normal consumption is four of the shared 100 daily slots. Busy/
quota responses defer. Failures differ from healthy HTTP. Receipts omit prompts,
credentials and response bodies. Expiry requires HTTP 410 and stops model calls.

Installed launchd job: `com.zmx.aiforbn-separator-monitor`. Supervisor reads its
deployment-bound, age-checked receipt using `service.ai-for-science-response` and
includes problems in existing daily reports. This is not an immediate push alert
or automatic repair. The global supervisor skill lists this authorized exception.
See [SERVICES](../../../SERVICES.md).

Initial live check: fresh public model completion in 21.275 seconds; Supervisor
projected `healthy / MODEL_RESPONSE_VERIFIED`. This proves response, not scientific
accuracy. Private validation evidence is under `.runtime/separator/revision3/`.
No external test message was sent.
