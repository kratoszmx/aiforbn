# Research source index

Updated: 2026-09-27. This directory holds source records and reference pointers. A meeting record is first-hand context, not an institutional policy or independently verified scientific result.

## Meeting source archive

Local source bundle: `meetings/2026-09-09_bn_research/`.

| File | Role |
| --- | --- |
| [SOURCE_MANIFEST.json](meetings/2026-09-09_bn_research/SOURCE_MANIFEST.json) | Original locations, sizes, SHA-256, provenance and reading limitations |
| [Original recording](meetings/2026-09-09_bn_research/audio/recording_original.mp4) | Original MP4 audio container, copied without conversion; 02:55:39.850 |
| [Reviewed transcript](meetings/2026-09-09_bn_research/transcripts/transcript_reviewed_zh.txt) | Preferred background-reviewed Chinese transcript |
| [Original transcript](meetings/2026-09-09_bn_research/transcripts/transcript_original_zh.txt) | Initial machine transcript, retained for comparison |
| [Context review](meetings/2026-09-09_bn_research/evidence/context_review.json) | Existing terminology edits and speaker-attribution uncertainty |
| [Minute comparison](meetings/2026-09-09_bn_research/evidence/minute_comparison.md) | Initial cross-provider comparison; additional provider transcripts are beside it |

The preferred transcript has 2,979 turns and was delivered on 2026-09-16. It describes 63 minutes of fourth-provider checks; the preserved original delivery notes describe the earlier 20-minute stage. Neither version is a human-approved transcript. Names remain inferred; disputed attribution and editorial notes must stay attached to their statements. The recording date is taken from the source filename, not independently established from its contents.

Audio, transcripts and review evidence are private local files excluded from Git. Only the metadata manifest and this index are tracked. A clone will not contain the source files; request an authorized source transfer if needed, rather than treating missing local files as proof that the recording never existed. Downloads originals remain intact. No audio was played, decoded, re-transcribed or sent to a provider during this archival task.

The named predecessor chat (`01a06a3e-1df1-7863-9f52-c6ed083f2a37`) supplied the earlier single-person MPhil context. Its available history concerns pre-meeting preparation; the later recording/transcripts were located separately in Downloads and matched against the existing delivery hashes.

## Public literature starting points

Checked using text on 2026-09-27. These are source leads, not a completed literature review or training corpus. No paper figures were inspected and no numerical dataset was extracted in this task.

| Source | Observed scope and intended use | Limit |
| --- | --- | --- |
| [Bouville & Deville, BN aqueous dispersion with cellulose](https://arxiv.org/abs/1710.04239) — journal DOI `10.1111/jace.12653` | Author-hosted abstract describes particle size, cellulose concentration and suspension behaviour; starting point for dispersion terminology and methods | Abstract inspected; full methods/data still require text review. Water suspensions are not automatically a target polymer or separator formulation |
| [Chen et al., PDA-functionalized BN in polypropylene](https://pmc.ncbi.nlm.nih.gov/articles/PMC5249180/) — DOI `10.1371/journal.pone.0170523` | Text describes surface modification and polymer-composite thermal conductivity; useful for separating process variables and downstream properties | Thermal conductivity is not a direct dispersion label; it does not establish separator performance |
| [Sleiti, PAO/hBN measured-viscosity dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC7907776/) — DOI `10.1016/j.dib.2021.106881` | Text and tables expose concentration, temperature and rheology measurements; candidate for testing a data schema | Oil-based nanofluid; repeated conditions do not equal independent formulations or studies. Relevance to the chosen target is unproven |
| [AbuShanab et al., PAO/hBN viscosity machine learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC10245067/) — DOI `10.1016/j.heliyon.2023.e16716` | Existing ML application provides prior work to assess novelty against | Published accuracy is the authors' result, not reproduced here and not evidence of cross-study or separator generalization |

Access note: initial PMC text reads returned a challenge/403; the existing text-fetch tool's curl transport with environment proxy subsequently returned each matching article title and substantive article text. No persistent network settings were changed. There are no unresolved source-access failures for these leads.

## Derived plans

- [Chinese plan for the user](../human_docs/next_steps_zh.md).
- [English plan for agents](../docs/research/next_steps_en.md).
- [Document reorganization and recovery record](../docs/research/document_reorganization_2026-09-27.json).
