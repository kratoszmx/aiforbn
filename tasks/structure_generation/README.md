# Structure generation follow-up for BN candidates

This task directory is now grounded by the ranking pipeline outputs rather than being a placeholder.

## Inputs

The main handoff artifacts are:

- `artifacts/demo_candidate_structure_generation_seeds.csv`
- `artifacts/demo_candidate_structure_generation_handoff.json`
- `artifacts/demo_candidate_structure_generation_reference_records.json`
- `artifacts/demo_candidate_structure_generation_job_plan.json`

They are produced by `main.py` after candidate ranking.

## What these artifacts mean

They do **not** claim that a new BN candidate already has a valid structure.
They only provide a deterministic bridge from:

- formula-level candidate ranking
- to observed BN analog reference records with structure summaries

Each shortlisted candidate is linked to nearby BN-containing `train+val` reference formulas and exemplar records.
These exemplar records are intended to be starting prototypes for later substitution, enumeration, and relaxation workflows.
The reference-record payload JSON now also carries the unique raw `atoms` objects needed to start that work directly.

## Recommended downstream workflow

For each candidate in `demo_candidate_structure_generation_job_plan.json`:

1. Start from the candidate's top 1-3 jobs rather than only the flat seed rows.
2. Inspect `job_action_label`, `workflow_steps`, and the linked `seed_reference_record_id` / `seed_reference_formula`.
3. Pull the corresponding raw structure from `demo_candidate_structure_generation_reference_records.json`.
4. If `direct_element_substitution_feasible` is true, start from the suggested substitution pairs.
5. Otherwise follow the planned path, e.g. stoichiometry adjustment, insertion/decoration, removal/vacancy, or mixed edit enumeration.
6. Enumerate a small set of plausible decorated structures rather than trusting a single prototype.
7. Run geometry relaxation / stability screening before making any scientific claim.
8. Keep provenance from candidate formula to job id and seed record id.

## Minimum metadata to preserve

When creating downstream structure jobs, keep at least:

- candidate formula
- general ranking rank
- BN-centered ranking rank
- shortlist membership flags
- seed reference formula
- seed reference record id
- seed reference source
- seed reference band gap / energy / exfoliation values
- seed reference structure summary columns

## Interpretation boundary

These artifacts are a **prototype handoff layer**, not:

- structure generation proof
- thermodynamic stability proof
- synthesis feasibility proof
- discovery claim

The goal is simply to make the current BN ranking output usable by a later structure-aware workflow.
