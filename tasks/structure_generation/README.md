# Structure generation follow-up for BN candidates

This task directory is now grounded by the ranking pipeline outputs rather than being a placeholder.

## Inputs

The main handoff artifacts are:

- `artifacts/demo_candidate_structure_generation_seeds.csv`
- `artifacts/demo_candidate_structure_generation_handoff.json`
- `artifacts/demo_candidate_structure_generation_reference_records.json`

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

For each candidate in `demo_candidate_structure_generation_handoff.json`:

1. Start from the candidate's top 1-3 prototype seeds.
2. Inspect `seed_reference_record_id`, `seed_reference_formula`, and `seed_reference_source`.
3. Pull the corresponding raw structure from the original dataset source.
4. Apply composition-aware prototype editing or substitution to move from the reference formula family toward the candidate formula.
5. Enumerate a small set of plausible decorated structures rather than trusting a single prototype.
6. Run geometry relaxation / stability screening before making any scientific claim.
7. Keep provenance from candidate formula back to seed record id.

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
