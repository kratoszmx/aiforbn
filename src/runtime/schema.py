from typing import Any

from pydantic import BaseModel, Field


FIXED_REPORT_ARTIFACT_NAMES = frozenset({
    'artifact_provenance.json',
    'benchmark_results.csv',
    'bn_candidate_compatible_evaluation.csv',
    'bn_evaluation_matrix.csv',
    'bn_family_benchmark_results.csv',
    'bn_family_predictions.csv',
    'bn_model_role_comparison.csv',
    'bn_slice.csv',
    'bn_slice_benchmark_results.csv',
    'bn_slice_predictions.csv',
    'bn_stratified_error_results.csv',
    'demo_candidate_bn_centered_ranking.csv',
    'demo_candidate_extrapolation_shortlist.csv',
    'demo_candidate_proposal_shortlist.csv',
    'demo_candidate_rank_stability_summary.csv',
    'demo_candidate_ranking.csv',
    'demo_candidate_ranking_uncertainty.csv',
    'demo_candidate_structure_followup_report.csv',
    'demo_candidate_structure_generation_first_pass_queue.json',
    'demo_candidate_structure_generation_followup_extrapolation_shortlist.csv',
    'demo_candidate_structure_generation_followup_shortlist.csv',
    'demo_candidate_structure_generation_handoff.json',
    'demo_candidate_structure_generation_job_plan.json',
    'demo_candidate_structure_generation_reference_records.json',
    'demo_candidate_structure_generation_seeds.csv',
    'experiment_summary.json',
    'manifest.json',
    'metrics.json',
    'parity_plot.png',
    'predictions.csv',
    'robustness_results.csv',
    'screened_candidates.csv',
})


STRUCTURE_EXECUTION_OUTPUT_ROLES = (
    (
        'structure_generation_first_pass_execution',
        'first_pass_execution_artifact',
        'artifact',
        '.json',
        'demo_candidate_structure_generation_first_pass_execution.json',
    ),
    (
        'structure_generation_first_pass_execution_summary',
        'first_pass_execution_summary_artifact',
        'summary_artifact',
        '.csv',
        'demo_candidate_structure_generation_first_pass_execution_summary.csv',
    ),
    (
        'structure_generation_first_pass_execution_variants',
        'first_pass_execution_variants_artifact',
        'variants_artifact',
        '.csv',
        'demo_candidate_structure_generation_first_pass_execution_variants.csv',
    ),
)


class DatasetManifest(BaseModel):
    name: str
    source: str
    retrieved_at: str
    target_column: str
    version_hint: str | None = None


class MaterialRecord(BaseModel):
    record_id: str | None = None
    source: str
    formula: str
    elements: list[str] = Field(default_factory=list)
    targets: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
