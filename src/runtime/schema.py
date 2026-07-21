from typing import Any

from pydantic import BaseModel, Field


STRUCTURE_EXECUTION_OUTPUT_ROLES = (
    (
        'structure_generation_first_pass_execution',
        'first_pass_execution_artifact',
        'artifact',
        '.json',
    ),
    (
        'structure_generation_first_pass_execution_summary',
        'first_pass_execution_summary_artifact',
        'summary_artifact',
        '.csv',
    ),
    (
        'structure_generation_first_pass_execution_variants',
        'first_pass_execution_variants_artifact',
        'variants_artifact',
        '.csv',
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
