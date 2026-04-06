from typing import Any

from pydantic import BaseModel, Field


class DatasetManifest(BaseModel):
    name: str
    source: str
    retrieved_at: str
    version_hint: str | None = None


class MaterialRecord(BaseModel):
    record_id: str | None = None
    source: str
    formula: str
    elements: list[str] = Field(default_factory=list)
    targets: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
