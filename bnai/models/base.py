from dataclasses import dataclass
from typing import Any


@dataclass
class ModelBundle:
    name: str
    model: Any
    feature_columns: list[str]
    target_column: str = 'target'
