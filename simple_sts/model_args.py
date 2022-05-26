from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SimpleSTSArgs:
    distance_matrix: str = "cosine"


@dataclass
class WordEmbeddingSTSArgs(SimpleSTSArgs):
    embedding_models: Dict[str, str] = field(default_factory=dict)