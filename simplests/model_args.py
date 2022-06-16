from dataclasses import dataclass, field


@dataclass
class SimpleSTSArgs:
    distance_matrix: str = "cosine"


@dataclass
class WordEmbeddingSTSArgs(SimpleSTSArgs):
    embedding_models: list = field(default_factory=list)
    remove_stopwords: bool = False
    language: str = "en"


@dataclass
class SentenceEmbeddingSTSArgs(SimpleSTSArgs):
    embedding_model: str = ""
    language: str = "en"

