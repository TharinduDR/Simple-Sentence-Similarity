import logging

import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from simplests.model_args import SentenceEmbeddingSTSArgs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentenceTransformerSTSMethod:
    def __init__(self, model_args: SentenceEmbeddingSTSArgs):
        self.model_args = model_args
        logging.info("Loading models ")
        self.model = SentenceTransformer(model_args.embedding_model)

    def predict(self, to_predict, batch_size=32):
        sims = []

        sentences_1 = list(zip(*to_predict))[0]
        sentences_2 = list(zip(*to_predict))[1]

        embeddings_1 = self.model.encode(sentences_1, batch_size=batch_size, show_progress_bar=True)
        embeddings_2 = self.model.encode(sentences_2, batch_size=batch_size, show_progress_bar=True)

        for embedding_1, embedding_2 in tqdm(zip(embeddings_1, embeddings_2), total=len(embeddings_1), desc="Calculating similarity "):
            cos_sim = np.dot(embedding_1, embedding_2) / (
                    norm(embedding_1) * norm(embedding_2))
            sims.append(cos_sim)

        return sims
