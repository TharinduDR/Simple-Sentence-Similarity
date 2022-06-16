import logging

import numpy as np
import tensorflow_hub as hub
from tqdm import tqdm

from simplests.model_args import SentenceEmbeddingSTSArgs
from simplests.util import batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniversalSentenceEncoderSTSMethod:
    def __init__(self, model_args: SentenceEmbeddingSTSArgs):

        self.model_args = model_args
        logging.info("Loading models ")
        self.model = hub.load(self.model_args.embedding_model)

    def predict(self, to_predict, batch_size=32):
        sims = []

        sentences_1 = list(zip(*to_predict))[0]
        sentences_2 = list(zip(*to_predict))[1]

        embeddings_1 = []
        embeddings_2 = []

        for x in tqdm(batch(sentences_1, batch_size), total=int(len(sentences_1) / batch_size) + (
                len(sentences_1) % batch_size > 0), desc="Embedding list 1 "):
            temp = self.model(x)
            for embedding in temp:
                embeddings_1.append(embedding.numpy())

        for x in tqdm(batch(sentences_2, batch_size), total=int(len(sentences_2) / batch_size) + (
                len(sentences_2) % batch_size > 0), desc="Embedding list 2 "):
            temp = self.model(x)
            for embedding in temp:
                embeddings_2.append(embedding.numpy())

        for embedding_1, embedding_2 in tqdm(zip(embeddings_1, embeddings_2), total=len(embeddings_1), desc="Calculating similarity "):
            sim = np.inner(embedding_1, embedding_2)
            sims.append(sim)

        return sims
