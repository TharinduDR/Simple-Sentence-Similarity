import logging

import numpy as np
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf
from tqdm import tqdm

from simplests.model_args import SentenceEmbeddingSTSArgs
from simplests.util import batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalization(embeds):
    norms = np.linalg.norm(embeds, 2, axis=1, keepdims=True)
    return embeds/norms


class LaBSESTSMethod:
    def __init__(self, model_args: SentenceEmbeddingSTSArgs):

        self.model_args = model_args
        logging.info("Loading models ")
        self.preprocessor = hub.KerasLayer(
            "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
        self.encoder = hub.KerasLayer(model_args.embedding_model)

    def predict(self, to_predict, batch_size=32):
        sims = []

        sentences_1 = list(zip(*to_predict))[0]
        sentences_2 = list(zip(*to_predict))[1]

        embeddings_1 = []
        embeddings_2 = []

        for x in tqdm(batch(sentences_1, batch_size), total=int(len(sentences_1) / batch_size) + (
                len(sentences_1) % batch_size > 0), desc="Embedding list 1 "):
            temp_sentences = tf.constant(x)
            temp_embeds = self.encoder(self.preprocessor(temp_sentences))["default"]
            temp_embeds = normalization(temp_embeds)
            for embedding in temp_embeds:
                embeddings_1.append(embedding.numpy())

        for x in tqdm(batch(sentences_2, batch_size), total=int(len(sentences_2) / batch_size) + (
                len(sentences_2) % batch_size > 0), desc="Embedding list 2 "):
            temp_sentences = tf.constant(x)
            temp_embeds = self.encoder(self.preprocessor(temp_sentences))["default"]
            temp_embeds = normalization(temp_embeds)
            for embedding in temp_embeds:
                embeddings_2.append(embedding.numpy())

        for embedding_1, embedding_2 in tqdm(zip(embeddings_1, embeddings_2), total=len(embeddings_1), desc="Calculating similarity "):
            sim = np.inner(embedding_1, embedding_2)
            sims.append(sim)

        return sims
