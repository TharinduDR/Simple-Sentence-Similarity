import logging
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
from simple_sts.util import batch

from simple_sts.model_args import SentenceEmbeddingSTSArgs

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

        for x in batch(sentences_1, batch_size):
            temp = self.model(x)
            for embedding in temp:
                embeddings_1.append(embedding.numpy())

        for x in batch(sentences_2, batch_size):
            temp = self.model(x)
            for embedding in temp:
                embeddings_2.append(embedding.numpy())

        for embedding_1, embedding_2 in zip(embeddings_1, embeddings_2):
            cos_sim = np.inner(embedding_1, embedding_2)
            sims.append(cos_sim)

        return sims
