import logging
import tensorflow_hub as hub
import numpy as np
import tensorflow_text

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


        return sims
