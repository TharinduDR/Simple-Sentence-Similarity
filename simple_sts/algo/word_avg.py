import logging

import numpy as np
from flair.data import Sentence
from flair.embeddings import StackedEmbeddings, WordEmbeddings, CharacterEmbeddings, TransformerWordEmbeddings, \
    ELMoEmbeddings
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm

from simple_sts.model_args import WordEmbeddingSTSArgs
from simple_sts.util import batch

logger = logging.getLogger(__name__)


class WordEmbeddingAverageSTSMethod:
    def __init__(self, model_args: WordEmbeddingSTSArgs):

        self.model_args = model_args

        logging.info("Loading models ")

        embedding_models = []
        for model_type, model_name in self.model_args.embedding_models.items():
            if model_type == "word":
                embedding_models.append(WordEmbeddings(model_name))
            elif model_type == "char":
                embedding_models.append(CharacterEmbeddings(model_name))
            elif model_type == "elmo":
                embedding_models.append(ELMoEmbeddings(model_name))
            elif model_type == "transformer":
                embedding_models.append(TransformerWordEmbeddings(model_name))

        if len(embedding_models) > 1:
            self.embedding_model = StackedEmbeddings(embedding_models)
        elif len(embedding_models) == 1:
            self.embedding_model = embedding_models[0]
        else:
            raise ValueError(
                "Please specify at least one embedding model"
            )

    def predict(self, to_predict, batch_size=32):

        sims = []

        sentences_1 = list(zip(*to_predict))[0]
        sentences_2 = list(zip(*to_predict))[1]

        processed_sentences_1 = []
        processed_sentences_2 = []

        for sentence_1, sentence_2 in zip(sentences_1, sentences_2):
            processed_sentences_1.append(Sentence(sentence_1))
            processed_sentences_2.append(Sentence(sentence_2))

        for x in tqdm(batch(processed_sentences_1 + processed_sentences_2, batch_size),
                      total=int(len(processed_sentences_1 + processed_sentences_2) / batch_size) + (
                              len(processed_sentences_1 + processed_sentences_2) % batch_size > 0), desc="Embedding sentences "):
            self.embedding_model.embed(x)

        for embed_sentence_1, embed_sentence_2 in zip(processed_sentences_1, processed_sentences_2):
            embedding1 = np.average([np.array(token1.embedding.data.tolist()) for token1 in embed_sentence_1], axis=0)
            embedding2 = np.average([np.array(token2.embedding.data.tolist()) for token2 in embed_sentence_2], axis=0)
            cos_sim = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
            sims.append(cos_sim)

        return sims
