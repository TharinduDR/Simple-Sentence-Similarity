from flair.embeddings import StackedEmbeddings, WordEmbeddings, CharacterEmbeddings, TransformerWordEmbeddings, ELMoEmbeddings
from flair.data import Sentence
import logging
import numpy as np

from simple_sts.model_args import WordEmbeddingSTSArgs

logger = logging.getLogger(__name__)


class WordEmbeddingAverageSTSMethod:
    def __init__(self, model_args: WordEmbeddingSTSArgs):

        self.model_args = model_args

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

        sentences_1 = list(zip(*to_predict))[0]
        sentences_2 = list(zip(*to_predict))[1]

        processed_sentences_1 = []
        processed_sentences_2 = []

        for sentence_1, sentence_2 in zip(sentences_1, sentences_2):
            processed_sentences_1.append(Sentence(sentence_1))
            processed_sentences_2.append(Sentence(sentence_2))

        self.embedding_model.encode(sentences_1)
        self.embedding_model.encode(sentences_2)

        embeddings_1 = []
        embeddings_2 = []

