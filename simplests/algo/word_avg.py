import logging

import numpy as np
from flair.data import Sentence
from flair.embeddings import StackedEmbeddings, WordEmbeddings, CharacterEmbeddings, TransformerWordEmbeddings, \
    ELMoEmbeddings
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm

from simplests.model_args import WordEmbeddingSTSArgs
from simplests.util import batch
from stop_words import get_stop_words

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WordEmbeddingAverageSTSMethod:
    def __init__(self, model_args: WordEmbeddingSTSArgs):

        self.model_args = model_args
        logging.info("Loading models ")

        embedding_models = []
        for model_type, model_name in self.model_args.embedding_models:
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
        if model_args.remove_stopwords:
            try:
                self.stop_words = get_stop_words(self.model_args.language)
            except KeyError as e:
                logging.warning("Stop words are not supported for {}. Please refer https://github.com/Alir3z4/python-stop-words to see supported languages.".format(model_args.language))
                logging.warning("Setting model_args.remove_stopwords to False")
                self.model_args.remove_stopwords = False

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

        for embed_sentence_1, embed_sentence_2 in tqdm(zip(processed_sentences_1, processed_sentences_2), total=len(processed_sentences_1), desc="Calculating similarity "):

            if self.model_args.remove_stopwords:
                embedding1 = np.average([np.array(token1.embedding.data.tolist()) for token1 in embed_sentence_1 if token1.text not in self.stop_words], axis=0)
                embedding2 = np.average([np.array(token2.embedding.data.tolist()) for token2 in embed_sentence_2 if token2.text not in self.stop_words], axis=0)

            else:
                embedding1 = np.average([np.array(token1.embedding.data.tolist()) for token1 in embed_sentence_1], axis=0)
                embedding2 = np.average([np.array(token2.embedding.data.tolist()) for token2 in embed_sentence_2], axis=0)

            cos_sim = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
            sims.append(cos_sim)

        return sims
