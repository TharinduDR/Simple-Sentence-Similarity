import logging

import numpy as np
from flair.data import Sentence
from flair.embeddings import StackedEmbeddings, WordEmbeddings, CharacterEmbeddings, TransformerWordEmbeddings, \
    ELMoEmbeddings
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
import math

from simplests.model_args import WordEmbeddingSTSArgs
from simplests.util import batch
from stop_words import get_stop_words
from gensim.corpora.dictionary import Dictionary

try:
    from pyemd import emd

    PYEMD_EXT = True
except ImportError:
    PYEMD_EXT = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gensim_logger = logging.getLogger('gensim')
gensim_logger.setLevel(logging.WARN)


class WordMoversDistanceSTSMethod:
    def __init__(self, model_args: WordEmbeddingSTSArgs):

        if not PYEMD_EXT:
            raise ImportError("Please install pyemd Python package to compute WMD.")

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

            sim = self.wmdistance(embed_sentence_1, embed_sentence_2)
            sims.append(sim)

        return sims

    def wmdistance(self, sentence1, sentence2):
        """
        Compute the Word Mover's Distance between two documents. When using this
        code, please consider citing the following papers:
        .. Ofir Pele and Michael Werman, "A linear time histogram metric for improved SIFT matching".
        .. Ofir Pele and Michael Werman, "Fast and robust earth mover's distances".
        .. Matt Kusner et al. "From Word Embeddings To Document Distances".
        Note that if one of the documents have no words that exist in the vocab, `float('inf')` (i.e. infinity) will be returned.
        This method only works if `pyemd` is installed (can be installed via pip, but requires a C compiler).
        """

        # Remove out-of-vocabulary words.

        if not self.model_args.remove_stopwords:
            document1 = [token.text for token in sentence1]
            document2 = [token.text for token in sentence2]

        else:
            document1 = [token.text for token in sentence1  if token.text not in self.stop_words]
            document2 = [token.text for token in sentence2  if token.text not in self.stop_words]

        dictionary = Dictionary(documents=[document1, document2])

        len_pre_oov1 = len(document1)
        len_pre_oov2 = len(document2)

        diff1 = len_pre_oov1 - len(document1)
        diff2 = len_pre_oov2 - len(document2)
        if diff1 > 0 or diff2 > 0:
            logger.info('Removed %d and %d OOV words from document 1 and 2 (respectively).', diff1, diff2)

        if len(document1) == 0 or len(document2) == 0:
            logger.warning(
                "At least one of the documents had no words that were in the vocabulary."
                "Aborting (returning inf)."
            )
            return float('inf')

        full_list = document1 + document2
        vocab_len = len(full_list)

        if len(set(full_list)) == 1:
            # Both documents are composed by a single unique token
            return 0.0

        # # Compute distance matrix.
        distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)

        for i, token1 in enumerate(document1):
            for j, token2 in enumerate(document2):
                # print("first sentence", sentence_1[i].text)
                # print("second sentence", sentence_2[j].text)
                distance_matrix[i, (len(document1) + j)] = np.sqrt(np.sum((np.array(
                    sentence1[i].embedding.data.tolist()) - np.array(sentence2[j].embedding.data.tolist())) ** 2))

        if np.sum(distance_matrix) == 0.0:
            logger.warning('The distance matrix is all zeros. Aborting (returning inf).')
            return float('inf')

        def nbow(document):
            d = np.zeros(vocab_len, dtype=np.double)
            nbow = dictionary.doc2bow(document)  # Word frequencies.
            doc_len = len(document)
            for idx, freq in nbow:
                d[idx] = freq / float(doc_len)  # Normalized word frequencies.
            return d

        # Compute nBOW representation of documents.
        d1 = nbow(document1)
        d2 = nbow(document2)

        # Compute WMD.
        distance = emd(d1, d2, distance_matrix)
        return 1 / (math.exp(distance))
