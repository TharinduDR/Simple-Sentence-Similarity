from __future__ import division  # py3 "true division"

import logging

import numpy as np
import tensorflow_hub as hub
# If pyemd C extension is available, import it.
# If pyemd is attempted to be used, but isn't installed, ImportError will be raised in wmdistance
from flair.data import Sentence

try:
    from pyemd import emd

    PYEMD_EXT = True
except ImportError:
    PYEMD_EXT = False

from numpy import double, zeros, sqrt, sum as np_sum
from gensim.corpora.dictionary import Dictionary
from six.moves import zip

logger = logging.getLogger(__name__)
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

embeddings_map1 = {}
embeddings_map2 = {}


def run_context_wmd_benchmark(sentences1, sentences2, model, use_stoplist=False):
    sims = []
    for (sent1, sent2) in zip(sentences1, sentences2):

        tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
        tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens

        tokens1 = [token for token in tokens1]
        tokens2 = [token for token in tokens2]

        if len(tokens1) == 0 or len(tokens2) == 0:
            tokens1 = [token for token in sent1.tokens if token in model]
            tokens2 = [token for token in sent2.tokens if token in model]

        flair_tokens1 = sent1.tokens
        flair_tokens2 = sent2.tokens

        flair_sent1 = Sentence(" ".join(flair_tokens1))
        flair_sent2 = Sentence(" ".join(flair_tokens2))

        model.embed(flair_sent1)
        model.embed(flair_sent2)

        for token in flair_sent1:
            embeddings_map1[token.text] = np.array(token.embedding.data.tolist())

        for token in flair_sent2:
            embeddings_map2[token.text] = np.array(token.embedding.data.tolist())

        sims.append(-wmdistance(tokens1, tokens2))

    return sims


def wmdistance(document1, document2):
    """
    Compute the Word Mover's Distance between two documents. When using this
    code, please consider citing the following papers:

    .. Ofir Pele and Michael Werman, "A linear time histogram metric for improved SIFT matching".
    .. Ofir Pele and Michael Werman, "Fast and robust earth mover's distances".
    .. Matt Kusner et al. "From Word Embeddings To Document Distances".

    Note that if one of the documents have no words that exist in the
    Word2Vec vocab, `float('inf')` (i.e. infinity) will be returned.

    This method only works if `pyemd` is installed (can be installed via pip, but requires a C compiler).
    """

    if not PYEMD_EXT:
        raise ImportError("Please install pyemd Python package to compute WMD.")

    # Remove out-of-vocabulary words.
    len_pre_oov1 = len(document1)
    len_pre_oov2 = len(document2)
    document1 = [token for token in document1]
    document2 = [token for token in document2]
    diff1 = len_pre_oov1 - len(document1)
    diff2 = len_pre_oov2 - len(document2)
    if diff1 > 0 or diff2 > 0:
        logger.info('Removed %d and %d OOV words from document 1 and 2 (respectively).', diff1, diff2)

    if len(document1) == 0 or len(document2) == 0:
        logger.info(
            "At least one of the documents had no words that werein the vocabulary. "
            "Aborting (returning inf)."
        )
        return float('inf')

    dictionary = Dictionary(documents=[document1, document2])
    vocab_len = len(dictionary)

    if vocab_len == 1:
        # Both documents are composed by a single unique token
        return 0.0

    # Sets for faster look-up.
    docset1 = set(document1)
    docset2 = set(document2)

    # Compute distance matrix.
    distance_matrix = zeros((vocab_len, vocab_len), dtype=double)
    for i, t1 in dictionary.items():
        for j, t2 in dictionary.items():
            if t1 not in docset1 or t2 not in docset2:
                continue
            # Compute Euclidean distance between word vectors.
            distance_matrix[i, j] = sqrt(np_sum((embeddings_map1[t1] - embeddings_map2[t2]) ** 2))

    if np_sum(distance_matrix) == 0.0:
        # `emd` gets stuck if the distance matrix contains only zeros.
        logger.info('The distance matrix is all zeros. Aborting (returning inf).')
        return float('inf')

    def nbow(document):
        d = zeros(vocab_len, dtype=double)
        nbow = dictionary.doc2bow(document)  # Word frequencies.
        doc_len = len(document)
        for idx, freq in nbow:
            d[idx] = freq / float(doc_len)  # Normalized word frequencies.
        return d

    # Compute nBOW representation of documents.
    d1 = nbow(document1)
    d2 = nbow(document2)

    # Compute WMD.
    return emd(d1, d2, distance_matrix)
