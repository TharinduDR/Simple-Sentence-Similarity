from __future__ import division  # py3 "true division"

import logging
import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# If pyemd C extension is available, import it.
# If pyemd is attempted to be used, but isn't installed, ImportError will be raised in wmdistance
from utility.hashing import convert_to_number

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


def run_elmo_wmd_benchmark(sentences1, sentences2, model_path, use_stoplist=False):
    sims = []
    for (sent1, sent2) in zip(sentences1, sentences2):

        tokens1 = sent1.tokens
        tokens2 = sent2.tokens

        file_name1 = str(convert_to_number(sent1.raw))
        file_name2 = str(convert_to_number(sent2.raw))

        if not os.path.isfile(os.path.join(model_path, '1', file_name1 + '.npy')):

            init = tf.initialize_all_variables()
            sess = tf.Session()
            sess.run(init)
            tf.logging.set_verbosity(tf.logging.ERROR)

            embeddings1 = \
                elmo(inputs={"tokens": [tokens1], "sequence_len": [len(tokens1)]}, signature="tokens", as_dict=True)[
                    "elmo"]
            raw_embedding1 = sess.run(embeddings1[0]).tolist()

            sess.close()

        else:
            raw_embedding1 = np.load(os.path.join(model_path, '1', file_name1 + '.npy')).tolist()

        formatted_embedding1 = []

        for embedding in raw_embedding1:
            formatted_embedding1.append(np.asarray(embedding, dtype=np.float32))

        if not os.path.isfile(os.path.join(model_path, '2', file_name2 + '.npy')):

            init = tf.initialize_all_variables()
            sess = tf.Session()
            sess.run(init)
            tf.logging.set_verbosity(tf.logging.ERROR)

            embeddings2 = \
                elmo(inputs={"tokens": [tokens2], "sequence_len": [len(tokens2)]}, signature="tokens", as_dict=True)[
                    "elmo"]
            raw_embedding2 = sess.run(embeddings2[0]).tolist()

            sess.close()

        else:
            raw_embedding2 = np.load(os.path.join(model_path, '2', file_name2 + '.npy')).tolist()

        formatted_embedding2 = []

        for embedding in raw_embedding2:
            formatted_embedding2.append(np.asarray(embedding, dtype=np.float32))

        related_tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
        related_tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens

        related_embeddings1 = []
        related_embeddings2 = []

        if len(related_tokens1) != len(tokens1):
            for related_token1 in related_tokens1:
                index = tokens1.index(related_token1)
                related_embeddings1.append(raw_embedding1[index])

        else:
            related_embeddings1 = raw_embedding1

        if len(related_tokens2) != len(tokens2):
            for related_token2 in related_tokens2:
                index = tokens2.index(related_token2)
                related_embeddings2.append(raw_embedding2[index])

        else:
            related_embeddings2 = raw_embedding2

        sims.append(-wmdistance(related_tokens1, related_tokens2, related_embeddings1, related_embeddings2))

    return sims


def wmdistance(document1, document2, embeddings1, embeddings2):
    if not PYEMD_EXT:
        raise ImportError("Please install pyemd Python package to compute WMD.")

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
            indext1 = document1.index(t1)
            indext2 = document2.index(t2)
            embeddingt1 = np.asarray(embeddings1[indext1], np.float32)
            embeddingt2 = np.asarray(embeddings2[indext2], np.float32)

            distance_matrix[i, j] = sqrt(np_sum((embeddingt1 - embeddingt2) ** 2))

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
