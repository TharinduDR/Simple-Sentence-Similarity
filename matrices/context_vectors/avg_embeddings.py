import math
import os
from collections import Counter

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

from utility.hashing import convert_to_number

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)


def run_avg_elmo_benchmark(sentences1, sentences2, model_path, doc_freqs=None, use_stoplist=False, ):
    sims = []

    counter = 0
    for (sent1, sent2) in zip(sentences1, sentences2):
        sims.append(get_similarity(sent1, sent2, model_path, doc_freqs, use_stoplist))
        counter = counter + 1

    return sims


def get_similarity(sent1, sent2, model_path, doc_freqs=None, use_stoplist=False):
    if doc_freqs is not None:
        N = doc_freqs["NUM_DOCS"]

    tokens1 = sent1.tokens
    tokens2 = sent2.tokens

    tokfreqs1 = Counter(tokens1)
    tokfreqs2 = Counter(tokens2)

    weights1 = [tokfreqs1[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
                for token in tokens1] if doc_freqs else None

    weights2 = [tokfreqs2[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
                for token in tokens2] if doc_freqs else None

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
        embedding1 = sess.run(embeddings1[0])

        sess.close()

    else:
        embedding1 = np.load(os.path.join(model_path, '1', file_name1 + '.npy'))

    if not os.path.isfile(os.path.join(model_path, '2', file_name2 + '.npy')):

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        tf.logging.set_verbosity(tf.logging.ERROR)

        embeddings2 = \
            elmo(inputs={"tokens": [tokens2], "sequence_len": [len(tokens2)]}, signature="tokens", as_dict=True)[
                "elmo"]
        embedding2 = sess.run(embeddings2[0])

        sess.close()

    else:
        embedding2 = np.load(os.path.join(model_path, '2', file_name2 + '.npy'))

    related_tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
    related_tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens

    related_embeddings1 = []
    related_embeddings2 = []

    related_weights1 = []
    related_weights2 = []

    if len(related_tokens1) != len(tokens1):
        for related_token1 in related_tokens1:
            index = tokens1.index(related_token1)
            related_embeddings1.append(embedding1[index])
            if doc_freqs is not None:
                related_weights1.append((weights1[index]))
            else:
                related_weights1 = None

        embedding1 = np.average([embedding for embedding in related_embeddings1], axis=0,
                                weights=related_weights1).reshape(1, -1)
    else:
        embedding1 = np.average([embedding for embedding in embedding1], axis=0,
                                weights=weights1).reshape(1, -1)

    if len(related_tokens2) != len(tokens2):
        for related_token2 in related_tokens2:
            index = tokens2.index(related_token2)
            related_embeddings2.append(embedding2[index])

            if doc_freqs is not None:
                related_weights2.append(weights2[index])

            else:
                related_weights2 = None

        embedding2 = np.average([embedding for embedding in related_embeddings2], axis=0,
                                weights=related_weights2).reshape(1, -1)

    else:
        embedding2 = np.average([embedding for embedding in embedding2], axis=0,
                                weights=weights2).reshape(1, -1)

    sim = cosine_similarity(embedding1, embedding2)[0][0]

    return sim
