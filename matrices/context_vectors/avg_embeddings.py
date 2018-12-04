import math
from collections import Counter

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

from utility.progress_bar import update_progress_bar

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)


def run_avg_elmo_benchmark(sentences1, sentences2, doc_freqs=None, use_stoplist=False):
    sims = []

    max_length = max(len(sentences1), len(sentences2))

    update_progress_bar(0, max_length, prefix='Progress:', suffix='Complete', length=50)
    counter = 0
    for (sent1, sent2) in zip(sentences1, sentences2):
        sims.append(get_similarity(sent1, sent2, doc_freqs, use_stoplist))
        update_progress_bar(counter, max_length, prefix='Progress:', suffix='Complete', length=100)
        counter = counter + 1

    return sims


def get_similarity(sent1, sent2, doc_freqs=None, use_stoplist=False):
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    tf.logging.set_verbosity(tf.logging.ERROR)

    if doc_freqs is not None:
        N = doc_freqs["NUM_DOCS"]

    tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
    tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens

    tokfreqs1 = Counter(tokens1)
    tokfreqs2 = Counter(tokens2)

    weights1 = [tokfreqs1[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
                for token in tokens1] if doc_freqs else None

    weights2 = [tokfreqs2[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
                for token in tokens2] if doc_freqs else None

    embeddings1 = elmo(inputs={"tokens": [tokens1], "sequence_len": [len(tokens1)]}, signature="tokens", as_dict=True)[
        "elmo"]

    embeddings2 = elmo(inputs={"tokens": [tokens2], "sequence_len": [len(tokens2)]}, signature="tokens", as_dict=True)[
        "elmo"]

    embedding1 = np.average([embedding for embedding in sess.run(embeddings1[0])], axis=0,
                            weights=weights1).reshape(1, -1)

    embedding2 = np.average([embedding for embedding in sess.run(embeddings2[0])], axis=0,
                            weights=weights2).reshape(1, -1)

    sim = cosine_similarity(embedding1, embedding2)[0][0]

    sess.close()

    return sim
