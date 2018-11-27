import math
from collections import Counter

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)


def run_avg_elmo_benchmark(sentences1, sentences2, doc_freqs=None, batch_size=100):
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    tf.logging.set_verbosity(tf.logging.ERROR)

    sims = []
    tokens_list1 = []
    tokens_list2 = []
    length_list1 = []
    length_list2 = []
    weights_list2 = []
    weights_list1 = []

    max_length1 = 0
    max_length2 = 0

    for (sent1, sent2) in zip(sentences1, sentences2):
        tokens1 = sent1.tokens
        tokens2 = sent2.tokens
        if len(tokens1) > max_length1:
            max_length1 = len(tokens1)

        if len(tokens2) > max_length2:
            max_length2 = len(tokens2)

    for i in range(len(sentences1 // batch_size)):

        if (batch_size * i) + batch_size < len(sentences1):
            sliced_setences1 = sentences1[batch_size * i:(batch_size * i) + batch_size]
            sliced_setences2 = sentences2[batch_size * i:(batch_size * i) + batch_size]

        else:
            sliced_setences1 = sentences1[batch_size * i:]
            sliced_setences2 = sentences2[batch_size * i:]

        for (sent1, sent2) in zip(sliced_setences1, sliced_setences2):
            if doc_freqs is not None:
                N = doc_freqs["NUM_DOCS"]

            tokens1 = sent1.tokens
            length_list1.append(len(tokens1))

            for i in range(max_length1 - len(tokens1) + 1):
                tokens1.append('')

            tokens2 = sent2.tokens
            length_list2.append(len(tokens2))

            for i in range(max_length2 - len(tokens2) + 1):
                tokens2.append('')

            tokens_list1.append(tokens1)
            tokens_list2.append(tokens2)

            tokfreqs1 = Counter(tokens1)
            tokfreqs2 = Counter(tokens2)

            weights1 = [tokfreqs1[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
                        for token in tokens1] if doc_freqs else None
            weights2 = [tokfreqs2[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
                        for token in tokens2] if doc_freqs else None

            weights_list1.append(weights1)
            weights_list2.append(weights2)

        embeddings_list1 = \
            elmo(inputs={"tokens": tokens_list1, "sequence_len": length_list1}, signature="tokens", as_dict=True)[
                "elmo"]

        embeddings_list2 = \
            elmo(inputs={"tokens": tokens_list2, "sequence_len": length_list2}, signature="tokens", as_dict=True)[
                "elmo"]

        for i in range(len(sliced_setences1)):
            embedding1 = np.average([embedding for embedding in sess.run(embeddings_list1[i])], axis=0,
                                    weights=weights_list1[i]).reshape(1, -1)

            embedding2 = np.average([embedding for embedding in sess.run(embeddings_list2[i])], axis=0,
                                    weights=weights_list2[i]).reshape(1, -1)

            sim = cosine_similarity(embedding1, embedding2)[0][0]
            sims.append(sim)

    return sims
