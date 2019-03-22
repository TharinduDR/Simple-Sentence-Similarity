import numpy as np
from flair.data import Sentence
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def remove_first_principal_component(X):
    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    svd.fit(X)
    pc = svd.components_
    XX = X - X.dot(pc.transpose()) * pc
    return XX


def run_context_sif_benchmark(sentences1, sentences2, model, freqs={}, use_stoplist=False, a=0.001):
    total_freq = sum(freqs.values())

    embeddings = []

    # SIF requires us to first collect all sentence embeddings and then perform
    # common component analysis.
    for (sent1, sent2) in zip(sentences1, sentences2):
        tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
        tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens

        flair_tokens1 = sent1.tokens
        flair_tokens2 = sent2.tokens

        flair_sent1 = Sentence(" ".join(flair_tokens1))
        flair_sent2 = Sentence(" ".join(flair_tokens2))

        model.embed(flair_sent1)
        model.embed(flair_sent2)

        embeddings_map1 = {}
        embeddings_map2 = {}

        for token in flair_sent1:
            embeddings_map1[token.text] = np.array(token.embedding.data.tolist())

        for token in flair_sent2:
            embeddings_map2[token.text] = np.array(token.embedding.data.tolist())

        tokens1 = [token for token in tokens1]
        tokens2 = [token for token in tokens2]

        weights1 = [a / (a + freqs.get(token, 0) / total_freq) for token in tokens1]
        weights2 = [a / (a + freqs.get(token, 0) / total_freq) for token in tokens2]

        embedding1 = np.average([embeddings_map1[token] for token in tokens1], axis=0, weights=weights1)
        embedding2 = np.average([embeddings_map2[token] for token in tokens2], axis=0, weights=weights2)

        embeddings.append(embedding1)
        embeddings.append(embedding2)

    embeddings = remove_first_principal_component(np.array(embeddings))
    sims = [cosine_similarity(embeddings[idx * 2].reshape(1, -1),
                              embeddings[idx * 2 + 1].reshape(1, -1))[0][0]
            for idx in range(int(len(embeddings) / 2))]

    return sims
