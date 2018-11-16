def percent_match_benchmark(sentences1, sentences2, use_stoplist=False):
    sims = []
    for (sent1, sent2) in zip(sentences1, sentences2):
        tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
        tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens

        sims.append(len(set(tokens1).intersection(set(tokens2))) / max(len(tokens1), len(tokens2)))

    return sims
