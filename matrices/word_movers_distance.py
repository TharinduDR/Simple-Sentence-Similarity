def run_wmd_benchmark(sentences1, sentences2, model, use_stoplist=False):
    sims = []
    for (sent1, sent2) in zip(sentences1, sentences2):

        tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
        tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens

        tokens1 = [token for token in tokens1 if token in model]
        tokens2 = [token for token in tokens2 if token in model]

        if len(tokens1) == 0 or len(tokens2) == 0:
            tokens1 = [token for token in sent1.tokens if token in model]
            tokens2 = [token for token in sent2.tokens if token in model]

        sims.append(-model.wmdistance(tokens1, tokens2))

    return sims
