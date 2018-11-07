from models.sentence import Sentence


def run_experiment(df, benchmark):
    sentences1 = [Sentence(s) for s in df['sent_1']]
    sentences2 = [Sentence(s) for s in df['sent_2']]

    sims = benchmark[1](sentences1, sentences2)

    return sims, benchmark[0]
