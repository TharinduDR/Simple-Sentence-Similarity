import scipy
from sklearn.metrics import mean_squared_error
from math import sqrt
from flair.embeddings import ELMoEmbeddings, TransformerWordEmbeddings, StackedEmbeddings, WordEmbeddings, \
    FlairEmbeddings
import functools as ft
import pandas as pd
from embeddings.load_embeddings import load_word2vec
from examples.english.config import w2v_path, IMAGE_PATH
from matrices.context_vectors.avg_embeddings import run_context_avg_benchmark
from matrices.word_vectors.avg_embeddings import run_avg_benchmark
from preprocessing.load_data import download_and_load_sts_data, download_and_load_sick_dataset
from preprocessing.normalize import normalize
from utility.frequency_loader import load_frequencies, load_doc_frequencies
from utility.run_experiment import run_experiment
import os

if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)

sts_dev, sts_test = download_and_load_sts_data()
print('Downloaded data')

frequency = load_frequencies("data/frequencies/frequencies.tsv")
doc_frequency = load_doc_frequencies("data/frequencies/doc_frequencies.tsv")
word2vec = load_word2vec(w2v_path)
elmo = ELMoEmbeddings('large')
bert = TransformerWordEmbeddings('bert-large-cased')
flair = StackedEmbeddings([WordEmbeddings('glove'), FlairEmbeddings('news-forward'),FlairEmbeddings('news-backward')])
elmo_bert = StackedEmbeddings([elmo, bert])

print("Loaded Resources")

benchmarks = [("AVG-W2V", ft.partial(run_avg_benchmark, model=word2vec, use_stoplist=False)),
              ("AVG-ELMO", ft.partial(run_context_avg_benchmark, model=elmo, use_stoplist=False)),
              ("AVG-BERT", ft.partial(run_context_avg_benchmark, model=bert, use_stoplist=False)),
              ("AVG-FLAIR", ft.partial(run_context_avg_benchmark, model=flair, use_stoplist=False)),
              ("AVG-ELMO⊕BERT", ft.partial(run_context_avg_benchmark, model=elmo_bert, use_stoplist=False)),

              ("AVG-W2V-STOP", ft.partial(run_avg_benchmark, model=word2vec, use_stoplist=True)),
              ("AVG-ELMO-STOP", ft.partial(run_context_avg_benchmark, model=elmo, use_stoplist=True)),
              ("AVG-BERT-STOP", ft.partial(run_context_avg_benchmark, model=bert, use_stoplist=True)),
              ("AVG-FLAIR-STOP", ft.partial(run_context_avg_benchmark, model=flair, use_stoplist=True)),
              ("AVG-ELMO⊕BERT-STOP", ft.partial(run_context_avg_benchmark, model=elmo_bert, use_stoplist=True)),

              ("AVG-W2V-TFIDF", ft.partial(run_avg_benchmark, model=word2vec, use_stoplist=False, doc_freqs=doc_frequency)),
              ("AVG-ELMO-TFIDF", ft.partial(run_context_avg_benchmark, model=elmo, use_stoplist=False, doc_freqs=doc_frequency)),
              ("AVG-BERT-TFIDF", ft.partial(run_context_avg_benchmark, model=bert, use_stoplist=False, doc_freqs=doc_frequency)),
              ("AVG-FLAIR-TFIDF", ft.partial(run_context_avg_benchmark, model=flair, use_stoplist=False, doc_freqs=doc_frequency)),
              ("AVG-ELMO⊕BERT-TFIDF", ft.partial(run_context_avg_benchmark, model=elmo_bert, use_stoplist=False, doc_freqs=doc_frequency)),

              ("AVG-W2V-TFIDF-STOP",ft.partial(run_avg_benchmark, model=word2vec, use_stoplist=True, doc_freqs=doc_frequency)),
              ("AVG-ELMO-TFIDF-STOP",ft.partial(run_context_avg_benchmark, model=elmo, use_stoplist=True, doc_freqs=doc_frequency)),
              ("AVG-BERT-TFIDF-STOP",ft.partial(run_context_avg_benchmark, model=bert, use_stoplist=True, doc_freqs=doc_frequency)),
              ("AVG-FLAIR-TFIDF-STOP",ft.partial(run_context_avg_benchmark, model=flair, use_stoplist=True, doc_freqs=doc_frequency)),
              ("AVG-ELMO⊕BERT-TFIDF-STOP",ft.partial(run_context_avg_benchmark, model=elmo_bert, use_stoplist=True, doc_freqs=doc_frequency))]

data = normalize(sts_test, ["sim"])

for benchmark in benchmarks:
    sims, topic = run_experiment(data, benchmark)
    pearson_correlation = scipy.stats.pearsonr(sims, data['sim'])[0]
    spearman_correlation = scipy.stats.spearmanr(sims, data['sim'])[0]
    rmse = sqrt(mean_squared_error(sims, data['sim']))
    textstr = 'RMSE=%.3f\n$Pearson Correlation=%.3f$\n$Spearman Correlation=%.3f$' % (rmse, pearson_correlation, spearman_correlation)
    data['predicted_sim'] = pd.Series(sims).values
    data = data.sort_values('sim')
    id = list(range(0, len(data.index)))
    data['id'] = pd.Series(id).values
    ax = data.plot(kind='scatter', x='id', y='sim', color='DarkBlue', label='Similarity', title=topic)
    data.plot(kind='scatter', x='id', y='predicted_sim', color='DarkGreen', label='Predicted Similarity',
                    ax=ax);
    ax.text(1000, 0, textstr, fontsize=8)
    fig = ax.get_figure()
    fig.savefig(os.path.join(IMAGE_PATH, topic+".png"))
    print(topic)
    print(textstr)