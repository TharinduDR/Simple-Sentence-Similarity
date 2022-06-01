import pandas as pd

from examples.evaluation import pearson_corr, spearman_corr, rmse
from simple_sts.algo.sif import WordEmbeddingSIFSTSMethod
from simple_sts.algo.word_avg import WordEmbeddingAverageSTSMethod
from simple_sts.model_args import WordEmbeddingSTSArgs

sick_test = pd.read_csv("examples/english/sick/data/SICK_test_annotated.txt", sep="\t")

to_predit = []
sims = []

sick_test = sick_test.reset_index()  # make sure indexes pair with number of rows
for index, row in sick_test.iterrows():
    to_predit.append([row['sentence_A'], row['sentence_B']])
    sims.append(row['relatedness_score'])

model_args = WordEmbeddingSTSArgs()
model_args.embedding_models = [["transformer", "bert-large-cased"],
                               ["transformer", "xlnet-large-cased"]]
model_args.language = "en"
model_args.remove_stopwords = True

# model = WordEmbeddingAverageSTSMethod(model_args=model_args)
model = WordEmbeddingSIFSTSMethod(model_args=model_args)

pred_sims = model.predict(to_predit)
print("Pearson correlation ", pearson_corr(sims, pred_sims))
print("Spearman correlation ", spearman_corr(sims, pred_sims))
print("RMSE ", rmse(sims, pred_sims))
