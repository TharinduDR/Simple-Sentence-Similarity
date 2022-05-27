import pandas as pd

from simple_sts.algo.word_avg import WordEmbeddingAverageSTSMethod
from simple_sts.model_args import WordEmbeddingSTSArgs

sick_test = pd.read_csv("examples/english/sick/data/SICK_test_annotated.txt", sep="\t")

to_predit = []

sick_test = sick_test.reset_index()  # make sure indexes pair with number of rows
for index, row in sick_test.iterrows():
    to_predit.append([row['sentence_A'], row['sentence_B']])

model_args = WordEmbeddingSTSArgs()
model_args.embedding_models = {"transformer": "bert-base-uncased",
                               "word": "glove"}

model = WordEmbeddingAverageSTSMethod(model_args=model_args)

sims = model.predict(to_predit)
print(len(to_predit))
print(len(sims))
