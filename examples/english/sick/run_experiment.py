import pandas as pd

from examples.evaluation import pearson_corr, spearman_corr, rmse
from simple_sts.algo.cls import TransformerCLSSTSMethod
from simple_sts.algo.labse import LaBSESTSMethod
from simple_sts.algo.laser import LASERSTSMethod
from simple_sts.algo.sbert import SentenceTransformerSTSMethod
from simple_sts.algo.sif import WordEmbeddingSIFSTSMethod
from simple_sts.algo.use import UniversalSentenceEncoderSTSMethod
from simple_sts.algo.wmd import WordMoversDistanceSTSMethod
from simple_sts.algo.word_avg import WordEmbeddingAverageSTSMethod
from simple_sts.model_args import WordEmbeddingSTSArgs, SentenceEmbeddingSTSArgs

sick_test = pd.read_csv("examples/english/sick/data/SICK_test_annotated.txt", sep="\t")

to_predit = []
sims = []

sick_test = sick_test.reset_index()  # make sure indexes pair with number of rows
for index, row in sick_test.iterrows():
    to_predit.append([row['sentence_A'], row['sentence_B']])
    sims.append(row['relatedness_score'])

model_args = WordEmbeddingSTSArgs()
model_args.embedding_models = [["transformer", "bert-base-multilingual-cased"]]
model_args.language = "en"
model_args.remove_stopwords = True

# model = WordEmbeddingAverageSTSMethod(model_args=model_args)
model = WordEmbeddingSIFSTSMethod(model_args=model_args)

pred_sims = model.predict(to_predit)
print("Pearson correlation ", pearson_corr(sims, pred_sims))
print("Spearman correlation ", spearman_corr(sims, pred_sims))
print("RMSE ", rmse(sims, pred_sims))

# -------------------------------------------------------------------------
model = WordMoversDistanceSTSMethod(model_args=model_args)

pred_sims = model.predict(to_predit)
print("Pearson correlation ", pearson_corr(sims, pred_sims))
print("Spearman correlation ", spearman_corr(sims, pred_sims))
print("RMSE ", rmse(sims, pred_sims))

#
# # -----------------------------------------------------------------------
#
# sentence_model_args = SentenceEmbeddingSTSArgs()
# sentence_model_args.embedding_model = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"
# sentence_model_args.language = "en"
#
#
# # model = WordEmbeddingAverageSTSMethod(model_args=model_args)
# model = UniversalSentenceEncoderSTSMethod(model_args=sentence_model_args)
#
# pred_sims = model.predict(to_predit)
# print("Pearson correlation ", pearson_corr(sims, pred_sims))
# print("Spearman correlation ", spearman_corr(sims, pred_sims))
# print("RMSE ", rmse(sims, pred_sims))
#
# # -----------------------------------------------------------------------
# # labse_model_args = SentenceEmbeddingSTSArgs()
# # labse_model_args.embedding_model = "https://tfhub.dev/google/LaBSE/2"
# # labse_model_args.language = "en"
# #
# #
# # model = WordEmbeddingAverageSTSMethod(model_args=model_args)
# # model = LaBSESTSMethod(model_args=labse_model_args)
# #
# # pred_sims = model.predict(to_predit)
# # print("Pearson correlation ", pearson_corr(sims, pred_sims))
# # print("Spearman correlation ", spearman_corr(sims, pred_sims))
# # print("RMSE ", rmse(sims, pred_sims))
#
# # -----------------------------------------------------------------------
# sbert_model_args = SentenceEmbeddingSTSArgs()
# sbert_model_args.embedding_model = "distiluse-base-multilingual-cased"
# sbert_model_args.language = "en"
#
#
# # model = WordEmbeddingAverageSTSMethod(model_args=model_args)
# model = SentenceTransformerSTSMethod(model_args=sbert_model_args)
#
# pred_sims = model.predict(to_predit)
# print("Pearson correlation ", pearson_corr(sims, pred_sims))
# print("Spearman correlation ", spearman_corr(sims, pred_sims))
# print("RMSE ", rmse(sims, pred_sims))
#
# # -----------------------------------------------------------------------
# cls_model_args = SentenceEmbeddingSTSArgs()
# cls_model_args.embedding_model = "bert-base-multilingual-cased"
# cls_model_args.language = "en"
#
#
# # model = WordEmbeddingAverageSTSMethod(model_args=model_args)
# model = TransformerCLSSTSMethod(model_args=cls_model_args)
#
# pred_sims = model.predict(to_predit)
# print("Pearson correlation ", pearson_corr(sims, pred_sims))
# print("Spearman correlation ", spearman_corr(sims, pred_sims))
# print("RMSE ", rmse(sims, pred_sims))

# -----------------------------------------------------------------------
laser_model_args = SentenceEmbeddingSTSArgs()
laser_model_args.language = "en"


# model = WordEmbeddingAverageSTSMethod(model_args=model_args)
model = LASERSTSMethod(model_args=laser_model_args)

pred_sims = model.predict(to_predit)
print("Pearson correlation ", pearson_corr(sims, pred_sims))
print("Spearman correlation ", spearman_corr(sims, pred_sims))
print("RMSE ", rmse(sims, pred_sims))


