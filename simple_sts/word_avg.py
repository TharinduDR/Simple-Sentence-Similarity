from flair.embeddings import StackedEmbeddings
import numpy as np


class WordEmbeddingAverageSTSMethod:
    def __init__(self, model: StackedEmbeddings):
        self.embedding_model = model

    def predict(self, to_predict, batch_size=32):
        """
          Performs predictions on a list of text.

          Args:
              to_predict: A python list of text (str) to be sent to the model for prediction.

          Returns:
              preds: A python list of the predictions (0 or 1) for each text.
          """
        sentences_1 = list(zip(*to_predict))[0]
        sentences_2 = list(zip(*to_predict))[1]

        self.embedding_model.encode(sentences_1)
        self.embedding_model.encode(sentences_2)

        embeddings_1 = []
        embeddings_2 = []
        for sentence_1 in sentences_1:
            for token in
