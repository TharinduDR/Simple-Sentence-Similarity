import logging

from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm

from simplests.model_args import SentenceEmbeddingSTSArgs
from simplests.util import batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerCLSSTSMethod:
    def __init__(self, model_args: SentenceEmbeddingSTSArgs):
        self.model_args = model_args
        logging.info("Loading models ")
        self.embedding_model = TransformerDocumentEmbeddings(model_args.embedding_model)

    def predict(self, to_predict, batch_size=32):
        sims = []

        sentences_1 = list(zip(*to_predict))[0]
        sentences_2 = list(zip(*to_predict))[1]

        processed_sentences_1 = []
        processed_sentences_2 = []

        for sentence_1, sentence_2 in zip(sentences_1, sentences_2):
            processed_sentences_1.append(Sentence(sentence_1))
            processed_sentences_2.append(Sentence(sentence_2))

        for x in tqdm(batch(processed_sentences_1 + processed_sentences_2, batch_size),
                      total=int(len(processed_sentences_1 + processed_sentences_2) / batch_size) + (
                              len(processed_sentences_1 + processed_sentences_2) % batch_size > 0),
                      desc="Embedding sentences "):
            self.embedding_model.embed(x)

        for embed_sentence_1, embed_sentence_2 in tqdm(zip(processed_sentences_1, processed_sentences_2),
                                                       total=len(processed_sentences_1),
                                                       desc="Calculating similarity "):
            embedding1 = embed_sentence_1.embedding.data.tolist()
            embedding2 = embed_sentence_2.embedding.data.tolist()

            cos_sim = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
            sims.append(cos_sim)

        return sims
