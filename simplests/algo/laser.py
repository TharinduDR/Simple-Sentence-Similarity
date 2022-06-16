import logging
import os
import sys
import urllib.request
from laserembeddings import Laser
from tqdm import tqdm
from numpy.linalg import norm
import numpy as np
from simplests.util import batch

from simplests.model_args import SentenceEmbeddingSTSArgs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url, dest):
    sys.stdout.flush()
    urllib.request.urlretrieve(url, dest)


def download_models(output_dir):
    logger.info('Downloading models into {}'.format(output_dir))

    download_file('https://dl.fbaipublicfiles.com/laser/models/93langs.fcodes',
                  os.path.join(output_dir, '93langs.fcodes'))
    download_file('https://dl.fbaipublicfiles.com/laser/models/93langs.fvocab',
                  os.path.join(output_dir, '93langs.fvocab'))
    download_file(
        'https://dl.fbaipublicfiles.com/laser/models/bilstm.93langs.2018-12-26.pt',
        os.path.join(output_dir, 'bilstm.93langs.2018-12-26.pt'))


class LASERSTSMethod:
    def __init__(self, model_args: SentenceEmbeddingSTSArgs):
        self.model_args = model_args
        logging.info("Loading models ")

        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            download_models(output_dir)

        DEFAULT_BPE_CODES_FILE = os.path.join(output_dir, '93langs.fcodes')
        DEFAULT_BPE_VOCAB_FILE = os.path.join(output_dir, '93langs.fvocab')
        DEFAULT_ENCODER_FILE = os.path.join(output_dir,
                                            'bilstm.93langs.2018-12-26.pt')
        self.embedding_model = Laser(DEFAULT_BPE_CODES_FILE, DEFAULT_BPE_VOCAB_FILE, DEFAULT_ENCODER_FILE)

    def predict(self, to_predict, batch_size=32):

        sentences_1 = list(zip(*to_predict))[0]
        sentences_2 = list(zip(*to_predict))[1]

        embeddings_1 = []
        embeddings_2 = []
        sims = []

        for x in tqdm(batch(sentences_1, batch_size), total=int(len(sentences_1) / batch_size) + (
                len(sentences_1) % batch_size > 0), desc="Embedding list 1 "):
            temp = self.embedding_model.embed_sentences(x, lang=self.model_args.language)
            for embedding in temp:
                embeddings_1.append(embedding)

        for x in tqdm(batch(sentences_2, batch_size), total=int(len(sentences_2) / batch_size) + (
                len(sentences_2) % batch_size > 0), desc="Embedding list 2 "):
            temp = self.embedding_model.embed_sentences(x, lang=self.model_args.language)
            for embedding in temp:
                embeddings_2.append(embedding)

        for embedding_1, embedding_2 in tqdm(zip(embeddings_1, embeddings_2), total=len(embeddings_1), desc="Calculating similarity "):
            cos_sim = np.dot(embedding_1, embedding_2) / (
                    norm(embedding_1) * norm(embedding_2))
            sims.append(cos_sim)

        return sims
