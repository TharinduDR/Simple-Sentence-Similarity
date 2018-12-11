import os
import pathlib

import numpy as np
import tensorflow as tf

from utility.hashing import convert_to_number
from utility.progress_bar import update_progress_bar


def save_vectors(sentences, path, model):
    max_length = len(sentences)
    counter = 0
    update_progress_bar(0, max_length, prefix='Progress:', suffix='Complete', length=50)

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    for sent in sentences:

        file_name = str(convert_to_number(sent.raw))

        if not os.path.isfile(os.path.join(path, file_name + '.npy')):
            init = tf.initialize_all_variables()
            sess = tf.Session()
            sess.run(init)
            tf.logging.set_verbosity(tf.logging.ERROR)

            tokens = sent.tokens
            tensor_embedding = \
                model(inputs={"tokens": [tokens], "sequence_len": [len(tokens)]}, signature="tokens", as_dict=True)[
                    "elmo"]
            embedding = sess.run(tensor_embedding[0])

            sess.close()

            with open(os.path.join(path, 'sentence.txt'), "a") as sent_file:
                sent_file.write(sent.raw + "\n")

            try:
                np.save(os.path.join(path, file_name), embedding)

            except OSError as exc:
                if exc.errno == 36:
                    print(sent.raw)
                else:
                    raise

        update_progress_bar(counter, max_length, prefix='Progress:', suffix='Complete', length=100)

        counter = counter + 1
