import os

import pandas as pd
import requests
import tensorflow as tf


def load_sts_dataset(filename):
    """
     Loads a subset of the STS dataset into a DataFrame.
     In particular both sentences and their human rated similarity score.
    :param filename:
    :return:
    """
    sent_pairs = []
    with tf.io.gfile.GFile(filename, "r") as f:
        for line in f:
            ts = line.strip().split("\t")
            sent_pairs.append((ts[5], ts[6], float(ts[4])))
    return pd.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])


def download_and_load_sts_data():
    sts_dataset = tf.keras.utils.get_file(
        fname="Stsbenchmark.tar.gz",
        origin="http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz",
        extract=True)

    sts_dev = load_sts_dataset(os.path.join(os.path.dirname(sts_dataset), "stsbenchmark", "sts-dev.csv"))
    sts_test = load_sts_dataset(os.path.join(os.path.dirname(sts_dataset), "stsbenchmark", "sts-test.csv"))

    return sts_dev, sts_test


def download_sick_dataset(url):
    response = requests.get(url).text

    lines = response.split("\n")[1:]
    lines = [l.split("\t") for l in lines if len(l) > 0]
    lines = [l for l in lines if len(l) == 5]

    df = pd.DataFrame(lines, columns=["idx", "sent_1", "sent_2", "sim", "label"])
    df['sim'] = pd.to_numeric(df['sim'])
    return df


def download_and_load_sick_dataset():
    sick_train = download_sick_dataset(
        "https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_train.txt")
    sick_dev = download_sick_dataset(
        "https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_trial.txt")
    sick_test = download_sick_dataset(
        "https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_test_annotated.txt")
    sick_all = sick_train.append(sick_test).append(sick_dev)

    return sick_all, sick_train, sick_test, sick_dev
