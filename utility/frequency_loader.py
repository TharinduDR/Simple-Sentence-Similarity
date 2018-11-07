import csv

def load_tsv(f):
    frequencies = {}
    with open(f) as tsv:
        tsv_reader = csv.reader(tsv, delimiter="\t")
        for row in tsv_reader:
            frequencies[row[0]] = int(row[1])

    return frequencies


def load_frequencies(path):
    return load_tsv(path)


def load_doc_frequencies(path):
    doc_frequencies = load_tsv(path)
    doc_frequencies["NUM_DOCS"] = 1288431
    return doc_frequencies
