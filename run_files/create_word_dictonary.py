import pandas as pd
import numpy as np
from pathlib import Path
import json
import codecs
import os

from lib.pre_processing import tokenize


def main():
    package_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    corpus_path = os.path.join(package_path, r"data\sentence_corpus\aug_sentence_corpus_2020_03_16.txt")

    with codecs.open(corpus_path, "r", 'utf-8') as f:
        corpus = f.readlines()

    tokens = []
    for sentence in corpus:
        if len(sentence)>1:
            tokens.append(tokenize(sentence))

    df = pd.DataFrame(data={"tokens":tokens})
    df.to_csv("tokens.csv",sep=";")



if __name__ == "__main__":
    main()

