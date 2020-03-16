import pandas as pd
import numpy as np
from pathlib import Path
import json
import codecs
import os

from lib.pre_processing import remove_special_chars


def main():
    package_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    corpus_path = os.path.join(package_path, r"data\sentence_corpus\raw_sentence_corpus_2020_03_16.txt")

    with codecs.open(corpus_path, "r", 'utf-8') as f:
        corpus = f.readlines()

    augmented_sentences = []
    for sentence in corpus:
        augmented_sentences.append(remove_special_chars(sentence))

    with open('aug_sentence_corpus_2020_03_16.txt', mode='w', encoding='utf-8') as f:
        for sentence in augmented_sentences:
            f.write("%s\n" % sentence)


if __name__ == "__main__":
    main()

