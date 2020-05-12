import pandas as pd
import numpy as np
from pathlib import Path
import json
import codecs
import os

from komma_ai.pre_processing import remove_special_chars


def main():
    package_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    corpus_path = os.path.join(package_path, r"data\sentence_corpus\sentence_corpus.txt")

    with codecs.open(corpus_path, "r", 'utf-8') as f:
        corpus = f.readlines()

    augmented_sentences = []
    for sentence in corpus:
        augmented_sentences.append(remove_special_chars(sentence))

    with open(os.path.join(package_path, r"data\sentence_corpus\aug_sentence_corpus.txt"),
              mode='w', encoding='utf-8') as f:
        for sentence in augmented_sentences:
            f.write(sentence)


if __name__ == "__main__":
    main()

