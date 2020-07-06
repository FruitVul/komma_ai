##
import json
import codecs
import os
import pandas as pd

from komma_ai.pre_processing import get_sentence_corpus,replace_abbreviations
##


package_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
three_path = os.path.join(package_path, r"komma_ai\data\3MILLSENTENCES\deu_news_2015_3M-sentences.txt")

content = []


##

with codecs.open(three_path, "r", 'utf-8') as f:
    sentences = f.readlines()
    processed_sentences = []
    for sentence in sentences[0:10]:
        if len(sentence) > 8:
            split_sentence = sentence.split(";")[-1]
            split_sentence = split_sentence.rstrip()
            processed_sentences.append(split_sentence)
