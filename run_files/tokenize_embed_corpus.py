import pandas as pd
import codecs
import os
import ujson
import swifter

from komma_ai.pre_processing import tokenize, create_word_dict, embed_tokens, make_input_embedding, make_output_embedding


def main():
    package_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    corpus_path = os.path.join(package_path, r"data\sentence_corpus\aug_sentence_corpus.txt")

    with codecs.open(corpus_path, "r", 'utf-8') as f:
        corpus_w_blanks = f.readlines()

    corpus = []

    for text in corpus_w_blanks:
        if len(text) > 2:
            corpus.append(text)

    df = pd.DataFrame(data={"sentence": corpus})

    df["tokens"] = df["sentence"].swifter.apply(tokenize)

    df["len"] = df["tokens"].swifter.apply(lambda x: len(x))

    dictionary = create_word_dict(df["tokens"])

    dictionary_path = os.path.join(package_path, r"data\sentence_corpus\\")

    with open(dictionary_path+'dictionary.json', 'w') as fp:
        ujson.dump(dictionary, fp)

    df["embedding"] = df["tokens"].swifter.apply(embed_tokens, args=(dictionary,))

    df["input_embedding"] = df["embedding"].swifter.apply(make_input_embedding, args=(dictionary,))
    df["output_embedding"] = df["embedding"].swifter.apply(make_output_embedding, args=(dictionary,))

    df_fin = df[["tokens", "input_embedding", "output_embedding"]]

    pre_processed_corpus = df_fin.to_dict('index')

    with open(dictionary_path+'pre_processed_corpus.json', 'w') as fp:
        ujson.dump(pre_processed_corpus, fp)



if __name__ == "__main__":
    main()

