import pandas as pd
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences

from komma_ai.neural_network import NeuralNetwork


def main():
    package_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    dictionary_path = os.path.join(package_path, r"data\sentence_corpus\dictionary.json")
    pre_processed_corpus_path = os.path.join(package_path, r"data\sentence_corpus\pre_processed_corpus.json")

    with open(dictionary_path) as handle:
        dictionary = json.loads(handle.read())

    corpus = json.load(open(pre_processed_corpus_path, "r"))
    df = pd.DataFrame.from_dict(corpus, orient="index")

    df["len"] = df["tokens"].apply(lambda tokens: len(tokens))

    max_len = 50

    df_len = df[df["len"] <= max_len]

    X = pad_sequences(df_len["input_embedding"], maxlen=max_len, dtype="int32",
                      padding="post", truncating="post", value=0.0)
    y = pad_sequences(df_len["output_embedding"], maxlen=max_len, dtype="int32",
                      padding="post", truncating="post", value=0.0)

    X = np.array([np.array(xi) for xi in X])
    y = np.array([np.array(yi) for yi in y])

    input_dim = list(dictionary.values())[-1]["id"]
    output_dim = max_len
    
    del df, df_len, corpus

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    del X, y

    nn = NeuralNetwork(input_dim, output_dim, max_len, optimizer="adam")
    nn.compile()
    nn.fit(32, 5, X_train, y_train, X_test, y_test, verbose=2)
    nn.save_model(os.path.join(package_path, r"data\models\baseline.h5"))


if __name__ == "__main__":
    main()

