import json
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np

from lib.pre_processing import tokenize, embed_tokens, make_input_embedding,split_up_sentence


class Inference:
    def __init__(self,dictionary_path):
        self.model = None
        self.dictionary = self.load_dictionary(dictionary_path)

    @staticmethod
    def load_dictionary(dictionary_path):
        with open(dictionary_path) as handle:
            return json.loads(handle.read())

    def load_model(self,model_path):
        print("Loading model...")
        self.model = load_model(model_path)
        print("...successful!")

    def predict(self, padded_embedding):
        return self.model.predict(padded_embedding)

    def predict_comma(self, sentence, nn_settings=None):
        if nn_settings is None:
            nn_settings = {"prediction_threshold": 0.5,
                           "max_len": 50}

        original_sentence_split = split_up_sentence(sentence)

        max_len = nn_settings["max_len"]
        prediction_threshold = nn_settings["prediction_threshold"]

        tokenized_sentence = tokenize(sentence)
        embedded_sentence = embed_tokens(tokenized_sentence, self.dictionary)
        input_embedding = make_input_embedding(embedded_sentence, self.dictionary)
        padded_input = pad_sequences([input_embedding], maxlen=max_len, dtype="int32",
                                     padding="post", truncating="post", value=0.0)

        X = np.array([np.array(xi) for xi in padded_input])
        predictions = self.model.predict(X)

        pred_list = []
        output_sentence = [" " if split == "," else split for split in original_sentence_split]

        for i, prediction in enumerate(predictions[0]):
            if prediction > prediction_threshold:
                output_sentence[i] = ", "
            if i >= len(output_sentence):
                break
            pred_list.append((original_sentence_split[i], np.round(prediction * 100, 2)))

        output_sentence = ''.join(output_sentence)

        return output_sentence, pred_list