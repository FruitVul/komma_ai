import json
from keras.models import load_model

class Inference:
    def __init__(self, production_nn,dictionary_path):
        self.model = production_nn
        self.dictonary = self.load_dictionary(dictionary_path)

    @staticmethod
    def load_dictionary(dictionary_path):
        with open(dictionary_path) as handle:
            return json.loads(handle.read())

    def predict(self, padded_embedding):
        return self.model.predict(padded_embedding)