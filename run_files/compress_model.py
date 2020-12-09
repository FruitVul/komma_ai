from tensorflow.python.keras.models import load_model
from tensorflow_model_optimization.sparsity import keras as sparsity
import tensorflow as tf
import numpy

import os

package_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
model_path = os.path.join(package_path, "data/models/baseline.h5")


def simple():
    model = load_model(model_path)

    model_json = model.to_json()
    with open("../model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")



if __name__ =="__main__":
    simple()