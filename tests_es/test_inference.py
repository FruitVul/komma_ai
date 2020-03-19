import os

from lib.inference import Inference


def main():
    package_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    dictionary_path = os.path.join(package_path, r"data\sentence_corpus\dictionary.json")

    inference = Inference(dictionary_path=dictionary_path)
    inference.load_model(model_path=os.path.join(package_path, r"data\models\baseline.h5"))








if __name__ == "__main__":
    main()
