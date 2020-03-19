import os

from lib.inference import Inference

def cmd_input():
    package_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    dictionary_path = os.path.join(package_path, r"data\sentence_corpus\dictionary.json")

    inference = Inference(dictionary_path=dictionary_path)
    inference.load_model(model_path=os.path.join(package_path, r"data\models\baseline.h5"))

    while True:
        input_sentence = input("Please enter sentence: ")

        output_sentence, pred_list = inference.predict_comma(input_sentence)

        print("Input:",input_sentence)
        print(">>>>",output_sentence)
        print(">>>>",pred_list)


if __name__ == "__main__":
    cmd_input()
