import os
import pyperclip

from komma_ai.inference import Inference

def cmd_input():
    package_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    dictionary_path = os.path.join(package_path, r"data\sentence_corpus\dictionary.json")

    inference = Inference(dictionary_path=dictionary_path)
    inference.load_model(model_path=os.path.join(package_path, r"data\models\baseline.h5"))

    while True:
        input_sentence = input("Please enter sentence: ")

        output_sentence, pred_list = inference.predict_comma(input_sentence)
        flesch_score = inference.german_flesch_score([input_sentence])
        pyperclip.copy(output_sentence)
        pyperclip.paste()
        print("Input:",input_sentence)
        print(">>>>",output_sentence)
        print(">>>>",pred_list)
        print("Flesch Sorce:",flesch_score)



if __name__ == "__main__":
    cmd_input()
