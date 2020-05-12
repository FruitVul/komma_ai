import os

from komma_ai.inference import Inference

def simple_test():
    package_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    dictionary_path = os.path.join(package_path, r"data\sentence_corpus\dictionary.json")

    inference = Inference(dictionary_path=dictionary_path)
    inference.load_model(model_path=os.path.join(package_path, r"data\models\baseline.h5"))

    input_sentence = "Er rannte los ohne auf die anderen Läufer zu achten."

    output_sentence,pred_list = inference.predict_comma(input_sentence)

    print("Input:",input_sentence)
    print(">>>>",output_sentence)
    print(">>>>",pred_list)

    input_sentence = "Die Klasse beschließt, von dem Ausflug völlig begeistert, dort noch einmal hinzufahren."

    output_sentence,pred_list = inference.predict_comma(input_sentence)

    print("Input:", input_sentence)
    print(">>>>", output_sentence)
    print(">>>>", pred_list)

if __name__ == "__main__":
    simple_test()
