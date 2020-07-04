from unittest import TestCase
import os
from komma_ai.inference import Inference
import tensorflow as tf

package_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
dictionary_path = os.path.join(package_path, r"data\sentence_corpus\dictionary.json")

sess = tf.Session()
inference = Inference(dictionary_path=dictionary_path, sess=sess)
inference.load_model(model_path=os.path.join(package_path, r"data\models\baseline.h5"))

test_sentences = ["Das hier ist ein Satz.",  # Basic Test
                  "Ä Ö Ü ä ü ö",
                  "ÄÖÜU ÖÄÜU 14147 { [ ] } } [.",
                  "Das hier sollte z.B. gut klappen.",
                  "Das hier ist ein Test der TestApfel Version."]


class TestInference(TestCase):
    def setUp(self):
        self.inference = Inference(dictionary_path=dictionary_path, sess=sess)


class TestInit(TestInference):
    def test_dictionary_type(self):
        dictionary = self.inference.dictionary
        assert isinstance(dictionary, dict), 'Dictionary is wrong type!'

    def test_model_load(self):
        self.inference.load_model(model_path=os.path.join(package_path, r"data\models\baseline.h5"))


class TestPrediction(TestCase):
    def test_input_equals_output_without_comma(self):
        for test_sentence in test_sentences:
            output_sentence, pred_list = inference.predict_comma(test_sentence)
            self.assertEqual(len(test_sentence), len(output_sentence.replace(",", "")))
