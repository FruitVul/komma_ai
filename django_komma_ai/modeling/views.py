from django.http import JsonResponse

import tensorflow as tf
import h5py
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

import os

from komma_ai.inference import Inference

dictionary_path = os.path.join(r"C:\Users\Philipp\Projekte\komma_ai\data\sentence_corpus\dictionary.json")


f = h5py.File(r"C:\Users\Philipp\Projekte\komma_ai\data\models\baseline.h5",'r+')
data_p = f.attrs['training_config']
data_p = data_p.decode().replace("learning_rate", "lr").encode()
f.attrs['training_config'] = data_p
f.close()


sess = tf.Session()
graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras!
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess)


inference = Inference(dictionary_path=dictionary_path, sess=sess)
graph = tf.get_default_graph()
model = load_model(r"C:\Users\Philipp\Projekte\komma_ai\data\models\baseline.h5")
inference.model = model
inference.graph = graph

#inference.load_model(model_path=os.path.join(r"C:\Users\Philipp\Projekte\webkomma\data\models\baseline.h5"))

# Create your views here.


def api_komma_pred(request):
    sentence = request.GET["sentence"]
    output_sentence, __ = inference.predict_comma(sentence)
    return JsonResponse(output_sentence, safe=False)