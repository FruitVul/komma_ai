import tensorflow as tf
from keras.models import Sequential
from keras import regularizers
from keras.layers import  Dropout
from keras.layers import  MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Dense, Conv1D
from keras.layers import Embedding
from keras.optimizers import RMSprop, Adam
from keras.models import load_model

class NeuralNetwork:
    def __init__(self,input_dim,output_dim,max_len,optimizer="adam"):

        self.optimizer = Adam(lr=0.001) if optimizer =="adam" else RMSprop(lr=0.001)

        self.model = Sequential()
        weight_decay = 1e-4

        self.model = Sequential()
        self.model.add(Embedding(input_dim=input_dim,
                                 output_dim=output_dim,
                                 input_length=max_len, trainable=True))
        self.model.add(Conv1D(256, 7, activation='relu', padding='same'))
        self.model.add(MaxPooling1D(2))
        self.model.add(Conv1D(128, 7, activation='relu', padding='same'))
        self.model.add(MaxPooling1D(2))
        self.model.add(Conv1D(64, 7, activation='relu', padding='same'))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Dense(max_len, activation='sigmoid'))

    def compile(self):
        self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer,metrics=['accuracy'])
        self.model.summary()

    def fit(self,batch_size, epochs, X_train, y_train, X_test, y_test,verbose=1):
        return self.model.fit(X_train, y_train, batch_size=batch_size,
                              epochs=epochs, validation_data=(X_test, y_test), verbose=verbose)

    def save_model(self, path):
        self.model.save(path)


class ProductionNN:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, padded_embedding):
        return self.model.predict(padded_embedding)
