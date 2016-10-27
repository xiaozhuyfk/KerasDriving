import globals
from model.util import load_model, save_model_to_file
from keras.layers import Input, LSTM, Dense, Embedding, Merge, Convolution2D, Activation, MaxPooling2D, Flatten, Dropout
from keras.models import Model, model_from_json, Sequential


class CNN(object):

    @staticmethod
    def model(nb_filter, nb_row, nb_col, shape):
        model = Sequential()
        model.add(Convolution2D(
            nb_filter=48,
            nb_row=11,
            nb_col=11,
            border_mode='same',
            input_shape=shape,
            activation="relu"))

        model.add(Convolution2D(32, 5, 5, activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(64, 3, 3, border_mode="same", activation="relu"))
        model.add(Convolution2D(64, 3, 3, activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256))

        model.add(Dense(128))
        model.add(Dropout(0.25))

        model.add(Dense(14))

        return model

    @staticmethod
    def load_model():
        model_struct = globals.config.get("Model", "model-struct")
        model_weights = globals.config.get("Model", "model-weights")
        return load_model(model_struct, model_weights)

    @staticmethod
    def store_model(model):
        model_struct = globals.config.get("Model", "model-struct")
        model_weights = globals.config.get("Model", "model-weights")
        save_model_to_file(model, model_struct, model_weights)
