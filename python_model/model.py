import random
import numpy as np
import tensorflow as tf

from keras import *
from keras.layers import *
from constants import INPUT_SIZE


def compile_model(random_seed: int) -> Model:
    print(f'Creating model with input size {INPUT_SIZE}...')
    __seed_random(random_seed)
    model = Sequential()
    model.add(Dense(12, input_dim=INPUT_SIZE))
    model.add(Activation('sigmoid'))
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def __seed_random(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)