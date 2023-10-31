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
    model.add(Conv1D(16, kernel_size=5, input_shape=(INPUT_SIZE, 1)))
    model.add(Activation('sigmoid'))
    model.add(Conv1D(32, kernel_size=3))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def __seed_random(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)