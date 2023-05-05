from keras import *
from keras.layers import *
from constants import INPUT_SIZE


def compile_model() -> Model:
    print(f'Creating model with input size {INPUT_SIZE}...')
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