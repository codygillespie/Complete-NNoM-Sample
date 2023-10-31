import os
import argparse

import numpy as np
from keras import Model
from sklearn.model_selection import train_test_split

from model import compile_model
from constants import MODEL_NAME
from headers import save_headers
from data import ALL_DATA
from evaluate import evaluate_model


def main():
    # Required command line arguments:
    # -s: double between 0 and 1, the fraction of the data to be used for training
    # -r: int, the random seed to be used for the random split of the data
    # -e: int, the number of epochs to train the model
    # python .\python_model\main.py -s .8 -r 42 -e 50
    parser = argparse.ArgumentParser(description='Train a neural network model.')
    parser.add_argument('-s', '--split', type=float, required=True,
                        help='fraction of data to use for training (between 0 and 1)')
    parser.add_argument('-r', '--random_seed', type=int, required=True,
                        help='random seed for data split')
    parser.add_argument('-e', '--epochs', type=int, required=True,
                        help='number of epochs to train the model')

    args = parser.parse_args()
    split: float = args.split
    random_seed: int = args.random_seed
    epochs: int = args.epochs
    
    train, test = train_test_split(ALL_DATA, train_size=split, random_state=random_seed)
    x_train = np.array([[v/128 for v in x['data']] for x in train])
    y_train = np.array([x['label'] for x in train])
    x_test = np.array([x['data'] for x in test])
    y_test = np.array([x['label'] for x in test])

    model: Model = compile_model(random_seed)
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_data=(x_test, y_test))
    model.summary()
    
    model.save(MODEL_NAME)
    loss, acc = model.evaluate(x_test, y_test)
    print(f'loss: {loss}, acc: {acc}')
    save_headers(x_test, y_test)
    
    # evaluation:
    evaluate_model(x_test, y_test)


if __name__ == "__main__":
    main()