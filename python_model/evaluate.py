import os
import numpy as np

from tensorflow.keras.models import load_model

from constants import MODEL_NAME, PY_RESULTS_PATH

def evaluate_model(data, labels):
    assert len(data) == len(labels), 'data and labels must have the same length'

    model = load_model(MODEL_NAME)
    print('Quantizing model...')
    for i, layer in enumerate(model.layers):
        config, weights, biases = layer.get_config(), layer.get_weights(), None
        if 'dense' in config['name']:
            weights, biases = weights[0], weights[1]
            int_bits = int(np.ceil(np.log2(max(abs(np.min(weights)), abs(np.max(weights))))))
            dec_bits = 7 - int_bits
            quantized_weights, quantized_biases = np.round(weights * 2 ** dec_bits), np.round(biases * 2 ** dec_bits)
            layer.set_weights([quantized_weights, quantized_biases])

    print('Evaluating model...')
    predictions = model.predict(data)
    is_cft = np.argmax(predictions, axis=1)
    results = np.equal(is_cft, labels)
    
    print(f'Made {len(results)} predictions.')
    print(f'Accuracy: {np.sum(results) / len(results)}')

    if not os.path.exists(PY_RESULTS_PATH):
        os.makedirs(PY_RESULTS_PATH)

    with open(f'{PY_RESULTS_PATH}/results.csv', 'w+') as f:
        for result in results:
            f.write(f'{1 if result else 0}\n')