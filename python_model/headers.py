from tensorflow.keras.models import load_model

from constants import INPUT_SIZE, MODEL_NAME, WEIGHTS_HEADER_PATH, TEST_DATA_HEADER_PATH, TEST_LABELS_HEADER_PATH
from nnom import nnom_utils


def save_headers(x_test, y_test) -> None:
    try:
        nnom_utils.generate_model(load_model(MODEL_NAME), x_test, name=WEIGHTS_HEADER_PATH)

        with open(TEST_DATA_HEADER_PATH, 'w+') as f:
            f.write(f'const int samples = {len(x_test)};\n')
            f.write(f'const int size_per_sample = {INPUT_SIZE};\n')
            f.write(f'const int test_data[{len(x_test)}][{INPUT_SIZE}] = {{\n')
            f.write(',\n'.join(['{' + (','.join([f'{str(z)}' for z in x]) + '}') for x in x_test]))
            f.write('\n};')

        with open(TEST_LABELS_HEADER_PATH, 'w+') as f:
            f.write(f'const int test_data_labels[{len(y_test)}] = {{\n')
            f.write(','.join([str(y) for y in y_test]))
            f.write('\n};')

        return
    except FileNotFoundError:
        pass
    raise FileNotFoundError(f'Could not find cpp_inference directory.')