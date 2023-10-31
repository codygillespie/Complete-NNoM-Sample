import os

INPUT_SIZE: int = 12

__directories: list[str] = ['../workspace', './workspace']
WORKSPACE_DIRECTORY: str = ''
for directory in __directories:
    if not os.path.isdir(directory):
        continue
    WORKSPACE_DIRECTORY = os.path.abspath(directory)
    break

if WORKSPACE_DIRECTORY == '':
    raise Exception('Workspace directory not found.')

MODEL_NAME: str = f'{WORKSPACE_DIRECTORY}/trained_model'

C_RESULTS_PATH: str = f'{WORKSPACE_DIRECTORY}\\c'
if os.path.exists(C_RESULTS_PATH):
    os.remove(C_RESULTS_PATH)
os.mkdir(C_RESULTS_PATH)

__C_ROOT_PATH: str = f'{WORKSPACE_DIRECTORY}/../cpp_inference'
WEIGHTS_HEADER_PATH: str = f'{__C_ROOT_PATH}/weights.h'
TEST_DATA_HEADER_PATH: str = f'{__C_ROOT_PATH}/test_data.h'
TEST_LABELS_HEADER_PATH: str = f'{__C_ROOT_PATH}/test_labels.h'
