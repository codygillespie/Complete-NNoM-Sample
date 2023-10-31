# NNoM Full Build

This is a fully self-contained example using the NNoM library located here: https://github.com/majianjia/nnom

# Requirements
- CMake
    - Required for building the C implementation of the NNOM model
- Python
    - numpy
    - tensorflow
    - keras
    - sklearn

# Build

1. Start by building a model in python. Run `./tools/RunPython.ps1` to train the model and output the necessary files for the C implementation.
2. Run `./tools/BuildC.ps1` to build the C implementation of the model. This will output a .exe file.
3. Run `./tools/RunC.ps1` to run the C implementation of the model. This will output the results of the model.
4. Run `./tools/RunCompare.ps1` to compare the results of the python and C implementations. This will output the results of the comparison.

Note: The previous four steps can be done with one command `./tools/Pipeline.ps1`. All output will be saved to a file called `output.log` in the root directory.
