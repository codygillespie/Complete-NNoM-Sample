cd cpp_inference

if(!(Test-Path -Path "./build")) {
    New-Item -ItemType Directory -Path "./build"
}

cd build

cmake ..
cmake --build .
cd ../..