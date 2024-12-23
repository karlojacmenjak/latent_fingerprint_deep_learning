#!/bin/sh

pushd $(dirname $(python -c 'print(__import__("tensorflow").__file__)'))
ln -svf ../nvidia/*/lib/*.so* .
popd

ln -sf $(find $(dirname $(dirname $(python -c "import nvidia.cuda_nvcc;         
print(nvidia.cuda_nvcc.__file__)"))/*/bin/) -name ptxas -print -quit) $VIRTUAL_ENV/bin/ptxas
