#!/usr/bin/env bash
# M1000: Graph and FlashAttention, with capture-compatible F32 projections.
# Override MUSA_ARCHITECTURES for other Moore Threads devices.
set -euo pipefail

root=$(dirname "$(dirname "$(realpath "$0")")")
build=${BUILD_DIR:-"$root/build-musa"}
export MUSA_PATH=${MUSA_PATH:-/usr/local/musa}
export PATH="$MUSA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$MUSA_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake -S "$root" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$MUSA_PATH/bin/clang" \
    -DCMAKE_CXX_COMPILER="$MUSA_PATH/bin/clang++" \
    -DCMAKE_BUILD_RPATH="$MUSA_PATH/lib" \
    -DGGML_MUSA=ON -DMUSA_ARCHITECTURES="${MUSA_ARCHITECTURES:-22}" \
    -DGGML_CUDA_NO_VMM=ON -DGGML_MUSA_GRAPHS=ON -DGGML_CUDA_FA=ON \
    -DVLA_BUILD_SERVER=ON -DVLA_SPM=ON -DVLA_BUILD_TESTS=ON "$@"
cmake --build "$build" --parallel "${JOBS:-4}"
ctest --test-dir "$build" --output-on-failure
