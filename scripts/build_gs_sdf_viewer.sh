#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
    echo "Usage: $0 <CUDA LibTorch directory>" >&2
    exit 2
fi
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
libtorch=$(realpath -e -- "$1")
test -r "$libtorch/share/cmake/Torch/TorchConfig.cmake"
build="$repo/build/gaussian-viewer"
mkdir -p "$build"

docker run --rm --init --pull=never --network none \
    --user "$(id -u):$(id -g)" --cpus 12 --memory 32g \
    --mount "type=bind,source=$repo,target=/src,readonly" \
    --mount "type=bind,source=$build,target=/build" \
    --mount "type=bind,source=$libtorch,target=/opt/viewer-libtorch,readonly" \
    --env TORCH_CUDA_ARCH_LIST=8.9 \
    "${GS_SDF_CUDA_IMAGE:-gs-sdf-cu013:cu118-jammy}" bash -euc '
    cmake -S /src -B /build -G Ninja -DENABLE_ROS=OFF -DENABLE_BACKWARD=OFF \
        -DTorch_DIR=/opt/viewer-libtorch/share/cmake/Torch \
        -DCMAKE_BUILD_RPATH=/opt/viewer-libtorch/lib \
        -DCMAKE_CUDA_ARCHITECTURES=89 -DTCNN_CUDA_ARCHITECTURES=89 \
        -DCMAKE_CUDA_STANDARD=17 \
        -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g0 -DNDEBUG" \
        -DCMAKE_CUDA_FLAGS_RELWITHDEBINFO="-O2 -DNDEBUG"
    cmake --build /build --target gs_sdf_viewer_worker --parallel 4
'
