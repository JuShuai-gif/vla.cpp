// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
#include "musa/vla_musa_ops.h"
#include "ggml-cuda/unary.cuh"

namespace vla {
namespace {
constexpr int WG = 256;
template<bool Gelu>
__global__ void vision_epilogue(const float * x, const float * bias, const float * residual,
                               float * dst, int h) {
    const int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (col >= h) return;
    const size_t i = (size_t) blockIdx.y*h + col;
    // Preserve the rounding boundary of ggml_add despite fast-math.
    const float v = __fadd_rn(x[i], bias[col]);
    if (Gelu) dst[i] = ggml_cuda_op_gelu_single(v);
    else dst[i] = __fadd_rn(residual[i], v);
}
}
void musa_vision_epilogue(ggml_tensor * t, bool gelu, void * stream_ptr) {
    const int h = (int)t->ne[0];
    const dim3 grid((h+WG-1)/WG,ggml_nrows(t));
    const auto stream = (musaStream_t)stream_ptr;
    const auto * x = (const float *)t->src[0]->data;
    const auto * bias = (const float *)t->src[1]->data;
    if (gelu) vision_epilogue<true><<<grid,WG,0,stream>>>(x,bias,nullptr,(float *)t->data,h);
    else vision_epilogue<false><<<grid,WG,0,stream>>>(x,bias,(const float *)t->src[2]->data,(float *)t->data,h);
}
} // namespace vla
