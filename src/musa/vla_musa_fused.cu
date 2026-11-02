// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
// F32 expert adaRMS and gated residuals, on the graph's own MUSA stream.
#include "layers/fused_ew.h"
#include "musa/vla_musa_ops.h"
#include <musa_runtime.h>
#include <musa_fp16.h>
#include <mublas.h>
#include <mutex>

extern "C" {
extern bool (*ggml_musa_ext_forward)(ggml_tensor *, void *, void *);
extern bool (*ggml_musa_ext_supports)(const ggml_tensor *);
}

namespace vla {
namespace {
constexpr int WG = 256;

const void * userdata(const ggml_tensor * t) {
    if (!t || t->op != GGML_OP_CUSTOM) return nullptr;
    struct Params { void * fun; int n_tasks; void * userdata; } p;
    std::memcpy(&p, t->op_params, sizeof(p));
    return p.userdata;
}

uint32_t magic(const ggml_tensor * t) {
    uint32_t m = 0;
    if (const auto * ud = userdata(t)) std::memcpy(&m, ud, sizeof(m));
    return m;
}

bool dense_f32(const ggml_tensor * t) {
    return t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}
bool dense_f16(const ggml_tensor * t) {
    return t && t->type == GGML_TYPE_F16 && ggml_is_contiguous(t);
}

bool supports(const ggml_tensor * t) {
    if (t->op == GGML_OP_GLU && ggml_get_glu_op(t) == GGML_GLU_OP_GEGLU && !t->src[1])
        return (dense_f16(t) || dense_f32(t)) && ggml_is_contiguous(t->src[0]) &&
               t->src[0]->type == t->type && t->src[0]->ne[0] == 2*t->ne[0];
    if (t->op == GGML_OP_MUL_MAT && dense_f16(t) && dense_f16(t->src[0]) && dense_f16(t->src[1]))
        return t->ne[2] == 1 && t->ne[3] == 1 && t->src[0]->ne[2] == 1 && t->src[0]->ne[3] == 1;
    if (t->op == GGML_OP_RMS_NORM)
        return dense_f16(t) && dense_f16(t->src[0]);
    const uint32_t m = magic(t);
    if (m != FUSED_ADARMS_MAGIC && m != FUSED_GATERES_MAGIC) return false;
    const auto * x = t->src[0];
    const auto * mod = t->src[m == FUSED_ADARMS_MAGIC ? 1 : 2];
    if (!(dense_f32(t) || (m == FUSED_ADARMS_MAGIC && dense_f16(t))) || !dense_f32(x) || !dense_f32(mod) ||
        !ggml_are_same_shape(t, x) || ggml_nelements(mod) != 3*x->ne[0]) return false;
    if (m == FUSED_GATERES_MAGIC &&
        (!(dense_f32(t->src[1]) || dense_f16(t->src[1])) || !ggml_are_same_shape(x, t->src[1]))) return false;
    return true;
}

template<typename T> __device__ float load(T x) { return x; }
template<> __device__ float load(__half x) { return __half2float(x); }
template<typename T> __device__ T store(float x) { return x; }
template<> __device__ __half store(float x) { return __float2half(x); }

template<typename S, typename D>
__global__ void adarms(const S * x, const float * mod, D * dst, int h, float eps) {
    const int tid = threadIdx.x;
    const size_t row = (size_t) blockIdx.x*h;
    __shared__ float sums[WG];
    float sum = 0.0f;
    for (int i = tid; i < h; i += WG) {
        const float a = load(x[row+i]);
        sum += a*a;
    }
    sums[tid] = sum;
    __syncthreads();
    for (int stride = WG/2; stride; stride /= 2) {
        if (tid < stride) sums[tid] += sums[tid+stride];
        __syncthreads();
    }
    const float inv = rsqrtf(sums[0]/h + eps);
    for (int i = tid; i < h; i += WG) {
        const float n = load(x[row+i])*inv;
        dst[row+i] = store<D>(mod ? (n + n*mod[i]) + mod[h+i] : n);
    }
}

template<typename T>
__global__ void gateres(const float * x, const T * o, const float * mod,
                        float * dst, int h) {
    const int col = blockIdx.x*blockDim.x + threadIdx.x;
    const size_t i = (size_t) blockIdx.y*h + col;
    if (col < h) dst[i] = x[i] + load(o[i])*mod[2*h + col];
}

template<typename T>
__global__ void merged_geglu(const T * src, T * dst, int h, bool swapped) {
    const int col = blockIdx.x*blockDim.x + threadIdx.x;
    const size_t row = (size_t) blockIdx.y*h;
    if (col >= h) return;
    const float g = load(src[2*row + col + (swapped ? h : 0)]);
    const float u = load(src[2*row + col + (swapped ? 0 : h)]);
    const float gelu = 0.5f*g*(1.0f+tanhf(0.7978845608028654f*g*(1.0f+0.044715f*g*g)));
    dst[row+col] = store<T>(gelu*u);
}

bool forward(ggml_tensor * t, void * stream_ptr, void * blas_ptr) {
    if (!supports(t)) return false;
    const auto stream = (musaStream_t) stream_ptr;
    const int h = (int) t->ne[0];
    const auto * x = (const float *) t->src[0]->data;
    if (t->op == GGML_OP_GLU) {
        const dim3 grid((h+WG-1)/WG, ggml_nrows(t));
        const bool swapped = ((const int32_t *) t->op_params)[1] != 0;
        if (t->type == GGML_TYPE_F16)
            merged_geglu<<<grid, WG, 0, stream>>>((const __half *)x, (__half *)t->data, h, swapped);
        else merged_geglu<<<grid, WG, 0, stream>>>(x, (float *)t->data, h, swapped);
    } else if (t->op == GGML_OP_MUL_MAT) {
        const auto * a = t->src[0];
        const auto * b = t->src[1];
        // MUSA 5.1 rejects F16 output with COMPUTE_32F (NOT_SUPPORTED).
        const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        const auto status = mublasGemmEx((mublasHandle_t) blas_ptr, MUBLAS_OP_T, MUBLAS_OP_N,
            a->ne[1], b->ne[1], a->ne[0], &alpha, a->data, MUSA_R_16F, a->ne[0],
            b->data, MUSA_R_16F, b->ne[0], &beta, t->data, MUSA_R_16F, t->ne[0],
            MUBLAS_COMPUTE_16F, MUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (status != MUBLAS_STATUS_SUCCESS) GGML_ABORT("MUSA F16 GEMM failed: %d", (int) status);
        return true;
    } else if (t->op == GGML_OP_RMS_NORM) {
        float eps;
        std::memcpy(&eps, t->op_params, sizeof(eps));
        adarms<<<ggml_nrows(t), WG, 0, stream>>>((const __half *) x, nullptr, (__half *) t->data, h, eps);
    } else if (magic(t) == FUSED_ADARMS_MAGIC) {
        const auto & spec = *(const FusedAdarmsSpec *) userdata(t);
        if (t->type == GGML_TYPE_F16)
            adarms<<<ggml_nrows(t), WG, 0, stream>>>(x, (const float *) t->src[1]->data,
                                                (__half *) t->data, h, spec.eps);
        else adarms<<<ggml_nrows(t), WG, 0, stream>>>(x, (const float *) t->src[1]->data,
                                                (float *) t->data, h, spec.eps);
    } else {
        const dim3 grid((h+WG-1)/WG, ggml_nrows(t));
        if (t->src[1]->type == GGML_TYPE_F16)
            gateres<<<grid, WG, 0, stream>>>(x, (const __half *) t->src[1]->data,
            (const float *) t->src[2]->data, (float *) t->data, h);
        else gateres<<<grid, WG, 0, stream>>>(x, (const float *) t->src[1]->data,
            (const float *) t->src[2]->data, (float *) t->data, h);
    }
    const auto err = musaGetLastError();
    if (err != musaSuccess) GGML_ABORT("MUSA fused kernel: %s", musaGetErrorString(err));
    return true;
}
} // namespace

__attribute__((visibility("default"))) bool musa_register_fused_ops() {
    static std::once_flag once;
    std::call_once(once, [] {
        ggml_musa_ext_supports = supports;
        ggml_musa_ext_forward = forward;
    });
    return true;
}
} // namespace vla
