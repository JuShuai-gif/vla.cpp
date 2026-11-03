// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
// F32 expert adaRMS and gated residuals, on the graph's own MUSA stream.
#include "layers/fused_ew.h"
#include "layers/action_edges.h"
#include "layers/rope_table.h"
#include "layers/u8_patch.h"
#include "layers/weight_prefetch.h"
#include "layers/small_gemm.h"
#include "layers/packed_norm.h"
#include <mma.h>
#include "layers/fused_ops.h"
#include "modules/vision_prune.h"
#include <cassert>
#include <cstdlib>
#include "musa/vla_musa_ops.h"
#include <musa_runtime.h>
#include <musa_fp16.h>
#include <mublas.h>
#include <mutex>
#ifdef VLA_MUSA_MUDNN_ATTN
#include <mudnn.h>
#endif
#include <map>
#include <memory>
#include <tuple>
#include <vector>
#include <cmath>

extern "C" {
extern bool (*ggml_musa_ext_forward)(ggml_tensor *, void *, void *);
extern bool (*ggml_musa_ext_supports)(const ggml_tensor *);
extern void (*ggml_musa_ext_release)(int, void *);
}

namespace vla {
bool musa_vision_pack_enabled() {
#ifdef VLA_MUSA_MUDNN_ATTN
    static const bool enabled = std::getenv("VLA_MUSA_VISION_SDPA") && std::getenv("VLA_MUSA_VISION_PACK");
    return enabled;
#else
    return false;
#endif
}
bool musa_vision_raw_qkv(const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v) {
    if (!musa_vision_pack_enabled() || !q || !k || !v ||
        q->ne[0]!=72 || q->ne[1]!=256 || q->ne[2]!=16 || q->ne[3]<1 || q->ne[3]>2 ||
        !ggml_are_same_shape(q,k) || !ggml_are_same_shape(q,v)) return false;
    for (const auto * t : {q,k,v}) {
        if(t->type!=GGML_TYPE_F32 || t->nb[0]!=4 || t->view_offs%16) return false;
        for(int j=1;j<4;++j) if(t->nb[j]%16 || t->nb[j]!=q->nb[j]) return false;
    }
    return true;
}
namespace {
bool gemma_direct_output() {
    static const bool enabled = std::getenv("VLA_MUSA_ATTN_DIRECT_OUT") != nullptr;
    return enabled;
}
bool gemma_flat(const char * key,int nq=0,int nk=0) {
    const char * value=std::getenv(key);
    return value && (std::strcmp(value,"denoise")!=0 || nq<nk);
}
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

bool tiled_attention(const ggml_tensor * t) {
    static const bool enabled = [] {
        const char * value = std::getenv("VLA_MUSA_F32_ATTN_TILE");
        return !value || std::strcmp(value, "0") != 0;
    }();
    if (!enabled || t->op != GGML_OP_MUL_MAT || !dense_f32(t) ||
        !dense_f32(t->src[0]) || !dense_f32(t->src[1])) return false;
    const auto * a = t->src[0]; const auto * b = t->src[1];
    return a->ne[2] > 1 && a->ne[2] == b->ne[2] && a->ne[3] == b->ne[3] &&
           a->ne[0] <= 256 && a->ne[1] <= 256 && b->ne[1] <= 256;
}

bool vector_pack(const ggml_tensor * t) {
    static const bool enabled = std::getenv("VLA_MUSA_NO_VECTOR_PACK") == nullptr;
    const auto * a = t->src[0];
    return enabled && t->op == GGML_OP_CPY && dense_f16(t) && a && a->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(t, a) && !ggml_is_contiguous(a) && a->nb[0] == sizeof(float) &&
           a->ne[0] % 4 == 0 && a->nb[1] % 16 == 0 && a->nb[2] % 16 == 0 && a->nb[3] % 16 == 0 &&
           a->view_offs % 16 == 0;
}

bool vision_sdpa(const ggml_tensor * t) {
#ifdef VLA_MUSA_MUDNN_ATTN
    static const bool enabled = std::getenv("VLA_MUSA_VISION_SDPA") != nullptr;
    if (!enabled || t->op != GGML_OP_FLASH_ATTN_EXT || !dense_f32(t) || t->src[3]) return false;
    const auto * q = t->src[0], * k = t->src[1], * v = t->src[2];
    float params[3]; std::memcpy(params, t->op_params, sizeof(params));
    return q->type == GGML_TYPE_F32 && q->ne[0] == 72 && q->ne[1] == 256 && q->ne[2] == 16 &&
        q->ne[3] >= 1 && q->ne[3] <= 2 && q->nb[0] == 4 && q->nb[1]%16 == 0 &&
        q->nb[2]%16 == 0 && q->nb[3]%16 == 0 && q->view_offs%16 == 0 &&
        ( (dense_f16(k) && dense_f16(v)) || musa_vision_raw_qkv(q,k,v)) && ggml_are_same_shape(q,k) && ggml_are_same_shape(k,v) &&
        params[1] == 0 && params[2] == 0;
#else
    (void)t; return false;
#endif
}

// Both supplied models have at most 576 prefix + 50 action tokens.
constexpr int GEMMA_ATTN_CAPACITY = 640;
bool gemma_sdpa(const ggml_tensor * t) {
    static const int mode = [] {
        const char * p=std::getenv("VLA_MUSA_GEMMA_ATTN");
        return p && std::strcmp(p,"fused-all")==0 ? 2 :
               p && std::strcmp(p,"fused-denoise")==0 ? 1 : 0;
    }();
    if(!mode || t->op!=GGML_OP_FLASH_ATTN_EXT || !dense_f32(t)) return false;
    for(int j=4;j<GGML_MAX_SRC;++j) if(t->src[j]) return false;
    const auto * q=t->src[0], * k=t->src[1], * v=t->src[2], * mask=t->src[3];
    float p[3]; std::memcpy(p,t->op_params,sizeof(p));
    if(!q || !dense_f16(k) || !dense_f16(v) || !ggml_are_same_shape(k,v) ||
       q->type!=GGML_TYPE_F32 || q->ne[0]!=256 || q->ne[2]!=8 || q->ne[3]!=1 ||
       k->ne[0]!=256 || k->ne[2]!=1 || k->ne[3]!=1 ||
       q->ne[1]>GEMMA_ATTN_CAPACITY || k->ne[1]>GEMMA_ATTN_CAPACITY || q->ne[1]>k->ne[1] ||
       q->nb[0]!=4 || q->nb[1]%16 || q->nb[2]%16 || q->view_offs%16 ||
       p[1]!=0 || p[2]!=0) return false;
    if(mode==1 && q->ne[1]>=k->ne[1]) return false;
    return !mask || (dense_f16(mask) && mask->ne[0]==k->ne[1] &&
                     mask->ne[1]>=q->ne[1] && mask->ne[2]==1 && mask->ne[3]==1);
}

bool supports(const ggml_tensor * t) {
    if (gemma_sdpa(t)) return true;
    if (vision_sdpa(t)) return true;
    if (vector_pack(t)) return true;
    if (tiled_attention(t)) return true;
    if (t->op == GGML_OP_GLU && ggml_get_glu_op(t) == GGML_GLU_OP_GEGLU && !t->src[1])
        return (dense_f16(t) || dense_f32(t)) && ggml_is_contiguous(t->src[0]) &&
               (t->src[0]->type == t->type ||
                (t->src[0]->type == GGML_TYPE_F32 && t->type == GGML_TYPE_F16)) &&
               t->src[0]->ne[0] == 2*t->ne[0];
    if (t->op == GGML_OP_MUL_MAT && dense_f16(t) && dense_f16(t->src[0]) && dense_f16(t->src[1]))
        return t->ne[2] == 1 && t->ne[3] == 1 && t->src[0]->ne[2] == 1 && t->src[0]->ne[3] == 1;
    if (t->op == GGML_OP_RMS_NORM)
        return dense_f16(t) && dense_f16(t->src[0]);
    const uint32_t m = magic(t);
    if (m == FUSED_MAGIC) {
        const auto & spec = *(const FusedSpec *) userdata(t);
        if (spec.kind != FusedKind::BiasGelu && spec.kind != FusedKind::BiasRes) return false;
        // Two-dimensional launch: larger tensors must use the fallback.
        return !spec.out_bf16 && ggml_nrows(t) <= 65535 && t->ne[0] <= INT32_MAX-WG &&
               dense_f32(t) && dense_f32(t->src[0]) &&
               ggml_are_same_shape(t, t->src[0]) && dense_f32(t->src[1]) &&
               ggml_nelements(t->src[1]) == t->ne[0] &&
               (spec.kind != FusedKind::BiasRes ||
                (dense_f32(t->src[2]) && ggml_are_same_shape(t, t->src[2])));
    }
    if(m==VISION_SCORE_MAGIC) {
        const auto * x=t->src[0];
        return dense_f32(t) && t->ne[0]==256 && ggml_nelements(t)==256 &&
               dense_f32(x) && x->ne[0]==256 && x->ne[1]==256 && x->ne[2]==16 && x->ne[3]==1;
    }
    if (m == VISION_KEEP_MAGIC)
        return t->type == GGML_TYPE_I32 && dense_f32(t->src[0]) &&
               ggml_is_vector(t->src[0]) && t->src[0]->ne[0] <= WG &&
               t->ne[0] > 0 && t->ne[0] <= t->src[0]->ne[0];
    if(m==PACKED_GEGLU_MAGIC) {
        const auto * x=t->src[0];
        return dense_f16(t) && dense_f32(x) && x->ne[0]==2*t->ne[0] &&
               x->ne[1]>0 && x->ne[1]<=64 && x->ne[2]==1 && x->ne[3]==1 &&
               t->ne[1]==64 && t->ne[2]==1 && t->ne[3]==1;
    }
    if(m==PACKED_NORM_MAGIC) {
        const auto spec=packed_norm_spec(t);const auto * x=t->src[0];
        if(!dense_f32(x) || x->ne[1]<1 || x->ne[1]>64 || x->ne[2]!=1 || x->ne[3]!=1 ||
           spec.kind<0 || spec.kind>2 || t->type!=GGML_TYPE_I8 || !ggml_is_contiguous(t))return false;
        if(spec.kind && !dense_f32(t->src[1]))return false;
        if(spec.kind==1 && (!ggml_are_same_shape(x,t->src[1]) || !dense_f32(t->src[2]) ||
                           ggml_nelements(t->src[2])!=3*x->ne[0]))return false;
        if(spec.kind==2 && ggml_nelements(t->src[1])!=x->ne[0])return false;
        const auto * mod=t->src[spec.kind==1?3:spec.kind==2?2:1];
        return dense_f32(mod) && ggml_nelements(mod)==3*x->ne[0] &&
               ggml_nbytes(t)==(spec.kind?ggml_nelements(x)*4:0)+x->ne[0]*64*2;
    }
    if(m==SMALL_PACK_MAGIC) {
        const auto * x=t->src[0];
        return dense_f16(t) && (dense_f16(x)||dense_f32(x)) && x->ne[0]==t->ne[0] &&
               x->ne[1]>0 && x->ne[1]<=64 && t->ne[1]==64 &&
               x->ne[2]==1 && x->ne[3]==1 && t->ne[2]==1 && t->ne[3]==1;
    }
    if(m==SMALL_GEMM_MAGIC) {
        if(!musa_small_gemm_supported())return false;
        const auto & spec=*(const SmallGemmSpec *)userdata(t);
        const auto * a=t->src[0],*b=t->src[1];
        return (spec.pipeline==0 || spec.pipeline==4) && (spec.tile==32 || spec.tile==64) && dense_f32(t) && dense_f16(a) && dense_f16(b) &&
               a->ne[0]%16==0 && a->ne[1]%spec.tile==0 && a->ne[0]==b->ne[0] && b->ne[1]==64 &&
               a->ne[2]==1 && a->ne[3]==1 && b->ne[2]==1 && b->ne[3]==1 &&
               t->ne[0]==a->ne[1] && t->ne[1]>0 && t->ne[1]<=64 && t->ne[2]==1 && t->ne[3]==1;
    }
    if(m==WEIGHT_PREFETCH_MAGIC) {
        const auto & spec=*(const WeightPrefetchSpec *)userdata(t);
        return spec.state && t->src[1] && (dense_f16(t)||dense_f32(t)) &&
               t->view_src && t->view_offs==0 && ggml_are_same_shape(t,t->src[0]) &&
               t->type==t->src[0]->type && (spec.join || (spec.bytes>0 && spec.bytes<=ggml_nbytes(t->src[1])));
    }
    if(m==F32_PATCH_MAGIC) {
        const int ps=((const U8PatchSpec *)userdata(t))->patch;
        const auto * x=t->src[0];
        return dense_f16(t) && dense_f32(x) && x->ne[0]==3 && x->ne[1]==x->ne[2] &&
               ps>0 && x->ne[1]%ps==0 && t->ne[0]==3*ps*ps &&
               t->ne[1]==(x->ne[1]/ps)*(x->ne[1]/ps) && t->ne[2]==x->ne[3] && t->ne[3]==1;
    }
    if(m==U8_PATCH_MAGIC) {
        const int ps=((const U8PatchSpec *)userdata(t))->patch;
        const auto * x=t->src[0];
        return dense_f16(t) && x && x->type==GGML_TYPE_I8 && ggml_is_contiguous(x) &&
               x->ne[0]==3 && x->ne[1]==x->ne[2] && ps>0 && x->ne[1]%ps==0 &&
               t->ne[0]==3*ps*ps && t->ne[1]==(x->ne[1]/ps)*(x->ne[1]/ps) &&
               t->ne[2]==x->ne[3] && t->ne[3]==1 && dense_f16(t->src[1]) && ggml_nelements(t->src[1])==256;
    }
    if(m==ROPE_TABLE_MAGIC)
        return dense_f32(t) && t->ne[0]>0 && t->ne[0]%2==0 && t->ne[2]==1 && t->ne[3]==1 &&
               t->src[0] && t->src[0]->type==GGML_TYPE_I32 && ggml_is_contiguous(t->src[0]) &&
               ggml_is_vector(t->src[0]) && t->src[0]->ne[0]==t->ne[1];
    if(m==ROPE_APPLY_MAGIC)
        return (dense_f32(t)||dense_f16(t)) && t->src[0]->type==GGML_TYPE_F32 &&
               ggml_are_same_shape(t,t->src[0]) && t->ne[0]>0 && t->ne[0]%2==0 && t->ne[3]==1 &&
               t->src[0]->nb[0]==4 && dense_f32(t->src[1]) &&
               t->src[1]->ne[0]==t->ne[0] && t->src[1]->ne[1]==t->ne[2] &&
               t->src[1]->ne[2]==1 && t->src[1]->ne[3]==1;
    if(m==FUSED_ACTION_LINEAR_MAGIC) {
        const auto & s=*(const FusedActionLinearSpec *)userdata(t);
        for(int i=0;i<(s.input?4:5);++i)if(!dense_f32(t->src[i]))return false;
        const auto * w=t->src[0],* x=t->src[1];
        if(s.packed)return s.input && w->ne[1]<=4096 && w->ne[0]==x->ne[0] &&
             w->ne[2]==1 && w->ne[3]==1 && x->ne[2]==1 && x->ne[3]==1 &&
             x->ne[1]>0 && x->ne[1]<=64 && t->type==GGML_TYPE_I8 && ggml_is_contiguous(t) &&
             ggml_nbytes(t)==w->ne[1]*(x->ne[1]*4+64*2);
        return (s.input?w->ne[1]:w->ne[0])<=4096 && dense_f32(t) && w->ne[0]==x->ne[0] && w->ne[2]==1 && w->ne[3]==1 &&
               x->ne[2]==1 && x->ne[3]==1 && t->ne[0]==w->ne[1] && t->ne[1]==x->ne[1] &&
               t->ne[2]==(s.input?2:1) && t->ne[3]==1;
    }
    if (m == FUSED_ACTION_STEP_MAGIC)
        return dense_f32(t) && dense_f32(t->src[0]) && dense_f32(t->src[1]) &&
               ggml_are_same_shape(t,t->src[0]) && ggml_are_same_shape(t,t->src[1]) &&
               dense_f32(t->src[2]) && ggml_nelements(t->src[2])==t->ne[0];
    if (m == FUSED_BIAS_ADARMS_MAGIC) {
        const auto * x=t->src[0];
        return dense_f32(t) && dense_f32(x) && dense_f32(t->src[1]) && dense_f32(t->src[2]) &&
               x->ne[2]==1 && x->ne[3]==1 && t->ne[0]==x->ne[0] && t->ne[1]==x->ne[1] &&
               t->ne[2]==2 && t->ne[3]==1 && ggml_nelements(t->src[1])==x->ne[0] &&
               ggml_nelements(t->src[2])==3*x->ne[0];
    }
    if (m == FUSED_RES_ADARMS_MAGIC) {
        const auto * x = t->src[0]; const auto * o = t->src[1];
        return dense_f32(t) && dense_f32(x) && (dense_f32(o) || dense_f16(o)) &&
               ggml_are_same_shape(x, o) && x->ne[2] == 1 && x->ne[3] == 1 &&
               t->ne[0] == x->ne[0] && t->ne[1] == x->ne[1] && t->ne[2] == 2 && t->ne[3] == 1 &&
               dense_f32(t->src[2]) && dense_f32(t->src[3]) &&
               ggml_nelements(t->src[2]) == 3*x->ne[0] && ggml_nelements(t->src[3]) == 3*x->ne[0];
    }
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

template<typename T, bool Bias=false>
__global__ void res_adarms(const float * x, const T * o, const float * gate,
                          const float * mod, float * dst, int h, int rows, float eps) {
    const int tid = threadIdx.x;
    const size_t row = (size_t) blockIdx.x*h;
    __shared__ float sums[WG];
    float sum = 0.f;
    for (int i = tid; i < h; i += WG) {
        const float a = Bias ? x[row+i]+load(o[i]) : x[row+i] + load(o[row+i])*gate[2*h+i];
        dst[row+i] = a;
        sum += a*a;
    }
    sums[tid] = sum;
    __syncthreads();
    // Match the existing AdaRMS reduction tree and arithmetic order exactly.
    for (int stride = WG/2; stride; stride /= 2) {
        if (tid < stride) sums[tid] += sums[tid+stride];
        __syncthreads();
    }
    const float inv = rsqrtf(sums[0]/h + eps);
    for (int i = tid; i < h; i += WG) {
        const float n = dst[row+i]*inv;
        dst[(size_t) h*rows+row+i] = (n + n*mod[i]) + mod[h+i];
    }
}

template<bool Input,bool Packed=false>
__global__ void action_linear_kernel(const float * w,const float * x,const float * b,
        const float * mod,const float * noise,float * y,int h,int cols,int outputs,int rows,float eps,float dt) {
    const int row=blockIdx.x,tid=threadIdx.x;
    half * normalized=Packed?(half *)(y+(size_t)h*rows):nullptr;
    if(Packed && row>=rows) {
        for(int i=tid;i<h;i+=WG)normalized[(size_t)row*h+i]=__float2half_rn(0.f);
        return;
    }
    extern __shared__ float shared[];
    float * values=shared,* sums=shared+h;
    float sum=0;
    for(int i=tid;i<h;i+=WG) {
        float v;
        if(Input) {
            v=0;for(int j=0;j<cols;++j)v=__fmaf_rn(w[(size_t)i*cols+j],x[(size_t)row*cols+j],v);
            v+=b[i];y[(size_t)row*h+i]=v;
        } else v=x[(size_t)row*h+i];
        values[i]=v;sum+=v*v;
    }
    sums[tid]=sum;__syncthreads();
    for(int stride=WG/2;stride;stride/=2){if(tid<stride)sums[tid]+=sums[tid+stride];__syncthreads();}
    const float inv=rsqrtf(sums[0]/h+eps);
    for(int i=tid;i<h;i+=WG) {
        const float n=values[i]*inv;
        values[i]=(n+n*mod[i])+mod[h+i];
        if(Input) {
            if(Packed)normalized[(size_t)row*h+i]=__float2half_rn(values[i]);
            else y[(size_t)h*rows+(size_t)row*h+i]=values[i];
        }
    }
    __syncthreads();
    if(!Input)for(int o=tid;o<outputs;o+=WG) {
        float v=0;for(int i=0;i<h;++i)v=__fmaf_rn(w[(size_t)o*h+i],values[i],v);
        const float velocity=v+b[o],step=velocity*dt;
        y[(size_t)row*outputs+o]=noise[(size_t)row*outputs+o]+step;
    }
}
__global__ void action_step(const float * x, const float * v, const float * b,
                            float * out, size_t count, int h, float dt) {
    const size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) {
        const float velocity=v[i]+b[i%h];
        const float step=velocity*dt;
        out[i]=x[i]+step;
    }
}
template<typename S, typename D>
__global__ void merged_geglu(const S * src, D * dst, int h, bool swapped) {
    const int col = blockIdx.x*blockDim.x + threadIdx.x;
    const size_t row = (size_t) blockIdx.y*h;
    if (col >= h) return;
    const float g = load(src[2*row + col + (swapped ? h : 0)]);
    const float u = load(src[2*row + col + (swapped ? 0 : h)]);
    const float gelu = 0.5f*g*(1.0f+tanhf(0.7978845608028654f*g*(1.0f+0.044715f*g*g)));
    dst[row+col] = store<D>(gelu*u);
}


// 256 patches: exact ranking, ties prefer the lower spatial index.
// Reproduce native mean(head=16), then its 32-lane mean(query=256) tree.
__global__ void vision_scores_kernel(const float * x,float * out) {
 const int lane=threadIdx.x/8,key=blockIdx.x*8+threadIdx.x%8;
 float sum=0;
 #pragma unroll
 for(int j=0;j<8;++j) {
  float v[16];
  #pragma unroll
  for(int h=0;h<16;++h)v[h]=x[key+256*(lane+j*32+256*h)];
  #pragma unroll
  for(int stride=8;stride;stride/=2) {
   #pragma unroll
   for(int h=0;h<stride;++h)v[h]+=v[h+stride];
  }
  sum+=v[0]/16.f;
 }
 __shared__ float values[256];values[threadIdx.x]=sum;__syncthreads();
 for(int stride=16;stride;stride/=2) {
  if(lane<stride)values[threadIdx.x]+=values[threadIdx.x+stride*8];
  __syncthreads();
 }
 if(!lane)out[key]=values[threadIdx.x]/256.f;
}

__global__ void vision_keep(const float * scores, int32_t * ids, int n, int keep) {
    __shared__ float values[WG];
    __shared__ int selected[WG];
    const int i = threadIdx.x;
    if (i < n) {
        assert(isfinite(scores[i]));
        values[i] = scores[i];
    }
    __syncthreads();
    int rank = 0;
    if (i < n) {
        for (int j = 0; j < n; ++j)
            rank += values[j] > values[i] || (values[j] == values[i] && j < i);
    }
    selected[i] = i < n && rank < keep;
    __syncthreads();
    if (selected[i]) {
        int out = 0;
        for (int j = 0; j < i; ++j) out += selected[j];
        ids[out] = i;
    }
}


// F32 batched QK/AV, avoiding muBLAS's capture-time host synchronization.
// Columns of A and B are contiguous; compute A^T B in 16x16 tiles.
__global__ void attention_matmul_f32(const float * a, const float * b, float * out,
                                      int k, int m, int n) {
    __shared__ float at[16][17], bt[16][17];
    const int x=threadIdx.x, y=threadIdx.y;
    const int row=blockIdx.x*16+x, col=blockIdx.y*16+y;
    a += (size_t)blockIdx.z*k*m;
    b += (size_t)blockIdx.z*k*n;
    out += (size_t)blockIdx.z*m*n;
    float sum=0.0f;
    for (int base=0; base<k; base+=16) {
        const int ar=blockIdx.x*16+y;
        at[y][x] = ar<m && base+x<k ? a[(size_t)ar*k+base+x] : 0.0f;
        bt[y][x] = col<n && base+x<k ? b[(size_t)col*k+base+x] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int j=0; j<16; ++j) sum=fmaf(at[x][j],bt[y][j],sum);
        __syncthreads();
    }
    if (row<m && col<n) out[(size_t)col*m+row]=sum;
}

__global__ void pack_f16x4(const float * src, __half2 * dst, size_t n4, int d4, int rows, int heads,
                           size_t s1, size_t s2, size_t s3) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    size_t row = i / d4;
    const size_t col = (i % d4) * 4;
    const size_t token = row % rows; row /= rows;
    const size_t head = row % heads, batch = row / heads;
    const float4 v = *(const float4 *)(src + col + token*s1 + head*s2 + batch*s3);
    dst[2*i]   = __floats2half2_rn(v.x, v.y);
    dst[2*i+1] = __floats2half2_rn(v.z, v.w);
}

#define VLA_ATTN_MU(call) do { const auto e=(call); if(e!=musaSuccess) GGML_ABORT("MUSA attention %s: %s",#call,musaGetErrorString(e)); } while(0)
#define VLA_ATTN_BLAS(call) do { const auto e=(call); if(e!=MUBLAS_STATUS_SUCCESS) GGML_ABORT("muBLAS attention %s: %d",#call,(int)e); } while(0)
struct GemmaSdpa {
    mublasHandle_t handle=nullptr;
    void * q=nullptr, * scores=nullptr, * probs=nullptr, * out=nullptr;
    int device;
    GemmaSdpa(int dev,musaStream_t stream):device(dev) {
        // Stable addresses across graph recaptures, bounded memory across token lengths.
        constexpr int nq=GEMMA_ATTN_CAPACITY,nk=GEMMA_ATTN_CAPACITY;
        VLA_ATTN_BLAS(mublasCreate(&handle)); VLA_ATTN_BLAS(mublasSetStream(handle,stream));
        VLA_ATTN_MU(musaMalloc(&q,(size_t)8*256*nq*2));
        VLA_ATTN_MU(musaMalloc(&scores,(size_t)8*nk*nq*4));
        VLA_ATTN_MU(musaMalloc(&probs,(size_t)8*nk*nq*2));
        if(!gemma_direct_output() || std::getenv("VLA_MUSA_GEMMA_FLAT_AV")) VLA_ATTN_MU(musaMalloc(&out,(size_t)8*256*nq*4));
    }
    ~GemmaSdpa() {
        int old=0;musaGetDevice(&old);musaSetDevice(device);
        musaFree(q);musaFree(scores);musaFree(probs);musaFree(out);
        mublasDestroy(handle);musaSetDevice(old);
    }
};
using GemmaKey=std::tuple<int,uintptr_t>;
struct GemmaCache {
    std::mutex mutex;
    std::map<GemmaKey,std::unique_ptr<GemmaSdpa>> entries;
};
GemmaCache & gemma_cache(){ static GemmaCache cache;return cache; }
void release_gemma_sdpa(int device,void * stream) {
    auto & cache=gemma_cache();std::lock_guard<std::mutex> guard(cache.mutex);
    for(auto it=cache.entries.begin();it!=cache.entries.end();)
        if(std::get<0>(it->first)==device && std::get<1>(it->first)==(uintptr_t)stream) it=cache.entries.erase(it);
        else ++it;
}

// One warp128 row; keep reductions F32 and fuse mask, scale and probability packing.
__global__ void gemma_probs(const float * scores,const __half * mask,__half * probs,int nq,int nk,float scale) {
    constexpr int threads=128;
    const int tid=threadIdx.x,row=blockIdx.x;
    __shared__ float reduction[threads];
    float vals[8], mx=-INFINITY;
#pragma unroll
    for(int j=0;j<8;++j){
        const int col=tid+j*threads;
        vals[j]=col<nk ? scores[(size_t)row*nk+col]*scale+(mask?__half2float(mask[(row%nq)*nk+col]):0.0f) : -INFINITY;
        mx=fmaxf(mx,vals[j]);
    }
    reduction[tid]=mx;__syncthreads();
    for(int s=threads/2;s;s/=2){if(tid<s)reduction[tid]=fmaxf(reduction[tid],reduction[tid+s]);__syncthreads();}
    mx=reduction[0];float sum=0;
#pragma unroll
    for(int j=0;j<8;++j){vals[j]=mx==-INFINITY?0.0f:expf(vals[j]-mx);sum+=vals[j];}
    __syncthreads();reduction[tid]=sum;__syncthreads();
    for(int s=threads/2;s;s/=2){if(tid<s)reduction[tid]+=reduction[tid+s];__syncthreads();}
    const float inv=reduction[0]>0 ? 1.0f/reduction[0] : 0.0f;
#pragma unroll
    for(int j=0;j<8;++j)if(tid+j*threads<nk)probs[(size_t)row*nk+tid+j*threads]=__float2half_rn(vals[j]*inv);
}
__global__ void gemma_unpack(const float4 * src,float4 * dst,int nq) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nq*8*64)return;
    const int d=i%64,h=(i/64)%8,q=i/(64*8);
    dst[i]=src[d+64*(q+nq*h)];
}
bool run_gemma_sdpa(ggml_tensor * t,musaStream_t stream) {
    int device=0;VLA_ATTN_MU(musaGetDevice(&device));
    const auto * q=t->src[0], * k=t->src[1], * v=t->src[2], * mask=t->src[3];
    const int nq=q->ne[1],nk=k->ne[1];
    auto & cache=gemma_cache();std::lock_guard<std::mutex> guard(cache.mutex);
    const GemmaKey key(device,(uintptr_t)stream);auto it=cache.entries.find(key);
    if(it==cache.entries.end()){
        musaStreamCaptureStatus capture;VLA_ATTN_MU(musaStreamIsCapturing(stream,&capture));
        if(capture!=musaStreamCaptureStatusNone)return false;
        it=cache.entries.emplace(key,std::make_unique<GemmaSdpa>(device,stream)).first;
    }
    auto & s=*it->second;const size_t n4=(size_t)8*256*nq/4;
    pack_f16x4<<<(n4+WG-1)/WG,WG,0,stream>>>((const float *)q->data,(__half2 *)s.q,n4,64,nq,8,q->nb[1]/4,q->nb[2]/4,q->nb[3]/4);
    const float one=1,zero=0;float scale;std::memcpy(&scale,t->op_params,sizeof(scale));
    if(gemma_flat("VLA_MUSA_GEMMA_FLAT_QK",nq,nk))
        VLA_ATTN_BLAS(mublasGemmEx(s.handle,MUBLAS_OP_T,MUBLAS_OP_N,nk,nq*8,256,
            &one,k->data,MUSA_R_16F,256,s.q,MUSA_R_16F,256,
            &zero,s.scores,MUSA_R_32F,nk,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP));
    else {
    VLA_ATTN_BLAS(mublasGemmStridedBatchedEx(s.handle,MUBLAS_OP_T,MUBLAS_OP_N,nk,nq,256,
        &one,k->data,MUSA_R_16F,256,0,s.q,MUSA_R_16F,256,256*nq,
        &zero,s.scores,MUSA_R_32F,nk,nk*nq,8,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    gemma_probs<<<8*nq,128,0,stream>>>((const float *)s.scores,mask?(const __half *)mask->data:nullptr,(__half *)s.probs,nq,nk,scale);
    const bool flat_av=gemma_flat("VLA_MUSA_GEMMA_FLAT_AV",nq,nk);
    if(flat_av)
        VLA_ATTN_BLAS(mublasGemmEx(s.handle,MUBLAS_OP_N,MUBLAS_OP_N,256,nq*8,nk,
            &one,v->data,MUSA_R_16F,256,s.probs,MUSA_R_16F,nk,
            &zero,s.out,MUSA_R_32F,256,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP));
    else {
    VLA_ATTN_BLAS(mublasGemmStridedBatchedEx(s.handle,MUBLAS_OP_N,MUBLAS_OP_N,256,nq,nk,
        &one,v->data,MUSA_R_16F,256,0,s.probs,MUSA_R_16F,nk,nk*nq,
        &zero,gemma_direct_output()?t->data:s.out,MUSA_R_32F,
        gemma_direct_output()?2048:256,gemma_direct_output()?256:256*nq,
        8,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    if(flat_av || !gemma_direct_output())
        gemma_unpack<<<(n4+WG-1)/WG,WG,0,stream>>>((const float4 *)s.out,(float4 *)t->data,nq);
    return true;
}

#ifdef VLA_MUSA_MUDNN_ATTN
#define VLA_DNN_CHECK(call) do { const auto e = (call); if (e != MUDNN_STATUS_SUCCESS) GGML_ABORT("muDNN %s: %d", #call, (int)e); } while(0)
#define VLA_MU_CHECK(call) do { const auto e = (call); if (e != musaSuccess) GGML_ABORT("MUSA %s: %s", #call, musaGetErrorString(e)); } while(0)
struct VisionSdpa {
    mudnnHandle_t handle = nullptr;
    mudnnScaledDotProductAttentionDescriptor_t attention = nullptr;
    // muDNN descriptors retain per-operand addresses; never alias Q/K/V/output descriptors.
    mudnnTensorDescriptor_t tensor[4] = {}, lse_desc = nullptr, empty = nullptr;
    void * q = nullptr, * k = nullptr, * v = nullptr, * out = nullptr, * lse = nullptr;
    int device = 0;
    VisionSdpa(int dev, int batch, float scale, musaStream_t stream): device(dev) {
        VLA_DNN_CHECK(mudnnCreate(&handle)); VLA_DNN_CHECK(mudnnSetStream(handle,stream));
        VLA_DNN_CHECK(mudnnCreateScaledDotProductAttentionDescriptor(&attention));
        VLA_DNN_CHECK(mudnnSetScaledDotProductAttentionDescriptor(attention,MUDNN_TENSOR_OP_MATH,
            128,batch,16,scale,false,false,false,false,true));
        for(auto & t:tensor) VLA_DNN_CHECK(mudnnCreateTensorDescriptor(&t));
        VLA_DNN_CHECK(mudnnCreateTensorDescriptor(&lse_desc)); VLA_DNN_CHECK(mudnnCreateTensorDescriptor(&empty));
        const int dims[]={batch,16,256,128}, strides[]={16*256*128,256*128,128,1};
        const int ld[]={batch,16,256,1}, ls[]={16*256,256,1,1};
        for(auto t:tensor) VLA_DNN_CHECK(mudnnSetTensorNdDescriptor(t,MUDNN_DATA_HALF,4,dims,strides));
        VLA_DNN_CHECK(mudnnSetTensorNdDescriptor(lse_desc,MUDNN_DATA_FLOAT,4,ld,ls));
        VLA_MU_CHECK(musaMalloc(&q,(size_t)batch*16*256*128*2));
        VLA_MU_CHECK(musaMalloc(&k,(size_t)batch*16*256*128*2));
        VLA_MU_CHECK(musaMalloc(&v,(size_t)batch*16*256*128*2));
        VLA_MU_CHECK(musaMalloc(&out,(size_t)batch*16*256*128*2));
        VLA_MU_CHECK(musaMalloc(&lse,(size_t)batch*16*256*4));
        // Input padding is immutable. Capture/replay only rewrites the 72 valid columns.
        for(void * ptr : {q,k,v})
            VLA_MU_CHECK(musaMemsetAsync(ptr,0,(size_t)batch*16*256*128*2,stream));
    }
    ~VisionSdpa() {
        int old=0;musaGetDevice(&old);musaSetDevice(device);
        musaFree(q);musaFree(k);musaFree(v);musaFree(out);musaFree(lse);
        for(auto t:tensor) mudnnDestroyTensorDescriptor(t);
        mudnnDestroyTensorDescriptor(lse_desc);mudnnDestroyTensorDescriptor(empty);
        mudnnDestroyScaledDotProductAttentionDescriptor(attention);mudnnDestroy(handle);
        musaSetDevice(old);
    }
};

// The library's supported head width is 128. Zero padding preserves QK and scale.
template<typename T>
__global__ void pad_sdpa(const T * src, __half2 * dst, int batch, size_t s1, size_t s2, size_t s3) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=batch*16*256*32)return;
    const int col=(i%32)*4, token=(i/32)%256, head=(i/(32*256))%16, b=i/(32*256*16);
    float a=0,c=0,d=0,e=0;
    if(col<72){
        const T * p=src+col+token*s1+head*s2+b*s3;
        a=load(p[0]);c=load(p[1]);d=load(p[2]);e=load(p[3]);
    }
    dst[2*i]=__floats2half2_rn(a,c); dst[2*i+1]=__floats2half2_rn(d,e);
}

// Fuse strided Q/K/V reads, F32->F16 conversion and head padding.
__global__ void pack_vision_qkv(const float * q,const float * k,const float * v,
        __half2 * qo,__half2 * ko,__half2 * vo,int batch,size_t s1,size_t s2,size_t s3) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=batch*16*256*18)return;
    const int d=i%18,token=(i/18)%256,head=(i/(18*256))%16,b=i/(18*256*16);
    const float * src=blockIdx.y==0?q:(blockIdx.y==1?k:v);
    __half2 * dst=blockIdx.y==0?qo:(blockIdx.y==1?ko:vo);
    const float4 x=*(const float4 *)(src+4*d+token*s1+head*s2+b*s3);
    const int out=d+32*(token+256*(head+16*b));
    dst[2*out]=__floats2half2_rn(x.x,x.y);
    dst[2*out+1]=__floats2half2_rn(x.z,x.w);
}

__global__ void unpack_sdpa(const __half2 * src, float4 * dst, int batch) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=batch*256*16*18)return;
    const int d4=i%18, head=(i/18)%16, token=(i/(18*16))%256, b=i/(18*16*256);
    const int j=d4+32*(token+256*(head+16*b));
    const float2 a=__half22float2(src[2*j]), c=__half22float2(src[2*j+1]);
    dst[i]=make_float4(a.x,a.y,c.x,c.y);
}

using VisionKey=std::tuple<int,uintptr_t,int,uint32_t>;
struct VisionCache {
    std::mutex mutex;
    std::map<VisionKey,std::unique_ptr<VisionSdpa>> entries;
};
VisionCache & vision_cache() { static VisionCache cache; return cache; }
void release_vision_sdpa(int device, void * stream) {
    auto & cache=vision_cache(); std::lock_guard<std::mutex> guard(cache.mutex);
    for(auto it=cache.entries.begin(); it!=cache.entries.end();)
        if(std::get<0>(it->first)==device && std::get<1>(it->first)==(uintptr_t)stream)
            it=cache.entries.erase(it);
        else ++it;
}

bool run_vision_sdpa(ggml_tensor * t, musaStream_t stream) {
    float scale;std::memcpy(&scale,t->op_params,sizeof(scale));
    uint32_t scale_bits;std::memcpy(&scale_bits,&scale,sizeof(scale));
    int device=0;VLA_MU_CHECK(musaGetDevice(&device));
    const auto * q=t->src[0];const int batch=q->ne[3];
    // One scratch set per stream/shape: nodes execute in order, including graph replay.
    auto & storage=vision_cache(); std::lock_guard<std::mutex> guard(storage.mutex);
    auto & cache=storage.entries;
    const VisionKey key(device,(uintptr_t)stream,batch,scale_bits);
    auto it=cache.find(key);
    if(it==cache.end()){
        musaStreamCaptureStatus capture;VLA_MU_CHECK(musaStreamIsCapturing(stream,&capture));
        if(capture!=musaStreamCaptureStatusNone)return false;
        it=cache.emplace(key,std::make_unique<VisionSdpa>(device,batch,scale,stream)).first;
    }
    auto & s=*it->second;
    const size_t n4=(size_t)batch*16*256*72/4;
    const int packed=batch*16*256*32;
    if(musa_vision_raw_qkv(q,t->src[1],t->src[2])) {
        pack_vision_qkv<<<dim3((n4+WG-1)/WG,3),WG,0,stream>>>(
            (const float *)q->data,(const float *)t->src[1]->data,(const float *)t->src[2]->data,
            (__half2 *)s.q,(__half2 *)s.k,(__half2 *)s.v,batch,q->nb[1]/4,q->nb[2]/4,q->nb[3]/4);
    } else {
    pad_sdpa<<<(packed+WG-1)/WG,WG,0,stream>>>((const float *)q->data,(__half2 *)s.q,batch,
        q->nb[1]/4,q->nb[2]/4,q->nb[3]/4);
    for(int j=1;j<=2;++j){
        const auto * a=t->src[j];
        pad_sdpa<<<(packed+WG-1)/WG,WG,0,stream>>>((const __half *)a->data,
            (__half2 *)(j==1?s.k:s.v),batch,a->nb[1]/2,a->nb[2]/2,a->nb[3]/2);
    }
    }
    VLA_DNN_CHECK(mudnnScaledDotProductAttentionFlashForward(s.handle,s.lse_desc,s.lse,
        s.tensor[0],s.q,s.tensor[1],s.k,s.tensor[2],s.v,s.empty,nullptr,
        s.attention,s.empty,nullptr,s.tensor[3],s.out));
    unpack_sdpa<<<(n4+WG-1)/WG,WG,0,stream>>>((const __half2 *)s.out,(float4 *)t->data,batch);
    return true;
}
#endif

void release_attention(int device,void * stream) {
    release_gemma_sdpa(device,stream);
#ifdef VLA_MUSA_MUDNN_ATTN
    release_vision_sdpa(device,stream);
#endif
}

__global__ void packed_geglu_kernel(const float * x,half * y,int h,int rows) {
 const int i=blockIdx.x*blockDim.x+threadIdx.x,r=blockIdx.y;
 if(i>=h)return;
 float value=0;
 if(r<rows) {
  const float g=x[r*2*h+i],u=x[r*2*h+h+i];
  const float gelu=0.5f*g*(1.0f+tanhf(0.7978845608028654f*g*(1.0f+0.044715f*g*g)));
  value=gelu*u;
 }
 y[r*h+i]=__float2half_rn(value);
}

template<int Kind>
__global__ void packed_norm_kernel(const float * x,const float * o,const float * gate,
        const float * mod,char * blob,int h,int rows,float eps) {
 const int tid=threadIdx.x,r=blockIdx.x;const size_t row=(size_t)r*h;
 float * residual=(float *)blob;
 half * norm=(half *)(blob+(Kind?h*rows*4:0));
 if(r>=rows){for(int i=tid;i<h;i+=WG)norm[row+i]=__float2half_rn(0.f);return;}
 __shared__ float sums[WG];float sum=0;
 for(int i=tid;i<h;i+=WG) {
  const float a=Kind==1?x[row+i]+o[row+i]*gate[2*h+i]:Kind==2?x[row+i]+o[i]:x[row+i];
  if(Kind)residual[row+i]=a;sum+=a*a;
 }
 sums[tid]=sum;__syncthreads();
 for(int stride=WG/2;stride;stride/=2){if(tid<stride)sums[tid]+=sums[tid+stride];__syncthreads();}
 const float inv=rsqrtf(sums[0]/h+eps);
 for(int i=tid;i<h;i+=WG) {
  const float n=(Kind?residual[row+i]:x[row+i])*inv;
  norm[row+i]=__float2half_rn((n+n*mod[i])+mod[h+i]);
 }
}

template<int TM,int Pipeline=0>
__global__ void small_gemm_kernel(const __half*w,const __half*x,float*y,int m,int n,int k) {
#if !defined(__MUSA_ARCH__) || __MUSA_ARCH__==220
 using namespace mtmusa::wmma;constexpr int NT=2,MT=TM/32,ld=68;
 const int row=blockIdx.x*TM,col=blockIdx.y*64;
 fragment<matrix_a,32,32,16,__half,row_major> a[MT];
 fragment<matrix_b,32,32,16,__half,col_major> b[NT];
 fragment<accumulator,32,32,16,float> acc[MT][NT];
 #pragma unroll
 for(int i=0;i<MT;i++){
  #pragma unroll
  for(int j=0;j<NT;j++)fill_fragment(acc[i][j],0.f);
 }
 #pragma unroll (Pipeline?Pipeline:1)
 for(int p=0;p<k;p+=16) {
  #pragma unroll
  for(int i=0;i<MT;i++)load_matrix_sync(a[i],w+(size_t)(row+i*32)*k+p,k);
  #pragma unroll
  for(int j=0;j<NT;j++)load_matrix_sync(b[j],x+(size_t)(col+j*32)*k+p,k);
  #pragma unroll
  for(int i=0;i<MT;i++){
   #pragma unroll
   for(int j=0;j<NT;j++)mma_sync(acc[i][j],a[i],b[j],acc[i][j]);
  }
 }
 __shared__ float out[TM*ld];
 #pragma unroll
 for(int i=0;i<MT;i++){
  #pragma unroll
  for(int j=0;j<NT;j++)store_matrix_sync(out+i*32*ld+j*32,acc[i][j],ld,mem_row_major);
 }
 __syncthreads();
 for(int i=threadIdx.x;i<TM*64;i+=blockDim.x) {
  const int c=row+i%TM,r=col+i/TM;
  if(c<m&&r<n)y[(size_t)r*m+c]=out[(i%TM)*ld+i/TM];
 }
#endif
}

template<typename S>
__global__ void small_pack_kernel(const S * x,half * y,int k,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=k*64)return;
    y[i]=i<k*n?__float2half_rn(load(x[i])):__float2half_rn(0.f);
}
struct WeightPrefetchState {
    musaStream_t stream=nullptr;
    musaEvent_t fork=nullptr,join=nullptr;
    unsigned * sink=nullptr;
};
__global__ void prefetch_lines(const unsigned * w,unsigned * sink,size_t words) {
    const size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    unsigned v=0;
    for(size_t j=i*32;j<words;j+=(size_t)gridDim.x*blockDim.x*32)
        v+=((const volatile unsigned *)w)[j];
    sink[i]=v;
}
bool forward(ggml_tensor * t, void * stream_ptr, void * blas_ptr) {
    if (!supports(t)) return false;
    const auto stream = (musaStream_t) stream_ptr;
    if (gemma_sdpa(t)) return run_gemma_sdpa(t,stream);
#ifdef VLA_MUSA_MUDNN_ATTN
    if (vision_sdpa(t)) return run_vision_sdpa(t, stream);
#endif
    const int h = (int) t->ne[0];
    const auto * x = (const float *) t->src[0]->data;
    if(magic(t)==PACKED_GEGLU_MAGIC) {
        packed_geglu_kernel<<<dim3((h+WG-1)/WG,64),WG,0,stream>>>(x,(half *)t->data,h,t->src[0]->ne[1]);
    } else if(magic(t)==PACKED_NORM_MAGIC) {
        const auto spec=packed_norm_spec(t);const auto * a=t->src[0];
        const float * o=spec.kind?(const float *)t->src[1]->data:nullptr;
        const float * gate=spec.kind==1?(const float *)t->src[2]->data:nullptr;
        const float * mod=(const float *)t->src[spec.kind==1?3:spec.kind==2?2:1]->data;
#define PACKED_NORM_CASE(K) case K: packed_norm_kernel<K><<<64,WG,0,stream>>>(x,o,gate,mod,(char *)t->data,a->ne[0],a->ne[1],spec.eps);break
        switch(spec.kind){PACKED_NORM_CASE(0);PACKED_NORM_CASE(1);PACKED_NORM_CASE(2);}
#undef PACKED_NORM_CASE
    } else if(magic(t)==SMALL_PACK_MAGIC) {
        const auto * a=t->src[0];const int k=a->ne[0],n=a->ne[1];
        if(a->type==GGML_TYPE_F16)small_pack_kernel<<<(k*64+WG-1)/WG,WG,0,stream>>>(
            (const half *)a->data,(half *)t->data,k,n);
        else small_pack_kernel<<<(k*64+WG-1)/WG,WG,0,stream>>>(
            (const float *)a->data,(half *)t->data,k,n);
    } else if(magic(t)==SMALL_GEMM_MAGIC) {
        const auto & spec=*(const SmallGemmSpec *)userdata(t);
        const int m=t->ne[0],n=t->ne[1],k=t->src[0]->ne[0];
#define SMALL_PIPE_CASE(P) case P: \
        if(spec.tile==64)small_gemm_kernel<64,P><<<dim3(m/64,1),128,0,stream>>>( \
            (const half *)t->src[0]->data,(const half *)t->src[1]->data,(float *)t->data,m,n,k); \
        else small_gemm_kernel<32,P><<<dim3(m/32,1),128,0,stream>>>( \
            (const half *)t->src[0]->data,(const half *)t->src[1]->data,(float *)t->data,m,n,k);break
        switch(spec.pipeline){SMALL_PIPE_CASE(4);default: SMALL_PIPE_CASE(0);}
#undef SMALL_PIPE_CASE
    } else if(magic(t)==WEIGHT_PREFETCH_MAGIC) {
        const auto & spec=*(const WeightPrefetchSpec *)userdata(t);
        auto & state=*(WeightPrefetchState *)spec.state;
        if(spec.join) {
            GGML_ASSERT(musaStreamWaitEvent(stream,state.join,0)==musaSuccess);
        } else {
            GGML_ASSERT(musaEventRecord(state.fork,stream)==musaSuccess);
            GGML_ASSERT(musaStreamWaitEvent(state.stream,state.fork,0)==musaSuccess);
            const int blocks=std::min(size_t(128),(spec.bytes+32767)/32768);
            prefetch_lines<<<blocks,WG,0,state.stream>>>((const unsigned *)t->src[1]->data,state.sink,spec.bytes/4);
            GGML_ASSERT(musaEventRecord(state.join,state.stream)==musaSuccess);
        }
    } else if (vector_pack(t)) {
        const auto * a = t->src[0];
        const size_t n4 = ggml_nelements(t)/4;
        pack_f16x4<<<(n4+WG-1)/WG, WG, 0, stream>>>(x, (__half2 *)t->data, n4,
            a->ne[0]/4, a->ne[1], a->ne[2], a->nb[1]/4, a->nb[2]/4, a->nb[3]/4);
    } else if (tiled_attention(t)) {
        const auto * a=t->src[0]; const auto * b=t->src[1];
        const dim3 grid((a->ne[1]+15)/16,(b->ne[1]+15)/16,a->ne[2]*a->ne[3]);
        attention_matmul_f32<<<grid,dim3(16,16),0,stream>>>(
            (const float *)a->data,(const float *)b->data,(float *)t->data,a->ne[0],a->ne[1],b->ne[1]);
    } else if (magic(t) == FUSED_MAGIC) {
        musa_vision_epilogue(t, ((const FusedSpec *) userdata(t))->kind == FusedKind::BiasGelu, stream_ptr);
    } else if(magic(t)==VISION_SCORE_MAGIC) {
        vision_scores_kernel<<<32,256,0,stream>>>(x,(float *)t->data);
    } else if (magic(t) == VISION_KEEP_MAGIC) {
        vision_keep<<<1, WG, 0, stream>>>(x, (int32_t *)t->data, t->src[0]->ne[0], h);
    } else if (t->op == GGML_OP_GLU) {
        const dim3 grid((h+WG-1)/WG, ggml_nrows(t));
        const bool swapped = ((const int32_t *) t->op_params)[1] != 0;
        if (t->type==GGML_TYPE_F16 && t->src[0]->type==GGML_TYPE_F32 && std::getenv("VLA_MUSA_GEGLU_FAST"))
            musa_geglu_fast_forward(t,stream_ptr);
        else if (t->type == GGML_TYPE_F16 && t->src[0]->type == GGML_TYPE_F32)
            merged_geglu<<<grid, WG, 0, stream>>>(x, (__half *)t->data, h, swapped);
        else if (t->type == GGML_TYPE_F16)
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
    } else if(magic(t)==F32_PATCH_MAGIC) {
        musa_f32_patch_forward(t,((const U8PatchSpec *)userdata(t))->patch,stream_ptr);
    } else if(magic(t)==U8_PATCH_MAGIC) {
        musa_u8_patch_forward(t,((const U8PatchSpec *)userdata(t))->patch,stream_ptr);
    } else if(magic(t)==ROPE_TABLE_MAGIC || magic(t)==ROPE_APPLY_MAGIC) {
        musa_cached_rope_forward(t,stream_ptr);
    } else if(magic(t)==FUSED_ACTION_LINEAR_MAGIC) {
        const auto & s=*(const FusedActionLinearSpec *)userdata(t);
        const auto * w=t->src[0]; const int hidden=s.input?w->ne[1]:w->ne[0];
        const size_t smem=(hidden+WG)*sizeof(float);
        if(s.packed)action_linear_kernel<true,true><<<64,WG,smem,stream>>>(
            (const float *)w->data,(const float *)t->src[1]->data,(const float *)t->src[2]->data,
            (const float *)t->src[3]->data,nullptr,(float *)t->data,hidden,w->ne[0],w->ne[1],t->src[1]->ne[1],s.eps,s.dt);
        else if(s.input)action_linear_kernel<true><<<t->ne[1],WG,smem,stream>>>(
            (const float *)w->data,(const float *)t->src[1]->data,(const float *)t->src[2]->data,
            (const float *)t->src[3]->data,nullptr,(float *)t->data,hidden,w->ne[0],w->ne[1],t->ne[1],s.eps,s.dt);
        else action_linear_kernel<false><<<t->ne[1],WG,smem,stream>>>(
            (const float *)w->data,(const float *)t->src[1]->data,(const float *)t->src[2]->data,
            (const float *)t->src[3]->data,(const float *)t->src[4]->data,(float *)t->data,
            hidden,w->ne[0],w->ne[1],t->ne[1],s.eps,s.dt);
    } else if (magic(t) == FUSED_ACTION_STEP_MAGIC) {
        const auto & spec=*(const FusedActionStepSpec *)userdata(t);
        action_step<<<(ggml_nelements(t)+WG-1)/WG,WG,0,stream>>>(x,
            (const float *)t->src[1]->data,(const float *)t->src[2]->data,
            (float *)t->data,ggml_nelements(t),h,spec.dt);
    } else if (magic(t) == FUSED_BIAS_ADARMS_MAGIC) {
        const auto & spec=*(const FusedResAdarmsSpec *)userdata(t);
        res_adarms<float,true><<<t->ne[1],WG,0,stream>>>(x,(const float *)t->src[1]->data,
            nullptr,(const float *)t->src[2]->data,(float *)t->data,h,t->ne[1],spec.eps);
    } else if (magic(t) == FUSED_RES_ADARMS_MAGIC) {
        const auto & spec = *(const FusedResAdarmsSpec *) userdata(t);
        if (t->src[1]->type == GGML_TYPE_F16)
            res_adarms<<<t->ne[1], WG, 0, stream>>>(x, (const __half *) t->src[1]->data,
                (const float *) t->src[2]->data, (const float *) t->src[3]->data,
                (float *) t->data, h, t->ne[1], spec.eps);
        else
            res_adarms<<<t->ne[1], WG, 0, stream>>>(x, (const float *) t->src[1]->data,
                (const float *) t->src[2]->data, (const float *) t->src[3]->data,
                (float *) t->data, h, t->ne[1], spec.eps);
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

__attribute__((visibility("default"))) bool musa_small_gemm_supported() {
    int device=-1;
    if(musaGetDevice(&device)!=musaSuccess)return false;
    static thread_local int cached=-1;
    static thread_local bool supported=false;
    if(cached!=device) {
        musaDeviceProp prop{};
        if(musaGetDeviceProperties(&prop,device)!=musaSuccess)return false;
        supported=prop.major==2 && prop.minor==2;cached=device;
    }
    return supported;
}

void musa_weight_prefetch_destroy(void * p) {
    if(!p)return;auto * state=(WeightPrefetchState *)p;
    if(state->stream)musaStreamSynchronize(state->stream);
    if(state->sink)musaFree(state->sink);
    if(state->fork)musaEventDestroy(state->fork);
    if(state->join)musaEventDestroy(state->join);
    if(state->stream)musaStreamDestroy(state->stream);
    delete state;
}
void * musa_weight_prefetch_create() {
    auto * state=new WeightPrefetchState;
    if(musaStreamCreateWithFlags(&state->stream,musaStreamNonBlocking)!=musaSuccess ||
       musaEventCreateWithFlags(&state->fork,musaEventDisableTiming)!=musaSuccess ||
       musaEventCreateWithFlags(&state->join,musaEventDisableTiming)!=musaSuccess ||
       musaMalloc(&state->sink,128*WG*sizeof(unsigned))!=musaSuccess) {
        musa_weight_prefetch_destroy(state);return nullptr;
    }
    return state;
}

bool musa_register_fused_ops() {
    static std::once_flag once;
    std::call_once(once, [] {
        ggml_musa_ext_supports = supports;
        ggml_musa_ext_forward = forward;
        ggml_musa_ext_release = release_attention;
    });
    return true;
}
} // namespace vla
