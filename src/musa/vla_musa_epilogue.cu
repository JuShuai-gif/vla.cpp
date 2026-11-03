#include <type_traits>
// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
#include "musa/vla_musa_ops.h"
#include "layers/rope_table.h"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/unary.cuh"

namespace vla {
namespace {
constexpr int WG = 256;
__global__ void make_rope_table(const int32_t * pos,float * table,int h,int rows,float scale) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=h/2*rows)return;
    const int row=i/(h/2),j=i%(h/2);
    const float theta=pos[row]*powf(scale,(float)j);
    table[2*i]=cosf(theta);table[2*i+1]=sinf(theta);
}
template<typename D>
__global__ void apply_rope_table(const float * x,const float * table,D * dst,int h,int heads,int rows,
                                size_t stride_head,size_t stride_row) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=h/2*heads*rows)return;
    const int j=i%(h/2),row=i/(h/2),head=row%heads,pos=row/heads;
    const float * src=x+pos*stride_row+head*stride_head;
    const float c=table[pos*h+2*j],s=table[pos*h+2*j+1];
    const float x0=src[j],x1=src[j+h/2];
    if constexpr (std::is_same<D,half>::value) {
        const float y0=__fmaf_rn(x0,c,-__fmul_rn(x1,s));
        const float y1=__fmaf_rn(x0,s,__fmul_rn(x1,c));
        dst[row*h+j]=__float2half_rn(y0);
        dst[row*h+j+h/2]=__float2half_rn(y1);
    } else {
        dst[row*h+j]=x0*c-x1*s;
        dst[row*h+j+h/2]=x0*s+x1*c;
    }
}
__global__ void fast_geglu(const float * x,half * y,int h,int n,bool swapped) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    const int c=i%h,r=i/h;
    const float g=x[2*r*h+c+(swapped?h:0)],u=x[2*r*h+c+(swapped?0:h)];
    y[i]=__float2half(ggml_cuda_op_gelu_single(g)*u);
}
__global__ void pack_u8_patch(const uint8_t * x,const half * lut,half * y,int ps,int S,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    const int grid=S/ps,pd=3*ps*ps,p=i%pd,tok=i/pd%(grid*grid),v=i/(pd*grid*grid);
    const int ch=p/(ps*ps),iy=(tok/grid)*ps+p/ps%ps,ix=(tok%grid)*ps+p%ps;
    y[i]=lut[x[(v*S*S+iy*S+ix)*3+ch]];
}
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
void musa_geglu_fast_forward(ggml_tensor * t,void * stream_ptr) {
    const int n=ggml_nelements(t);const bool swapped=((const int32_t *)t->op_params)[1]!=0;
    fast_geglu<<<(n+WG-1)/WG,WG,0,(musaStream_t)stream_ptr>>>((const float *)t->src[0]->data,
        (half *)t->data,t->ne[0],n,swapped);
}
void musa_u8_patch_forward(ggml_tensor * t,int patch,void * stream_ptr) {
    const int n=ggml_nelements(t);
    pack_u8_patch<<<(n+WG-1)/WG,WG,0,(musaStream_t)stream_ptr>>>((const uint8_t *)t->src[0]->data,
        (const half *)t->src[1]->data,(half *)t->data,patch,t->src[0]->ne[1],n);
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
void musa_cached_rope_forward(ggml_tensor * t,void * stream_ptr) {
    const auto stream=(musaStream_t)stream_ptr;
    struct Params {void * fun;int tasks;void * userdata;} p;
    std::memcpy(&p,t->op_params,sizeof(p));
    uint32_t magic;std::memcpy(&magic,p.userdata,sizeof(magic));
    const int h=t->ne[0];
    if(magic==ROPE_TABLE_MAGIC) {
        const auto & spec=*(const RopeTableSpec *)p.userdata;
        const float scale=powf(spec.base,-2.f/h);
        const int n=h/2*t->ne[1];
        make_rope_table<<<(n+WG-1)/WG,WG,0,stream>>>((const int32_t *)t->src[0]->data,
            (float *)t->data,h,t->ne[1],scale);
    } else {
        const auto * x=t->src[0];const int n=h/2*t->ne[1]*t->ne[2];
        if(t->type==GGML_TYPE_F16)
            apply_rope_table<<<(n+WG-1)/WG,WG,0,stream>>>((const float *)x->data,
                (const float *)t->src[1]->data,(half *)t->data,h,t->ne[1],t->ne[2],x->nb[1]/4,x->nb[2]/4);
        else
            apply_rope_table<<<(n+WG-1)/WG,WG,0,stream>>>((const float *)x->data,
                (const float *)t->src[1]->data,(float *)t->data,h,t->ne[1],t->ne[2],x->nb[1]/4,x->nb[2]/4);
    }
}
} // namespace vla
