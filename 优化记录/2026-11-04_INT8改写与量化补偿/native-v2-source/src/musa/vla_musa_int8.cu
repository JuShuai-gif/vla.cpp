// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
#include "musa/vla_musa_ops.h"
#include "layers/musa_int8.h"
#include "musa/int8_kernels.cuh"
#include <cstring>
namespace vla {
static uint32_t kind(const ggml_tensor * t){
    if(!t || t->op!=GGML_OP_CUSTOM)return 0;
    struct Params {void *fun;int tasks;void*ud;} p;
    std::memcpy(&p,t->op_params,sizeof(p));uint32_t k=0;
    if(p.ud)std::memcpy(&k,p.ud,4);return k;
}
bool musa_int8_supports(const ggml_tensor * t){
    const auto k=kind(t);
    if(k!=MUSA_INT8_PACK && k!=MUSA_INT8_GEMM && k!=MUSA_INT8_GEGLU)return false;
    if(!musa_small_gemm_supported() || !ggml_is_contiguous(t))return false;
    const auto *a=t->src[0],*b=t->src[1];
    if(!a || !b || !ggml_is_contiguous(a) || !ggml_is_contiguous(b))return false;
    if(k==MUSA_INT8_PACK)
        return t->type==GGML_TYPE_I8 && (a->type==GGML_TYPE_F32 || a->type==GGML_TYPE_F16) &&
          a->ne[0]%32==0 && a->ne[2]==1 && a->ne[3]==1 && b->type==GGML_TYPE_F32 &&
          ggml_nelements(b)==a->ne[0] && t->ne[0]%(a->ne[0]+4)==0;
    if(k==MUSA_INT8_GEGLU){
        const auto *x=t->src[3];
        if(!x || x->ne[1]<1 || x->ne[1]>64 || t->ne[1]!=64 || t->src[4])return false;
    }
    if(t->src[4]){
        const auto *x=t->src[3],*extra=t->src[4],*indices=t->src[5];
        if(!x || !indices || !ggml_is_contiguous(x) || !ggml_is_contiguous(extra) || !ggml_is_contiguous(indices) ||
           (x->type!=GGML_TYPE_F16 && x->type!=GGML_TYPE_F32) || extra->type!=GGML_TYPE_F16 ||
           indices->type!=GGML_TYPE_I32 || extra->ne[0]!=ggml_nelements(indices) ||
           extra->ne[1]!=t->ne[0] || x->ne[0]!=a->ne[0] || x->ne[1]!=t->ne[1])return false;
    }
    return t->type==(k==MUSA_INT8_GEGLU?GGML_TYPE_F16:GGML_TYPE_F32) && a->type==GGML_TYPE_I8 && a->ne[0]%32==0 &&
        a->ne[1]%128==0 && t->ne[0]*(k==MUSA_INT8_GEGLU?2:1)==a->ne[1] && b->type==GGML_TYPE_F32 &&
        ggml_nelements(b)==a->ne[1] && t->src[2] && t->src[2]->type==GGML_TYPE_I8 &&
        ggml_is_contiguous(t->src[2]) && t->src[2]->ne[0]%(a->ne[0]+4)==0 &&
        t->src[2]->ne[0]/(a->ne[0]+4)>=t->ne[1];
}
bool musa_int8_forward(ggml_tensor *t,void *stream){
    const auto st=(musaStream_t)stream;const auto k=kind(t);const auto *a=t->src[0];
    if(k==MUSA_INT8_PACK){
        const int K=a->ne[0],N=a->ne[1],NP=t->ne[0]/(K+4);
        auto *q=(signed char*)t->data;auto *s=(float*)(q+(size_t)NP*K);
        const auto *inv=(const float*)t->src[1]->data;
        if(a->type==GGML_TYPE_F16)int8_native::pack<<<NP,128,0,st>>>((const half*)a->data,inv,q,s,K,N);
        else int8_native::pack<<<NP,128,0,st>>>((const float*)a->data,inv,q,s,K,N);
    }else if(k==MUSA_INT8_GEMM || k==MUSA_INT8_GEGLU){
        const int K=a->ne[0],M=a->ne[1],N=k==MUSA_INT8_GEGLU?t->src[3]->ne[1]:t->ne[1],NP=t->src[2]->ne[0]/(K+4);
        const auto *w=(const signed char*)a->data,*x=(const signed char*)t->src[2]->data;
        const auto *sw=(const float*)t->src[1]->data,*sx=(const float*)(x+(size_t)NP*K);
        if(k==MUSA_INT8_GEGLU){
            int8_native::row_wmma<64,64,1,false,true><<<dim3(M/64,1),128,0,st>>>(w,x,sw,sx,(float*)t->data,M,N,K);
        }else if(t->src[4]){
            const auto *extra=(const half*)t->src[4]->data;const auto *indices=(const int*)t->src[5]->data;
            const int R=t->src[4]->ne[0];const bool half_input=t->src[3]->type==GGML_TYPE_F16;
            if(N<=64)int8_native::row_wmma<32,64,1,true><<<dim3(M/32,NP/64),128,0,st>>>(w,x,sw,sx,(float*)t->data,M,N,K,extra,t->src[3]->data,indices,R,half_input);
            else int8_native::row_wmma<64,160,2,true><<<dim3(M/128,NP/160),256,0,st>>>(w,x,sw,sx,(float*)t->data,M,N,K,extra,t->src[3]->data,indices,R,half_input);
        }else{
            if(N<=64)int8_native::row_wmma<32,64,1><<<dim3(M/32,NP/64),128,0,st>>>(w,x,sw,sx,(float*)t->data,M,N,K);
            else int8_native::row_wmma<64,160,2><<<dim3(M/128,NP/160),256,0,st>>>(w,x,sw,sx,(float*)t->data,M,N,K);
        }
    }else return false;
    return musaGetLastError()==musaSuccess;
}
}
