// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "loader.h"
#include "ggml.h"
#include "layers/packed_norm.h"
namespace vla {
constexpr uint32_t MUSA_INT8_PACK=0x38504d49,MUSA_INT8_GEMM=0x38474d49,MUSA_INT8_GEGLU=0x384c4d49;
inline void musa_int8_cpu(ggml_tensor*,int,int,void*) {
    GGML_ABORT("experimental INT8 requires the MUSA MP22 backend; CPU fallback refused");
}
inline ggml_tensor * musa_int8_linear(ggml_context * C,const WeightLoader::QWeight & q,ggml_tensor * x,bool geglu=false) {
    const int64_t K=q.w->ne[0],M=q.w->ne[1],N=x->ne[1],tile=N<=64?64:160,NP=(N+tile-1)/tile*tile;
    GGML_ASSERT(q.inverse && q.bits==8 && K%32==0 && M%128==0 && N>0 && N<=4096 &&
        x->ne[0]==K && x->ne[2]==1 && x->ne[3]==1 && ggml_is_contiguous(x) &&
        (x->type==GGML_TYPE_F16 || x->type==GGML_TYPE_F32));
    GGML_ASSERT(!geglu || (!q.extra && N<=64));
    static const uint32_t pack=MUSA_INT8_PACK,gemm=MUSA_INT8_GEMM,glu=MUSA_INT8_GEGLU;
    ggml_tensor * args[]={x,q.inverse};
    ggml_tensor * blob=nullptr;
    auto * producer=x->view_src;
    if(producer && producer->op==GGML_OP_CUSTOM){
        const auto spec=packed_norm_spec(producer);
        if(spec.magic==PACKED_NORM_MAGIC && spec.quantized && spec.kind==1 &&
           producer->src[4]==q.inverse && N<=64)
            blob=ggml_view_1d(C,producer,(K+4)*64,K*N*4+K*64*2);
    }
    if(!blob)blob=ggml_custom_4d(C,GGML_TYPE_I8,(K+4)*NP,1,1,1,args,2,musa_int8_cpu,1,(void*)&pack);
    ggml_format_name(blob,"%s.activations",q.w->name);
    ggml_tensor * inputs[]={q.w,q.s,blob,x,q.extra,q.indices};
    auto * out=ggml_custom_4d(C,geglu?GGML_TYPE_F16:GGML_TYPE_F32,geglu?M/2:M,geglu?64:N,1,1,inputs,q.extra?6:geglu?4:3,musa_int8_cpu,1,(void*)(geglu?&glu:&gemm));
    ggml_format_name(out,"%s.native",q.w->name);
    return geglu?ggml_view_2d(C,out,M/2,N,out->nb[1],0):out;
}
}
