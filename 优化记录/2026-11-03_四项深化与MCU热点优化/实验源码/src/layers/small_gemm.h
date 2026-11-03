// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "layers/custom_params.h"
#include "musa/vla_musa_ops.h"
#include <cmath>
#include <cstdlib>
namespace vla {
constexpr uint32_t SMALL_GEMM_MAGIC=0x4d47534d,SMALL_PACK_MAGIC=0x4b50534d;
struct SmallGemmSpec {uint32_t magic=SMALL_GEMM_MAGIC;int tile=32;int pipeline=0;};
inline void small_pack_cpu(ggml_tensor * t,int ith,int nth,void *) {
 const auto * x=t->src[0];auto * y=(ggml_fp16_t *)t->data;const int64_t n=ggml_nelements(x);
 for(int64_t i=ith;i<ggml_nelements(t);i+=nth)
  y[i]=i>=n ? ggml_fp32_to_fp16(0) : x->type==GGML_TYPE_F16 ?
       ((const ggml_fp16_t *)x->data)[i] : ggml_fp32_to_fp16(((const float *)x->data)[i]);
}
inline void small_gemm_cpu(ggml_tensor * t,int ith,int nth,void *) {
 const auto * w=(const ggml_fp16_t *)t->src[0]->data,*x=(const ggml_fp16_t *)t->src[1]->data;
 auto * y=(float *)t->data;const int k=t->src[0]->ne[0],m=t->ne[0];
 for(int64_t i=ith;i<ggml_nelements(t);i+=nth) {
  const int row=i%m,col=i/m;float sum=0;
  for(int p=0;p<k;++p)sum=std::fma(ggml_fp16_to_fp32(w[(int64_t)row*k+p]),ggml_fp16_to_fp32(x[(int64_t)col*k+p]),sum);
  y[i]=sum;
 }
}
inline ggml_tensor * small_gemm(ggml_context * C,ggml_tensor * w,ggml_tensor * x,int tile=32) {
 GGML_ASSERT(w->type==GGML_TYPE_F16 && (x->type==GGML_TYPE_F16 || x->type==GGML_TYPE_F32) &&
             ggml_is_contiguous(w) && ggml_is_contiguous(x) && w->ne[0]==x->ne[0] &&
             w->ne[0]%16==0 && w->ne[1]%tile==0 && x->ne[1]>0 && x->ne[1]<=64 &&
             w->ne[2]==1 && w->ne[3]==1 && x->ne[2]==1 && x->ne[3]==1 && (tile==32 || tile==64));
 static const uint32_t pack=SMALL_PACK_MAGIC;ggml_tensor * inputs[]={x};
 size_t view_offset=0;std::memcpy(&view_offset,x->op_params,sizeof(view_offset));
 auto * padded=x->op==GGML_OP_VIEW && x->type==GGML_TYPE_F16 && view_offset==0 &&
  x->src[0] && x->src[0]->type==GGML_TYPE_F16 && ggml_is_contiguous(x->src[0]) &&
  x->src[0]->ne[0]==x->ne[0] && x->src[0]->ne[1]==64 &&
  x->src[0]->ne[2]==1 && x->src[0]->ne[3]==1 ? x->src[0] :
  ggml_custom_4d(C,GGML_TYPE_F16,x->ne[0],64,1,1,inputs,1,small_pack_cpu,GGML_N_TASKS_MAX,(void *)&pack);
 static const SmallGemmSpec specs[2][4]={
  {{SMALL_GEMM_MAGIC,32,0},{SMALL_GEMM_MAGIC,32,1},{SMALL_GEMM_MAGIC,32,4},{SMALL_GEMM_MAGIC,32,8}},
  {{SMALL_GEMM_MAGIC,64,0},{SMALL_GEMM_MAGIC,64,1},{SMALL_GEMM_MAGIC,64,4},{SMALL_GEMM_MAGIC,64,8}}};
 const char * value=std::getenv("VLA_MUSA_SMALL_PIPE");
 const int mode=value?std::atoi(value):0;
 const auto & spec=specs[tile==64][mode==1?1:mode==4?2:mode==8?3:0];ggml_tensor * args[]={w,padded};
 auto * out=ggml_custom_4d(C,GGML_TYPE_F32,w->ne[1],x->ne[1],1,1,args,2,small_gemm_cpu,GGML_N_TASKS_MAX,(void *)&spec);
 custom_parameter_snapshot(out,spec);return out;
}
inline ggml_tensor * maybe_small_gemm(ggml_context * C,ggml_tensor * w,ggml_tensor * x) {
#ifdef GGML_USE_MUSA
 const char * flag=std::getenv("VLA_MUSA_SMALL_GEMM");
 if(flag && musa_small_gemm_supported() && w->buffer && !ggml_backend_buffer_is_host(w->buffer) &&
    w->type==GGML_TYPE_F16 && (x->type==GGML_TYPE_F16 || x->type==GGML_TYPE_F32) &&
    ggml_is_contiguous(w) && ggml_is_contiguous(x) && w->ne[0]==x->ne[0] &&
    w->ne[0]%16==0 && w->ne[1]%32==0 && x->ne[1]>0 && x->ne[1]<=64 &&
    w->ne[2]==1 && w->ne[3]==1 && x->ne[2]==1 && x->ne[3]==1) {
  int tile=std::atoi(flag);if(tile!=32 && tile!=64)tile=w->ne[1]==1024 && w->ne[0]==2048?64:32;
  if(w->ne[1]%tile)tile=32;
  return small_gemm(C,w,x,tile);
 }
#endif
 return nullptr;
}
}
