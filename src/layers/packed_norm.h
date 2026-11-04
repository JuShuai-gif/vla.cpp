// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "layers/fused_ew.h"
#include "layers/small_gemm.h"
namespace vla {
constexpr uint32_t PACKED_NORM_MAGIC=0x4e50524d;
struct PackedNormSpec {uint32_t magic=PACKED_NORM_MAGIC;float eps=1e-6f;int kind=0;int quantized=0;};
inline PackedNormSpec packed_norm_spec(const ggml_tensor * t) {
 PackedNormSpec s;std::memcpy(&s,(const char *)t->op_params+32,sizeof(s));return s;
}
inline bool packed_norm_enabled() {
#ifdef GGML_USE_MUSA
 return std::getenv("VLA_PI05_NORM_PACK") && std::getenv("VLA_MUSA_SMALL_GEMM") &&
        musa_small_gemm_supported();
#else
 return false;
#endif
}
// Byte-preserving typed views keep both allocations visible to ggml's scheduler.
inline ggml_tensor * packed_typed_view(ggml_context * C,ggml_tensor * blob,ggml_type type,
                                      int64_t h,int64_t rows,size_t offset) {
 GGML_ASSERT(blob->type==GGML_TYPE_I8);
 const size_t bytes=h*rows*ggml_type_size(type);
 auto * v=ggml_view_1d(C,blob,bytes,offset);
 v->type=type;v->ne[0]=h*rows;v->nb[0]=ggml_type_size(type);
 return ggml_reshape_2d(C,v,h,rows);
}
inline void packed_norm_cpu(ggml_tensor * t,int ith,int nth,void *) {
 const auto s=packed_norm_spec(t);const auto * x=t->src[0];
 GGML_ASSERT(!s.quantized);
 const int h=x->ne[0],rows=x->ne[1];
 const float * xp=(const float *)x->data,* op=s.kind?(const float *)t->src[1]->data:nullptr;
 const float * gate=s.kind==1?(const float *)t->src[2]->data:nullptr;
 const float * mod=(const float *)t->src[s.kind==1?3:s.kind==2?2:1]->data;
 auto * res=(float *)t->data;
 auto * norm=(ggml_fp16_t *)((char *)t->data+(s.kind?h*rows*4:0));
 for(int r=ith;r<64;r+=nth) {
  if(r>=rows){for(int i=0;i<h;++i)norm[r*h+i]=ggml_fp32_to_fp16(0);continue;}
  double sum=0;
  for(int i=0;i<h;++i) {
   const float a=s.kind==1?xp[r*h+i]+op[r*h+i]*gate[2*h+i]:
                 s.kind==2?xp[r*h+i]+op[i]:xp[r*h+i];
   if(s.kind)res[r*h+i]=a;sum+=(double)a*a;
  }
  const float inv=1.f/std::sqrt((float)(sum/h)+s.eps);
  for(int i=0;i<h;++i) {
   const float n=(s.kind?res[r*h+i]:xp[r*h+i])*inv;
   norm[r*h+i]=ggml_fp32_to_fp16((n+n*mod[i])+mod[h+i]);
  }
 }
}
inline ggml_tensor * packed_norm(ggml_context * C,ggml_tensor * x,ggml_tensor * o,
        ggml_tensor * gate,ggml_tensor * mod,float eps,int kind,ggml_tensor ** residual=nullptr,ggml_tensor * inverse=nullptr) {
 GGML_ASSERT(x->type==GGML_TYPE_F32 && ggml_is_contiguous(x) && x->ne[1]>0 && x->ne[1]<=64 &&
             x->ne[2]==1 && x->ne[3]==1 && kind>=0 && kind<=2 && mod->ne[0]==3*x->ne[0]);
 GGML_ASSERT(mod->type==GGML_TYPE_F32 && ggml_is_contiguous(mod) && (!kind || residual));
 GGML_ASSERT(kind!=1 || (gate && gate->type==GGML_TYPE_F32 && ggml_is_contiguous(gate)));
 GGML_ASSERT(!kind || (o && o->type==GGML_TYPE_F32 && ggml_is_contiguous(o)));
 GGML_ASSERT(kind!=1 || (ggml_are_same_shape(x,o) && gate && gate->ne[0]==3*x->ne[0]));
 GGML_ASSERT(kind!=2 || ggml_nelements(o)==x->ne[0]);
 GGML_ASSERT(!inverse || (kind==1 && inverse->type==GGML_TYPE_F32 && ggml_nelements(inverse)==x->ne[0] && x->ne[0]%32==0));
 ggml_tensor * args[]={x,kind?o:mod,kind==1?gate:mod,mod,inverse};
 static const uint32_t magic=PACKED_NORM_MAGIC;
 const size_t offset=kind?ggml_nelements(x)*4:0;
 auto * blob=ggml_custom_4d(C,GGML_TYPE_I8,offset+x->ne[0]*64*2+(inverse?(x->ne[0]+4)*64:0),1,1,1,args,
                           inverse?5:kind==1?4:kind==2?3:2,packed_norm_cpu,GGML_N_TASKS_MAX,(void *)&magic);
 custom_parameter_snapshot(blob,PackedNormSpec{PACKED_NORM_MAGIC,eps,kind,inverse?1:0});
 if(kind)*residual=packed_typed_view(C,blob,GGML_TYPE_F32,x->ne[0],x->ne[1],0);
 auto * pad=packed_typed_view(C,blob,GGML_TYPE_F16,x->ne[0],64,offset);
 return ggml_view_2d(C,pad,x->ne[0],x->ne[1],pad->nb[1],0);
}
constexpr uint32_t PACKED_GEGLU_MAGIC=0x4750504d;
inline void packed_geglu_cpu(ggml_tensor * t,int ith,int nth,void *) {
 const auto * x=(const float *)t->src[0]->data;auto * y=(ggml_fp16_t *)t->data;
 const int h=t->ne[0],rows=t->src[0]->ne[1];
 for(int64_t i=ith;i<h*64;i+=nth) {
  const int r=i/h,c=i%h;
  y[i]=ggml_fp32_to_fp16(r<rows?fused_gelu_tanh(x[r*2*h+c])*x[r*2*h+h+c]:0.f);
 }
}
inline ggml_tensor * packed_geglu(ggml_context * C,ggml_tensor * x) {
 GGML_ASSERT(x->type==GGML_TYPE_F32 && ggml_is_contiguous(x) && x->ne[0]%2==0 &&
             x->ne[1]>0 && x->ne[1]<=64 && x->ne[2]==1 && x->ne[3]==1);
 static const uint32_t magic=PACKED_GEGLU_MAGIC;ggml_tensor * inputs[]={x};
 auto * pad=ggml_custom_4d(C,GGML_TYPE_F16,x->ne[0]/2,64,1,1,inputs,1,
                          packed_geglu_cpu,GGML_N_TASKS_MAX,(void *)&magic);
 return ggml_view_2d(C,pad,pad->ne[0],x->ne[1],pad->nb[1],0);
}

}
