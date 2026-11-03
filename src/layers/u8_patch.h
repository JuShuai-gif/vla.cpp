// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ggml.h"
#include "layers/custom_params.h"
#include <cstdint>
namespace vla {
constexpr uint32_t U8_PATCH_MAGIC=0x38505455;
struct U8PatchSpec {uint32_t magic=U8_PATCH_MAGIC;int patch=14;};
inline void u8_patch_cpu(ggml_tensor * t,int ith,int nth,void * ud) {
 const int ps=((const U8PatchSpec *)ud)->patch,S=t->src[0]->ne[1],grid=S/ps,pd=3*ps*ps;
 const auto * x=(const uint8_t *)t->src[0]->data;const auto * lut=(const ggml_fp16_t *)t->src[1]->data;
 auto * y=(ggml_fp16_t *)t->data;
 for(int64_t i=ith;i<ggml_nelements(t);i+=nth) {
  const int p=i%pd,tok=i/pd%(grid*grid),v=i/(pd*grid*grid);
  const int ch=p/(ps*ps),py=p/ps%ps,px=p%ps;
  const int iy=(tok/grid)*ps+py,ix=(tok%grid)*ps+px;
  y[i]=lut[x[((int64_t)v*S*S+iy*S+ix)*3+ch]];
 }
}
inline ggml_tensor * u8_patch(ggml_context * C,ggml_tensor * image,ggml_tensor * lut,const U8PatchSpec & spec) {
 GGML_ASSERT(image->type==GGML_TYPE_I8 && ggml_is_contiguous(image) && image->ne[0]==3 &&
             image->ne[1]==image->ne[2] && image->ne[1]%spec.patch==0);
 GGML_ASSERT(lut->type==GGML_TYPE_F16 && ggml_is_contiguous(lut) && ggml_nelements(lut)==256);
 int64_t grid=image->ne[1]/spec.patch;ggml_tensor * args[]={image,lut};
 auto * t=ggml_custom_4d(C,GGML_TYPE_F16,3*spec.patch*spec.patch,grid*grid,image->ne[3],1,
                        args,2,u8_patch_cpu,GGML_N_TASKS_MAX,(void *)&spec);
 custom_parameter_snapshot(t,spec);return t;
}
}
