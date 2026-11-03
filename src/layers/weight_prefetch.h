// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ggml.h"
#include "layers/custom_params.h"
#include <cstdint>
namespace vla {
constexpr uint32_t WEIGHT_PREFETCH_MAGIC=0x46525057;
struct WeightPrefetchSpec {uint32_t magic=WEIGHT_PREFETCH_MAGIC;int join=0;void * state=nullptr;size_t bytes=0;};
inline void weight_prefetch_cpu(ggml_tensor *,int,int,void *) {}
inline ggml_tensor * weight_prefetch(ggml_context * C,ggml_tensor * x,ggml_tensor * dependency,
                                     const WeightPrefetchSpec & spec) {
 GGML_ASSERT(spec.state && ggml_is_contiguous(x));
 ggml_tensor * args[]={dependency};
 auto * t=ggml_custom_inplace(C,x,args,1,weight_prefetch_cpu,1,(void *)&spec);
 custom_parameter_snapshot(t,spec);return t;
}
}
