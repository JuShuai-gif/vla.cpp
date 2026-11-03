#pragma once
#include "ggml.h"
#include <cstring>
#include <type_traits>
namespace vla {
// ggml CUSTOM stores callback/task/userdata in the first 24 bytes. Graph
// compatibility must also see constants behind userdata, even at reused addresses.
template<class Spec>
inline void custom_parameter_snapshot(ggml_tensor * t,const Spec & spec) {
    static_assert(std::is_trivially_copyable<Spec>::value,"custom constants must be trivially copyable");
    static_assert(32+sizeof(Spec)<=sizeof(t->op_params),"custom constants exceed reserved params");
    GGML_ASSERT(t->op==GGML_OP_CUSTOM);
    std::memcpy((char *)t->op_params+32,&spec,sizeof(spec));
}
}
