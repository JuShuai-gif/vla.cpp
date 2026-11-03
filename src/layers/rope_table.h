#pragma once
#include "ggml.h"
#include "layers/custom_params.h"
#include <cmath>
#include <cstring>
namespace vla {
constexpr uint32_t ROPE_TABLE_MAGIC=0x42545052, ROPE_APPLY_MAGIC=0x50415052;
struct RopeTableSpec {uint32_t magic=ROPE_TABLE_MAGIC;float base=10000.f;};
inline bool rope_table_is_apply(const ggml_tensor * t) {
    if(t->op!=GGML_OP_CUSTOM)return false;
    struct Params {void * fun;int tasks;void * userdata;} p;
    std::memcpy(&p,t->op_params,sizeof(p));
    uint32_t magic=0;if(p.userdata)std::memcpy(&magic,p.userdata,sizeof(magic));
    return magic==ROPE_APPLY_MAGIC;
}
inline void rope_table_cpu(ggml_tensor * t,int ith,int nth,void * ud) {
    const auto & spec=*(const RopeTableSpec *)ud;
    const auto * pos=(const int32_t *)t->src[0]->data;auto * dst=(float *)t->data;
    const int h=t->ne[0];const float scale=std::pow(spec.base,-2.f/h);
    for(int64_t i=ith;i<ggml_nelements(t)/2;i+=nth) {
        const int row=i/(h/2),j=i%(h/2);
        const float theta=pos[row]*std::pow(scale,(float)j);
        dst[2*i]=std::cos(theta);dst[2*i+1]=std::sin(theta);
    }
}
inline void rope_apply_cpu(ggml_tensor * t,int ith,int nth,void *) {
    const auto * x=t->src[0];const auto * table=(const float *)t->src[1]->data;
    const int h=t->ne[0],heads=t->ne[1];
    for(int64_t row=ith;row<heads*t->ne[2];row+=nth) {
        const int pos=row/heads,head=row%heads;
        const auto * a=(const float *)((const char *)x->data+pos*x->nb[2]+head*x->nb[1]);
        auto * out=(char *)t->data+pos*t->nb[2]+head*t->nb[1];
        for(int j=0;j<h/2;++j) {
            const float c=table[pos*h+2*j],s=table[pos*h+2*j+1];
            const float v0=a[j]*c-a[j+h/2]*s,v1=a[j]*s+a[j+h/2]*c;
            if(t->type==GGML_TYPE_F16){((ggml_fp16_t *)out)[j]=ggml_fp32_to_fp16(v0);((ggml_fp16_t *)out)[j+h/2]=ggml_fp32_to_fp16(v1);}
            else {((float *)out)[j]=v0;((float *)out)[j+h/2]=v1;}
        }
    }
}
inline ggml_tensor * rope_table(ggml_context * C,ggml_tensor * positions,int64_t h,const RopeTableSpec & spec) {
    GGML_ASSERT(positions->type==GGML_TYPE_I32 && ggml_is_contiguous(positions) && h>0 && h%2==0);
    GGML_ASSERT(spec.base>0 && std::isfinite(spec.base));
    ggml_tensor * args[]={positions};
    auto * t=ggml_custom_4d(C,GGML_TYPE_F32,h,positions->ne[0],1,1,args,1,rope_table_cpu,GGML_N_TASKS_MAX,(void *)&spec);
    custom_parameter_snapshot(t,spec);return t;
}
inline ggml_tensor * rope_apply(ggml_context * C,ggml_tensor * x,ggml_tensor * table) {
    GGML_ASSERT(x->type==GGML_TYPE_F32 && x->ne[3]==1 && x->nb[0]==4);
    GGML_ASSERT(table->type==GGML_TYPE_F32 && table->ne[0]==x->ne[0] && table->ne[1]==x->ne[2]);
    static const uint32_t spec=ROPE_APPLY_MAGIC;
    ggml_tensor * args[]={x,table};
    return ggml_custom_4d(C,GGML_TYPE_F32,x->ne[0],x->ne[1],x->ne[2],1,args,2,
                          rope_apply_cpu,GGML_N_TASKS_MAX,(void *)&spec);
}
}
