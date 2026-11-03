#pragma once
#include "ggml.h"
#include "layers/custom_params.h"
#include "layers/packed_norm.h"
#include <cmath>
#include <vector>
namespace vla {
constexpr uint32_t FUSED_ACTION_LINEAR_MAGIC=0x4e4c4341;
struct FusedActionLinearSpec {
    uint32_t magic=FUSED_ACTION_LINEAR_MAGIC;
    int input=0;
    float eps=1e-6f,dt=0;
    int packed=0;
};
inline void action_linear_cpu(ggml_tensor * dst,int ith,int nth,void * ud) {
    const auto & s=*(const FusedActionLinearSpec *)ud;
    const auto * w=(const float *)dst->src[0]->data;
    const auto * x=(const float *)dst->src[1]->data;
    const auto * b=(const float *)dst->src[2]->data;
    const auto * mod=(const float *)dst->src[3]->data;
    const int h=s.input?dst->src[0]->ne[1]:dst->src[0]->ne[0];
    const int cols=dst->src[0]->ne[0],outputs=dst->src[0]->ne[1],rows=dst->src[1]->ne[1];
    auto * y=(float *)dst->data;std::vector<float> values(h);
    for(int row=ith;row<(s.packed?64:rows);row+=nth) {
        auto * half=s.packed?(ggml_fp16_t *)(y+h*rows):nullptr;
        if(row>=rows){for(int i=0;i<h;++i)half[row*h+i]=ggml_fp32_to_fp16(0.f);continue;}
        double sum=0;
        for(int i=0;i<h;++i) {
            float v=0;
            if(s.input) {
                v=0;
                for(int j=0;j<cols;++j)v=std::fma(w[i*cols+j],x[row*cols+j],v);
                v+=b[i];y[row*h+i]=v;
            } else v=x[row*h+i];
            values[i]=v;sum+=(double)v*v;
        }
        const float inv=1.f/std::sqrt((float)(sum/h)+s.eps);
        for(int i=0;i<h;++i){const float n=values[i]*inv;values[i]=(n+n*mod[i])+mod[h+i];}
        if(s.input)for(int i=0;i<h;++i) {
            if(s.packed)half[row*h+i]=ggml_fp32_to_fp16(values[i]);
            else y[h*rows+row*h+i]=values[i];
        }
        else for(int o=0;o<outputs;++o) {
            float v=0;for(int i=0;i<h;++i)v=std::fma(w[o*h+i],values[i],v);
            const float velocity=v+b[o],step=velocity*s.dt;
            y[row*outputs+o]=((const float*)dst->src[4]->data)[row*outputs+o]+step;
        }
    }
}
inline ggml_tensor * fused_action_linear(ggml_context * C,ggml_tensor * w,ggml_tensor * x,
        ggml_tensor * b,ggml_tensor * mod,ggml_tensor * noise,const FusedActionLinearSpec & spec,
        ggml_tensor ** residual=nullptr) {
    GGML_ASSERT(w->type==GGML_TYPE_F32 && x->type==GGML_TYPE_F32 && b->type==GGML_TYPE_F32);
    GGML_ASSERT(w->ne[2]==1 && w->ne[3]==1 && x->ne[2]==1 && x->ne[3]==1);
    GGML_ASSERT(w->ne[0]==x->ne[0] && b->ne[0]==w->ne[1]);
    const int64_t h=spec.input?w->ne[1]:w->ne[0];
    GGML_ASSERT(mod->type==GGML_TYPE_F32 && ggml_nelements(mod)==3*h);
    if(!spec.input)GGML_ASSERT(noise && noise->type==GGML_TYPE_F32 && noise->ne[0]==w->ne[1] && noise->ne[1]==x->ne[1]);
    ggml_tensor * args[]={w,x,b,mod,noise};
    GGML_ASSERT(!spec.packed || (spec.input && x->ne[1]>0 && x->ne[1]<=64));
    auto * t=ggml_custom_4d(C,spec.packed?GGML_TYPE_I8:GGML_TYPE_F32,
                           spec.packed?h*x->ne[1]*4+h*64*2:w->ne[1],
                           spec.packed?1:x->ne[1],spec.packed?1:spec.input?2:1,1,args,
                           spec.input?4:5,action_linear_cpu,GGML_N_TASKS_MAX,(void *)&spec);
    custom_parameter_snapshot(t,spec);
    ggml_set_name(t,spec.input?"action_in_linear_adarms":"action_out_adarms_linear_step");
    if(!spec.input)return t;
    GGML_ASSERT(residual);
    if(spec.packed) {
        *residual=packed_typed_view(C,t,GGML_TYPE_F32,h,x->ne[1],0);
        auto * pad=packed_typed_view(C,t,GGML_TYPE_F16,h,64,h*x->ne[1]*4);
        return ggml_view_2d(C,pad,h,x->ne[1],pad->nb[1],0);
    }
    *residual=ggml_view_2d(C,t,h,x->ne[1],t->nb[1],0);
    return ggml_view_2d(C,t,h,x->ne[1],t->nb[1],t->nb[2]);
}
}
