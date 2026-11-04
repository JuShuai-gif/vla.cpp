// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <musa_runtime.h>
#include <musa_fp16.h>
#include <mma.h>
#define INT8_WARP 128
#define PREFETCH_STEPS 2
namespace vla { namespace int8_native {
template<int TM,int TN,int NW=1,bool Compensate=false,bool Geglu=false> __global__ void row_wmma(const signed char*w,const signed char*x,const float*sw,const float*sx,float*y,int M,int N,int K,const half *extra=nullptr,const void *original=nullptr,const int *indices=nullptr,int R=0,bool half_input=false) {
    using namespace mtmusa::wmma;
    constexpr int MT=TM/32,NT=TN/32,LD=TN+4;
    int warp=threadIdx.x/INT8_WARP,row=(blockIdx.x*NW+warp)*(Geglu?TM/2:TM),col=blockIdx.y*TN,t=threadIdx.x%INT8_WARP;
    fragment<accumulator,32,32,32,int> cf[MT][NT];
    #pragma unroll
    for(int i=0;i<MT;++i){
      #pragma unroll
      for(int j=0;j<NT;++j)fill_fragment(cf[i][j],0);
    }
    // Group independent loads before MMA to expose more memory-level parallelism.
    for(int kb=0;kb<K;kb+=32*PREFETCH_STEPS){
      fragment<matrix_a,32,32,32,signed char,row_major> af[PREFETCH_STEPS][MT];
      fragment<matrix_b,32,32,32,signed char,col_major> bf[PREFETCH_STEPS][NT];
      #pragma unroll
      for(int u=0;u<PREFETCH_STEPS;++u){
        if(kb+32*u<K){
          #pragma unroll
          for(int i=0;i<MT;++i){const int *p=(const int*)(w+((size_t)(row/32+(Geglu?(i%(MT/2)+(i/(MT/2))*(M/64)):i))*(K/32)+kb/32+u)*1024);
              ((int*)&af[u][i])[0]=p[t];((int*)&af[u][i])[1]=p[128+t];}
          #pragma unroll
          for(int j=0;j<NT;++j){const int *p=(const int*)(x+((size_t)(col/32+j)*(K/32)+kb/32+u)*1024);
              ((int*)&bf[u][j])[0]=p[t];((int*)&bf[u][j])[1]=p[128+t];}
        }
      }
      #pragma unroll
      for(int u=0;u<PREFETCH_STEPS;++u){
        if(kb+32*u<K){
          #pragma unroll
          for(int i=0;i<MT;++i){
            #pragma unroll
            for(int j=0;j<NT;++j)mma_sync(cf[i][j],af[u][i],bf[u][j],cf[i][j]);
          }
        }
      }
    }
    // MP22 fragment layout is verified by probe_musa_int8_layout.cpp.
    // Gate and up share a warp so GEGLU needs no intermediate tensor.
    #pragma unroll
    for(int i=0;i<(Geglu?MT/2:MT);++i){
      #pragma unroll
      for(int j=0;j<NT;++j){
        #pragma unroll
        for(int v=0;v<8;++v){
          const int m=row+i*32+((t&31)>>3)+4*(t>>6)+8*(v>>1);
          const int n=col+j*32+(t&7)+8*((t>>5)&1)+16*(v&1);
          float value=0;
          if(n<N){
            value=float(cf[i][j].x[v])*sw[m]*sx[n];
            if constexpr(Compensate){
              float correction=0;
              for(int e=0;e<R;++e){
                const size_t pos=(size_t)n*K+indices[e];
                const float input=half_input?float(((const half*)original)[pos]):float(__float2half(((const float*)original)[pos]));
                correction=__fmaf_rn(float(extra[(size_t)m*R+e]),input,correction);
              }
              value+=correction;
            }
            if constexpr(Geglu){
              const float up=float(cf[i+MT/2][j].x[v])*sw[m+M/2]*sx[n];
              const float gelu=0.5f*value*(1.0f+tanhf(0.7978845608028654f*value*(1.0f+0.044715f*value*value)));
              value=gelu*up;
            }
          }
          if constexpr(Geglu)((half*)y)[(size_t)n*(M/2)+m]=__float2half_rn(value);
          else if(n<N)y[(size_t)n*M+m]=value;
        }
      }
    }
}

template<typename T> __global__ void pack(const T*x,const float*inverse,signed char*q,float*s,int K,int N) {
    const int n=blockIdx.x,t=threadIdx.x;float mx=0;
    for(int k=t;k<K;k+=128)mx=fmaxf(mx,n<N?fabsf(float(x[(size_t)n*K+k])*inverse[k]):0.f);
    __shared__ float reduce[128];reduce[t]=mx;__syncthreads();
    for(int d=64;d;d>>=1){if(t<d)reduce[t]=fmaxf(reduce[t],reduce[t+d]);__syncthreads();}
    const float scale=reduce[0]>0?__fdiv_rn(reduce[0],127.f):1.f;
    if(t==0)s[n]=scale;
    for(int k=t;k<K;k+=128){
        const size_t i=((size_t)(n/32)*(K/32)+k/32)*1024+(n%32/16)*512+((n%16)*4+(k%16)/4+(k%32/16)*64)*4+k%4;
        q[i]=n<N?(signed char)rintf(fminf(127.f,fmaxf(-127.f,__fdiv_rn(float(x[(size_t)n*K+k])*inverse[k],scale)))):0;
    }
}

}}
#undef PREFETCH_STEPS
#undef INT8_WARP
