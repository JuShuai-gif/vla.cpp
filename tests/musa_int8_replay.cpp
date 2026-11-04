// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
// Offline operator replay only; transfers and static packing are outside timing.
#define main unused_row_quant_benchmark_main
#include "bench_musa_row_quant_direct.cpp"
#undef main

__global__ void quantize_smooth(const float*x,const float*inverse,signed char*q,float*s,int K,int N,int NP) {
    int n=blockIdx.x,t=threadIdx.x;float mx=0;
    for(int k=t;k<K;k+=128)mx=fmaxf(mx,n<N?fabsf(x[(size_t)n*K+k]*inverse[k]):0.f);
    __shared__ float reduce[128];reduce[t]=mx;__syncthreads();
    for(int d=64;d;d>>=1){if(t<d)reduce[t]=fmaxf(reduce[t],reduce[t+d]);__syncthreads();}
    float scale=reduce[0]>0?__fdiv_rn(reduce[0],127.f):1.f;
    if(t==0)s[n]=scale;
    for(int k=t;k<K;k+=128){
        const size_t pos=((size_t)(n/32)*(K/32)+k/32)*1024+(n%32/16)*512+((n%16)*4+(k%16)/4+(k%32/16)*64)*4+k%4;
        q[pos]=n<N?(signed char)rintf(fminf(127.f,fmaxf(-127.f,__fdiv_rn(x[(size_t)n*K+k]*inverse[k],scale)))):0;
    }
}
__global__ void gather_extra(const float*x,const int*indices,half*y,int K,int N,int R){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N*R)y[i]=__float2half(x[(size_t)(i/R)*K+indices[i%R]]);
}
struct Replay {
    int K,M,R,capacity,TN;
    signed char *w,*q;float *sw,*inverse,*x,*sx,*y;half *extra,*xe,*wf,*xf;int *indices;
    mublasHandle_t blas; musaEvent_t begin,end;
};
template<class T>static void alloc(T**p,size_t n){check(musaMalloc(p,n*sizeof(T)));}
extern "C" void *int8_replay_create(const signed char*codes,const float*scales,const float*inverse,
                                   const half*extra,const int*indices,const half*original,
                                   int K,int M,int maxN,int R) {
    if(K<32||K%32||M<128||M%128||maxN<1||R<0||R>=K)return nullptr;
    auto *p=new Replay{};p->K=K;p->M=M;p->R=R;p->TN=maxN<=64?64:160;p->capacity=(maxN+p->TN-1)/p->TN*p->TN;
    alloc(&p->w,(size_t)M*K);alloc(&p->q,(size_t)p->capacity*K);alloc(&p->sw,M);alloc(&p->inverse,K);
    alloc(&p->x,(size_t)p->capacity*K);alloc(&p->sx,p->capacity);alloc(&p->y,(size_t)p->capacity*M);
    alloc(&p->wf,(size_t)M*K);alloc(&p->xf,(size_t)p->capacity*K);
    std::vector<signed char> packed((size_t)M*K);
    for(int m=0;m<M;++m)for(int k=0;k<K;++k)
      packed[((size_t)(m/32)*(K/32)+k/32)*1024+(m%32/16)*512+((m%16)*4+(k%16)/4+(k%32/16)*64)*4+k%4]=codes[(size_t)m*K+k];
    check(musaMemcpy(p->w,packed.data(),packed.size(),musaMemcpyHostToDevice));
    check(musaMemcpy(p->sw,scales,M*4,musaMemcpyHostToDevice));check(musaMemcpy(p->inverse,inverse,K*4,musaMemcpyHostToDevice));
    check(musaMemcpy(p->wf,original,(size_t)M*K*2,musaMemcpyHostToDevice));
    if(R){alloc(&p->extra,(size_t)M*R);alloc(&p->xe,(size_t)p->capacity*R);alloc(&p->indices,R);
      check(musaMemcpy(p->extra,extra,(size_t)M*R*2,musaMemcpyHostToDevice));
      check(musaMemcpy(p->indices,indices,R*4,musaMemcpyHostToDevice));}
    blas(mublasCreate(&p->blas));check(musaEventCreate(&p->begin));check(musaEventCreate(&p->end));return p;
}
extern "C" int int8_replay_run(void*handle,const float*x,int N,int runs,int mode,float*out,float*times) {
    auto *p=(Replay*)handle;if(!p||N<1||N>p->capacity||runs<1||(mode!=0&&mode!=1))return 1;
    const int K=p->K,M=p->M,R=p->R,NP=(N+p->TN-1)/p->TN*p->TN;
    check(musaMemcpy(p->x,x,(size_t)N*K*4,musaMemcpyHostToDevice));
    float one=1,zero=0;
    for(int rep=0;rep<runs+3;++rep){
      check(musaEventRecord(p->begin));
      if(mode==0){
        quantize_smooth<<<NP,128>>>(p->x,p->inverse,p->q,p->sx,K,N,NP);
        if(p->TN==64)row_wmma<32,64,1><<<dim3(M/32,NP/64),128>>>(p->w,p->q,p->sw,p->sx,p->y,M,N,K);
        else row_wmma<64,160,2><<<dim3(M/128,NP/160),256>>>(p->w,p->q,p->sw,p->sx,p->y,M,N,K);
        if(R){
          gather_extra<<<(N*R+255)/256,256>>>(p->x,p->indices,p->xe,K,N,R);
          blas(mublasGemmEx(p->blas,MUBLAS_OP_T,MUBLAS_OP_N,M,N,R,&one,p->extra,MUSA_R_16F,R,p->xe,MUSA_R_16F,R,&one,p->y,MUSA_R_32F,M,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
      }else{
        cast_input<<<(N*K+255)/256,256>>>(p->x,p->xf,N*K);
        blas(mublasGemmEx(p->blas,MUBLAS_OP_T,MUBLAS_OP_N,M,N,K,&one,p->wf,MUSA_R_16F,K,p->xf,MUSA_R_16F,K,&zero,p->y,MUSA_R_32F,M,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }
      check(musaEventRecord(p->end));check(musaEventSynchronize(p->end));
      float ms;check(musaEventElapsedTime(&ms,p->begin,p->end));if(rep>=3)times[rep-3]=ms;
    }
    check(musaMemcpy(out,p->y,(size_t)M*N*4,musaMemcpyDeviceToHost));return 0;
}
extern "C" void int8_replay_destroy(void*handle){
    auto*p=(Replay*)handle;if(!p)return;
    musaFree(p->w);musaFree(p->q);musaFree(p->sw);musaFree(p->inverse);musaFree(p->x);musaFree(p->sx);musaFree(p->y);
    musaFree(p->wf);musaFree(p->xf);if(p->R){musaFree(p->extra);musaFree(p->xe);musaFree(p->indices);}
    mublasDestroy(p->blas);musaEventDestroy(p->begin);musaEventDestroy(p->end);delete p;
}
