// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
// Prototype: block-32 Q8_0 weights, dynamic block-32 Q8 activations, INT8 WMMA.
#include <musa_runtime.h>
#include <musa_fp16.h>
#include <mma.h>
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

static void check(musaError_t error) {
    if(error!=musaSuccess){std::fprintf(stderr,"%s\n",musaGetErrorString(error));std::exit(2);}
}

__global__ void quant_input(const float * x, signed char * q, float * scales, int K, int N, int NP) {
    int i=(blockIdx.x*blockDim.x+threadIdx.x)*4;
    if(i>=K*NP)return;
    float4 v=i<K*N?((const float4 *)x)[i/4]:make_float4(0,0,0,0);
    float mx=fmaxf(fmaxf(fabsf(v.x),fabsf(v.y)),fmaxf(fabsf(v.z),fabsf(v.w)));
    for(int d=4;d>0;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d,32));
    float inv=mx>0?127.f/mx:0.f;
    ((char4 *)q)[i/4]=make_char4(roundf(v.x*inv),roundf(v.y*inv),roundf(v.z*inv),roundf(v.w*inv));
    if(i%32==0)scales[i/32]=mx>0?1.f/inv:0.f;
}

__global__ void quant_gemm(const unsigned char * weights, const signed char * input, const float * scales,
                         float * output, int M, int N, int K) {
#if !defined(__MUSA_ARCH__) || __MUSA_ARCH__==220
    using namespace mtmusa::wmma;
    const int row=blockIdx.x*32,col=blockIdx.y*32,tid=threadIdx.x;
    __shared__ signed char a[1024],b[1024];
    __shared__ int c[1024];
    __shared__ float sw[32],sx[32];
    float accum[8]={};
    for(int kb=0;kb<K/32;++kb) {
        for(int i=tid;i<1024;i+=128) {
            int r=i/32,k=i%32;
            const unsigned char * block=weights+((size_t)(row+r)*(K/32)+kb)*34;
            a[i]=row+r<M?(signed char)block[2+k]:0;
            b[i]=input[(size_t)(col+r)*K+kb*32+k];
        }
        if(tid<32) {
            const unsigned char * block=weights+((size_t)(row+tid)*(K/32)+kb)*34;
            sw[tid]=row+tid<M?__half2float(*(const __half *)block):0;
            sx[tid]=scales[(size_t)(col+tid)*(K/32)+kb];
        }
        __syncthreads();
        fragment<matrix_a,32,32,32,signed char,row_major> af;
        fragment<matrix_b,32,32,32,signed char,col_major> bf;
        fragment<accumulator,32,32,32,int> cf;
        load_matrix_sync(af,a,32);load_matrix_sync(bf,b,32);fill_fragment(cf,0);
        mma_sync(cf,af,bf,cf);
        store_matrix_sync(c,cf,32,mem_row_major);
        __syncthreads();
        #pragma unroll
        for(int j=0;j<8;++j){int i=tid+j*128;accum[j]+=float(c[i])*sw[i/32]*sx[i%32];}
        __syncthreads();
    }
    #pragma unroll
    for(int j=0;j<8;++j){int i=tid+j*128,r=row+i/32,n=col+i%32;if(r<M&&n<N)output[(size_t)n*M+r]=accum[j];}
#endif
}

int main(int argc,char **argv) {
    const int K=argc>1?std::atoi(argv[1]):256,M=argc>2?std::atoi(argv[2]):64,N=argc>3?std::atoi(argv[3]):50;
    const int runs=argc>4?std::atoi(argv[4]):10,NP=(N+31)/32*32;
    if(K<32||K%32||M<1||N<1||runs<1)return 2;
    std::vector<float>w((size_t)K*M),x((size_t)K*N),dw(w.size()),qx(x.size()),result((size_t)M*N);
    for(size_t i=0;i<w.size();++i)w[i]=float(int((i*2654435761ULL+17)%65521)-32760)/65536.f;
    for(size_t i=0;i<x.size();++i)x[i]=(i%997==0?16.f:1.f)*float(int((i*40503ULL+37)%8191)-4095)/4096.f;
    std::fill(x.begin(),x.begin()+32,0.f);
    std::vector<unsigned char>packed(w.size()/32*34);
    ggml_quantize_chunk(GGML_TYPE_Q8_0,w.data(),packed.data(),0,M,K,nullptr);
    ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(packed.data(),dw.data(),dw.size());
    for(size_t b=0;b<x.size();b+=32){float mx=0;for(size_t i=b;i<b+32;++i)mx=std::max(mx,std::fabs(x[i]));
        float inv=mx>0?127.f/mx:0,scale=mx>0?1.f/inv:0;
        for(size_t i=b;i<b+32;++i)qx[i]=std::round(x[i]*inv)*scale;}
    unsigned char *gw;signed char *gq;float *gx,*gs,*gy;
    check(musaMalloc(&gw,packed.size()));check(musaMalloc(&gx,x.size()*4));
    check(musaMalloc(&gq,(size_t)K*NP));check(musaMalloc(&gs,(size_t)K*NP/32*4));
    check(musaMalloc(&gy,result.size()*4));
    check(musaMemcpy(gw,packed.data(),packed.size(),musaMemcpyHostToDevice));
    check(musaMemcpy(gx,x.data(),x.size()*4,musaMemcpyHostToDevice));
    musaEvent_t begin,end;check(musaEventCreate(&begin));check(musaEventCreate(&end));
    std::vector<float>times;
    for(int i=0;i<3+runs;++i){
        check(musaEventRecord(begin));
        quant_input<<<(K*NP/4+127)/128,128>>>(gx,gq,gs,K,N,NP);
        quant_gemm<<<dim3((M+31)/32,NP/32),128>>>(gw,gq,gs,gy,M,N,K);
        check(musaEventRecord(end));check(musaEventSynchronize(end));
        float ms;check(musaEventElapsedTime(&ms,begin,end));if(i>=3)times.push_back(ms);
    }
    check(musaMemcpy(result.data(),gy,result.size()*4,musaMemcpyDeviceToHost));
    bool finite=std::all_of(result.begin(),result.end(),[](float v){return std::isfinite(v);});
    size_t checks=std::min<size_t>(4096,result.size());int failed=0;double max_error=0,sq=0;
    for(size_t s=0;s<checks;++s){size_t index=checks==result.size()?s:s*(result.size()-1)/(checks-1),m=index%M,n=index/M;
        double expected=0;for(int k=0;k<K;++k)expected+=double(dw[m*K+k])*qx[n*K+k];
        double e=std::fabs(expected-result[index]);max_error=std::max(e,max_error);sq+=e*e;
        if(e>0.003+0.0003*std::fabs(expected))++failed;
        if(s<4)std::fprintf(stderr,"sample=%zu expected=%g actual=%g\n",index,expected,result[index]);}
    std::sort(times.begin(),times.end());
    std::printf("{\"kernel\":\"Q8-WMMA-32\",\"K\":%d,\"M\":%d,\"N\":%d,\"runs\":%d,\"gpu_p50_ms\":%.6f,\"activation_quant_included\":true,\"checks\":%zu,\"failed\":%d,\"finite\":%s,\"max_error\":%.9g,\"rmse\":%.9g}\n",K,M,N,runs,times[times.size()/2],checks,failed,finite?"true":"false",max_error,std::sqrt(sq/checks));
    musaEventDestroy(begin);musaEventDestroy(end);musaFree(gw);musaFree(gx);musaFree(gq);musaFree(gs);musaFree(gy);
    return finite&&!failed?0:1;
}
