// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
#include <musa_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

__global__ void primitives(const float * x, signed char * q, float * scales, const int * a, const int * b, int * dots) {
    const int t=blockIdx.x*blockDim.x+threadIdx.x;
    const float4 v=((const float4 *)x)[t];
    float mx=fmaxf(fmaxf(fabsf(v.x),fabsf(v.y)),fmaxf(fabsf(v.z),fabsf(v.w)));
    for(int offset=4;offset>0;offset>>=1) mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,offset,32));
    const float inv=127.0f/mx;
    ((char4 *)q)[t]=make_char4(roundf(v.x*inv),roundf(v.y*inv),roundf(v.z*inv),roundf(v.w*inv));
    if(t%8==0) scales[t/8]=1.0f/inv;
    dots[t]=__dp4a(a[t],b[t],17);
}

int main() {
    musaDeviceProp prop{}; musaGetDeviceProperties(&prop,0);
    std::printf("warp_size=%d sm_count=%d\n",prop.warpSize,prop.multiProcessorCount);
    constexpr int T=1024;
    std::vector<float> x(T*4),scales(T/8);
    std::vector<int> a(T),b(T),dots(T);
    std::vector<signed char> q(T*4);
    for(int i=0;i<T*4;++i)x[i]=float((i*40503+37)%8191-4095)/4096.f;
    std::fill(x.begin(),x.begin()+32,0.f);
    for(int i=0;i<T;++i){a[i]=uint32_t(i)*2654435761U;b[i]=uint32_t(i)*175971027U+29;}
    float *dx,*ds;int *da,*db,*dd;signed char *dq;
    auto check=[](musaError_t s){if(s!=musaSuccess){std::fprintf(stderr,"%s\n",musaGetErrorString(s));std::exit(2);}};
    check(musaMalloc(&dx,x.size()*4));check(musaMalloc(&ds,scales.size()*4));check(musaMalloc(&dq,q.size()));
    check(musaMalloc(&da,T*4));check(musaMalloc(&db,T*4));check(musaMalloc(&dd,T*4));
    check(musaMemcpy(dx,x.data(),x.size()*4,musaMemcpyHostToDevice));
    check(musaMemcpy(da,a.data(),T*4,musaMemcpyHostToDevice));check(musaMemcpy(db,b.data(),T*4,musaMemcpyHostToDevice));
    primitives<<<T/128,128>>>(dx,dq,ds,da,db,dd);check(musaDeviceSynchronize());
    check(musaMemcpy(q.data(),dq,q.size(),musaMemcpyDeviceToHost));
    check(musaMemcpy(scales.data(),ds,scales.size()*4,musaMemcpyDeviceToHost));
    check(musaMemcpy(dots.data(),dd,T*4,musaMemcpyDeviceToHost));
    int quant_errors=0,scale_errors=0,dot_errors=0;
    for(int i=0;i<T;++i){
        const signed char *aa=(const signed char*)&a[i],*bb=(const signed char*)&b[i];
        int expected=17;for(int j=0;j<4;++j)expected+=int(aa[j])*bb[j];
        if(expected!=dots[i]){++dot_errors;if(dot_errors<5)std::fprintf(stderr,"dot %d expected=%d actual=%d\n",i,expected,dots[i]);}
    }
    for(int i=0;i<T*4;i+=32){
        float mx=0;for(int j=i;j<i+32;++j)mx=std::max(mx,std::fabs(x[j]));
        float inv=mx>0?127.f/mx:0.f, scale=mx>0?1.f/inv:0.f;
        if(std::fabs(scale-scales[i/32])>1e-6)++scale_errors;
        for(int j=i;j<i+32;++j)if(int(q[j])!=int(std::round(x[j]*inv)))++quant_errors;
    }
    std::printf("{\"dp4a_errors\":%d,\"quant_errors\":%d,\"scale_errors\":%d}\n",dot_errors,quant_errors,scale_errors);
    musaFree(dx);musaFree(ds);musaFree(dq);musaFree(da);musaFree(db);musaFree(dd);
    return dot_errors||quant_errors||scale_errors ? 1:0;
}
