// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
// Compare complete dynamic W8A8 consumption with FP16 BLAS on identical shapes.
#include <musa_runtime.h>
#include <musa_fp16.h>
#include <mublas.h>
#include <mma.h>
#include <mudnncxx/mudnn.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(musaError_t e) { if(e!=musaSuccess){fprintf(stderr,"MUSA: %s\n",musaGetErrorString(e));exit(2);} }
static void dnn(musa::dnn::Status e) {if(e!=musa::dnn::Status::SUCCESS){fprintf(stderr,"muDNN: %d\n",int(e));exit(4);}}
static void blas(mublasStatus_t e) { if(e!=MUBLAS_STATUS_SUCCESS){fprintf(stderr,"BLAS: %d\n",int(e));exit(3);} }
__global__ void quantize(const float*x,signed char*q,float*s,int K,int N,int NP,bool packed) {
    int n=blockIdx.x,t=threadIdx.x; float mx=0;
    for(int k=t;k<K;k+=128)mx=fmaxf(mx,n<N?fabsf(x[(size_t)n*K+k]):0.f);
    __shared__ float reduce[128];reduce[t]=mx;__syncthreads();
    for(int d=64;d;d>>=1){if(t<d)reduce[t]=fmaxf(reduce[t],reduce[t+d]);__syncthreads();}
    float scale=reduce[0]>0?fmaxf(__fdiv_rn(reduce[0],127.f),1.17549435e-38f):1.f;
    float inverse_scale=__fdiv_rn(1.f,scale);
    if(t==0)s[n]=scale;
    for(int k=t*4;k<K;k+=512){
      unsigned word=0;
      #pragma unroll
      for(int j=0;j<4;++j){
        signed char v=n<N?(signed char)rintf(fminf(127.f,fmaxf(-127.f,x[(size_t)n*K+k+j]*inverse_scale))):0;
        word|=unsigned((unsigned char)v)<<(j*8);
      }
      size_t dst=packed?((size_t)(n/32)*(K/32)+k/32)*1024+(n%32/16)*512+((n%16)*4+(k%16)/4+(k%32/16)*64)*4+k%4:(size_t)n*K+k;
      *((unsigned*)(q+dst))=word;
    }
}
__global__ void epilogue(const int*x,const float*sw,const float*sx,float*y,int M,int N) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<(size_t)M*N)y[i]=float(x[i])*sw[i%M]*sx[i/M];
}
__global__ void cast_input(const float*x,__half*y,int size) {int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<size)y[i]=__float2half(x[i]);}
#ifndef SPLIT_K
#define SPLIT_K 1
#endif
#ifndef INT8_WARP
#define INT8_WARP 128
#endif
#ifndef PREFETCH_STEPS
#define PREFETCH_STEPS 2
#endif

template<int TM,int TN,int NW=1> __global__ void row_wmma(const signed char*w,const signed char*x,const float*sw,const float*sx,float*y,int M,int N,int K) {
    using namespace mtmusa::wmma;
    constexpr int MT=TM/32,NT=TN/32,LD=TN+4;
    int warp=threadIdx.x/INT8_WARP,row=(blockIdx.x*NW+warp)*TM,col=blockIdx.y*TN,t=threadIdx.x%INT8_WARP;
    fragment<accumulator,32,32,32,int> cf[MT][NT];
    #pragma unroll
    for(int i=0;i<MT;++i){
      #pragma unroll
      for(int j=0;j<NT;++j)fill_fragment(cf[i][j],0);
    }
    // Group independent loads before MMA to expose more memory-level parallelism.
    for(int kb=blockIdx.z*(K/SPLIT_K);kb<(blockIdx.z+1)*(K/SPLIT_K);kb+=32*PREFETCH_STEPS){
      fragment<matrix_a,32,32,32,signed char,row_major> af[PREFETCH_STEPS][MT];
      fragment<matrix_b,32,32,32,signed char,col_major> bf[PREFETCH_STEPS][NT];
      #pragma unroll
      for(int u=0;u<PREFETCH_STEPS;++u){
        if(kb+32*u<(blockIdx.z+1)*(K/SPLIT_K)){
          #pragma unroll
          for(int i=0;i<MT;++i){const int *p=(const int*)(w+((size_t)(row/32+i)*(K/32)+kb/32+u)*1024);
              ((int*)&af[u][i])[0]=p[t];((int*)&af[u][i])[1]=p[128+t];}
          #pragma unroll
          for(int j=0;j<NT;++j){const int *p=(const int*)(x+((size_t)(col/32+j)*(K/32)+kb/32+u)*1024);
              ((int*)&bf[u][j])[0]=p[t];((int*)&bf[u][j])[1]=p[128+t];}
        }
      }
      #pragma unroll
      for(int u=0;u<PREFETCH_STEPS;++u){
        if(kb+32*u<(blockIdx.z+1)*(K/SPLIT_K)){
          #pragma unroll
          for(int i=0;i<MT;++i){
            #pragma unroll
            for(int j=0;j<NT;++j)mma_sync(cf[i][j],af[u][i],bf[u][j],cf[i][j]);
          }
        }
      }
    }
    // MP22 fragment layout is verified by probe_musa_int8_layout.cpp.
    // Fuse scaling into the accumulator store without an INT32 intermediate.
    #pragma unroll
    for(int i=0;i<MT;++i){
      #pragma unroll
      for(int j=0;j<NT;++j){
        #pragma unroll
        for(int v=0;v<8;++v){
          const int m=row+i*32+((t&31)>>3)+4*(t>>6)+8*(v>>1);
          const int n=col+j*32+(t&7)+8*((t>>5)&1)+16*(v&1);
          if(n<N){
            if constexpr(SPLIT_K>1)((int*)y)[(size_t)blockIdx.z*M*N+(size_t)n*M+m]=cf[i][j].x[v];
            else y[(size_t)n*M+m]=float(cf[i][j].x[v])*sw[m]*sx[n];
          }
        }
      }
    }
}

__global__ void reduce_split(const int *partial,const float *sw,const float *sx,float *out,int M,int N){
 size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<(size_t)M*N){
  int sum=0;
  #pragma unroll
  for(int split=0;split<SPLIT_K;++split)sum+=partial[(size_t)split*M*N+i];
  out[i]=float(sum)*sw[i%M]*sx[i/M];
 }
}
int main(int argc,char**argv) {
    int K=argc>1?atoi(argv[1]):256,M=argc>2?atoi(argv[2]):64,N=argc>3?atoi(argv[3]):50;
    int runs=argc>4?atoi(argv[4]):10,TM=argc>5?atoi(argv[5]):64,TN=argc>6?atoi(argv[6]):64,NW=argc>7?atoi(argv[7]):1,NP=(N+TN-1)/TN*TN;
    if(K%(32*SPLIT_K)||M%(TM*NW)||K<32||M<1||N<1||runs<1)return 2;
    std::vector<float>w((size_t)M*K),x((size_t)N*K),sw(M),sx(N),result((size_t)M*N);
    std::vector<signed char>qw(w.size()),qx(x.size());std::vector<__half>wh(w.size());
    for(size_t i=0;i<w.size();++i)w[i]=float(int((i*2654435761ULL+17)%65521)-32760)/65536.f;
    for(size_t i=0;i<x.size();++i)x[i]=(i%997==0?16.f:1.f)*float(int((i*40503ULL+37)%8191)-4095)/4096.f;
    std::fill(x.begin(),x.begin()+K,0.f);
    auto pack=[K](const std::vector<float>&a,std::vector<signed char>&q,std::vector<float>&s){
      for(size_t n=0;n<s.size();++n){float mx=0;for(int k=0;k<K;++k)mx=std::max(mx,std::fabs(a[n*K+k]));s[n]=mx>0?mx/127.f:1;
        for(int k=0;k<K;++k)q[n*K+k]=(signed char)std::lrint(std::fmin(127.f,std::fmax(-127.f,a[n*K+k]/s[n])));}};
    pack(w,qw,sw);pack(x,qx,sx);
    for(int n=0;n<N;++n){float inverse_scale=1.f/sx[n];for(int k=0;k<K;++k)qx[(size_t)n*K+k]=(signed char)std::lrint(std::fmin(127.f,std::fmax(-127.f,x[(size_t)n*K+k]*inverse_scale)));}for(size_t i=0;i<w.size();++i)wh[i]=__float2half(w[i]);
    signed char *gw,*gq;float *gx,*gsw,*gsx,*gy;int*gi;__half *gwh,*gxh;
    check(musaMalloc(&gw,qw.size()));check(musaMalloc(&gq,(size_t)NP*K));check(musaMalloc(&gx,x.size()*4));
    check(musaMalloc(&gsw,M*4));check(musaMalloc(&gsx,NP*4));check(musaMalloc(&gy,(size_t)M*NP*4));
    check(musaMalloc(&gi,(size_t)M*NP*4*SPLIT_K));check(musaMalloc(&gwh,wh.size()*2));check(musaMalloc(&gxh,(size_t)NP*K*2));
    std::vector<signed char> packed_w(qw.size());
    for(int m=0;m<M;++m)for(int k=0;k<K;++k)packed_w[((size_t)(m/32)*(K/32)+k/32)*1024+(m%32/16)*512+((m%16)*4+(k%16)/4+(k%32/16)*64)*4+k%4]=qw[(size_t)m*K+k];
    check(musaMemcpy(gw,std::getenv("BENCH_MUDNN")?qw.data():packed_w.data(),qw.size(),musaMemcpyHostToDevice));check(musaMemcpy(gx,x.data(),x.size()*4,musaMemcpyHostToDevice));
    check(musaMemcpy(gsw,sw.data(),M*4,musaMemcpyHostToDevice));check(musaMemcpy(gwh,wh.data(),wh.size()*2,musaMemcpyHostToDevice));
    mublasHandle_t h;blas(mublasCreate(&h));musaEvent_t b,e;check(musaEventCreate(&b));check(musaEventCreate(&e));
    bool use_dnn=std::getenv("BENCH_MUDNN")!=nullptr;
    musa::dnn::Handle dh; musa::dnn::Tensor da,db,dc; musa::dnn::MatMul mm;
    void *workspace=nullptr;size_t workspace_size=0;
    if(use_dnn){dnn(da.SetType(musa::dnn::Tensor::Type::INT8));dnn(db.SetType(musa::dnn::Tensor::Type::INT8));dnn(dc.SetType(musa::dnn::Tensor::Type::INT32));
      dnn(da.SetNdInfo({NP,K}));dnn(db.SetNdInfo({M,K}));dnn(dc.SetNdInfo({NP,M}));dnn(da.SetAddr(gq));dnn(db.SetAddr(gw));dnn(dc.SetAddr(gi));
      dnn(mm.SetTranspose(false,true));dnn(mm.SetComputeMode(musa::dnn::MatMul::ComputeMode::TENSOR));
      dnn(mm.GetWorkspaceSize(dh,workspace_size,dc,da,db));if(workspace_size)check(musaMalloc(&workspace,workspace_size));
    }
    musa::dnn::MemoryMaintainer allocator=[&](size_t n){if(n>workspace_size)std::abort();return musa::dnn::MemoryHandler(workspace,[](void*){});};
    musaDeviceProp prop{};check(musaGetDeviceProperties(&prop,0));fprintf(stderr,"device warpSize=%d WMMA threads=%d\n",prop.warpSize,INT8_WARP);
    fprintf(stderr,"tile=%dx%d warps=%d prefetch_steps=%d\n",TM,TN,NW,PREFETCH_STEPS);
    for(int mode=0;mode<2;++mode){std::vector<float>times;int one=1,zero=0;float fone=1,fzero=0;
      for(int r=0;r<3+runs;++r){check(musaEventRecord(b));
        if(mode==0){quantize<<<NP,128>>>(gx,gq,gsx,K,N,NP,!use_dnn);
          if(use_dnn){dnn(mm.Run(dh,dc,da,db,allocator));epilogue<<<((size_t)M*N+255)/256,256>>>(gi,gsw,gsx,gy,M,N);}else{
#define LAUNCH(A,B,W) row_wmma<A,B,W><<<dim3(M/(A*W),NP/B,SPLIT_K),INT8_WARP*W>>>(gw,gq,gsw,gsx,SPLIT_K>1?(float*)gi:gy,M,N,K)
#define TILE(A,B) if(TM==A&&TN==B){if(NW==1){LAUNCH(A,B,1);}else if(NW==2){LAUNCH(A,B,2);}else if(NW==4){LAUNCH(A,B,4);}else{return 2;}}else
          TILE(32,32) TILE(32,64) TILE(64,32) TILE(64,64) TILE(64,160) {fprintf(stderr,"unsupported tile\n");return 2;}
#undef LAUNCH
#undef TILE
          if constexpr(SPLIT_K>1)reduce_split<<<((size_t)M*N+255)/256,256>>>(gi,gsw,gsx,gy,M,N);

          }
        }else{cast_input<<<((size_t)K*N+255)/256,256>>>(gx,gxh,K*N);
          blas(mublasGemmEx(h,MUBLAS_OP_T,MUBLAS_OP_N,M,N,K,&fone,gwh,MUSA_R_16F,K,gxh,MUSA_R_16F,K,&fzero,gy,MUSA_R_32F,M,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP));}
        check(musaEventRecord(e));check(musaEventSynchronize(e));float ms;check(musaEventElapsedTime(&ms,b,e));if(r>=3)times.push_back(ms);
      }
      check(musaMemcpy(result.data(),gy,result.size()*4,musaMemcpyDeviceToHost));
      int failed=0;double max_error=0;size_t checks=std::min<size_t>(4096,result.size());
      for(size_t j=0;j<checks;++j){size_t i=checks==result.size()?j:j*(result.size()-1)/(checks-1),m=i%M,n=i/M;double ref=0;
        for(int k=0;k<K;++k)ref+=mode==0?double(qw[m*K+k])*qx[n*K+k]:double(__half2float(wh[m*K+k]))*__half2float(__float2half(x[n*K+k]));
        if(mode==0)ref*=double(sw[m])*sx[n];double err=std::fabs(ref-result[i]);max_error=std::max(max_error,err);
        if(!std::isfinite(result[i])||err>0.003+0.0003*std::fabs(ref))++failed;
      }
      printf("{\"mode\":%d,\"gpu_samples_ms\":[",mode);
      for(size_t i=0;i<times.size();++i)printf("%s%.6f",i?",":"",times[i]);
      printf("]}\n");
      std::sort(times.begin(),times.end());
      printf("{\"path\":\"%s\",\"K\":%d,\"M\":%d,\"N\":%d,\"gpu_p50_ms\":%.6f,\"checks\":%zu,\"failed\":%d,\"max_error\":%.9g}\n",mode==0?(use_dnn?"row-W8A8-muDNN":"row-W8A8-WMMA-split-K"):"F16-BLAS",K,M,N,times[times.size()/2],checks,failed,max_error);
      if(failed)return 1;
    }
    if(workspace)musaFree(workspace);mublasDestroy(h);musaEventDestroy(b);musaEventDestroy(e);musaFree(gw);musaFree(gq);musaFree(gx);musaFree(gsw);musaFree(gsx);musaFree(gy);musaFree(gi);musaFree(gwh);musaFree(gxh);
}
