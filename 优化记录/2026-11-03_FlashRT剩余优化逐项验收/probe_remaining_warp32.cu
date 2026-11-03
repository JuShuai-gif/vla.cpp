
#include <musa_runtime.h>
#include <musa_fp16.h>
#include <mublas.h>
#include <mma.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#define MU(x) do{auto e=(x);if(e!=musaSuccess){fprintf(stderr,"%s:%s\n",#x,musaGetErrorString(e));exit(2);}}while(0)
#define BL(x) do{auto e=(x);if(e!=MUBLAS_STATUS_SUCCESS){fprintf(stderr,"%s:%d\n",#x,(int)e);exit(3);}}while(0)
__global__ void transpose(const float* x,float* y,int m,int n) {
 size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<(size_t)m*n)y[i]=x[(i%m)*n+i/m];
}
__global__ void touch(const unsigned* w,unsigned* sink,size_t n) {
 size_t i=((size_t)blockIdx.x*blockDim.x+threadIdx.x)*32;
 unsigned v=0;
 for(;i<n;i+=(size_t)gridDim.x*blockDim.x*32) v+=((const volatile unsigned*)w)[i];
 sink[(size_t)blockIdx.x*blockDim.x+threadIdx.x]=v;
}
__global__ void geglu(const float*x,__half*y,int h,int n) {
 size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=(size_t)h*n)return;
 int c=i%h,r=i/h;
 float g=x[(size_t)r*2*h+c],u=x[(size_t)r*2*h+h+c];
 float v=0.5f*g*(1.f+tanhf(0.7978845608028654f*g*(1.f+0.044715f*g*g)));
 y[i]=__float2half(v*u);
}
__global__ void mma_geglu(const __half*w,const __half*x,__half*y,int h,int n,int k) {
 using namespace mtmusa::wmma;
 const int row=blockIdx.x*16,col=blockIdx.y*16;
 fragment<matrix_a,16,16,16,__half,row_major> a,u;
 fragment<matrix_b,16,16,16,__half,col_major> b;
 fragment<accumulator,16,16,16,float> ag,au;
 fill_fragment(ag,0.f);fill_fragment(au,0.f);
 for(int p=0;p<k;p+=16) {
  load_matrix_sync(a,w+(size_t)row*k+p,k);
  load_matrix_sync(u,w+(size_t)(row+h)*k+p,k);
  load_matrix_sync(b,x+(size_t)col*k+p,k);
  mma_sync(ag,a,b,ag);mma_sync(au,u,b,au);
 }
 __shared__ float gs[256],us[256];
 store_matrix_sync(gs,ag,16,mem_row_major);store_matrix_sync(us,au,16,mem_row_major);
 __syncthreads();
 for(int i=threadIdx.x;i<256;i+=blockDim.x) {
  int c=row+i/16,r=col+i%16;
  if(c<h && r<n) {
   float g=gs[i],u0=us[i];
   float v=0.5f*g*(1.f+tanhf(0.7978845608028654f*g*(1.f+0.044715f*g*g)));
   y[(size_t)r*h+c]=__float2half(v*u0);
  }
 }
}
int main(int argc,char**argv) {
 if(argc!=5)return 1;
 int m=atoi(argv[1]),n=atoi(argv[2]),k=atoi(argv[3]);bool glu=atoi(argv[4]);
 int np=(n+15)/16*16;
 size_t na=(size_t)m*k,nb=(size_t)np*k,nc=(size_t)m*n;
 std::vector<_Float16> ha(na),hb(nb);
 for(size_t i=0;i<na;i++)ha[i]=_Float16(((int)(i*17%101)-50)*.002f);
 for(size_t i=0;i<nb;i++)hb[i]=_Float16(((int)(i*13%97)-48)*.002f);
 __half *a,*a1,*b,*halfout;float *c,*temp;unsigned*sink;
 MU(musaMalloc(&a,na*2));MU(musaMalloc(&a1,na*2));MU(musaMalloc(&b,nb*2));
 MU(musaMalloc(&c,nc*4));MU(musaMalloc(&temp,nc*4));MU(musaMalloc(&halfout,nc*2));MU(musaMalloc(&sink,128*256*4));
 MU(musaMemcpy(a,ha.data(),na*2,musaMemcpyHostToDevice));
 std::reverse(ha.begin(),ha.end());MU(musaMemcpy(a1,ha.data(),na*2,musaMemcpyHostToDevice));
 MU(musaMemcpy(b,hb.data(),nb*2,musaMemcpyHostToDevice));
 musaStream_t s,t;MU(musaStreamCreate(&s));MU(musaStreamCreate(&t));
 musaEvent_t fork,join,start,end;
 MU(musaEventCreate(&fork));MU(musaEventCreate(&join));MU(musaEventCreate(&start));MU(musaEventCreate(&end));
 mublasHandle_t handle;BL(mublasCreate(&handle));BL(mublasSetStream(handle,s));
 float one=1,zero=0;
 auto gemm=[&](__half* w,float*out,bool swap) {
  if(!swap)BL(mublasGemmEx(handle,MUBLAS_OP_T,MUBLAS_OP_N,m,n,k,&one,w,MUSA_R_16F,k,b,MUSA_R_16F,k,&zero,out,MUSA_R_32F,m,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP));
  else BL(mublasGemmEx(handle,MUBLAS_OP_T,MUBLAS_OP_N,n,m,k,&one,b,MUSA_R_16F,k,w,MUSA_R_16F,k,&zero,temp,MUSA_R_32F,n,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP));
  if(swap)transpose<<<(nc+255)/256,256,0,s>>>(temp,out,m,n);
 };
 std::vector<float>ref(nc),got(nc);
 std::vector<__half> refh(nc/2),goth(nc/2);
 for(int mode=0;mode<(glu?2:4);++mode) {
  auto run=[&](){
   if(glu) {
    if(mode==0){gemm(a,c,false);geglu<<<(nc/2+255)/256,256,0,s>>>(c,halfout,m/2,n);}
    else mma_geglu<<<dim3((m/2)/16,(n+15)/16),32,0,s>>>(a,b,halfout,m/2,n,k);
   } else if(mode>=2) {
    if(mode==3) {
     MU(musaEventRecord(fork,s));MU(musaStreamWaitEvent(t,fork,0));
     touch<<<128,256,0,t>>>((const unsigned*)a1,sink,na/2);MU(musaEventRecord(join,t));
    }
    gemm(a,c,false);
    if(mode==3)MU(musaStreamWaitEvent(s,join,0));
    gemm(a1,c,false);
   } else gemm(a,c,mode==1);
  };
  for(int i=0;i<3;++i)run();MU(musaDeviceSynchronize());
  MU(musaStreamBeginCapture(s,musaStreamCaptureModeGlobal));run();
  musaGraph_t graph;MU(musaStreamEndCapture(s,&graph));musaGraphExec_t exec;
  MU(musaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
  std::vector<double> times;
  for(int r=0;r<3;++r) {
   MU(musaEventRecord(start,s));for(int i=0;i<20;++i)MU(musaGraphLaunch(exec,s));
   MU(musaEventRecord(end,s));MU(musaEventSynchronize(end));float ms;MU(musaEventElapsedTime(&ms,start,end));times.push_back(ms/20);
  }
  double err=0,peak=0;
  if(glu) {
   MU(musaMemcpy(goth.data(),halfout,nc,musaMemcpyDeviceToHost));if(mode==0)refh=goth;
   for(size_t i=0;i<goth.size();++i){double x=__half2float(goth[i]),y=__half2float(refh[i]);if(!std::isfinite(x))return 4;err=std::max(err,fabs(x-y));peak=std::max(peak,fabs(y));}
  } else {
   MU(musaMemcpy(got.data(),c,nc*4,musaMemcpyDeviceToHost));if(mode==0||mode==2)ref=got;
   for(size_t i=0;i<got.size();++i){if(!std::isfinite(got[i]))return 4;err=std::max(err,fabs((double)got[i]-ref[i]));peak=std::max(peak,fabs((double)ref[i]));}
  }
  printf("{\"m\":%d,\"n\":%d,\"k\":%d,\"glu\":%d,\"mode\":%d,\"mean_ms\":%.7f,\"relative_peak\":%.9g,\"max_abs\":%.9g,\"round_ms\":[%.7f,%.7f,%.7f]}\n",m,n,k,glu,mode,(times[0]+times[1]+times[2])/3,err/std::max(peak,1e-12),err,times[0],times[1],times[2]);fflush(stdout);
  MU(musaGraphExecDestroy(exec));MU(musaGraphDestroy(graph));
 }
 MU(musaDeviceSynchronize());
}
