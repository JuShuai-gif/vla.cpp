#include <musa_runtime.h>
#include <musa_fp16.h>
#include <mublas.h>
#include <mma.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#define MU(x) do{auto e=x;if(e!=musaSuccess){fprintf(stderr,"%s:%s\n",#x,musaGetErrorString(e));exit(2);}}while(0)
#define BL(x) do{auto e=x;if(e){fprintf(stderr,"%s:%d\n",#x,(int)e);exit(3);}}while(0)

template<int TM>
__global__ void small_gemm(const __half*w,const __half*x,float*y,int m,int n,int k) {
 using namespace mtmusa::wmma;constexpr int NT=2,MT=TM/32,ld=68;
 const int row=blockIdx.x*TM,col=blockIdx.y*64;
 fragment<matrix_a,32,32,16,__half,row_major> a[MT];
 fragment<matrix_b,32,32,16,__half,col_major> b[NT];
 fragment<accumulator,32,32,16,float> acc[MT][NT];
 #pragma unroll
 for(int i=0;i<MT;i++){
  #pragma unroll
  for(int j=0;j<NT;j++)fill_fragment(acc[i][j],0.f);
 }
 #pragma unroll
 for(int i=0;i<MT;i++)load_matrix_sync(a[i],w+(size_t)(row+i*32)*k,k);
 #pragma unroll
 for(int j=0;j<NT;j++)load_matrix_sync(b[j],x+(size_t)(col+j*32)*k,k);
 for(int p=16;p<k;p+=16) {
  fragment<matrix_a,32,32,16,__half,row_major> next_a[MT];
  fragment<matrix_b,32,32,16,__half,col_major> next_b[NT];
  #pragma unroll
  for(int i=0;i<MT;i++)load_matrix_sync(next_a[i],w+(size_t)(row+i*32)*k+p,k);
  #pragma unroll
  for(int j=0;j<NT;j++)load_matrix_sync(next_b[j],x+(size_t)(col+j*32)*k+p,k);
  #pragma unroll
  for(int i=0;i<MT;i++){
   #pragma unroll
   for(int j=0;j<NT;j++)mma_sync(acc[i][j],a[i],b[j],acc[i][j]);
  }
  #pragma unroll
  for(int i=0;i<MT;i++)a[i]=next_a[i];
  #pragma unroll
  for(int j=0;j<NT;j++)b[j]=next_b[j];
 }
 #pragma unroll
 for(int i=0;i<MT;i++){
  #pragma unroll
  for(int j=0;j<NT;j++)mma_sync(acc[i][j],a[i],b[j],acc[i][j]);
 }
 __shared__ float out[TM*ld];
 #pragma unroll
 for(int i=0;i<MT;i++){
  #pragma unroll
  for(int j=0;j<NT;j++)store_matrix_sync(out+i*32*ld+j*32,acc[i][j],ld,mem_row_major);
 }
 __syncthreads();
 for(int i=threadIdx.x;i<TM*64;i+=blockDim.x) {
  const int c=row+i%TM,r=col+i/TM;
  if(c<m&&r<n)y[(size_t)r*m+c]=out[(i%TM)*ld+i/TM];
 }
}

int main(int argc,char**argv) {
 if(argc<4)return 1;int m=atoi(argv[1]),n=atoi(argv[2]),k=atoi(argv[3]);bool once=argc>4;
 size_t na=(size_t)m*k,nb=(size_t)((n+63)/64*64)*k,nc=(size_t)m*n;
 std::vector<__half> ah(na),bh(nb);unsigned seed=31;
 auto rnd=[&](){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return ((int)(seed%2049)-1024)*.0001f;};
 for(auto&v:ah)v=__float2half(rnd());for(auto&v:bh)v=__float2half(rnd());
 __half*a,*b,*hc;float*c;MU(musaMalloc(&a,na*2));MU(musaMalloc(&b,nb*2));
 MU(musaMalloc(&c,nc*4));MU(musaMalloc(&hc,nc*2));
 MU(musaMemcpy(a,ah.data(),na*2,musaMemcpyHostToDevice));MU(musaMemcpy(b,bh.data(),nb*2,musaMemcpyHostToDevice));
 musaStream_t s;MU(musaStreamCreate(&s));mublasHandle_t h;BL(mublasCreate(&h));BL(mublasSetStream(h,s));
 musaEvent_t begin,end;MU(musaEventCreate(&begin));MU(musaEventCreate(&end));
 float one=1,zero=0;__half oh=__float2half(1),zh=__float2half(0);
 std::vector<float>ref(nc),got(nc);std::vector<__half> gh(nc);
 for(int mode=0;mode<(once?1:3);++mode)for(int split:{1,2,4,8}) {
  if(split!=1)continue;int chunk=m/split;if(chunk*split!=m)continue;
  auto run=[&](){
   if(mode==1){small_gemm<32><<<dim3(m/32,(n+63)/64),128,0,s>>>(a,b,c,m,n,k);return MUBLAS_STATUS_SUCCESS;}
   if(mode==2){small_gemm<64><<<dim3(m/64,(n+63)/64),128,0,s>>>(a,b,c,m,n,k);return MUBLAS_STATUS_SUCCESS;}
   return mublasGemmEx(h,MUBLAS_OP_T,MUBLAS_OP_N,m,n,k,&one,a,MUSA_R_16F,k,b,MUSA_R_16F,k,&zero,c,MUSA_R_32F,m,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP);
  };
  auto rc=run();if(rc){printf("{\"mode\":%d,\"split\":%d,\"status\":%d}\n",mode,split,(int)rc);continue;}
  MU(musaStreamSynchronize(s));if(once){printf("profiled one GEMM\n");return 0;}
  for(int i=0;i<3;i++)BL(run());MU(musaStreamSynchronize(s));
  MU(musaStreamBeginCapture(s,musaStreamCaptureModeGlobal));BL(run());musaGraph_t graph;MU(musaStreamEndCapture(s,&graph));
  musaGraphExec_t ex;MU(musaGraphInstantiate(&ex,graph,nullptr,nullptr,0));
  float total=0;for(int r=0;r<3;r++){MU(musaEventRecord(begin,s));for(int j=0;j<20;j++)MU(musaGraphLaunch(ex,s));MU(musaEventRecord(end,s));MU(musaEventSynchronize(end));float ms;MU(musaEventElapsedTime(&ms,begin,end));total+=ms/20;}
  MU(musaMemcpy(got.data(),c,nc*4,musaMemcpyDeviceToHost));
  if(mode==0)ref=got;double diff=0,peak=0;
  for(size_t i=0;i<nc;i++){if(!std::isfinite(got[i]))return 4;diff=fmax(diff,fabs(got[i]-ref[i]));peak=fmax(peak,fabs(ref[i]));}
  printf("{\"m\":%d,\"n\":%d,\"k\":%d,\"mode\":%d,\"split\":%d,\"ms\":%.7f,\"max_abs\":%.9g,\"relative_peak\":%.9g}\n",m,n,k,mode,split,total/3,diff,diff/fmax(peak,1e-12));fflush(stdout);
  MU(musaGraphExecDestroy(ex));MU(musaGraphDestroy(graph));
 }
}