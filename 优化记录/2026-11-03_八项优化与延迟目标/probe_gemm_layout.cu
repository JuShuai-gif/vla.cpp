#include <musa_runtime.h>
#include <musa_fp16.h>
#include <mublas.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#define MU(x) do{auto e=x;if(e!=musaSuccess){fprintf(stderr,"%s: %s\n",#x,musaGetErrorString(e));exit(2);}}while(0)
#define BL(x) do{auto e=x;if(e){fprintf(stderr,"%s: %d\n",#x,(int)e);exit(3);}}while(0)
int main(int argc,char **argv) {
 if(argc!=4)return 1;int m=atoi(argv[1]),n=atoi(argv[2]),k=atoi(argv[3]);
 std::vector<__half> ah((size_t)m*k),at(ah.size()),bh((size_t)n*k),bt(bh.size());
 uint32_t seed=47;auto rnd=[&](){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return ((int)(seed%2049)-1024)/4096.f;};
 for(int r=0;r<m;r++)for(int p=0;p<k;p++)at[(size_t)p*m+r]=ah[(size_t)r*k+p]=__float2half(rnd());
 for(int c=0;c<n;c++)for(int p=0;p<k;p++)bt[(size_t)p*n+c]=bh[(size_t)c*k+p]=__float2half(rnd());
 __half *a,*aN,*b,*bT;float *c;
 MU(musaMalloc(&a,ah.size()*2));MU(musaMalloc(&aN,at.size()*2));
 MU(musaMalloc(&b,bh.size()*2));MU(musaMalloc(&bT,bt.size()*2));MU(musaMalloc(&c,(size_t)m*n*4));
 MU(musaMemcpy(a,ah.data(),ah.size()*2,musaMemcpyHostToDevice));MU(musaMemcpy(aN,at.data(),at.size()*2,musaMemcpyHostToDevice));
 MU(musaMemcpy(b,bh.data(),bh.size()*2,musaMemcpyHostToDevice));MU(musaMemcpy(bT,bt.data(),bt.size()*2,musaMemcpyHostToDevice));
 musaStream_t s;MU(musaStreamCreate(&s));mublasHandle_t h;BL(mublasCreate(&h));BL(mublasSetStream(h,s));
 musaEvent_t begin,end;MU(musaEventCreate(&begin));MU(musaEventCreate(&end));
 float one=1,zero=0;std::vector<float> ref((size_t)m*n),got(ref.size());
 for(int layout=0;layout<4;layout++)for(int mode:{0,1,2,3,16})for(int algo:{1,0}) {
  auto sm=mublasSetMathMode(h,(mublasMath_t)mode);if(sm){printf("{\"layout\":%d,\"math\":%d,\"set\":%d}\n",layout,mode,(int)sm);continue;}
  bool an=layout&1,tt=layout&2;
  auto run=[&](){return mublasGemmEx(h,an?MUBLAS_OP_N:MUBLAS_OP_T,tt?MUBLAS_OP_T:MUBLAS_OP_N,
    m,n,k,&one,an?aN:a,MUSA_R_16F,an?m:k,tt?bT:b,MUSA_R_16F,tt?n:k,&zero,c,MUSA_R_32F,m,
    MUBLAS_COMPUTE_32F,(mublasGemmAlgo_t)algo);};
  auto stat=run();if(stat){printf("{\"layout\":%d,\"math\":%d,\"algo\":%d,\"status\":%d}\n",layout,mode,algo,(int)stat);continue;}
  for(int i=0;i<2;i++)BL(run());MU(musaStreamSynchronize(s));
  MU(musaMemcpy(got.data(),c,got.size()*4,musaMemcpyDeviceToHost));if(layout==0&&mode==0&&algo==1)ref=got;
  double err=0,peak=0;for(size_t i=0;i<got.size();i++){if(!std::isfinite(got[i]))return 4;err=fmax(err,fabs(got[i]-ref[i]));peak=fmax(peak,fabs(ref[i]));}
  MU(musaStreamBeginCapture(s,musaStreamCaptureModeGlobal));BL(run());musaGraph_t graph;MU(musaStreamEndCapture(s,&graph));
  musaGraphExec_t ex;MU(musaGraphInstantiate(&ex,graph,nullptr,nullptr,0));
  double total=0;
  for(int r=0;r<3;r++){MU(musaEventRecord(begin,s));for(int j=0;j<10;j++)MU(musaGraphLaunch(ex,s));MU(musaEventRecord(end,s));MU(musaEventSynchronize(end));float ms;MU(musaEventElapsedTime(&ms,begin,end));total+=ms/10;}
  printf("{\"m\":%d,\"n\":%d,\"k\":%d,\"layout\":%d,\"math\":%d,\"algo\":%d,\"ms\":%.7f,\"max_abs\":%.9g,\"relative_peak\":%.9g}\n",m,n,k,layout,mode,algo,total/3,err,err/fmax(peak,1e-12));fflush(stdout);
  MU(musaGraphExecDestroy(ex));MU(musaGraphDestroy(graph));
 }
}