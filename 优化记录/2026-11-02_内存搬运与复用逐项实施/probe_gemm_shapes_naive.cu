#include <musa_runtime.h>
#include <musa_fp16.h>
#include <mublas.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#define MU(x) do{auto e=(x);if(e!=musaSuccess){fprintf(stderr,"%s: %s\n",#x,musaGetErrorString(e));exit(2);}}while(0)
#define BL(x) do{auto e=(x);if(e!=MUBLAS_STATUS_SUCCESS){fprintf(stderr,"%s: %d\n",#x,(int)e);exit(3);}}while(0)
__global__ void pack(const float *src,__half *dst,int k,int n,int kp,int np,int batch) {
 size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=(size_t)kp*np*batch)return;
 int col=i%kp,row=(i/kp)%np,b=i/(kp*np);
 dst[i]=__float2half_rn(row<n && col<k?src[col+(size_t)k*(row+n*b)]:0.f);
}
// Initial scalar restoration kernel, retained for the naive-crop measurements.
__global__ void crop(const float *src,float *dst,int m,int n,int mp,int np,int batch) {
 size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=(size_t)m*n*batch)return;
 int col=i%m,row=(i/m)%n,b=i/(m*n);
 dst[i]=src[col+(size_t)mp*(row+np*b)];
}
struct Shape {int m,n,k,b;};
int main(int argc,char**argv) {
 if(argc<5)return 1;
 int m=atoi(argv[1]),n=atoi(argv[2]),k=atoi(argv[3]),batch=atoi(argv[4]);
 std::vector<Shape> variants={{m,n,k,batch}};
 for(int j=5;j<argc;j++) {
  Shape v={m,n,k,batch};
  if(std::strchr(argv[j],':')) { if(std::sscanf(argv[j],"%d:%d:%d:%d",&v.m,&v.n,&v.k,&v.b)!=4)return 1; }
  else v.n=atoi(argv[j]);
  if(v.m<m || v.k<k || (v.b==batch?v.n<n:v.b!=1 || v.n!=n*batch))return 1;
  variants.push_back(v);
 }
 if(batch>1)variants.push_back({m,n*batch,k,1});
 size_t ca=0,cb=0,cc=0;
 for(auto v:variants){ca=std::max(ca,(size_t)v.m*v.k);cb=std::max(cb,(size_t)v.k*v.n*v.b);cc=std::max(cc,(size_t)v.m*v.n*v.b);}
 musaStream_t stream;MU(musaStreamCreate(&stream));mublasHandle_t h;BL(mublasCreate(&h));BL(mublasSetStream(h,stream));
 size_t na=(size_t)m*k,nb=(size_t)k*n*batch,nc=(size_t)m*n*batch;
 std::vector<_Float16> ha(na);std::vector<float> hb(nb),ref(nc),got(nc);
 for(size_t i=0;i<na;i++)ha[i]=_Float16(((int)(i*17%101)-50)*.002f);
 for(size_t i=0;i<nb;i++)hb[i]=((int)(i*13%97)-48)*.002f;
 void *a,*b;float *src,*c,*out;
 MU(musaMalloc(&a,ca*2));MU(musaMalloc(&src,nb*4));MU(musaMalloc(&b,cb*2));
 MU(musaMalloc(&c,cc*4));MU(musaMalloc(&out,nc*4));
 MU(musaMemcpy(a,ha.data(),na*2,musaMemcpyHostToDevice));MU(musaMemcpy(src,hb.data(),nb*4,musaMemcpyHostToDevice));
 float one=1,zero=0;musaEvent_t start,end;MU(musaEventCreate(&start));MU(musaEventCreate(&end));
 bool first=true;
 for(auto variant:variants) {
  bool flat=variant.b!=batch;int nn=flat?n*batch:n,np=variant.n,bb=variant.b,mp=variant.m,kp=variant.k;
  bool copy=mp!=m || np!=nn;
  std::vector<_Float16> weights((size_t)mp*kp,0);
  for(int row=0;row<m;row++)std::copy(ha.begin()+(size_t)row*k,ha.begin()+(size_t)(row+1)*k,weights.begin()+(size_t)row*kp);
  MU(musaMemcpy(a,weights.data(),weights.size()*2,musaMemcpyHostToDevice));
  auto run=[&](){
   size_t num=(size_t)kp*np*bb;
   pack<<<(num+255)/256,256,0,stream>>>(src,(__half*)b,k,nn,kp,np,bb);
   if(bb==1)BL(mublasGemmEx(h,MUBLAS_OP_T,MUBLAS_OP_N,mp,np,kp,&one,a,MUSA_R_16F,kp,b,MUSA_R_16F,kp,&zero,c,MUSA_R_32F,mp,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP));
   else BL(mublasGemmStridedBatchedEx(h,MUBLAS_OP_T,MUBLAS_OP_N,mp,np,kp,&one,a,MUSA_R_16F,kp,0,b,MUSA_R_16F,kp,(long long)kp*np,&zero,c,MUSA_R_32F,mp,(long long)mp*np,bb,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP));
   if(copy)crop<<<(nc+255)/256,256,0,stream>>>(c,out,m,nn,mp,np,bb);
  };
  for(int j=0;j<3;j++)run();MU(musaStreamSynchronize(stream));
  MU(musaStreamBeginCapture(stream,musaStreamCaptureModeGlobal));run();musaGraph_t graph;MU(musaStreamEndCapture(stream,&graph));musaGraphExec_t exec;
  MU(musaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
  std::vector<float> samples;
  for(int round=0;round<3;round++) {
   MU(musaEventRecord(start,stream));for(int j=0;j<20;j++)MU(musaGraphLaunch(exec,stream));
   MU(musaEventRecord(end,stream));MU(musaEventSynchronize(end));float ms;MU(musaEventElapsedTime(&ms,start,end));samples.push_back(ms/20);
  }
  MU(musaMemcpy(got.data(),copy?out:c,nc*4,musaMemcpyDeviceToHost));
  if(first)ref=got;
  double peak=0,err=0;for(size_t i=0;i<nc;i++){peak=std::max(peak,fabs((double)ref[i]));err=std::max(err,fabs((double)ref[i]-got[i]));}
  printf("{\"m\":%d,\"n\":%d,\"k\":%d,\"batch\":%d,\"gemm_m\":%d,\"gemm_n\":%d,\"gemm_k\":%d,\"gemm_batch\":%d,\"variant\":\"%s\",\"mean_ms\":%.7f,\"round_ms\":[%.7f,%.7f,%.7f],\"relative_peak\":%.9g,\"max_abs\":%.9g}\n",
   m,n,k,batch,mp,np,kp,bb,flat?"flatten":(first?"baseline":"pad"),(samples[0]+samples[1]+samples[2])/3,samples[0],samples[1],samples[2],err/std::max(peak,1e-12),err);fflush(stdout);
  first=false;
  MU(musaGraphExecDestroy(exec));MU(musaGraphDestroy(graph));
 }
 MU(musaFree(a));MU(musaFree(b));MU(musaFree(src));MU(musaFree(c));MU(musaFree(out));
 MU(musaEventDestroy(start));MU(musaEventDestroy(end));BL(mublasDestroy(h));MU(musaStreamDestroy(stream));
}
