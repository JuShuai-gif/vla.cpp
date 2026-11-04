// Diagnostic capability probe, not an inference implementation.
// Exact integer reference, QINT8 scales=1; supported status alone is insufficient.
#include <musa_runtime.h>
#include <mudnncxx/mudnn.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
using namespace musa::dnn;
static void gpu(musaError_t s){if(s!=musaSuccess){fprintf(stderr,"%s\n",musaGetErrorString(s));exit(2);}}
int main(){
 const int M=64,N=64,K=256;
 std::vector<signed char>a(M*K),b(N*K);
 for(int i=0;i<M*K;i++)a[i]=(i*17+3)%7-3;
 for(int i=0;i<N*K;i++)b[i]=(i*13+2)%5-2;
 void *ga,*gb,*gc;gpu(musaMalloc(&ga,a.size()));gpu(musaMalloc(&gb,b.size()));gpu(musaMalloc(&gc,M*N*4));
 gpu(musaMemcpy(ga,a.data(),a.size(),musaMemcpyHostToDevice));gpu(musaMemcpy(gb,b.data(),b.size(),musaMemcpyHostToDevice));
 Handle h;
 Tensor::Type ins[]={Tensor::Type::INT8,Tensor::Type::QINT8};
 Tensor::Type outs[]={Tensor::Type::INT32,Tensor::Type::FLOAT};
 const char *in_names[]={"INT8","QINT8"},*out_names[]={"INT32","FLOAT"};
 MemoryMaintainer alloc=[](size_t size){void *p=nullptr;gpu(musaMalloc(&p,size));return MemoryHandler(p,[](void *v){musaFree(v);});};
 int correctness_failures=0;
 for(int it=0;it<2;it++)for(int ot=0;ot<2;ot++)for(int mode=0;mode<2;mode++){
  Tensor x,w,y;MatMul mm;
  Status s=x.SetType(ins[it]);
  if(s==Status::SUCCESS)s=w.SetType(ins[it]);
  if(s==Status::SUCCESS)s=y.SetType(outs[ot]);
  if(s==Status::SUCCESS)s=x.SetQuantizationInfo(1.f);
  if(s==Status::SUCCESS)s=w.SetQuantizationInfo(1.f);
  if(s==Status::SUCCESS)s=y.SetQuantizationInfo(1.f);
  if(s==Status::SUCCESS)s=x.SetNdInfo({M,K});
  if(s==Status::SUCCESS)s=w.SetNdInfo({N,K});
  if(s==Status::SUCCESS)s=y.SetNdInfo({M,N});
  if(s==Status::SUCCESS)s=x.SetAddr(ga);
  if(s==Status::SUCCESS)s=w.SetAddr(gb);
  if(s==Status::SUCCESS)s=y.SetAddr(gc);
  if(s==Status::SUCCESS)s=mm.SetTranspose(false,true);
  if(s==Status::SUCCESS)s=mm.SetComputeMode(mode?MatMul::ComputeMode::SCALAR:MatMul::ComputeMode::TENSOR);
  size_t ws=0;
  if(s==Status::SUCCESS)s=mm.GetWorkspaceSize(h,ws,y,x,w);
  const int workspace_status=int(s);
  if(s==Status::SUCCESS)s=mm.Run(h,y,x,w,alloc);
  int failed=0;double error=0;
  if(s==Status::SUCCESS){
   gpu(musaDeviceSynchronize());
   std::vector<int>ci(M*N);std::vector<float>cf(M*N);
   gpu(musaMemcpy(ot?(void*)cf.data():(void*)ci.data(),gc,M*N*4,musaMemcpyDeviceToHost));
   for(int m=0;m<M;m++)for(int n=0;n<N;n++){int ref=0;for(int k=0;k<K;k++)ref+=int(a[m*K+k])*int(b[n*K+k]);
    double actual=ot?cf[m*N+n]:ci[m*N+n];double e=std::abs(actual-ref);if(!std::isfinite(actual)||e>1e-5)failed++;error=std::max(error,e);}
   correctness_failures+=failed;
  }
  printf("{\"input\":\"%s\",\"output\":\"%s\",\"mode\":\"%s\",\"workspace_status\":%d,\"run_status\":%d,\"workspace_bytes\":%zu,\"failed\":%d,\"max_error\":%.9g}\n",in_names[it],out_names[ot],mode?"SCALAR":"TENSOR",workspace_status,int(s),ws,failed,error);fflush(stdout);
 }
 musaFree(ga);musaFree(gb);musaFree(gc);return correctness_failures?1:0;
}
