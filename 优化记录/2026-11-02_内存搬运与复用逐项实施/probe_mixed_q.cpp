#include <initializer_list>
#include <utility>
#include <musa_runtime.h>
#include <mublas.h>
#include <cstdio>
int main() {
 mublasHandle_t h=nullptr; if(mublasCreate(&h))return 2;
 for(auto shape : {std::pair<int,int>{50,328},{50,568},{278,278},{518,518}}) {
  int nq=shape.first,nk=shape.second;
  void *a=nullptr,*b=nullptr,*c=nullptr;
  musaMalloc(&a,256*nk*2);musaMalloc(&b,256*8*nq*4);musaMalloc(&c,nk*nq*8*4);
  musaMemset(a,0,256*nk*2);musaMemset(b,0,256*8*nq*4);
  float one=1,zero=0;
  auto status=mublasGemmStridedBatchedEx(h,MUBLAS_OP_T,MUBLAS_OP_N,nk,nq,256,
   &one,a,MUSA_R_16F,256,0,b,MUSA_R_32F,2048,256,&zero,c,MUSA_R_32F,nk,nk*nq,8,
   MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP);
  auto sync=musaDeviceSynchronize();
  printf("{\"nq\":%d,\"nk\":%d,\"mixed_f16_f32_status\":%d,\"sync\":%d}\n",nq,nk,(int)status,(int)sync);
  musaFree(a);musaFree(b);musaFree(c);
 }
 mublasDestroy(h);
}
