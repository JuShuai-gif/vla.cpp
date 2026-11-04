// Diagnostic only: check INT8 library layouts against an exact CPU integer reference.
#include <musa_runtime.h>
#include <mublas.h>
#include <mublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
static void gpu(musaError_t s){if(s!=musaSuccess){fprintf(stderr,"%s\n",musaGetErrorString(s));exit(2);}}
static void ok(mublasStatus_t s){if(s!=MUBLAS_STATUS_SUCCESS){fprintf(stderr,"setup status %d\n",int(s));exit(3);}}
int main(){
 const int M=64,N=64,K=256;const size_t ws=32*1024*1024;
 void *a,*b,*c,*workspace;gpu(musaMalloc(&a,M*K));gpu(musaMalloc(&b,N*K));gpu(musaMalloc(&c,M*N*4));gpu(musaMalloc(&workspace,ws));
 mublasHandle_t h;ok(mublasCreate(&h));mublasLtHandle_t lt;ok(mublasLtCreate(&lt));
 std::vector<int>ref(M*N),ci(M*N);std::vector<float>cf(M*N);
 for(int n=0;n<N;n++)for(int m=0;m<M;m++)for(int k=0;k<K;k++)ref[n*M+m]+=((m*17+k*3)%13-6)*((n*7+k*11)%9-4);
 int total_failures=0;
 for(int ta=0;ta<2;ta++)for(int tb=0;tb<2;tb++)for(int output=0;output<2;output++){
  auto opa=ta?MUBLAS_OP_T:MUBLAS_OP_N,opb=tb?MUBLAS_OP_T:MUBLAS_OP_N;
  int lda=ta?K:M,ldb=tb?N:K;
  std::vector<signed char>ha(M*K),hb(N*K);
  for(int m=0;m<M;m++)for(int k=0;k<K;k++)ha[ta?m*K+k:k*M+m]=(m*17+k*3)%13-6;
  for(int n=0;n<N;n++)for(int k=0;k<K;k++)hb[tb?k*N+n:n*K+k]=(n*7+k*11)%9-4;
  gpu(musaMemcpy(a,ha.data(),ha.size(),musaMemcpyHostToDevice));gpu(musaMemcpy(b,hb.data(),hb.size(),musaMemcpyHostToDevice));
  auto dtype=output?MUSA_R_32F:MUSA_R_32I;auto compute=output?MUBLAS_COMPUTE_32F:MUBLAS_COMPUTE_32I;
  int one=1,zero=0;float fone=1,fzero=0;const void*alpha=output?(void*)&fone:(void*)&one;const void*beta=output?(void*)&fzero:(void*)&zero;
  auto report=[&](const char*path,int algo,mublasStatus_t status,int heuristic,int count){
    int failures=0,checks=0;double error=0;
    if(status==MUBLAS_STATUS_SUCCESS){gpu(musaDeviceSynchronize());gpu(musaMemcpy(output?(void*)cf.data():(void*)ci.data(),c,M*N*4,musaMemcpyDeviceToHost));
      checks=M*N;for(int i=0;i<M*N;i++){double value=output?cf[i]:ci[i];double e=std::fabs(value-ref[i]);if(!std::isfinite(value)||e>1e-5)++failures;error=std::max(error,e);}}
    total_failures+=failures;
    printf("{\"path\":\"%s\",\"transa\":%d,\"transb\":%d,\"float_output\":%d,\"algo\":%d,\"status\":%d,\"heuristic_status\":%d,\"algo_count\":%d,\"checks\":%d,\"failures\":%d,\"max_error\":%.9g}\n",path,ta,tb,output,algo,int(status),heuristic,count,checks,failures,error);fflush(stdout);
  };
  for(int algo=0;algo<2;algo++){
    gpu(musaMemset(c,0,M*N*4));
    auto status=mublasGemmEx(h,opa,opb,M,N,K,alpha,a,MUSA_R_8I,lda,b,MUSA_R_8I,ldb,beta,c,dtype,M,compute,algo?MUBLAS_GEMM_DEFAULT_TENSOR_OP:MUBLAS_GEMM_DEFAULT);
    report("GemmEx",algo,status,-1,0);
  }
  mublasLtMatmulDesc_t desc;ok(mublasLtMatmulDescCreate(&desc,compute,dtype));
  ok(mublasLtMatmulDescSetAttribute(desc,MUBLASLT_MATMUL_DESC_TRANSA,&opa,sizeof(opa)));
  ok(mublasLtMatmulDescSetAttribute(desc,MUBLASLT_MATMUL_DESC_TRANSB,&opb,sizeof(opb)));
  mublasLtMatrixLayout_t ad,bd,cd;
  ok(mublasLtMatrixLayoutCreate(&ad,MUSA_R_8I,ta?K:M,ta?M:K,lda));
  ok(mublasLtMatrixLayoutCreate(&bd,MUSA_R_8I,tb?N:K,tb?K:N,ldb));
  ok(mublasLtMatrixLayoutCreate(&cd,dtype,M,N,M));
  mublasLtMatmulPreference_t pref;ok(mublasLtMatmulPreferenceCreate(&pref));
  ok(mublasLtMatmulPreferenceSetAttribute(pref,MUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws)));
  mublasLtMatmulHeuristicResult_t result{};int count=0;
  auto hs=mublasLtMatmulAlgoGetHeuristic(lt,desc,ad,bd,cd,cd,pref,1,&result,&count);
  gpu(musaMemset(c,0,M*N*4));
  auto status=mublasLtMatmul(lt,desc,alpha,a,ad,b,bd,beta,c,cd,c,cd,count?&result.algo:nullptr,workspace,ws,0);
  report("LtMatmul",-1,status,int(hs),count);
  mublasLtMatmulPreferenceDestroy(pref);mublasLtMatmulDescDestroy(desc);mublasLtMatrixLayoutDestroy(ad);mublasLtMatrixLayoutDestroy(bd);mublasLtMatrixLayoutDestroy(cd);
 }
 mublasDestroy(h);mublasLtDestroy(lt);musaFree(a);musaFree(b);musaFree(c);musaFree(workspace);return total_failures?1:0;
}
