#include <musa_runtime.h>
#include <musa_fp16.h>
#include <mublas.h>
#include <mublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <functional>
#include <cstring>
#define MU(x) do{auto e=(x);if(e!=musaSuccess){fprintf(stderr,"%s: %s\n",#x,musaGetErrorString(e));exit(2);}}while(0)
#define BL(x) do{auto e=(x);if(e!=MUBLAS_STATUS_SUCCESS){fprintf(stderr,"%s: %d\n",#x,(int)e);exit(3);}}while(0)
int main(int argc,char**argv){
 if(argc!=5)return 1;
 int m=atoi(argv[1]),n=atoi(argv[2]),k=atoi(argv[3]),batch=atoi(argv[4]);
 if(m<=0||n<=0||k<=0||batch<=0)return 1;
 musaStream_t stream;MU(musaStreamCreate(&stream));
 mublasHandle_t blas;BL(mublasCreate(&blas));BL(mublasSetStream(blas,stream));
 mublasLtHandle_t lt;BL(mublasLtCreate(&lt));
 size_t na=(size_t)m*k,nb=(size_t)k*n*batch,nc=(size_t)m*n*batch;
 std::vector<_Float16> ha(na),hb(nb);
 for(size_t i=0;i<na;i++)ha[i]=_Float16(((int)(i*17%101)-50)*.002f);
 for(size_t i=0;i<nb;i++)hb[i]=_Float16(((int)(i*13%97)-48)*.002f);
 _Float16 *a,*b;float *c,*ref;void *ws;
 MU(musaMalloc(&a,na*2));MU(musaMalloc(&b,nb*2));
 MU(musaMalloc(&c,nc*4));MU(musaMalloc(&ref,nc*4));MU(musaMalloc(&ws,64<<20));
 MU(musaMemcpy(a,ha.data(),na*2,musaMemcpyHostToDevice));MU(musaMemcpy(b,hb.data(),nb*2,musaMemcpyHostToDevice));
 float alpha=1,beta=0;
 auto base=[&](){if(batch==1)return mublasGemmEx(blas,MUBLAS_OP_T,MUBLAS_OP_N,m,n,k,&alpha,a,MUSA_R_16F,k,b,MUSA_R_16F,k,&beta,ref,MUSA_R_32F,m,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP);return mublasGemmStridedBatchedEx(blas,MUBLAS_OP_T,MUBLAS_OP_N,m,n,k,
 &alpha,a,MUSA_R_16F,k,0,b,MUSA_R_16F,k,(long long)k*n,&beta,ref,MUSA_R_32F,m,
 (long long)m*n,batch,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP);};
 musaEvent_t start,end;MU(musaEventCreate(&start));MU(musaEventCreate(&end));
 auto timing=[&](const std::function<mublasStatus_t()>&fn){
   for(int i=0;i<3;i++)BL(fn());MU(musaStreamSynchronize(stream));
   MU(musaEventRecord(start,stream));for(int i=0;i<10;i++)BL(fn());MU(musaEventRecord(end,stream));
   MU(musaEventSynchronize(end));float ms;MU(musaEventElapsedTime(&ms,start,end));return ms/10;
 };
 float base_ms=timing(base);
 std::vector<float> href(nc),hout(nc);MU(musaMemcpy(href.data(),ref,nc*4,musaMemcpyDeviceToHost));
 mublasLtMatmulDesc_t op;BL(mublasLtMatmulDescCreate(&op,MUBLAS_COMPUTE_32F,MUSA_R_32F));
 mublasOperation_t ta=MUBLAS_OP_T;BL(mublasLtMatmulDescSetAttribute(op,MUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
 mublasLtMatrixLayout_t ad,bd,cd;
 BL(mublasLtMatrixLayoutCreate(&ad,MUSA_R_16F,k,m,k));BL(mublasLtMatrixLayoutCreate(&bd,MUSA_R_16F,k,n,k));
 BL(mublasLtMatrixLayoutCreate(&cd,MUSA_R_32F,m,n,m));
 for(auto desc:{ad,bd,cd})BL(mublasLtMatrixLayoutSetAttribute(desc,MUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&batch,sizeof(batch)));
 int64_t sa=0,sb=(int64_t)k*n,sc=(int64_t)m*n;
 BL(mublasLtMatrixLayoutSetAttribute(ad,MUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&sa,sizeof(sa)));
 BL(mublasLtMatrixLayoutSetAttribute(bd,MUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&sb,sizeof(sb)));
 BL(mublasLtMatrixLayoutSetAttribute(cd,MUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&sc,sizeof(sc)));
 mublasLtMatmulPreference_t pref;BL(mublasLtMatmulPreferenceCreate(&pref));
 printf("{\"kind\":\"baseline\",\"m\":%d,\"n\":%d,\"k\":%d,\"batch\":%d,\"ms\":%.6f,\"lt_version\":%zu}\n",m,n,k,batch,base_ms,mublasLtGetVersion());fflush(stdout);
 std::vector<mublasLtMatmulAlgo_t> seen;
 float best_ms=1e30; mublasLtMatmulAlgo_t best{};size_t best_ws=0;
 for(size_t work:{size_t(0),size_t(8<<20),size_t(32<<20),size_t(64<<20)}){
  BL(mublasLtMatmulPreferenceSetAttribute(pref,MUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&work,sizeof(work)));
  mublasLtMatmulHeuristicResult_t results[16];int count=0;
  auto status=mublasLtMatmulAlgoGetHeuristic(lt,op,ad,bd,cd,cd,pref,16,results,&count);
  printf("{\"kind\":\"query\",\"workspace\":%zu,\"status\":%d,\"count\":%d}\n",work,(int)status,count);fflush(stdout);
  if(status!=MUBLAS_STATUS_SUCCESS)continue;
  for(int i=0;i<count;i++){
   auto &h=results[i];if(h.state!=MUBLAS_STATUS_SUCCESS)continue;
   for(auto attr:{MUBLASLT_ALGO_CAP_TILE_IDS,MUBLASLT_ALGO_CAP_STAGES_IDS,MUBLASLT_ALGO_CAP_SPLITK_SUPPORT,MUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT,MUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX}) {
    int data[256]={};size_t written=0;
    auto cap=mublasLtMatmulAlgoCapGetAttribute(&h.algo,attr,data,sizeof(data),&written);
    printf("{\"kind\":\"cap\",\"attr\":%d,\"status\":%d,\"bytes\":%zu,\"values\":[",(int)attr,(int)cap,written);
    for(size_t ci=0;ci<written/4 && ci<256;++ci)printf("%s%d",ci?",":"",data[ci]);puts("]}");fflush(stdout);
   }

   bool duplicate=false;for(auto &x:seen)if(!memcmp(&x,&h.algo,sizeof(x)))duplicate=true;
   if(duplicate)continue;seen.push_back(h.algo);
   auto run=[&](){return mublasLtMatmul(lt,op,&alpha,a,ad,b,bd,&beta,c,cd,c,cd,&h.algo,ws,work,stream);};
   auto r=run();if(r!=MUBLAS_STATUS_SUCCESS){printf("{\"kind\":\"candidate\",\"id\":%zu,\"status\":%d}\n",seen.size()-1,(int)r);continue;}
   float ms=timing(run);MU(musaMemcpy(hout.data(),c,nc*4,musaMemcpyDeviceToHost));
   double peak=0,diff=0;for(size_t j=0;j<nc;j++){peak=fmax(peak,fabs(href[j]));diff=fmax(diff,fabs(href[j]-hout[j]));}
   double rel=diff/fmax(peak,1e-12);
   printf("{\"kind\":\"candidate\",\"id\":%zu,\"workspace\":%zu,\"ms\":%.6f,\"relative_peak\":%.9g}\n",seen.size()-1,work,ms,rel);fflush(stdout);
   if(rel<1e-3 && ms<best_ms){best_ms=ms;best=h.algo;best_ws=work;}
  }
 }
 if(best_ms<1e29){
   MU(musaStreamSynchronize(stream));MU(musaStreamBeginCapture(stream,musaStreamCaptureModeGlobal));
   auto status=mublasLtMatmul(lt,op,&alpha,a,ad,b,bd,&beta,c,cd,c,cd,&best,ws,best_ws,stream);
   musaGraph_t graph=nullptr;auto capture=musaStreamEndCapture(stream,&graph);
   int instantiate=-1,launch=-1;double replay_error=-1;float replay_ms=-1;
   if(status==MUBLAS_STATUS_SUCCESS && capture==musaSuccess && graph){
     musaGraphExec_t exec=nullptr;instantiate=(int)musaGraphInstantiate(&exec,graph,nullptr,nullptr,0);
     if(instantiate==0){
       launch=(int)musaGraphLaunch(exec,stream);MU(musaStreamSynchronize(stream));
       MU(musaMemcpy(hout.data(),c,nc*4,musaMemcpyDeviceToHost));
       double peak=0,diff=0;for(size_t j=0;j<nc;j++){peak=fmax(peak,fabs(href[j]));diff=fmax(diff,fabs(href[j]-hout[j]));}
       replay_error=diff/fmax(peak,1e-12);
       MU(musaEventRecord(start,stream));for(int i=0;i<10;i++)MU(musaGraphLaunch(exec,stream));
       MU(musaEventRecord(end,stream));MU(musaEventSynchronize(end));MU(musaEventElapsedTime(&replay_ms,start,end));replay_ms/=10;
       MU(musaGraphExecDestroy(exec));
     }MU(musaGraphDestroy(graph));
   }
   printf("{\"kind\":\"best\",\"ms\":%.6f,\"baseline_ms\":%.6f,\"capture_call\":%d,\"capture_end\":%d,\"instantiate\":%d,\"launch\":%d,\"replay_error\":%.9g,\"replay_ms\":%.6f}\n",best_ms,base_ms,(int)status,(int)capture,instantiate,launch,replay_error,replay_ms);
 }
 mublasLtMatmulPreferenceDestroy(pref);mublasLtMatrixLayoutDestroy(ad);mublasLtMatrixLayoutDestroy(bd);mublasLtMatrixLayoutDestroy(cd);mublasLtMatmulDescDestroy(op);mublasLtDestroy(lt);mublasDestroy(blas);
 MU(musaFree(a));MU(musaFree(b));MU(musaFree(c));MU(musaFree(ref));MU(musaFree(ws));MU(musaEventDestroy(start));MU(musaEventDestroy(end));MU(musaStreamDestroy(stream));
}
