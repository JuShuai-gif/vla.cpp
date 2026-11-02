#include <musa_runtime.h>
#include <mudnn.h>
#include <mublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#define MU(x) do{auto e=(x);if(e!=musaSuccess){printf("MUSA_ERROR %s %s\n",#x,musaGetErrorString(e));return 2;}}while(0)
#define DN(x) do{auto e=(x);printf("API %s status=%d\n",#x,(int)e);if(e!=MUDNN_STATUS_SUCCESS)return 3;}while(0)
int main(int argc,char**argv){
 setbuf(stdout,nullptr);
 if(argc==1){
   mublasLtHandle_t h;mublasLtCreate(&h);int ids[256],count=0;
   auto e=mublasLtMatmulAlgoGetIds(h,MUBLAS_COMPUTE_32F,MUSA_R_32F,MUSA_R_16F,MUSA_R_16F,MUSA_R_32F,MUSA_R_32F,256,ids,&count);
   printf("LT_IDS status=%d count=%d\n",(int)e,count);for(int i=0;i<count;i++)printf("%d\n",ids[i]);mublasLtDestroy(h);return 0;
 }
 if(argc!=7)return 1;
 int d=atoi(argv[1]),sq=atoi(argv[2]),sk=atoi(argv[3]),hq=atoi(argv[4]),hk=atoi(argv[5]),embed=atoi(argv[6]);
 mudnnHandle_t h;DN(mudnnCreate(&h));
 mudnnScaledDotProductAttentionDescriptor_t sd;DN(mudnnCreateScaledDotProductAttentionDescriptor(&sd));
 DN(mudnnSetScaledDotProductAttentionDescriptor(sd,MUDNN_TENSOR_OP_MATH,embed,1,hq,1.0/sqrt(d),false,false,false,false,true));
 mudnnTensorDescriptor_t q,k,v,o,l,empty;
 DN(mudnnCreateTensorDescriptor(&q));DN(mudnnCreateTensorDescriptor(&k));DN(mudnnCreateTensorDescriptor(&v));
 DN(mudnnCreateTensorDescriptor(&o));DN(mudnnCreateTensorDescriptor(&l));DN(mudnnCreateTensorDescriptor(&empty));
 int dq[]={1,hq,sq,d},dk[]={1,hk,sk,d},stq[]={hq*sq*d,sq*d,d,1},stk[]={hk*sk*d,sk*d,d,1};
 DN(mudnnSetTensorNdDescriptor(q,MUDNN_DATA_HALF,4,dq,stq));DN(mudnnSetTensorNdDescriptor(o,MUDNN_DATA_HALF,4,dq,stq));
 DN(mudnnSetTensorNdDescriptor(k,MUDNN_DATA_HALF,4,dk,stk));DN(mudnnSetTensorNdDescriptor(v,MUDNN_DATA_HALF,4,dk,stk));
 int dl[]={1,hq,sq,1},stl[]={hq*sq,sq,1,1};DN(mudnnSetTensorNdDescriptor(l,MUDNN_DATA_FLOAT,4,dl,stl));
 void *pq,*pk,*pv,*po,*pl;size_t nq=(size_t)hq*sq*d,nk=(size_t)hk*sk*d;
 MU(musaMalloc(&pq,nq*2));MU(musaMalloc(&pk,nk*2));MU(musaMalloc(&pv,nk*2));MU(musaMalloc(&po,nq*2));MU(musaMalloc(&pl,hq*sq*4));
 MU(musaMemset(pq,0,nq*2));MU(musaMemset(pk,0,nk*2));
 std::vector<_Float16> hv(nk);for(size_t i=0;i<nk;i++)hv[i]=_Float16((int(i%17)-8)*.0625f);
 MU(musaMemcpy(pv,hv.data(),nk*2,musaMemcpyHostToDevice));
 MU(musaMemset(po,0x7f,nq*2));
 auto e=mudnnScaledDotProductAttentionFlashForward(h,l,pl,q,pq,k,pk,v,pv,empty,nullptr,sd,empty,nullptr,o,po);
 printf("FORWARD d=%d sq=%d sk=%d hq=%d hk=%d embed=%d status=%d\n",d,sq,sk,hq,hk,embed,(int)e);
 if(e==MUDNN_STATUS_SUCCESS){
   MU(musaDeviceSynchronize());std::vector<_Float16> out(nq);MU(musaMemcpy(out.data(),po,nq*2,musaMemcpyDeviceToHost));
   for(auto x:out)if(!std::isfinite(float(x))){printf("NONFINITE_OR_UNWRITTEN_OUTPUT\n");return 8;}
   double maxerr=0;for(int head=0;head<hq;head++)for(int j=0;j<d;j++){
     double sum=0;for(int t=0;t<sk;t++)sum+=float(hv[((head/(hq/hk))*sk+t)*d+j]);sum/=sk;
     for(int t=0;t<sq;t++)maxerr=fmax(maxerr,fabs(float(out[(head*sq+t)*d+j])-sum));
   }printf("UNIFORM_ATTN_MAX_ABS %.9g\n",maxerr);if(maxerr>.001)return 4;
   std::vector<_Float16> hqv(nq),hkv(nk);for(size_t i=0;i<nq;i++)hqv[i]=_Float16(sin(i*.071)*1.3);
   for(size_t i=0;i<nk;i++)hkv[i]=_Float16(cos(i*.053)*1.1);
   MU(musaMemcpy(pq,hqv.data(),nq*2,musaMemcpyHostToDevice));MU(musaMemcpy(pk,hkv.data(),nk*2,musaMemcpyHostToDevice));
   auto run=[&](){return mudnnScaledDotProductAttentionFlashForward(h,l,pl,q,pq,k,pk,v,pv,empty,nullptr,sd,empty,nullptr,o,po);};
   DN(run());MU(musaDeviceSynchronize());MU(musaMemcpy(out.data(),po,nq*2,musaMemcpyDeviceToHost));
   for(auto x:out)if(!std::isfinite(float(x))){printf("NONFINITE_RANDOM_OUTPUT\n");return 8;}
   maxerr=0;double peak=0;std::vector<double> scores(sk);
   for(int head=0;head<hq;head++)for(int iq=0;iq<sq;iq++){
     double mx=-1e30,sum=0;int kh=head/(hq/hk);
     for(int ik=0;ik<sk;ik++){double dot=0;for(int j=0;j<d;j++)dot+=float(hqv[(head*sq+iq)*d+j])*float(hkv[(kh*sk+ik)*d+j]);scores[ik]=dot/sqrt(d);mx=fmax(mx,scores[ik]);}
     for(auto &x:scores){x=exp(x-mx);sum+=x;}
     for(int j=0;j<d;j++){double ref=0;for(int ik=0;ik<sk;ik++)ref+=scores[ik]/sum*float(hv[(kh*sk+ik)*d+j]);maxerr=fmax(maxerr,fabs(float(out[(head*sq+iq)*d+j])-ref));peak=fmax(peak,fabs(ref));}
   }printf("RANDOM_ATTN_MAX_ABS %.9g RELATIVE_PEAK %.9g\n",maxerr,maxerr/fmax(peak,1e-12));if(maxerr>.002)return 6;
   const auto expected=out;
   musaEvent_t begin,end;MU(musaEventCreate(&begin));MU(musaEventCreate(&end));
   for(int i=0;i<3;i++)if(run()!=MUDNN_STATUS_SUCCESS)return 7;
   MU(musaDeviceSynchronize());MU(musaEventRecord(begin));for(int i=0;i<20;i++)if(run()!=MUDNN_STATUS_SUCCESS)return 7;
   MU(musaEventRecord(end));MU(musaEventSynchronize(end));float ms;MU(musaEventElapsedTime(&ms,begin,end));printf("EVENT_MS %.6f\n",ms/20);
   musaStream_t stream;MU(musaStreamCreate(&stream));DN(mudnnSetStream(h,stream));
   DN(run());MU(musaStreamSynchronize(stream));MU(musaStreamBeginCapture(stream,musaStreamCaptureModeGlobal));
   auto call=run();musaGraph_t graph=nullptr;auto capture=musaStreamEndCapture(stream,&graph);
   printf("CAPTURE_CALL %d END %d\n",(int)call,(int)capture);
   if(call==MUDNN_STATUS_SUCCESS && capture==musaSuccess && graph){
     musaGraphExec_t exec=nullptr;auto inst=musaGraphInstantiate(&exec,graph,nullptr,nullptr,0);printf("INSTANTIATE %d\n",(int)inst);
     if(inst==musaSuccess){
       MU(musaMemset(po,0x7f,nq*2));
       MU(musaGraphLaunch(exec,stream));MU(musaStreamSynchronize(stream));
       MU(musaMemcpy(out.data(),po,nq*2,musaMemcpyDeviceToHost));
       double replay_err=0;
       for(size_t i=0;i<nq;i++){
          if(!std::isfinite(float(out[i]))){printf("NONFINITE_GRAPH_OUTPUT\n");return 9;}
          replay_err=fmax(replay_err,fabs(float(out[i])-float(expected[i])));
       }
       printf("GRAPH_MAX_ABS %.9g\n",replay_err);if(replay_err>.001)return 10;
       MU(musaGraphExecDestroy(exec));
     }
     MU(musaGraphDestroy(graph));
   }
   MU(musaStreamDestroy(stream));MU(musaEventDestroy(begin));MU(musaEventDestroy(end));

 }
 musaFree(pq);musaFree(pk);musaFree(pv);musaFree(po);musaFree(pl);
 for(auto desc:{q,k,v,o,l,empty})mudnnDestroyTensorDescriptor(desc);
 mudnnDestroyScaledDotProductAttentionDescriptor(sd);mudnnDestroy(h);return e==MUDNN_STATUS_SUCCESS?0:5;
}
