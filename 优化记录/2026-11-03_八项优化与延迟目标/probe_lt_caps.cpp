#include <musa_runtime.h>
#include <mublasLt.h>
#include <dlfcn.h>
#include <cstdio>
#include <vector>
int main() {
 if(!dlopen("libmusa.so.1",RTLD_NOW|RTLD_GLOBAL)){puts(dlerror());return 2;}
 if(musaSetDevice(0)!=musaSuccess)return 3;
 mublasLtHandle_t lt;auto status=mublasLtCreate(&lt);if(status)return 4;
 for(auto compute:{MUBLAS_COMPUTE_32F,MUBLAS_COMPUTE_32F_PEDANTIC,MUBLAS_COMPUTE_32F_FAST_16F,MUBLAS_COMPUTE_16F}) {
  bool half=compute==MUBLAS_COMPUTE_16F;auto scale=half?MUSA_R_16F:MUSA_R_32F;auto out=scale;
  int ids[256]={},count=0;
  auto s=mublasLtMatmulAlgoGetIds(lt,compute,scale,MUSA_R_16F,MUSA_R_16F,out,out,256,ids,&count);
  printf("{\"compute\":%d,\"ids_status\":%d,\"count\":%d,\"ids\":[",(int)compute,(int)s,count);
  for(int i=0;i<count&&i<256;++i)printf("%s%d",i?",":"",ids[i]);puts("]}");fflush(stdout);
  if(s)continue;
  for(int i=0;i<count&&i<256;++i) {
   mublasLtMatmulAlgo_t a{};
   auto init=mublasLtMatmulAlgoInit(lt,compute,scale,MUSA_R_16F,MUSA_R_16F,out,out,ids[i],&a);
   printf("{\"id\":%d,\"compute\":%d,\"init\":%d}\n",ids[i],(int)compute,(int)init);
   if(init)continue;
   for(auto attr:{MUBLASLT_ALGO_CAP_TILE_IDS,MUBLASLT_ALGO_CAP_STAGES_IDS,MUBLASLT_ALGO_CAP_SPLITK_SUPPORT,
       MUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK,MUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT,MUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX}) {
    size_t size=0;auto q=mublasLtMatmulAlgoCapGetAttribute(&a,attr,nullptr,0,&size);
    std::vector<int> data(4096,0);size_t written=0;int rc=-1;
    if(!q && size && size<=data.size()*4)rc=mublasLtMatmulAlgoCapGetAttribute(&a,attr,data.data(),size,&written);
    printf("{\"id\":%d,\"compute\":%d,\"attr\":%d,\"query\":%d,\"bytes\":%zu,\"read\":%d,\"values\":[",ids[i],(int)compute,(int)attr,(int)q,size,rc);
    for(size_t j=0;j<written/4 && j<data.size();++j)printf("%s%d",j?",":"",data[j]);puts("]}");fflush(stdout);
   }
  }
 }
 mublasLtDestroy(lt);
}
