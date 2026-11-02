#include <musa_runtime.h>
#include <cstdio>
#include <initializer_list>
#include <dlfcn.h>
int main() {
 musaDeviceProp p{};auto e=musaGetDeviceProperties(&p,0);
 if(e!=musaSuccess){printf("error=%s\n",musaGetErrorString(e));return 1;}
 printf("name=%s cc=%d.%d MPs=%d warp=%d shared_block=%zu shared_optin=%zu L2=%d persist_L2_max=%d access_window_max=%d integrated=%d map_host=%d async_engines=%d\n",
 p.name,p.major,p.minor,p.multiProcessorCount,p.warpSize,p.sharedMemPerBlock,p.sharedMemPerBlockOptin,p.l2CacheSize,p.persistingL2CacheMaxSize,p.accessPolicyMaxWindowSize,p.integrated,p.canMapHostMemory,p.asyncEngineCount);
 void * driver=dlopen("libmusa.so.1",RTLD_NOW|RTLD_GLOBAL);
 printf("driver_global_loaded=%d error=%s\n",driver!=nullptr,driver?"none":dlerror());
 for(const char * lib:{"libmublasLt.so","libmudnn.so"}) {
   void * h=dlopen(lib,RTLD_NOW|RTLD_LOCAL);printf("%s loaded=%d error=%s\n",lib,h!=nullptr,h?"none":dlerror());
   if(h)for(const char * name:{"mublasLtMatmulAlgoGetHeuristic","mublasLtMatmulAlgoGetIds","mudnnScaledDotProductAttentionFlashForward","mudnnCreateScaledDotProductAttentionDescriptor"})printf("  %s=%d\n",name,dlsym(h,name)!=nullptr);
   if(h)dlclose(h);
 }
}
