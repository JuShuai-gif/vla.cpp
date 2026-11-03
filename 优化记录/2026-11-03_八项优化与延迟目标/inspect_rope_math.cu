#include <musa_runtime.h>
#include <musa_fp16.h>
__global__ void rfloat(const float * x,float * y) {int i=threadIdx.x;float a=x[i],b=x[i+128],c=x[i+256],s=x[i+384];y[i]=a*c-b*s;y[i+128]=a*s+b*c;}
__global__ void rhalf(const float * x,half * y) {int i=threadIdx.x;float a=x[i],b=x[i+128],c=x[i+256],s=x[i+384];y[i]=__float2half_rn(a*c-b*s);y[i+128]=__float2half_rn(a*s+b*c);}
