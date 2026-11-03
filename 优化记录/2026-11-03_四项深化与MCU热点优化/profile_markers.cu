#include <musa_runtime.h>
__global__ void vla_r2_begin() {}
__global__ void vla_r2_end() {}
extern "C" int r2_begin() {
 auto e=musaDeviceSynchronize();if(e!=musaSuccess)return int(e);
 vla_r2_begin<<<1,128>>>();return int(musaDeviceSynchronize());
}
extern "C" int r2_end() {
 vla_r2_end<<<1,128>>>();return int(musaDeviceSynchronize());
}
