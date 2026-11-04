// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
#include <musa_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
static void check(musaError_t e){if(e!=musaSuccess){fprintf(stderr,"%s\n",musaGetErrorString(e));exit(2);}}
__global__ void probe(const signed char *a,int *frag,int *positions) {
    using namespace mtmusa::wmma;
    fragment<matrix_a,32,32,32,signed char,row_major> af;
    for(int i=0;i<af.num_elements;++i)af.x[i]=-1;
    load_matrix_sync(af,a,32);
    for(int i=0;i<af.num_elements;++i)frag[threadIdx.x*af.num_elements+i]=(unsigned char)af.x[i];
    fragment<accumulator,32,32,32,int> cf;
    for(int i=0;i<cf.num_elements;++i)cf.x[i]=threadIdx.x*100+i;
    store_matrix_sync(positions,cf,32,mem_col_major);
}
int main(){
    signed char *a;int *f,*p;check(musaMalloc(&a,1024));check(musaMalloc(&f,128*32*4));check(musaMalloc(&p,1024*4));
    std::vector<signed char> in(1024);std::vector<int> lo(4096),hi(4096),pos(1024);
    for(int mode=0;mode<2;++mode){
        for(int i=0;i<1024;++i)in[i]=mode?i/256:i%256;
        check(musaMemcpy(a,in.data(),1024,musaMemcpyHostToDevice));probe<<<1,128>>>(a,f,p);
        check(musaDeviceSynchronize());check(musaMemcpy((mode?hi:lo).data(),f,4096*4,musaMemcpyDeviceToHost));
    }
    check(musaMemcpy(pos.data(),p,1024*4,musaMemcpyDeviceToHost));
    printf("{\"input_indices\":[");
    for(int i=0;i<4096;++i)printf("%s%d",i?",":"",lo[i]+256*hi[i]);
    printf("],\"output_registers\":[");
    for(int i=0;i<1024;++i)printf("%s%d",i?",":"",pos[i]);
    printf("]}\n");
    musaFree(a);musaFree(f);musaFree(p);
}
