// Probe the actual mp_22 WMMA paths; not a throughput benchmark.
#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <musa_fp8.h>
#include <mma.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
using namespace mtmusa::wmma;
#if KIND == 0
using A=__half;using B=__half;using C=float;
constexpr int M=32,N=32,K=16;constexpr unsigned ABITS=0x3c00,BBITS=0x3c00;
#elif KIND == 1
using A=__mt_bfloat16;using B=__mt_bfloat16;using C=float;
constexpr int M=32,N=32,K=16;constexpr unsigned ABITS=0x3f80,BBITS=0x3f80;
#elif KIND == 2
using A=signed char;using B=signed char;using C=int;
constexpr int M=32,N=32,K=32;constexpr unsigned ABITS=1,BBITS=1;
#elif KIND == 3
using A=signed char;using B=__half;using C=float;
constexpr int M=16,N=16,K=16;constexpr unsigned ABITS=1,BBITS=0x3c00;
#elif KIND == 4
using A=__mt_fp8_e4m3;using B=__mt_fp8_e4m3;using C=float;
constexpr int M=16,N=16,K=16;constexpr unsigned ABITS=0x38,BBITS=0x38;
#elif KIND == 5
using A=float;using B=float;using C=float;
constexpr int M=32,N=32,K=8;constexpr unsigned ABITS=0x3f800000,BBITS=0x3f800000;
#endif
__global__ void multiply(const A*a,const B*b,C*c){
#if KIND == 5
 fragment<matrix_a,M,N,K,precision::tf32,row_major> af;
 fragment<matrix_b,M,N,K,precision::tf32,col_major> bf;
#else
 fragment<matrix_a,M,N,K,A,row_major> af;
 fragment<matrix_b,M,N,K,B,col_major> bf;
#endif
 fragment<accumulator,M,N,K,C> cf;
 fill_fragment(cf,C(0));load_matrix_sync(af,a,K);load_matrix_sync(bf,b,K);
 mma_sync(cf,af,bf,cf);store_matrix_sync(c,cf,N,mem_row_major);
}
static void gpu(musaError_t s){if(s!=musaSuccess){fprintf(stderr,"%s\n",musaGetErrorString(s));exit(2);}}
int main(){
 std::vector<A>a(M*K);std::vector<B>b(N*K);std::vector<C>c(M*N);
 auto av=[](int m,int k){return (m*3+k*7+1)%11<5;};
 auto bv=[](int n,int k){return (n*7+k*3+2)%13<6;};
 for(int m=0;m<M;m++)for(int k=0;k<K;k++){unsigned bits=av(m,k)?ABITS:0;std::memcpy(&a[m*K+k],&bits,sizeof(A));}
 for(int n=0;n<N;n++)for(int k=0;k<K;k++){unsigned bits=bv(n,k)?BBITS:0;std::memcpy(&b[n*K+k],&bits,sizeof(B));}
 A*ga;B*gb;C*gc;gpu(musaMalloc(&ga,a.size()*sizeof(A)));gpu(musaMalloc(&gb,b.size()*sizeof(B)));gpu(musaMalloc(&gc,c.size()*sizeof(C)));
 gpu(musaMemcpy(ga,a.data(),a.size()*sizeof(A),musaMemcpyHostToDevice));gpu(musaMemcpy(gb,b.data(),b.size()*sizeof(B),musaMemcpyHostToDevice));gpu(musaMemset(gc,0xff,c.size()*sizeof(C)));
 multiply<<<1,128>>>(ga,gb,gc);gpu(musaGetLastError());gpu(musaDeviceSynchronize());gpu(musaMemcpy(c.data(),gc,c.size()*sizeof(C),musaMemcpyDeviceToHost));
 int failed=0;double error=0;
 for(int m=0;m<M;m++)for(int n=0;n<N;n++){int ref=0;for(int k=0;k<K;k++)ref+=av(m,k)*bv(n,k);double e=std::abs(double(c[m*N+n])-ref);if(!std::isfinite(double(c[m*N+n]))||e>1e-5)failed++;error=std::max(error,e);}
 printf("{\"kind\":%d,\"checks\":%d,\"failed\":%d,\"max_error\":%.9g}\n",KIND,M*N,failed,error);
 musaFree(ga);musaFree(gb);musaFree(gc);return failed?1:0;
}
