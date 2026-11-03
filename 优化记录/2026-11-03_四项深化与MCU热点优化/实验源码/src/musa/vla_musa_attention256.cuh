// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
// MP22: one warp owns 16 queries; QK, F32 softmax and AV share scratch.
// Normalized probabilities still round to F16, matching the muBLAS path.
__global__ void pack_gemma_tiled_q(const float * src,half * dst,int nq,int padded,size_t s1,size_t s2) {
 const int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=256*padded*8)return;
 const int d=i%256,q=i/256%padded,h=i/(256*padded);
 dst[i]=q<nq?__float2half_rn(src[d+q*s1+h*s2]):__float2half_rn(0.f);
}
template<bool Parallel=false>
__global__ void gemma_tiled_256(const half * q,const half * k,const half * v,const half * mask,
        float * out,int nq,int nk,int qpad,float scale) {
#if !defined(__MUSA_ARCH__) || __MUSA_ARCH__==220
 using namespace mtmusa::wmma;
 const int tid=threadIdx.x,q0=blockIdx.x*16,h=blockIdx.y;
 __shared__ float scores[16*640];
 __shared__ half kv[32*256];
 __shared__ float reduction[128];
 float * tile=(float *)kv;
 const half * query=q+(h*qpad+q0)*256;
 for(int kb=0;kb<nk;kb+=32) {
  const half * keys=k+kb*256;
  if(kb+32>nk) {
   for(int i=tid;i<32*256;i+=128)kv[i]=kb+i/256<nk?keys[i]:__float2half_rn(0.f);
   __syncthreads();keys=kv;
  }
  fragment<matrix_a,32,32,16,half,row_major> a;
  fragment<matrix_b,32,32,16,half,col_major> b;
  fragment<accumulator,32,32,16,float> c;fill_fragment(c,0.f);
  for(int d=0;d<256;d+=16) {
   load_matrix_sync(a,query+d,256);load_matrix_sync(b,keys+d,256);
   mma_sync(c,a,b,c);
  }
  __syncthreads();
  store_matrix_sync(tile,c,32,mem_row_major);__syncthreads();
  for(int i=tid;i<16*32;i+=128)
   if(kb+i%32<nk)scores[(i/32)*640+kb+i%32]=tile[i];
  __syncthreads();
 }
 // Ascending rows permit an in-place F32->F16 compaction after each row's
 // scores have been read into registers. No global scores/probabilities.
 half * probs=(half *)scores;
 if constexpr(Parallel) {
  // Eight rows at once reduce block barriers; retain F32 reductions and
  // normalized F16 probabilities. This variant has independent accuracy gates.
  const int lane=tid%16;
  for(int base=0;base<16;base+=8) {
   const int r=base+tid/16;float vals[40],mx=-INFINITY;
   #pragma unroll
   for(int j=0;j<40;++j) {
    const int col=lane+j*16;
    vals[j]=col<nk && q0+r<nq?scores[r*640+col]*scale+
       (mask?__half2float(mask[(q0+r)*nk+col]):0.f):-INFINITY;
    mx=fmaxf(mx,vals[j]);
   }
   reduction[tid]=mx;__syncthreads();
   for(int stride=8;stride;stride/=2) {
    if(lane<stride)reduction[tid]=fmaxf(reduction[tid],reduction[tid+stride]);
    __syncthreads();
   }
   mx=reduction[(tid/16)*16];float sum=0;
   #pragma unroll
   for(int j=0;j<40;++j){vals[j]=mx==-INFINITY?0.f:expf(vals[j]-mx);sum+=vals[j];}
   __syncthreads();reduction[tid]=sum;__syncthreads();
   for(int stride=8;stride;stride/=2) {
    if(lane<stride)reduction[tid]+=reduction[tid+stride];__syncthreads();
   }
   const float total=reduction[(tid/16)*16],inv=total>0?1.f/total:0.f;
   #pragma unroll
   for(int j=0;j<40;++j)probs[r*640+lane+j*16]=__float2half_rn(vals[j]*inv);
   __syncthreads();
  }
 } else {
 for(int r=0;r<16;++r) {
  float vals[8],mx=-INFINITY;
  #pragma unroll
  for(int j=0;j<8;++j) {
   const int col=tid+j*128;
   vals[j]=col<nk && q0+r<nq?scores[r*640+col]*scale+
      (mask?__half2float(mask[(q0+r)*nk+col]):0.f):-INFINITY;
   mx=fmaxf(mx,vals[j]);
  }
  reduction[tid]=mx;__syncthreads();
  for(int s=64;s;s/=2){if(tid<s)reduction[tid]=fmaxf(reduction[tid],reduction[tid+s]);__syncthreads();}
  mx=reduction[0];float sum=0;
  #pragma unroll
  for(int j=0;j<8;++j){vals[j]=mx==-INFINITY?0.f:expf(vals[j]-mx);sum+=vals[j];}
  __syncthreads();reduction[tid]=sum;__syncthreads();
  for(int s=64;s;s/=2){if(tid<s)reduction[tid]+=reduction[tid+s];__syncthreads();}
  const float inv=reduction[0]>0?1.f/reduction[0]:0.f;
  #pragma unroll
  for(int j=0;j<5;++j)probs[r*640+tid+j*128]=__float2half_rn(vals[j]*inv);
  __syncthreads();
 }
 }
 for(int i=16*640+tid;i<32*640;i+=128)probs[i]=__float2half_rn(0.f);
 __syncthreads();
 fragment<accumulator,32,32,16,float> accum[8];
 #pragma unroll
 for(int d=0;d<8;++d)fill_fragment(accum[d],0.f);
 for(int kb=0;kb<nk;kb+=16) {
  const half * values=v+kb*256;
  if(kb+16>nk) {
   for(int i=tid;i<16*256;i+=128)kv[i]=kb+i/256<nk?values[i]:__float2half_rn(0.f);
   __syncthreads();values=kv;
  }
  fragment<matrix_b,32,32,16,half,col_major> b;
  load_matrix_sync(b,probs+kb,640);
  #pragma unroll
  for(int d=0;d<8;++d) {
   fragment<matrix_a,32,32,16,half,col_major> a;
   load_matrix_sync(a,values+d*32,256);mma_sync(accum[d],a,b,accum[d]);
  }
  if(kb+16>nk)__syncthreads();
 }
 #pragma unroll
 for(int d=0;d<8;++d) {
  store_matrix_sync(tile,accum[d],32,mem_col_major);__syncthreads();
  for(int i=tid;i<16*32;i+=128)
   if(q0+i/32<nq)out[(q0+i/32)*2048+h*256+d*32+i%32]=tile[i];
  __syncthreads();
 }
#endif
}
