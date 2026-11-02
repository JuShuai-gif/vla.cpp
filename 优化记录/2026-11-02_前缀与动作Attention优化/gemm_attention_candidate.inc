// Candidate evaluated before promotion into src/musa/vla_musa_fused.cu.
bool gemma_sdpa(const ggml_tensor * t) {
    static const int mode = [] {
        const char * p=std::getenv("VLA_MUSA_GEMMA_ATTN");
        return p && std::strcmp(p,"fused-all")==0 ? 2 :
               p && std::strcmp(p,"fused-denoise")==0 ? 1 : 0;
    }();
    if(!mode || t->op!=GGML_OP_FLASH_ATTN_EXT || !dense_f32(t)) return false;
    for(int j=4;j<GGML_MAX_SRC;++j) if(t->src[j]) return false;
    const auto * q=t->src[0], * k=t->src[1], * v=t->src[2], * mask=t->src[3];
    float p[3]; std::memcpy(p,t->op_params,sizeof(p));
    if(!q || !dense_f16(k) || !dense_f16(v) || !ggml_are_same_shape(k,v) ||
       q->type!=GGML_TYPE_F32 || q->ne[0]!=256 || q->ne[2]!=8 || q->ne[3]!=1 ||
       k->ne[0]!=256 || k->ne[2]!=1 || k->ne[3]!=1 ||
       q->ne[1]>1024 || k->ne[1]>1024 || q->ne[1]>k->ne[1] ||
       q->nb[0]!=4 || q->nb[1]%16 || q->nb[2]%16 || q->view_offs%16 ||
       p[1]!=0 || p[2]!=0) return false;
    if(mode==1 && q->ne[1]>=k->ne[1]) return false;
    return !mask || (dense_f16(mask) && mask->ne[0]==k->ne[1] &&
                     mask->ne[1]>=q->ne[1] && mask->ne[2]==1 && mask->ne[3]==1);
}

#define VLA_ATTN_MU(call) do { const auto e=(call); if(e!=musaSuccess) GGML_ABORT("MUSA attention %s: %s",#call,musaGetErrorString(e)); } while(0)
#define VLA_ATTN_BLAS(call) do { const auto e=(call); if(e!=MUBLAS_STATUS_SUCCESS) GGML_ABORT("muBLAS attention %s: %d",#call,(int)e); } while(0)
struct GemmaSdpa {
    mublasHandle_t handle=nullptr;
    void * q=nullptr, * scores=nullptr, * probs=nullptr, * out=nullptr;
    int device;
    GemmaSdpa(int dev,int nq,int nk,musaStream_t stream):device(dev) {
        VLA_ATTN_BLAS(mublasCreate(&handle)); VLA_ATTN_BLAS(mublasSetStream(handle,stream));
        VLA_ATTN_MU(musaMalloc(&q,(size_t)8*256*nq*2));
        VLA_ATTN_MU(musaMalloc(&scores,(size_t)8*nk*nq*4));
        VLA_ATTN_MU(musaMalloc(&probs,(size_t)8*nk*nq*2));
        VLA_ATTN_MU(musaMalloc(&out,(size_t)8*256*nq*4));
    }
    ~GemmaSdpa() {
        int old=0;musaGetDevice(&old);musaSetDevice(device);
        musaFree(q);musaFree(scores);musaFree(probs);musaFree(out);
        mublasDestroy(handle);musaSetDevice(old);
    }
};
using GemmaKey=std::tuple<int,uintptr_t,int,int>;
struct GemmaCache {
    std::mutex mutex;
    std::map<GemmaKey,std::unique_ptr<GemmaSdpa>> entries;
};
GemmaCache & gemma_cache(){ static GemmaCache cache;return cache; }
void release_gemma_sdpa(int device,void * stream) {
    auto & cache=gemma_cache();std::lock_guard<std::mutex> guard(cache.mutex);
    for(auto it=cache.entries.begin();it!=cache.entries.end();)
        if(std::get<0>(it->first)==device && std::get<1>(it->first)==(uintptr_t)stream) it=cache.entries.erase(it);
        else ++it;
}

// One warp128 row; keep reductions F32 and fuse mask, scale and probability packing.
__global__ void gemma_probs(const float * scores,const __half * mask,__half * probs,int nq,int nk,float scale) {
    constexpr int threads=128;
    const int tid=threadIdx.x,row=blockIdx.x;
    __shared__ float reduction[threads];
    float vals[8], mx=-INFINITY;
#pragma unroll
    for(int j=0;j<8;++j){
        const int col=tid+j*threads;
        vals[j]=col<nk ? scores[(size_t)row*nk+col]*scale+(mask?__half2float(mask[(row%nq)*nk+col]):0.0f) : -INFINITY;
        mx=fmaxf(mx,vals[j]);
    }
    reduction[tid]=mx;__syncthreads();
    for(int s=threads/2;s;s/=2){if(tid<s)reduction[tid]=fmaxf(reduction[tid],reduction[tid+s]);__syncthreads();}
    mx=reduction[0];float sum=0;
#pragma unroll
    for(int j=0;j<8;++j){vals[j]=mx==-INFINITY?0.0f:expf(vals[j]-mx);sum+=vals[j];}
    __syncthreads();reduction[tid]=sum;__syncthreads();
    for(int s=threads/2;s;s/=2){if(tid<s)reduction[tid]+=reduction[tid+s];__syncthreads();}
    const float inv=reduction[0]>0 ? 1.0f/reduction[0] : 0.0f;
#pragma unroll
    for(int j=0;j<8;++j)if(tid+j*threads<nk)probs[(size_t)row*nk+tid+j*threads]=__float2half_rn(vals[j]*inv);
}
__global__ void gemma_unpack(const float4 * src,float4 * dst,int nq) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nq*8*64)return;
    const int d=i%64,h=(i/64)%8,q=i/(64*8);
    dst[i]=src[d+64*(q+nq*h)];
}
bool run_gemma_sdpa(ggml_tensor * t,musaStream_t stream) {
    int device=0;VLA_ATTN_MU(musaGetDevice(&device));
    const auto * q=t->src[0], * k=t->src[1], * v=t->src[2], * mask=t->src[3];
    const int nq=q->ne[1],nk=k->ne[1];
    auto & cache=gemma_cache();std::lock_guard<std::mutex> guard(cache.mutex);
    const GemmaKey key(device,(uintptr_t)stream,nq,nk);auto it=cache.entries.find(key);
    if(it==cache.entries.end()){
        musaStreamCaptureStatus capture;VLA_ATTN_MU(musaStreamIsCapturing(stream,&capture));
        if(capture!=musaStreamCaptureStatusNone)return false;
        it=cache.entries.emplace(key,std::make_unique<GemmaSdpa>(device,nq,nk,stream)).first;
    }
    auto & s=*it->second;const size_t n4=(size_t)8*256*nq/4;
    pack_f16x4<<<(n4+WG-1)/WG,WG,0,stream>>>((const float *)q->data,(__half2 *)s.q,n4,64,nq,8,q->nb[1]/4,q->nb[2]/4,q->nb[3]/4);
    const float one=1,zero=0;float scale;std::memcpy(&scale,t->op_params,sizeof(scale));
    VLA_ATTN_BLAS(mublasGemmStridedBatchedEx(s.handle,MUBLAS_OP_T,MUBLAS_OP_N,nk,nq,256,
        &one,k->data,MUSA_R_16F,256,0,s.q,MUSA_R_16F,256,256*nq,
        &zero,s.scores,MUSA_R_32F,nk,nk*nq,8,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP));
    gemma_probs<<<8*nq,128,0,stream>>>((const float *)s.scores,mask?(const __half *)mask->data:nullptr,(__half *)s.probs,nq,nk,scale);
    VLA_ATTN_BLAS(mublasGemmStridedBatchedEx(s.handle,MUBLAS_OP_N,MUBLAS_OP_N,256,nq,nk,
        &one,v->data,MUSA_R_16F,256,0,s.probs,MUSA_R_16F,nk,nk*nq,
        &zero,s.out,MUSA_R_32F,256,256*nq,8,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP));
    gemma_unpack<<<(n4+WG-1)/WG,WG,0,stream>>>((const float4 *)s.out,(float4 *)t->data,nq);
    return true;
}
