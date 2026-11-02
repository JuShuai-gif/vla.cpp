from pathlib import Path
r=Path('/home/skysi/code/vla.cpp')
p=r/'src/musa/vla_musa_fused.cu';s=p.read_text()
s=s.replace('namespace vla {\nnamespace {','''namespace vla {
bool musa_vision_pack_enabled() {
#ifdef VLA_MUSA_MUDNN_ATTN
    static const bool enabled = std::getenv("VLA_MUSA_VISION_SDPA") && std::getenv("VLA_MUSA_VISION_PACK");
    return enabled;
#else
    return false;
#endif
}
bool musa_vision_raw_qkv(const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v) {
    if (!musa_vision_pack_enabled() || !q || !k || !v ||
        q->ne[0]!=72 || q->ne[1]!=256 || q->ne[2]!=16 || q->ne[3]<1 || q->ne[3]>2 ||
        !ggml_are_same_shape(q,k) || !ggml_are_same_shape(q,v)) return false;
    for (const auto * t : {q,k,v}) {
        if(t->type!=GGML_TYPE_F32 || t->nb[0]!=4 || t->view_offs%16) return false;
        for(int j=1;j<4;++j) if(t->nb[j]%16 || t->nb[j]!=q->nb[j]) return false;
    }
    return true;
}
namespace {
bool gemma_direct_output() {
    static const bool enabled = std::getenv("VLA_MUSA_ATTN_DIRECT_OUT") != nullptr;
    return enabled;
}''',1)
s=s.replace('dense_f16(k) && dense_f16(v) && ggml_are_same_shape(q,k)', '( (dense_f16(k) && dense_f16(v)) || musa_vision_raw_qkv(q,k,v)) && ggml_are_same_shape(q,k)',1)
s=s.replace('VLA_ATTN_MU(musaMalloc(&out,(size_t)8*256*nq*4));','if(!gemma_direct_output()) VLA_ATTN_MU(musaMalloc(&out,(size_t)8*256*nq*4));',1)
a='''        &zero,s.out,MUSA_R_32F,256,256*nq,8,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP));
    gemma_unpack<<<(n4+WG-1)/WG,WG,0,stream>>>((const float4 *)s.out,(float4 *)t->data,nq);'''
b='''        &zero,gemma_direct_output()?t->data:s.out,MUSA_R_32F,
        gemma_direct_output()?2048:256,gemma_direct_output()?256:256*nq,
        8,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP));
    if(!gemma_direct_output())
        gemma_unpack<<<(n4+WG-1)/WG,WG,0,stream>>>((const float4 *)s.out,(float4 *)t->data,nq);'''
assert a in s;s=s.replace(a,b,1)
a='''        VLA_MU_CHECK(musaMalloc(&lse,(size_t)batch*16*256*4));'''
b=a+'''
        // Input padding is immutable. Capture/replay only rewrites the 72 valid columns.
        for(void * ptr : {q,k,v})
            VLA_MU_CHECK(musaMemsetAsync(ptr,0,(size_t)batch*16*256*128*2,stream));'''
assert a in s;s=s.replace(a,b,1)
pos='__global__ void unpack_sdpa('
kernel='''// Fuse strided Q/K/V reads, F32->F16 conversion and head padding.
__global__ void pack_vision_qkv(const float * q,const float * k,const float * v,
        __half2 * qo,__half2 * ko,__half2 * vo,int batch,size_t s1,size_t s2,size_t s3) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=batch*16*256*18)return;
    const int d=i%18,token=(i/18)%256,head=(i/(18*256))%16,b=i/(18*256*16);
    const float * src=blockIdx.y==0?q:(blockIdx.y==1?k:v);
    __half2 * dst=blockIdx.y==0?qo:(blockIdx.y==1?ko:vo);
    const float4 x=*(const float4 *)(src+4*d+token*s1+head*s2+b*s3);
    const int out=d+32*(token+256*(head+16*b));
    dst[2*out]=__floats2half2_rn(x.x,x.y);
    dst[2*out+1]=__floats2half2_rn(x.z,x.w);
}

'''
assert pos in s;s=s.replace(pos,kernel+pos,1)
a='''    pad_sdpa<<<(packed+WG-1)/WG,WG,0,stream>>>((const float *)q->data,(__half2 *)s.q,batch,'''
b='''    if(musa_vision_raw_qkv(q,t->src[1],t->src[2])) {
        pack_vision_qkv<<<dim3((n4+WG-1)/WG,3),WG,0,stream>>>(
            (const float *)q->data,(const float *)t->src[1]->data,(const float *)t->src[2]->data,
            (__half2 *)s.q,(__half2 *)s.k,(__half2 *)s.v,batch,q->nb[1]/4,q->nb[2]/4,q->nb[3]/4);
    } else {
    pad_sdpa<<<(packed+WG-1)/WG,WG,0,stream>>>((const float *)q->data,(__half2 *)s.q,batch,'''
assert a in s;s=s.replace(a,b,1)
s=s.replace('''    VLA_DNN_CHECK(mudnnScaledDotProductAttentionFlashForward''','''    }
    VLA_DNN_CHECK(mudnnScaledDotProductAttentionFlashForward''',1)
p.write_text(s)
p=r/'src/musa/vla_musa_ops.h';s=p.read_text().replace('bool musa_register_fused_ops();','''bool musa_register_fused_ops();
bool musa_vision_pack_enabled();
bool musa_vision_raw_qkv(const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v);''');p.write_text(s)
p=r/'src/layers/attn.h';s=p.read_text().replace('#include "ggml.h"','#include "ggml.h"\n#ifdef GGML_USE_MUSA\n#include "musa/vla_musa_ops.h"\n#endif')
a='''    ggml_tensor * kf = K->type == GGML_TYPE_F16 ? K : ggml_cast(C, K, GGML_TYPE_F16);'''
b='''    bool raw_vision = false;
#ifdef GGML_USE_MUSA
    raw_vision = !mask && musa_vision_raw_qkv(Q,K,V);
#endif
    ggml_tensor * kf = raw_vision || K->type == GGML_TYPE_F16 ? K : ggml_cast(C, K, GGML_TYPE_F16);'''
assert a in s;s=s.replace(a,b,1).replace('ggml_tensor * vf = V->type','ggml_tensor * vf = raw_vision || V->type',1);p.write_text(s)
p=r/'scripts/bench_pi05.py';s=p.read_text().replace('"VLA_PI05_HOST_INPUT",','"VLA_MUSA_ATTN_DIRECT_OUT", "VLA_MUSA_VISION_PACK", "VLA_PI05_HOST_INPUT",');p.write_text(s)
print('layout candidates applied')
