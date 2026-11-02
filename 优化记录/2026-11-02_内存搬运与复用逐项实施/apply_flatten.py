from pathlib import Path
r=Path('/home/skysi/code/vla.cpp');d=r/'优化记录/2026-11-02_内存搬运与复用逐项实施'
p=r/'scripts/patch_ggml_musa.py';s=p.read_text()
addition='''#ifdef GGML_USE_MUSA
    // vla.cpp: shared weights and consecutive batches can form one wider GEMM.
    static const bool flatten_shared = std::getenv("VLA_MUSA_FLATTEN_GEMM") != nullptr;
    if (flatten_shared && compute_type == GGML_TYPE_F16 && cu_data_type == CUDA_R_32F &&
        ne02 == 1 && ne03 == 1 && ne12 > 1 && ne13 == 1 &&
        s12 == s11*ne11 && ne11 <= INT32_MAX/ne12) {
        CUBLAS_CHECK(cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11*ne12, ne10,
                alpha, src0_ptr, cu_data_type_a, s01,
                       src1_ptr, cu_data_type_b, s11,
                beta, dst_ptr, cu_data_type, ne0,
                cu_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        return;
    }
#endif
'''
block="\n    # Batch flattening is an independent opt-in, after dtype/stride normalization.\n    path = root / 'ggml/src/ggml-cuda/ggml-cuda.cu'\n    text = path.read_text()\n    anchor = '    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.'\n    addition = "+repr(addition)+"\n    if addition not in text:\n        assert text.count(anchor) == 1\n        path.write_text(text.replace(anchor, addition + anchor))\n\n"
s=s.replace('\nif __name__ == "__main__":',block+'\nif __name__ == "__main__":',1);p.write_text(s)
p=r/'scripts/bench_pi05.py';s=p.read_text().replace("'VLA_MUSA_ATTN_DIRECT_OUT',","'VLA_MUSA_FLATTEN_GEMM', 'VLA_MUSA_ATTN_DIRECT_OUT',",1);p.write_text(s)
p=r/'tests/CMakeLists.txt';s=p.read_text().replace('    add_test(NAME musa_ops COMMAND test_musa_ops)','''    add_test(NAME musa_ops COMMAND test_musa_ops)
    add_test(NAME musa_flatten_gemm COMMAND test_musa_ops)
    set_tests_properties(musa_flatten_gemm PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180
        ENVIRONMENT "VLA_MUSA_FLATTEN_GEMM=1")''',1);p.write_text(s)
# Faster output restoration for the isolated shape study, retaining the initial measurements.
(d/'gemm-shapes').rename(d/'gemm-shapes-naive-crop')
p=d/'probe_gemm_shapes.cu';s=p.read_text()
a=s.index('__global__ void crop(');b=s.index('struct Shape',a)
s=s[:a]+'''__global__ void crop4(const float4 *src,float4 *dst,int m4,int n,int mp4,int np) {
 int col=blockIdx.x*blockDim.x+threadIdx.x,row=blockIdx.y%n,b=blockIdx.y/n;
 if(col<m4)dst[col+(size_t)m4*(row+n*b)]=src[col+(size_t)mp4*(row+np*b)];
}
'''+s[b:]
s=s.replace('''   if(copy)crop<<<(nc+255)/256,256,0,stream>>>(c,out,m,nn,mp,np,bb);''','''   if(copy && mp==m) {
    for(int j=0;j<bb;j++)MU(musaMemcpyAsync(out+(size_t)j*m*nn,c+(size_t)j*m*np,(size_t)m*nn*4,musaMemcpyDeviceToDevice,stream));
   } else if(copy)crop4<<<dim3((m/4+255)/256,nn*bb),256,0,stream>>>((float4*)c,(float4*)out,m/4,nn,mp/4,np);''',1)
p.write_text(s)
print('flatten candidate and corrected shape-probe restoration applied')
