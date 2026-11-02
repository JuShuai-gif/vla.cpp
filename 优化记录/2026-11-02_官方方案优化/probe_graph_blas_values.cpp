#include <musa_runtime.h>
#include <mublas.h>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <cmath>
int main() {
    musaStream_t stream; musaStreamCreate(&stream);
    mublasHandle_t h; mublasCreate(&h); mublasSetStream(h, stream);
    void *a, *b, *c;
    musaMalloc(&a, 16*256*256*4); musaMalloc(&b, 16*256*256*4); musaMalloc(&c, 16*256*256*4);
    musaMemset(a,0,16*256*256*4); musaMemset(b,0,16*256*256*4);
    const float alpha=1, beta=0;
    std::vector<float> fp(16*256*256,1.0f);
    std::vector<uint16_t> hp(16*256*256,0x3c00);
    for (int kind=0; kind<4; ++kind) {
        auto run = [&]() {
            if (kind==0) return mublasSgemm(h,MUBLAS_OP_T,MUBLAS_OP_N,256,256,256,&alpha,(float*)a,256,(float*)b,256,&beta,(float*)c,256);
            if (kind==1) return mublasGemmEx(h,MUBLAS_OP_T,MUBLAS_OP_N,256,256,256,&alpha,a,MUSA_R_16F,256,b,MUSA_R_16F,256,&beta,c,MUSA_R_32F,256,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (kind==3) return mublasGemmEx(h,MUBLAS_OP_T,MUBLAS_OP_N,256,256,256,&alpha,a,MUSA_R_32F,256,b,MUSA_R_32F,256,&beta,c,MUSA_R_32F,256,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP);
            return mublasGemmStridedBatchedEx(h,MUBLAS_OP_T,MUBLAS_OP_N,256,256,256,&alpha,a,MUSA_R_32F,256,256*256,b,MUSA_R_32F,256,256*256,&beta,c,MUSA_R_32F,256,256*256,16,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP);
        };
        if (kind == 1) { musaMemcpy(a,hp.data(),hp.size()*2,musaMemcpyHostToDevice); musaMemcpy(b,hp.data(),hp.size()*2,musaMemcpyHostToDevice); }
        else { musaMemcpy(a,fp.data(),fp.size()*4,musaMemcpyHostToDevice); musaMemcpy(b,fp.data(),fp.size()*4,musaMemcpyHostToDevice); }
        auto warm = run(); musaStreamSynchronize(stream);
        musaGetLastError();
        auto begin=musaStreamBeginCapture(stream,musaStreamCaptureModeRelaxed);
        auto status=run();
        auto last=musaGetLastError();
        musaGraph_t graph=nullptr;
        auto end=musaStreamEndCapture(stream,&graph);
        std::printf("kind=%d warm=%d begin=%d blas=%d last=%d (%s) end=%d\n",kind,int(warm),int(begin),int(status),int(last),musaGetErrorString(last),int(end));
        if (end==musaSuccess && graph) {
            musaGraphExec_t exec=nullptr;
            auto instantiate=musaGraphInstantiate(&exec,graph,nullptr,nullptr,0);
            auto launch=instantiate==musaSuccess ? musaGraphLaunch(exec,stream) : instantiate;
            auto sync=musaStreamSynchronize(stream);
            std::printf("  instantiate=%d launch=%d sync=%d\n",int(instantiate),int(launch),int(sync));
            if (instantiate == musaSuccess) {
                std::vector<float> result(256*256);
                musaMemcpy(result.data(),c,result.size()*4,musaMemcpyDeviceToHost);
                float error=0; for(float v:result) error=std::fmax(error,std::fabs(v-256));
                std::printf("  first launch max_error=%g\n",error);
                musaMemset(c,0,result.size()*4); musaGraphLaunch(exec,stream); musaStreamSynchronize(stream);
                musaMemcpy(result.data(),c,result.size()*4,musaMemcpyDeviceToHost);
                error=0; for(float v:result) error=std::fmax(error,std::fabs(v-256));
                std::printf("  replay after output clear max_error=%g\n",error);
            }
            if(exec) musaGraphExecDestroy(exec);
            musaGraphDestroy(graph);
        }
    }
    musaFree(a); musaFree(b); musaFree(c); mublasDestroy(h); musaStreamDestroy(stream);
}
