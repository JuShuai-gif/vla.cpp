#include <musa_runtime.h>
#include <mublas.h>
#include <cstdio>
#include <vector>
#include <cstdint>
int main() {
    musaStream_t stream; musaStreamCreate(&stream);
    mublasHandle_t h; mublasCreate(&h); mublasSetStream(h, stream);
    void *a, *b, *c;
    musaMalloc(&a, 16*256*256*4); musaMalloc(&b, 16*256*256*4); musaMalloc(&c, 16*256*256*4);
    musaMemset(a,0,16*256*256*4); musaMemset(b,0,16*256*256*4);
    const float alpha=1, beta=0;
    for (int kind=0; kind<4; ++kind) {
        auto run = [&]() {
            if (kind==0) return mublasSgemm(h,MUBLAS_OP_T,MUBLAS_OP_N,256,256,256,&alpha,(float*)a,256,(float*)b,256,&beta,(float*)c,256);
            if (kind==1) return mublasGemmEx(h,MUBLAS_OP_T,MUBLAS_OP_N,256,256,256,&alpha,a,MUSA_R_16F,256,b,MUSA_R_16F,256,&beta,c,MUSA_R_32F,256,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (kind==3) return mublasGemmEx(h,MUBLAS_OP_T,MUBLAS_OP_N,256,256,256,&alpha,a,MUSA_R_32F,256,b,MUSA_R_32F,256,&beta,c,MUSA_R_32F,256,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP);
            return mublasGemmStridedBatchedEx(h,MUBLAS_OP_T,MUBLAS_OP_N,256,256,256,&alpha,a,MUSA_R_32F,256,256*256,b,MUSA_R_32F,256,256*256,&beta,c,MUSA_R_32F,256,256*256,16,MUBLAS_COMPUTE_32F,MUBLAS_GEMM_DEFAULT_TENSOR_OP);
        };
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
            if(exec) musaGraphExecDestroy(exec);
            musaGraphDestroy(graph);
        }
    }
    musaFree(a); musaFree(b); musaFree(c); mublasDestroy(h); musaStreamDestroy(stream);
}
