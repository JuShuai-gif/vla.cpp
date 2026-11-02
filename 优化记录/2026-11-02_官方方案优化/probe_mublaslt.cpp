#include <musa_runtime.h>
#include <mublasLt.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>

#define CHECK_MUSA(call) do { auto e = (call); if (e != musaSuccess) { std::fprintf(stderr, "%s: %s\n", #call, musaGetErrorString(e)); return 1; } } while (0)
#define CHECK_BLAS(call) do { auto e = (call); if (e != MUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "%s: %d\n", #call, int(e)); return 1; } } while (0)

int main() {
    mublasLtHandle_t handle;
    CHECK_BLAS(mublasLtCreate(&handle));
    for (int k : {257, 1024}) for (bool output_f32 : {false, true}) {
        const int m = 256, n = 50;
        mublasLtMatmulDesc_t desc;
        mublasLtMatrixLayout_t a_desc, b_desc, c_desc;
        mublasLtMatmulPreference_t pref;
        CHECK_BLAS(mublasLtMatmulDescCreate(&desc, MUBLAS_COMPUTE_32F, MUSA_R_32F));
        mublasOperation_t trans = MUBLAS_OP_T;
        CHECK_BLAS(mublasLtMatmulDescSetAttribute(desc, MUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
        CHECK_BLAS(mublasLtMatrixLayoutCreate(&a_desc, MUSA_R_16F, k, m, k));
        CHECK_BLAS(mublasLtMatrixLayoutCreate(&b_desc, MUSA_R_16F, k, n, k));
        CHECK_BLAS(mublasLtMatrixLayoutCreate(&c_desc, output_f32 ? MUSA_R_32F : MUSA_R_16F, m, n, m));
        CHECK_BLAS(mublasLtMatmulPreferenceCreate(&pref));
        size_t work_bytes = 4*1024*1024;
        CHECK_BLAS(mublasLtMatmulPreferenceSetAttribute(pref, MUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &work_bytes, sizeof(work_bytes)));
        mublasLtMatmulHeuristicResult_t results[8] = {};
        int count = 0;
        auto status = mublasLtMatmulAlgoGetHeuristic(handle, desc, a_desc, b_desc, c_desc, c_desc, pref, 8, results, &count);
        std::printf("k=%d m=%d n=%d output=%s compute=f32 heuristic_status=%d count=%d\n", k, m, n, output_f32 ? "f32" : "f16", int(status), count);
        if (status == MUBLAS_STATUS_SUCCESS && count > 0) {
            void *a, *b, *c, *work;
            CHECK_MUSA(musaMalloc(&a, k*m*2));
            CHECK_MUSA(musaMalloc(&b, k*n*2));
            CHECK_MUSA(musaMalloc(&c, m*n*(output_f32 ? 4 : 2)));
            CHECK_MUSA(musaMalloc(&work, work_bytes));
            std::vector<uint16_t> ones(k*m, 0x3c00);
            CHECK_MUSA(musaMemcpy(a, ones.data(), k*m*2, musaMemcpyHostToDevice));
            CHECK_MUSA(musaMemcpy(b, ones.data(), k*n*2, musaMemcpyHostToDevice));
            const float alpha = 1, beta = 0;
            for (int i=0; i<count; ++i) {
                auto run = mublasLtMatmul(handle, desc, &alpha, a, a_desc, b, b_desc, &beta, c, c_desc, c, c_desc, &results[i].algo, work, work_bytes, nullptr);
                std::printf("  algo=%d state=%d workspace=%zu run_status=%d", i, int(results[i].state), results[i].workspaceSize, int(run));
                if (run == MUBLAS_STATUS_SUCCESS) {
                    CHECK_MUSA(musaDeviceSynchronize());
                    double error = 0;
                    if (output_f32) {
                        std::vector<float> out(m*n);
                        CHECK_MUSA(musaMemcpy(out.data(), c, out.size()*4, musaMemcpyDeviceToHost));
                        for (float v : out) error = std::fmax(error, std::fabs(v-k));
                    } else {
                        std::vector<uint16_t> out(m*n);
                        CHECK_MUSA(musaMemcpy(out.data(), c, out.size()*2, musaMemcpyDeviceToHost));
                        for (uint16_t v : out) {
                            float value = std::ldexp(1.0f+(v&1023)/1024.0f, ((v>>10)&31)-15);
                            if (v&0x8000) value=-value;
                            error = std::fmax(error, std::fabs(value-k));
                        }
                    }
                    std::printf(" max_error=%g", error);
                    if (error != 0) return 2;
                }
                std::printf("\n");
            }
            CHECK_MUSA(musaFree(a)); CHECK_MUSA(musaFree(b)); CHECK_MUSA(musaFree(c)); CHECK_MUSA(musaFree(work));
        }
        CHECK_BLAS(mublasLtMatmulPreferenceDestroy(pref));
        CHECK_BLAS(mublasLtMatrixLayoutDestroy(a_desc)); CHECK_BLAS(mublasLtMatrixLayoutDestroy(b_desc)); CHECK_BLAS(mublasLtMatrixLayoutDestroy(c_desc));
        CHECK_BLAS(mublasLtMatmulDescDestroy(desc));
    }
    CHECK_BLAS(mublasLtDestroy(handle));
}
