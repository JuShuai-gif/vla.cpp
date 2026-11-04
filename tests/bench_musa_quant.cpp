// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
// Check Q8_0 MMQ against its D4 activation quantization, including real GEMM
// shapes. CPU reference samples large outputs; small shapes check every value.
#include "ggml-cuda.h"
#include "backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static double milliseconds() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    const int K = argc > 1 ? std::atoi(argv[1]) : 256;
    const int M = argc > 2 ? std::atoi(argv[2]) : 64;
    const int N = argc > 3 ? std::atoi(argv[3]) : 50;
    const int runs = argc > 4 ? std::atoi(argv[4]) : 10;
    const bool blas = std::getenv("GGML_MUSA_QUANT_ROUTE") &&
                      std::strcmp(std::getenv("GGML_MUSA_QUANT_ROUTE"), "blas") == 0;
    if (K < 32 || K % 32 || M < 1 || N < 9 || runs < 1) return 2;
    auto backend = vla::backend_init("quant-check", 4);
    if (!backend.is_musa) return 77;
    auto * ctx = ggml_init({8*1024*1024, nullptr, true});
    auto * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, M);
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_set_name(w, "quant_check_q8");
    ggml_set_input(w); ggml_set_input(x);
    auto * out = ggml_mul_mat(ctx, w, x);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    ggml_set_output(out);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.handle));
    if (!ggml_gallocr_alloc_graph(alloc, graph)) return 3;
    std::vector<float> weights(size_t(K)*M), inputs(size_t(K)*N), decoded(weights.size()), qinputs(inputs.size());
    // Dyadic values avoid CPU/GPU rounding tie ambiguity; zeros, outliers and
    // partial tiles catch padding/scale/addressing errors.
    for (size_t i=0; i<weights.size(); ++i)
        weights[i] = float(int((i*2654435761ULL+17)%65521)-32760)/65536.0f;
    for (size_t i=0; i<inputs.size(); ++i)
        inputs[i] = (i%997 == 0 ? 16.0f : 1.0f)*float(int((i*40503ULL+37)%8191)-4095)/4096.0f;
    std::fill(inputs.begin(), inputs.begin()+32, 0.0f);
    std::vector<unsigned char> packed(ggml_nbytes(w));
    if (ggml_quantize_chunk(GGML_TYPE_Q8_0, weights.data(), packed.data(), 0, M, K, nullptr) != packed.size()) return 4;
    ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(packed.data(), decoded.data(), decoded.size());
    for (size_t b=0; b<inputs.size(); b+=32) {
        float amax=0;
        for (size_t i=b; i<b+32; ++i) amax=std::max(amax,std::fabs(inputs[i]));
        const float inv=amax==0 ? 0 : 127.0f/amax;
        const float scale=amax==0 ? 0 : 1.0f/inv;
        for (size_t i=b; i<b+32; ++i)
            qinputs[i]=blas ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(inputs[i])) : std::round(inputs[i]*inv)*scale;
    }
    if (blas) for (auto & v:decoded) v=ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
    ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(x, inputs.data(), 0, inputs.size()*sizeof(float));
    std::vector<double> times;
    for (int i=0; i<3+runs; ++i) {
        double started=milliseconds();
        if (ggml_backend_graph_compute(backend.handle, graph)!=GGML_STATUS_SUCCESS) return 5;
        ggml_backend_synchronize(backend.handle);
        if(i>=3) times.push_back(milliseconds()-started);
    }
    std::vector<float> result(size_t(M)*N);
    ggml_backend_tensor_get(out,result.data(),0,result.size()*sizeof(float));
    bool finite=std::all_of(result.begin(),result.end(),[](float x){return std::isfinite(x);});
    double max_error=0, max_ref=0, sum_sq=0; int fails=0;
    const size_t checks=std::min<size_t>(result.size(),4096);
    for (size_t s=0;s<checks;++s) {
        const size_t index=checks==result.size()?s:s*(result.size()-1)/(checks-1);
        const size_t m=index%M, n=index/M;
        double expected=0;
        for(int k=0;k<K;++k) expected+=double(decoded[m*K+k])*qinputs[n*K+k];
        const double err=std::fabs(expected-result[index]);
        if (s < 8) std::fprintf(stderr, "sample index=%zu expected=%.9g actual=%.9g\n", index, expected, result[index]);
        max_error=std::max(max_error,err);max_ref=std::max(max_ref,std::fabs(expected));sum_sq+=err*err;
        if (err>0.003+0.0003*std::fabs(expected)) ++fails;
    }
    std::sort(times.begin(),times.end());
    std::printf("{\"type\":\"Q8_0\",\"reference\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"runs\":%d,\"p50_ms\":%.6f,\"checks\":%zu,\"finite\":%s,\"failed_values\":%d,\"max_error\":%.9g,\"rmse\":%.9g,\"max_reference\":%.9g}\n",
        blas?"dequant-F16-BLAS":"MMQ-D4",M,N,K,runs,times[times.size()/2],checks,finite?"true":"false",fails,max_error,std::sqrt(sum_sq/checks),max_ref);
    ggml_gallocr_free(alloc);ggml_free(ctx);ggml_backend_free(backend.handle);
    return finite && !fails ? 0 : 1;
}
