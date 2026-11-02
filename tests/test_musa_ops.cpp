// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0

// No checkpoint: exercise the actual Gemma graph builders used by pi0.5,
// including cached-prefix MQA, head_dim=256, non-aligned sequence lengths,
// adaRMS/gated residuals and the SigLIP patch embedding. Compare device output
// against the identical CPU graph, with identical resident weight types.
#include "backend.h"
#include "cuda/vla_cuda_ops.h"
#include "options.h"
#include "modules/gemma_expert.h"
#include "ggml-alloc.h"
#include "modules/vision_prune.h"
#include "musa/vla_musa_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace {

struct Graph {
    ggml_context * ctx = ggml_init({32*1024*1024, nullptr, true});
    ggml_gallocr_t alloc = nullptr;
    std::vector<ggml_tensor *> inputs;

    ~Graph() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx) ggml_free(ctx);
    }

    ggml_tensor * input(ggml_type type, std::initializer_list<int64_t> shape) {
        auto * t = ggml_new_tensor(ctx, type, (int) shape.size(), shape.begin());
        ggml_set_input(t);
        ggml_format_name(t, "input_%zu", inputs.size());
        inputs.push_back(t);
        return t;
    }

    void upload() {
        for (size_t j = 0; j < inputs.size(); ++j) {
            auto * t = inputs[j];
            const size_t n = (size_t) ggml_nelements(t);
            if (t->type == GGML_TYPE_I32) {
                std::vector<int32_t> data(n);
                for (size_t i = 0; i < n; ++i) data[i] = (int32_t) i + 17;
                ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
                continue;
            }
            std::vector<float> data(n);
            for (size_t i = 0; i < n; ++i)
                data[i] = ((int) ((i*37 + j*13) % 67) - 33) / 128.0f;
            if (std::strcmp(ggml_get_name(t), "geglu_values") == 0)
                for (auto & v : data) v *= 40.0f;
            if (std::strcmp(ggml_get_name(t), "mask") == 0)
                for (size_t i = 0; i < n; ++i)
                    data[i] = (i % (size_t)t->ne[0]) % 7 == 6 ? -INFINITY : 0.0f;
            if (t->type == GGML_TYPE_BF16) {
                std::vector<ggml_bf16_t> packed(n);
                ggml_fp32_to_bf16_row(data.data(), packed.data(), (int64_t) n);
                ggml_backend_tensor_set(t, packed.data(), 0, ggml_nbytes(t));
            } else if (t->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> packed(n);
                ggml_fp32_to_fp16_row(data.data(), packed.data(), (int64_t) n);
                ggml_backend_tensor_set(t, packed.data(), 0, ggml_nbytes(t));
            } else {
                ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
            }
        }
    }

    std::vector<float> run(ggml_backend_t backend, const std::vector<ggml_tensor *> & outputs) {
        auto * gf = ggml_new_graph(ctx);
        for (auto * t : outputs) {
            ggml_set_output(t);
            ggml_build_forward_expand(gf, t);
        }
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            auto * t = ggml_graph_node(gf, i);
            if (!ggml_backend_supports_op(backend, t)) {
                std::fprintf(stderr, "%s rejects %s (%s)\n", ggml_backend_name(backend),
                             ggml_op_name(t->op), ggml_get_name(t));
                return {};
            }
        }
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!alloc || !ggml_gallocr_alloc_graph(alloc, gf)) return {};
        std::vector<float> result;
        // Reuse the allocation and graph: catches stream/lifetime problems too.
        for (int repeat = 0; repeat < 2; ++repeat) {
            // gallocr may reuse input memory after its last consumer; restore
            // every input just as predict restores each new observation.
            upload();
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) return {};
            for (auto * t : outputs) {
                const size_t offset = result.size();
                result.resize(offset + (size_t) ggml_nelements(t));
                if (t->type == GGML_TYPE_F16) {
                    std::vector<ggml_fp16_t> half((size_t) ggml_nelements(t));
                    ggml_backend_tensor_get(t, half.data(), 0, ggml_nbytes(t));
                    ggml_fp16_to_fp32_row(half.data(), result.data()+offset, half.size());
                } else {
                    ggml_backend_tensor_get(t, result.data() + offset, 0, ggml_nbytes(t));
                }
            }
        }
        return result;
    }
};

vla::GemmaLayerW weights(Graph & g, ggml_type type, int64_t hidden, int64_t qdim) {
    vla::GemmaLayerW w;
    w.ln_in = g.input(GGML_TYPE_F32, {hidden});
    w.ln_post = g.input(GGML_TYPE_F32, {hidden});
    w.Wq = g.input(type, {hidden, qdim});
    w.Wk = g.input(type, {hidden, 256});
    w.Wv = g.input(type, {hidden, 256});
    w.Wo = g.input(type, {qdim, hidden});
    w.Wgate = g.input(type, {hidden, 2*hidden});
    w.Wup = g.input(type, {hidden, 2*hidden});
    w.Wdown = g.input(type, {2*hidden, hidden});
    return w;
}

std::vector<float> gemma(ggml_backend_t backend, ggml_type type, int64_t prefix, bool packed = false) {
    Graph g;
    if (!g.ctx) return {};
    auto * C = g.ctx;
    vla::Config cfg;
    cfg.hidden = 128;
    cfg.n_q_heads = 8;
    cfg.n_kv_heads = 1;
    cfg.head_dim = cfg.kv_full_dim = 256;
    cfg.rms_eps = 1e-6f;
    cfg.rope_freq_base = 10000.0f;
    auto w = weights(g, type, cfg.hidden, cfg.n_q_heads*cfg.head_dim);
    if (packed) {
        w.Wqkv = ggml_concat(C, ggml_concat(C, w.Wq, w.Wk, 1), w.Wv, 1);
        w.Wgu = ggml_concat(C, w.Wgate, w.Wup, 1);
    }
    auto * x = g.input(GGML_TYPE_F32, {cfg.hidden, prefix});
    auto * pos = g.input(GGML_TYPE_I32, {prefix});
    ggml_tensor * k = nullptr, * v = nullptr;
    auto * pre = vla::gemma_layer(C, w, x, pos, cfg, prefix, nullptr, nullptr, nullptr, &k, &v);

    constexpr int64_t chunk = 50;
    auto * h = g.input(GGML_TYPE_F32, {cfg.hidden, chunk});
    auto * suffix_pos = g.input(GGML_TYPE_I32, {chunk});
    auto * mod = g.input(GGML_TYPE_F32, {3*cfg.hidden});
    auto * scale = ggml_view_1d(C, mod, cfg.hidden, 0);
    auto * shift = ggml_view_1d(C, mod, cfg.hidden, cfg.hidden*sizeof(float));
    auto * gate = ggml_view_1d(C, mod, cfg.hidden, 2*cfg.hidden*sizeof(float));
    auto adarms = [&](ggml_tensor * t) {
        auto * norm = ggml_rms_norm(C, t, cfg.rms_eps);
        return ggml_add(C, ggml_add(C, norm, ggml_mul(C, norm, scale)), shift);
    };
    for (int step = 0; step < 2; ++step) {
        auto * att = vla::gemma_attn(C, w, adarms(h), suffix_pos, cfg, chunk,
                                    k, v, nullptr, nullptr, nullptr, GGML_TYPE_F32, false);
        h = ggml_add(C, h, ggml_mul(C, att, gate));
        auto * mlp = vla::gemma_mlp(C, w, adarms(h), GGML_TYPE_F32);
        h = ggml_add(C, h, ggml_scale(C, ggml_mul(C, mlp, gate), -0.1f));
    }
    return g.run(backend, {pre, h});
}

std::vector<float> vision(ggml_backend_t backend, ggml_type type) {
    Graph g;
    if (!g.ctx) return {};
    vla::SigLipTower tower;
    tower.enc.cfg.hidden = 128;
    tower.patch_w = g.input(GGML_TYPE_F32, {14, 14, 3, 128});
    tower.patch_b = g.input(GGML_TYPE_F32, {128});
    tower.pos = g.input(GGML_TYPE_F32, {128, 256});
    auto * pixels = g.input(GGML_TYPE_F32, {224, 224, 3, 2});
    auto * h = tower.embed_conv(g.ctx, pixels, 14, 16);
    auto * W = g.input(type, {128, 256});
    auto * bias = g.input(GGML_TYPE_F32, {256});
    h = ggml_gelu(g.ctx, ggml_add(g.ctx, vla::mm_f32(g.ctx, W, ggml_norm(g.ctx, h, 1e-6f)), bias));
    return g.run(backend, {h});
}

std::vector<float> softmax(ggml_backend_t backend, int64_t width, bool masked) {
    Graph g;
    if (!g.ctx) return {};
    auto * x = g.input(GGML_TYPE_F32, {width, 3, 8});
    auto * mask = masked ? g.input(GGML_TYPE_F16, {width, 3}) : nullptr;
    if (mask) ggml_set_name(mask, "mask");
    auto result = g.run(backend, {ggml_soft_max_ext(g.ctx, x, mask, 0.0625f, 0.0f)});
    for (size_t row = 0; row < result.size(); row += (size_t) width) {
        double sum = 0;
        for (int64_t col = 0; col < width; ++col) {
            const float v = result[row + col];
            if (!std::isfinite(v) || v < 0 || (masked && col % 7 == 6 && v != 0)) return {};
            sum += v;
        }
        if (std::fabs(sum - 1.0) > 2e-5) return {};
    }
    return result;
}

bool compare(const char * name, ggml_type type, const std::vector<float> & ref,
             const std::vector<float> & got, double tolerance = 0.0) {
    if (ref.empty() || ref.size() != got.size()) {
        std::fprintf(stderr, "FAIL %s/%s: missing output\n", name, ggml_type_name(type));
        return false;
    }
    double peak = 0, error = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(ref[i]) || !std::isfinite(got[i])) {
            std::fprintf(stderr, "FAIL %s/%s: non-finite output\n", name, ggml_type_name(type));
            return false;
        }
        peak = std::max(peak, std::fabs((double) ref[i]));
        error = std::max(error, std::fabs((double) ref[i] - got[i]));
    }
    const double relative = error / std::max(peak, 1e-6);
    // muBLAS narrows activations for reduced-precision weights; CPU kernels
    // use different rounding/accumulation. Allow 0.5% for F16 and 1% for BF16
    // on these multi-op chains, not as a full-policy accuracy criterion.
    if (tolerance == 0.0)
        tolerance = type == GGML_TYPE_BF16 ? 1e-2 : type == GGML_TYPE_F16 ? 5e-3 : 2e-3;
    const bool ok = peak > 0 && relative < tolerance;
    std::printf("%s %s/%s: %zu values, max_abs=%.7g normalized=%.7g tolerance=%.4g\n",
                ok ? "PASS" : "FAIL", name, ggml_type_name(type), got.size(), error, relative, tolerance);
    return ok;
}

bool large_weight_buffers(ggml_backend_t backend) {
    // Reproduce the >4 GiB model allocation without touching most of its pages.
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    std::vector<ggml_tensor *> tensors;
    for (int i=0; i<5; ++i) tensors.push_back(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)256*1024*1024));
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    bool ok = buffer != nullptr;
    if (ok) {
        for (int i=0; i<5; ++i) {
            const float value = 100.0f + i;
            ggml_backend_tensor_set(tensors[i], &value, 0, sizeof(value));
        }
        for (int i=0; i<5; ++i) {
            float value = 0;
            ggml_backend_tensor_get(tensors[i], &value, 0, sizeof(value));
            ok &= value == 100.0f + i;
            ok &= ggml_backend_buffer_get_size(tensors[i]->buffer) <= (size_t)2*1024*1024*1024;
        }
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(ctx);
    std::printf("%s segmented weights above 4 GiB\n", ok ? "PASS" : "FAIL");
    return ok;
}

std::vector<float> patch_scores(ggml_backend_t backend) {
    Graph g;
    auto * a = g.input(GGML_TYPE_F32, {256, 256, 16});
    return g.run(backend, {vla::vision_patch_scores(g.ctx, a)});
}

std::vector<float> fused_expert(ggml_backend_t backend, int64_t h, int64_t rows, bool fused, bool half = false) {
    Graph g;
    auto * x = g.input(GGML_TYPE_F32, {h, rows});
    auto * o = g.input(GGML_TYPE_F32, {h, rows});
    auto * mod = g.input(GGML_TYPE_F32, {3*h});
    const vla::FusedAdarmsSpec norm_spec;
    const vla::FusedGateResSpec gate_spec;
    ggml_tensor * norm, * out;
    if (fused) {
        norm = vla::fused_adarms(g.ctx, x, mod, half ? GGML_TYPE_F16 : GGML_TYPE_F32, norm_spec);
        if (half) o = ggml_cast(g.ctx, o, GGML_TYPE_F16);
        auto * residual = half ? ggml_cast(g.ctx, norm, GGML_TYPE_F32) : norm;
        out = vla::fused_gateres(g.ctx, residual, o, mod, gate_spec);
    } else {
        auto * n = ggml_rms_norm(g.ctx, x, norm_spec.eps);
        auto * scale = ggml_view_1d(g.ctx, mod, h, 0);
        auto * shift = ggml_view_1d(g.ctx, mod, h, h*sizeof(float));
        auto * gate = ggml_view_1d(g.ctx, mod, h, 2*h*sizeof(float));
        norm = ggml_add(g.ctx, ggml_add(g.ctx, n, ggml_mul(g.ctx, n, scale)), shift);
        out = ggml_add(g.ctx, norm, ggml_mul(g.ctx, o, gate));
    }
    return g.run(backend, {norm, out});
}

std::vector<float> f16_gemm_norm(ggml_backend_t backend, bool native) {
    Graph g;
    auto * w = g.input(GGML_TYPE_F16, {257, 33});
    auto * x = g.input(GGML_TYPE_F16, {257, 51});
    auto * product = native ? vla::mul_mat_t(g.ctx, w, x, GGML_TYPE_F16)
        : ggml_cast(g.ctx, ggml_mul_mat(g.ctx, ggml_cast(g.ctx, w, GGML_TYPE_F32),
                                      ggml_cast(g.ctx, x, GGML_TYPE_F32)), GGML_TYPE_F16);
    auto * norm = native ? ggml_rms_norm(g.ctx, product, 1e-6f)
        : ggml_cast(g.ctx, ggml_rms_norm(g.ctx, ggml_cast(g.ctx, product, GGML_TYPE_F32), 1e-6f), GGML_TYPE_F16);
    auto * b = g.input(GGML_TYPE_F32, {33});
    auto * scaled = native ? ggml_mul(g.ctx, norm, b) : ggml_cast(g.ctx,
        ggml_mul(g.ctx, ggml_cast(g.ctx, norm, GGML_TYPE_F32), b), GGML_TYPE_F16);
    auto * result = native ? ggml_add(g.ctx, scaled, b) : ggml_cast(g.ctx,
        ggml_add(g.ctx, ggml_cast(g.ctx, scaled, GGML_TYPE_F32), b), GGML_TYPE_F16);
    return g.run(backend, {product, norm, result});
}

std::vector<float> merged_glu(ggml_backend_t backend, ggml_type type, bool native, bool swapped) {
    Graph g;
    auto * x = g.input(type, {2*4097, 3});
    ggml_set_name(x, "geglu_values");
    if (!native && type == GGML_TYPE_F16) x = ggml_cast(g.ctx, x, GGML_TYPE_F32);
    auto * y = swapped ? ggml_geglu_swapped(g.ctx, x) : ggml_geglu(g.ctx, x);
    if (!native && type == GGML_TYPE_F16) y = ggml_cast(g.ctx, y, type);
    return g.run(backend, {y});
}

std::vector<float> merged_glu_reference(ggml_type type, bool swapped) {
    // Independent float tanh reference: ggml CPU GELU can use an F16 lookup table.
    constexpr int h = 4097, rows = 3;
    std::vector<float> result;
    auto input = [&](int i) {
        float v = ((i*37)%67-33)/128.0f*40.0f;
        return type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)) : v;
    };
    for (int repeat = 0; repeat < 2; ++repeat)
        for (int r = 0; r < rows; ++r)
            for (int i = 0; i < h; ++i) {
                float g = input(r*2*h+i+(swapped ? h : 0));
                float u = input(r*2*h+i+(swapped ? 0 : h));
                float y = vla::fused_gelu_tanh(g)*u;
                result.push_back(type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(y)) : y);
            }
    return result;
}

}  // namespace

int main(int argc, char ** argv) {
    const auto b = vla::backend_init("musa_ops", 4);
    if (!b.handle) return 1;
    if (argc == 2 && std::strcmp(argv[1], "--expect-cpu") == 0) {
        const bool ok = !b.is_musa && !b.is_cuda && std::strcmp(ggml_backend_name(b.handle), "CPU") == 0;
        ggml_backend_free(b.handle);
        return ok ? 0 : 1;
    }
    if (!b.is_musa) {
        std::printf("SKIP: no MUSA device selected\n");
        ggml_backend_free(b.handle);
        return 77;
    }
    // The upstream GGML_USE_CUDA alias must not enable NVIDIA-only hooks.
    if (b.is_cuda || vla::enable_fused_ops("musa_test", b.is_cuda)) return 1;
    auto * cpu = ggml_backend_cpu_init();
    if (!cpu) return 1;
    ggml_backend_cpu_set_n_threads(cpu, 4);
    bool ok = true;
    if (!vla::musa_register_fused_ops()) return 1;
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16})
        for (bool swapped : {false, true})
            ok = compare("merged_geglu_rows", type, merged_glu_reference(type, swapped),
                merged_glu(b.handle, type, true, swapped), type == GGML_TYPE_F32 ? 2e-6 : 2e-3) && ok;
    ok = compare("f16_native_gemm_rms", GGML_TYPE_F16,
        f16_gemm_norm(cpu, false), f16_gemm_norm(b.handle, true), 2e-3) && ok;
    for (int64_t h : {1, 33, 257, 1024, 1025}) {
        for (int64_t rows : {1, 50}) {
            const auto fused = fused_expert(b.handle, h, rows, true);
            ok = compare("fused_expert_vs_cpu", GGML_TYPE_F32,
                fused_expert(cpu, h, rows, false), fused, 3e-6) && ok;
            ok = compare("fused_expert_vs_musa", GGML_TYPE_F32,
                fused_expert(b.handle, h, rows, false), fused, 3e-6) && ok;
            ok = compare("f16_fused_expert", GGML_TYPE_F16,
                fused_expert(cpu, h, rows, true, true), fused_expert(b.handle, h, rows, true, true), 2e-3) && ok;
        }
    }
    ok = large_weight_buffers(b.handle) && ok;
    ok = compare("visual_patch_scores", GGML_TYPE_F32, patch_scores(cpu), patch_scores(b.handle), 2e-5) && ok;
    for (int64_t width : {1, 32, 51, 256, 579, 8192, 32769}) {
        for (bool masked : {false, true}) {
            char name[64];
            std::snprintf(name, sizeof(name), "softmax_%lld%s", (long long)width, masked ? "_mask" : "");
            ok = compare(name, GGML_TYPE_F32, softmax(cpu, width, masked),
                         softmax(b.handle, width, masked), 2e-5) && ok;
        }
    }
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
        ok = compare("vision", type, vision(cpu, type), vision(b.handle, type)) && ok;
        for (int64_t prefix : {1, 17, 529}) {
            const auto ref = gemma(cpu, type, prefix);
            const auto got = gemma(b.handle, type, prefix);
            const char * name = prefix == 1 ? "gemma_1" : prefix == 17 ? "gemma_17" : "gemma_529";
            ok = compare(name, type, ref, got) && ok;
            if (prefix == 17) {
                const auto packed = gemma(b.handle, type, prefix, true);
                ok = compare("gemma_merged_vs_cpu", type, ref, packed) && ok;
                ok = compare("gemma_merged_vs_musa", type, got, packed) && ok;
            }
        }
    }
    ggml_backend_free(cpu);
    ggml_backend_free(b.handle);
    return ok ? 0 : 1;
}
