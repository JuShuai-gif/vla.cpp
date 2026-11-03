#include "layers/packed_norm.h"
#include "layers/action_edges.h"
#include "layers/u8_patch.h"
// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0

// No checkpoint: exercise the actual Gemma graph builders used by pi0.5,
// including cached-prefix MQA, head_dim=256, non-aligned sequence lengths,
// adaRMS/gated residuals and the SigLIP patch embedding. Compare device output
// against the identical CPU graph, with identical resident weight types.
#include "backend.h"
#include "cuda/vla_cuda_ops.h"
#include "options.h"
#include "layers/attn.h"
#include "layers/fused_ops.h"
#include "modules/gemma_expert.h"
#include "ggml-alloc.h"
#include "modules/vision_prune.h"
#include "musa/vla_musa_ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace {

float f32_patch_value(size_t i) {
    const float edges[]={0.f,1.f,.5f,std::nextafter(.5f,0.f),std::nextafter(.5f,1.f),
                          1.f/255.f,.1234567f,1e-12f};
    return i%13<8?edges[i%13]:float(uint32_t(i*2654435761u)>>8)*(1.f/16777216.f);
}

struct Graph {
    ggml_context * ctx = ggml_init({32*1024*1024, nullptr, true});
    ggml_gallocr_t alloc = nullptr;
    std::vector<ggml_tensor *> inputs;
    void * mapped_input = nullptr;

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
            if(t->type==GGML_TYPE_I8) {
                std::vector<uint8_t> raw(n);for(size_t i=0;i<n;++i)raw[i]=(i*37)&255;
                ggml_backend_tensor_set(t,raw.data(),0,raw.size());continue;
            }
            std::vector<float> data(n);
            for (size_t i = 0; i < n; ++i)
                data[i] = ((int) ((i*37 + j*13) % 67) - 33) / 128.0f;
            if (std::strcmp(ggml_get_name(t), "prune_score_values") == 0)
                for(size_t i=0;i<n;++i)data[i]=.01f+std::fabs(std::sin(float(i)*.017f));
            if (std::strcmp(ggml_get_name(t), "attention_values") == 0)
                for (size_t i=0; i<n; ++i) data[i] = 2.0f*std::sin(float(i)*0.017f+float(j));
            if (std::strcmp(ggml_get_name(t), "epilogue_values") == 0)
                for (size_t i=0; i<n; ++i) data[i] = 10.0f*std::sin(float(i)*0.017f+float(j));
            if (std::strcmp(ggml_get_name(t), "geglu_values") == 0)
                for (auto & v : data) v *= 40.0f;
            if (std::strcmp(ggml_get_name(t), "attention_mask") == 0)
                for (size_t i=0; i<n; ++i) data[i]=(i%(size_t)t->ne[0])%7==6 ? -INFINITY : -0.125f*(i%5);
            if (std::strcmp(ggml_get_name(t), "mask") == 0)
                for (size_t i = 0; i < n; ++i)
                    data[i] = (i % (size_t)t->ne[0]) % 7 == 6 ? -INFINITY : 0.0f;
            if(std::strcmp(ggml_get_name(t),"u8_lut")==0)
                for(size_t i=0;i<n;++i)data[i]=(i/255.0f-.5f)/.5f;
            if(std::strcmp(ggml_get_name(t),"u8_chw")==0)
                for(size_t i=0;i<n;++i) {
                    const size_t S=t->ne[0],x=i%S,y=i/S%S,c=i/(S*S)%3,v=i/(S*S*3);
                    const size_t hwc=((v*S+y)*S+x)*3+c;
                    data[i]=(((hwc*37)&255)/255.0f-.5f)/.5f;
                }
            if(std::strcmp(ggml_get_name(t),"f32_hwc")==0)
                for(size_t i=0;i<n;++i)data[i]=f32_patch_value(i);
            if(std::strcmp(ggml_get_name(t),"f32_chw")==0)
                for(size_t i=0;i<n;++i) {
                    const size_t S=t->ne[0],x=i%S,y=i/S%S,c=i/(S*S)%3,v=i/(S*S*3);
                    data[i]=(f32_patch_value(((v*S+y)*S+x)*3+c)-.5f)/.5f;
                }
            if (j==0 && mapped_input) {
                std::memcpy(mapped_input,data.data(),ggml_nbytes(t));continue;
            }
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

    std::vector<float> run(ggml_backend_t backend, const std::vector<ggml_tensor *> & outputs, ggml_tensor * dynamic_scale = nullptr, bool switch_dtype = false, int timed_runs = 0, bool rebind_output = false) {
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
        auto * target=outputs.front();
        void * original_data=target->data;
        auto original_buffer=target->buffer;
        ggml_backend_buffer_t external[2]={nullptr,nullptr};
        if(rebind_output) for(auto & buffer:external) buffer=ggml_backend_alloc_buffer(backend,ggml_nbytes(target));
        std::vector<float> result;
        double elapsed_ms = 0;
        // Reuse the allocation and graph: catches stream/lifetime problems too.
        for (int repeat = 0; repeat < ((dynamic_scale || switch_dtype || rebind_output) ? 9 : 3) + timed_runs; ++repeat) {
            if(rebind_output) {
                target->buffer=external[(repeat/3)%2];
                target->data=ggml_backend_buffer_get_base(target->buffer);
            }
            if (switch_dtype) inputs[0]->type = repeat/3 == 1 ? GGML_TYPE_BF16 : GGML_TYPE_F16;
            if (dynamic_scale) {
                float scale = 0.5f * (repeat/3 + 1);
                std::memcpy(dynamic_scale->op_params, &scale, sizeof(scale));
            }
            // gallocr may reuse input memory after its last consumer; restore
            // every input just as predict restores each new observation.
            upload();
            const auto start = std::chrono::steady_clock::now();
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) return {};
            if (repeat >= 3) elapsed_ms += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            if (timed_runs && repeat >= 3) continue;
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
        if (timed_runs) std::printf("ATTENTION_BENCH {\"mean_ms\":%.6f,\"runs\":%d,\"graph_bytes\":%zu}\n",
            elapsed_ms/timed_runs,timed_runs,ggml_gallocr_get_buffer_size(alloc,0));
        target->data=original_data;target->buffer=original_buffer;
        for(auto buffer:external) if(buffer) ggml_backend_buffer_free(buffer);
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

std::vector<float> gemma(ggml_backend_t backend, ggml_type type, int64_t prefix, bool packed = false, bool flash = false, bool masked = false, bool recapture = false, bool reuse = false) {
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
    auto * prefix_mask = masked ? g.input(GGML_TYPE_F32, {prefix, prefix}) : nullptr;
    if (prefix_mask) ggml_set_name(prefix_mask, "mask");
    vla::RopeTableSpec rope_spec;
    auto * prefix_angles=vla::env_flag("VLA_PI05_ROPE_TABLE")?vla::rope_table(C,pos,cfg.head_dim,rope_spec):nullptr;
    auto * pre = vla::gemma_layer(C, w, x, pos, cfg, prefix, nullptr, nullptr, prefix_mask, &k, &v,
                                GGML_TYPE_F32, flash,nullptr,false,prefix_angles);

    constexpr int64_t chunk = 50;
    auto * h = g.input(GGML_TYPE_F32, {cfg.hidden, chunk});
    auto * suffix_pos = g.input(GGML_TYPE_I32, {chunk});
    auto * suffix_angles=vla::env_flag("VLA_PI05_ROPE_TABLE")?vla::rope_table(C,suffix_pos,cfg.head_dim,rope_spec):nullptr;
    auto * suffix_mask = masked ? g.input(GGML_TYPE_F32, {prefix + chunk, chunk}) : nullptr;
    if (suffix_mask) ggml_set_name(suffix_mask, "mask");
    auto * mod = g.input(GGML_TYPE_F32, {3*cfg.hidden});
    auto * scale = ggml_view_1d(C, mod, cfg.hidden, 0);
    auto * shift = ggml_view_1d(C, mod, cfg.hidden, cfg.hidden*sizeof(float));
    auto * gate = ggml_view_1d(C, mod, cfg.hidden, 2*cfg.hidden*sizeof(float));
    auto adarms = [&](ggml_tensor * t) {
        auto * norm = ggml_rms_norm(C, t, cfg.rms_eps);
        return ggml_add(C, ggml_add(C, norm, ggml_mul(C, norm, scale)), shift);
    };
    vla::KvSlot cache;
    cache.concat_only = true;
    for (int step = 0; step < 3; ++step) {
        auto * att = vla::gemma_attn(C, w, adarms(h), suffix_pos, cfg, chunk,
                                    k, v, suffix_mask, nullptr, nullptr, GGML_TYPE_F32, flash,
                                    nullptr, nullptr, reuse ? &cache : nullptr,nullptr,nullptr,nullptr,nullptr,false,suffix_angles);
        h = ggml_add(C, h, ggml_mul(C, att, gate));
        auto * mlp = vla::gemma_mlp(C, w, adarms(h), GGML_TYPE_F32);
        h = ggml_add(C, h, ggml_scale(C, ggml_mul(C, mlp, gate), -0.1f));
    }
    auto * scaled = recapture ? ggml_scale(C, h, 0.5f) : nullptr;
    return g.run(backend, {pre, scaled ? scaled : h}, scaled);
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

std::vector<float> mapped_input(ggml_backend_t backend, bool mapped) {
    Graph g;
    auto * x=g.input(GGML_TYPE_F32,{33,17});
    ggml_backend_buffer_t buffer=nullptr;
    if(mapped) {
        buffer=ggml_backend_musa_mapped_buffer(backend,ggml_nbytes(x),&g.mapped_input);
        if(!buffer) return {};
        x->buffer=buffer;x->data=ggml_backend_buffer_get_base(buffer);
    }
    auto result=g.run(backend,{ggml_scale(g.ctx,x,0.5f)},nullptr,false,0,true);
    if(buffer) ggml_backend_buffer_free(buffer);
    return result;
}

std::vector<float> vision_epilogue(ggml_backend_t backend, int64_t h, int64_t seq, int64_t batch, bool fused) {
    Graph g;
    auto * x = g.input(GGML_TYPE_F32, {h,seq,batch});
    ggml_set_name(x, "epilogue_values"); // Cover saturated and near-zero activation values.
    auto * bias = g.input(GGML_TYPE_F32, {h});
    auto * residual = g.input(GGML_TYPE_F32, {h,seq,batch});
    auto * gelu = fused ? vla::fused_bias_gelu(g.ctx,x,bias) : ggml_gelu(g.ctx,ggml_add(g.ctx,x,bias));
    auto * res = fused ? vla::fused_bias_res(g.ctx,x,bias,residual) : ggml_add(g.ctx,residual,ggml_add(g.ctx,x,bias));
    return g.run(backend,{gelu,res},nullptr,false,5);
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

std::vector<float> graph_source_dtype(ggml_backend_t backend) {
    Graph g;
    auto * a = g.input(GGML_TYPE_F16, {128, 64});
    auto * b = g.input(GGML_TYPE_F32, {128, 50});
    return g.run(backend, {ggml_mul_mat(g.ctx, a, b)}, nullptr, true);
}

std::vector<float> action_projection(ggml_backend_t backend, int64_t k, int64_t m, int64_t n) {
    Graph g;
    auto * a=g.input(GGML_TYPE_F32,{k,m});
    auto * b=g.input(GGML_TYPE_F32,{k,n});
    return g.run(backend,{ggml_mul_mat(g.ctx,a,b)});
}

std::vector<float> vision_attention(ggml_backend_t backend, bool flash, int64_t heads=4, int64_t nv=2, bool force_cast=false) {
    Graph g;
    constexpr int64_t hd=72, seq=256;
    auto * q=g.input(GGML_TYPE_F32, {hd*heads, seq, nv});
    auto * k=g.input(GGML_TYPE_F32, {hd*heads, seq, nv});
    auto * v=g.input(GGML_TYPE_F32, {hd*heads, seq, nv});
    auto view = [&](ggml_tensor * t) {
        return ggml_permute(g.ctx, ggml_reshape_4d(g.ctx,t,hd,heads,seq,nv),0,2,1,3);
    };
    auto * kview=view(k), * vview=view(v);
    if(force_cast) { kview=ggml_cast(g.ctx,kview,GGML_TYPE_F16);vview=ggml_cast(g.ctx,vview,GGML_TYPE_F16); }
    auto * out = flash
        ? vla::flash_attention(g.ctx,view(q),kview,vview,nullptr,1.f/std::sqrt(float(hd)))
        : vla::attention(g.ctx,vla::to_heads(g.ctx,q,hd,heads,seq,nv),
                         vla::to_heads(g.ctx,k,hd,heads,seq,nv),
                         vla::to_heads_v(g.ctx,v,hd,heads,seq,nv),nullptr,
                         1.f/std::sqrt(float(hd)),hd*heads,seq,nv);
    return g.run(backend,{out});
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
    if (!ok && ref.size() % 3 == 0) {
        const size_t n = ref.size()/3;
        for (size_t repeat=0; repeat<3; ++repeat) {
            double e=0;
            for (size_t i=repeat*n; i<(repeat+1)*n; ++i)
                e=std::max(e,std::fabs(double(ref[i])-got[i]));
            std::printf("  repeat %zu max_abs=%.7g\n",repeat,e);
        }
    }
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

std::vector<float> selected_rows(ggml_backend_t backend, int count, int keep) {
    Graph g;
    auto * scores = g.input(GGML_TYPE_F32, {count});
    auto * emb = g.input(GGML_TYPE_F32, {37, count});
    // upload's repeating 67-value pattern deliberately creates score ties.
    return g.run(backend, {vla::vision_keep_rows(g.ctx, emb, scores, keep)});
}

std::vector<float> attention256(ggml_backend_t backend, int nq, int nk, bool masked, int mode) {
    Graph g;
    auto * q = g.input(GGML_TYPE_F32,{256,8,nq});
    auto * k = g.input(GGML_TYPE_F16,{256,nk,1});
    auto * v = g.input(GGML_TYPE_F16,{256,nk,1});
    for(auto * t:{q,k,v}) ggml_set_name(t,"attention_values");
    auto * mask = masked ? g.input(GGML_TYPE_F32,{nk,nq}) : nullptr;
    if(mask) ggml_set_name(mask,"attention_mask");
    q=ggml_permute(g.ctx,q,0,2,1,3);
    ggml_tensor * out;
    if(mode==1) out=vla::flash_attention(g.ctx,q,k,v,mask?ggml_cast(g.ctx,mask,GGML_TYPE_F16):nullptr,1.0f/16);
    else {
        k=ggml_cast(g.ctx,k,GGML_TYPE_F32);v=ggml_cast(g.ctx,v,GGML_TYPE_F32);
        out=vla::attention(g.ctx,q,k,ggml_cont(g.ctx,ggml_transpose(g.ctx,v)),mask,1.0f/16,2048,nq);
    }
    std::printf("ATTENTION_CASE q=%d k=%d mask=%d mode=%d\n",nq,nk,masked,mode);
    return g.run(backend,{out},nullptr,false,mode?15:0);
}

std::vector<float> strided_pack(ggml_backend_t backend, int64_t hd, int64_t seq, int64_t batch) {
    Graph g;
    auto * x = g.input(GGML_TYPE_F32, {hd, 3*16, seq, batch});
    // A head-aligned slice inside merged QKV, with gaps between token rows.
    auto * q = ggml_view_4d(g.ctx, x, hd, 16, seq, batch, x->nb[1], x->nb[2], x->nb[3], 16*x->nb[1]);
    auto * transposed = ggml_permute(g.ctx, q, 0, 2, 1, 3);
    return g.run(backend, {ggml_cast(g.ctx, transposed, GGML_TYPE_F16)});
}

std::vector<float> patch_scores(ggml_backend_t backend,bool fused=false) {
    Graph g;
    auto * a = g.input(GGML_TYPE_F32, {256, 256, 16});ggml_set_name(a,"prune_score_values");
    return g.run(backend, {fused?vla::fused_vision_scores(g.ctx,a):vla::vision_patch_scores(g.ctx, a)});
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

std::vector<float> action_linear(ggml_backend_t,int64_t,int64_t,int,bool,bool,bool);

std::vector<float> small_padded_view(ggml_backend_t backend,int offset,bool fused) {
 Graph g;auto * w=g.input(GGML_TYPE_F16,{128,64});auto * pad=g.input(GGML_TYPE_F16,{128,64});
 auto * x=ggml_view_2d(g.ctx,pad,128,50,pad->nb[1],offset*pad->nb[1]);
 return g.run(backend,{fused?vla::small_gemm(g.ctx,w,x):ggml_mul_mat(g.ctx,w,x)});
}

std::vector<float> packed_geglu_graph(ggml_backend_t backend,int h,int rows,bool fused) {
 Graph g;auto * x=g.input(GGML_TYPE_F32,{2*h,rows});ggml_set_name(x,"geglu_values");
 auto * out=fused?vla::packed_geglu(g.ctx,x):vla::retype(ggml_geglu(g.ctx,x),GGML_TYPE_F16);
 return g.run(backend,{out});
}

std::vector<float> packed_norm_graph(ggml_backend_t backend,int h,int rows,int kind,bool fused) {
 Graph g;auto * x=g.input(GGML_TYPE_F32,{h,rows});
 auto * o=kind?g.input(GGML_TYPE_F32,{h,kind==2?1:rows}):nullptr;
 auto * gate=kind==1?g.input(GGML_TYPE_F32,{3*h}):nullptr;auto * mod=g.input(GGML_TYPE_F32,{3*h});
 ggml_tensor * res=nullptr,*norm=nullptr;
 vla::FusedAdarmsSpec a;vla::FusedResAdarmsSpec b;
 if(fused)norm=vla::packed_norm(g.ctx,x,o,gate,mod,a.eps,kind,&res);
 else {
  if(kind==1)norm=vla::fused_res_adarms(g.ctx,x,o,gate,mod,b,&res);
  else if(kind==2){b.magic=vla::FUSED_BIAS_ADARMS_MAGIC;norm=vla::fused_bias_adarms(g.ctx,x,o,mod,b,&res);}
  else norm=vla::fused_adarms(g.ctx,x,mod,GGML_TYPE_F32,a);
  norm=ggml_cast(g.ctx,norm,GGML_TYPE_F16);
 }
 std::vector<ggml_tensor *> out;if(kind)out.push_back(res);out.push_back(norm);
 if(fused)out.push_back(norm->src[0]);
 auto result=g.run(backend,out);
 if(fused) {
  const size_t valid=(kind?2:1)*h*rows;
  const size_t stride=valid+h*64;
  if(result.size()!=3*stride)return {};
  std::vector<float> compact;
  for(int repeat=0;repeat<3;++repeat) {
   for(int i=h*rows;i<h*64;++i)if(result[repeat*stride+valid+i]!=0.f)return {};
   compact.insert(compact.end(),result.begin()+repeat*stride,result.begin()+repeat*stride+valid);
  }
  return compact;
 }
 return result;
}

std::vector<float> res_adarms(ggml_backend_t backend, int64_t h, int64_t rows, ggml_type type, bool fused) {
    Graph g;
    auto * x = g.input(GGML_TYPE_F32, {h, rows});
    auto * o = g.input(type, {h, rows});
    auto * gate = g.input(GGML_TYPE_F32, {3*h});
    auto * mod = g.input(GGML_TYPE_F32, {3*h});
    vla::FusedAdarmsSpec a; vla::FusedGateResSpec b; vla::FusedResAdarmsSpec c;
    ggml_tensor * residual = nullptr;
    ggml_tensor * norm = nullptr;
    if (fused) norm = vla::fused_res_adarms(g.ctx, x, o, gate, mod, c, &residual);
    else {
        residual = vla::fused_gateres(g.ctx, x, o, gate, b);
        norm = vla::fused_adarms(g.ctx, residual, mod, GGML_TYPE_F32, a);
    }
    return g.run(backend, {residual, norm});
}

#ifdef GGML_USE_MUSA
std::vector<float> prefetch_mlp_graph(ggml_backend_t backend,bool prefetch) {
    std::unique_ptr<void,decltype(&vla::musa_weight_prefetch_destroy)> state(
        vla::musa_weight_prefetch_create(),vla::musa_weight_prefetch_destroy);
    if(!state)return {};
    Graph g;vla::GemmaLayerW w;
    w.Wgu=g.input(GGML_TYPE_F16,{32,128});w.Wdown=g.input(GGML_TYPE_F16,{64,32});
    auto * next=ggml_view_2d(g.ctx,w.Wgu,32,80,w.Wgu->nb[1],0);
    auto * x=g.input(GGML_TYPE_F32,{32,17});
    if(prefetch) {
        w.prefetch_next=next;w.prefetch_begin.state=state.get();w.prefetch_end.state=state.get();
        w.prefetch_begin.bytes=ggml_nbytes(next);
    }
    for(int i=0;i<3;++i)x=vla::gemma_mlp(g.ctx,w,x,GGML_TYPE_F32);
    return g.run(backend,{x});
}
#endif
std::vector<float> small_gemm_graph(ggml_backend_t backend,int k,int m,int n,ggml_type type,int tile) {
    Graph g;auto * w=g.input(GGML_TYPE_F16,{k,m});auto * x=g.input(type,{k,n});
    ggml_tensor * out=nullptr;
    if(tile<0) {
        // CPU's native F16 dot may accumulate in half; reference this kernel's F32 accumulation explicitly.
        auto * wf=ggml_cast(g.ctx,w,GGML_TYPE_F32);
        auto * xf=ggml_cast(g.ctx,ggml_cast(g.ctx,x,GGML_TYPE_F16),GGML_TYPE_F32);
        out=ggml_mul_mat(g.ctx,wf,xf);ggml_mul_mat_set_prec(out,GGML_PREC_F32);
    } else out=tile?vla::small_gemm(g.ctx,w,x,tile):ggml_mul_mat(g.ctx,w,x);
    return g.run(backend,{out});
}
std::vector<float> u8_patch_graph(ggml_backend_t backend,int nv,bool fused) {
    Graph g;ggml_tensor * out;
    vla::U8PatchSpec spec;
    if(fused) {
        auto * raw=g.input(GGML_TYPE_I8,{3,224,224,nv});
        auto * lut=g.input(GGML_TYPE_F16,{256});ggml_set_name(lut,"u8_lut");
        out=u8_patch(g.ctx,raw,lut,spec);
    } else {
        auto * chw=g.input(GGML_TYPE_F32,{224,224,3,nv});ggml_set_name(chw,"u8_chw");
        auto * w=g.input(GGML_TYPE_F32,{14,14,3,1});
        out=ggml_im2col(g.ctx,w,chw,14,14,0,0,1,1,true,GGML_TYPE_F16);
    }
    return g.run(backend,{out});
}

std::vector<float> f32_patch_graph(ggml_backend_t backend,int nv,bool fused) {
    Graph g;ggml_tensor * out;
    vla::U8PatchSpec spec{vla::F32_PATCH_MAGIC,14};
    if(fused) {
        auto * raw=g.input(GGML_TYPE_F32,{3,224,224,nv});ggml_set_name(raw,"f32_hwc");
        out=vla::f32_patch(g.ctx,raw,spec);
    } else {
        auto * chw=g.input(GGML_TYPE_F32,{224,224,3,nv});ggml_set_name(chw,"f32_chw");
        auto * w=g.input(GGML_TYPE_F32,{14,14,3,1});
        out=ggml_im2col(g.ctx,w,chw,14,14,0,0,1,1,true,GGML_TYPE_F16);
    }
    return g.run(backend,{out});
}

bool custom_parameter_replay(ggml_backend_t backend) {
    Graph g;
    auto * x=g.input(GGML_TYPE_F32,{32,50}),*v=g.input(GGML_TYPE_F32,{32,50}),*b=g.input(GGML_TYPE_F32,{32});
    vla::FusedActionStepSpec spec{vla::FUSED_ACTION_STEP_MAGIC,-1.f/3};
    auto * y=vla::fused_action_step(g.ctx,x,v,b,spec);
    ggml_set_output(y);auto * graph=ggml_new_graph(g.ctx);ggml_build_forward_expand(graph,y);
    g.alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if(!ggml_gallocr_alloc_graph(g.alloc,graph))return false;
    bool ok=true;
    for(int repeat=0;repeat<9;++repeat) {
        spec.dt=repeat/3==1?-.1f:-1.f/3;
        vla::custom_parameter_snapshot(y,spec);
        g.upload();
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return false;
        std::vector<float> got(1600),ref(1600);
        ggml_backend_tensor_get(y,got.data(),0,got.size()*4);
        auto value=[](int i,int j){return ((i*37+j*13)%67-33)/128.f;};
        for(int i=0;i<1600;++i) {
            volatile float velocity=value(i,1)+value(i%32,2);
            volatile float step=velocity*spec.dt;
            ref[i]=value(i,0)+step;
        }
        ok=compare("custom_dt_graph_replay",GGML_TYPE_F32,ref,got,1e-6)&&ok;
    }
    return ok;
}

std::vector<float> cached_rope_graph(ggml_backend_t backend,int64_t heads,int64_t seq,ggml_type type,bool cached) {
    Graph g;constexpr int64_t h=256;
    auto * raw=g.input(GGML_TYPE_F32,{h,heads+2,seq});
    ggml_set_name(raw,"attention_values");
    auto * pos=g.input(GGML_TYPE_I32,{seq});
    auto * x=ggml_view_3d(g.ctx,raw,h,heads,seq,raw->nb[1],raw->nb[2],h*4);
    vla::RopeTableSpec spec;
    auto * y=cached ? vla::rope_apply(g.ctx,x,vla::rope_table(g.ctx,pos,h,spec)) :
        ggml_rope_ext(g.ctx,x,pos,nullptr,h,GGML_ROPE_TYPE_NEOX,0,10000.f,1.f,0.f,1.f,32.f,1.f);
    if(type==GGML_TYPE_F16)y=cached?vla::retype(y,type):ggml_cast(g.ctx,y,type);
    return g.run(backend,{y});
}

std::vector<float> last_prefix_kv(ggml_backend_t backend,int64_t seq,bool kv_only) {
    Graph g;
    vla::Config cfg;cfg.hidden=2048;cfg.n_q_heads=8;cfg.n_kv_heads=1;
    cfg.head_dim=cfg.kv_full_dim=256;cfg.rms_eps=1e-6f;cfg.rope_freq_base=10000.f;
    vla::GemmaLayerW w;
    w.Wqkv=g.input(GGML_TYPE_F16,{2048,2560});
    w.Wo=ggml_view_2d(g.ctx,w.Wqkv,2048,2048,w.Wqkv->nb[1],0);
    w.ln_in=g.input(GGML_TYPE_F32,{2048});
    auto * x=g.input(GGML_TYPE_F32,{2048,seq});
    auto * pos=g.input(GGML_TYPE_I32,{seq});
    ggml_tensor * k=nullptr,*v=nullptr;
    if(kv_only)vla::gemma_layer(g.ctx,w,x,pos,cfg,seq,nullptr,nullptr,nullptr,&k,&v,GGML_TYPE_F32,true,nullptr,true);
    else vla::gemma_attn(g.ctx,w,vla::rms_norm(g.ctx,x,w.ln_in,cfg.rms_eps),pos,cfg,seq,
                        nullptr,nullptr,nullptr,&k,&v,GGML_TYPE_F32,true);
    return g.run(backend,{k,v});
}

std::vector<float> action_linear(ggml_backend_t backend,int64_t h,int64_t rows,int steps,bool fused,bool packed=false,bool narrow=false) {
    Graph g;
    auto * wi=g.input(GGML_TYPE_F32,{32,h}),*wo=g.input(GGML_TYPE_F32,{h,32});
    auto * x=g.input(GGML_TYPE_F32,{32,rows}),*bi=g.input(GGML_TYPE_F32,{h}),*bo=g.input(GGML_TYPE_F32,{32});
    auto * mi=g.input(GGML_TYPE_F32,{3*h}),*mo=g.input(GGML_TYPE_F32,{3*h});
    vla::FusedActionLinearSpec in{vla::FUSED_ACTION_LINEAR_MAGIC,1,1e-6f,-1.f/steps};
    vla::FusedActionLinearSpec out{vla::FUSED_ACTION_LINEAR_MAGIC,0,1e-6f,-1.f/steps};
    in.packed=packed?1:0;
    ggml_tensor * hidden=nullptr,*norm=nullptr,*y=nullptr;
    auto adarms=[&](ggml_tensor * a,ggml_tensor * m) {
        auto * n=ggml_rms_norm(g.ctx,a,1e-6f);
        auto * scale=ggml_view_1d(g.ctx,m,h,0),*shift=ggml_view_1d(g.ctx,m,h,h*4);
        return ggml_add(g.ctx,ggml_add(g.ctx,n,ggml_mul(g.ctx,n,scale)),shift);
    };
    if(fused) {
        norm=vla::fused_action_linear(g.ctx,wi,x,bi,mi,nullptr,in,&hidden);
        y=vla::fused_action_linear(g.ctx,wo,hidden,bo,mo,x,out);
    } else {
        hidden=ggml_add(g.ctx,ggml_mul_mat(g.ctx,wi,x),bi);
        norm=adarms(hidden,mi);
        y=ggml_add(g.ctx,x,ggml_scale(g.ctx,ggml_add(g.ctx,ggml_mul_mat(g.ctx,wo,adarms(hidden,mo)),bo),out.dt));
    }
    if(narrow && !packed)norm=ggml_cast(g.ctx,norm,GGML_TYPE_F16);
    return g.run(backend,{hidden,norm,y});
}

std::vector<float> action_edges(ggml_backend_t backend, int64_t h, int64_t rows, int steps, bool direct) {
    Graph g;
    auto * x=g.input(GGML_TYPE_F32,{h,rows});
    auto * b=g.input(GGML_TYPE_F32,{h});
    auto * mod=g.input(GGML_TYPE_F32,{3*h});
    vla::FusedResAdarmsSpec spec{vla::FUSED_BIAS_ADARMS_MAGIC,1e-6f};
    vla::FusedAdarmsSpec old;
    vla::FusedActionStepSpec step{vla::FUSED_ACTION_STEP_MAGIC,-1.f/steps};
    ggml_tensor * res=nullptr;
    ggml_tensor * norm=nullptr;
    if(direct) norm=vla::fused_bias_adarms(g.ctx,x,b,mod,spec,&res);
    else { res=ggml_add(g.ctx,x,b);norm=vla::fused_adarms(g.ctx,res,mod,GGML_TYPE_F32,old); }
    auto * y=direct ? vla::fused_action_step(g.ctx,x,norm,b,step) :
                     ggml_add(g.ctx,x,ggml_scale(g.ctx,ggml_add(g.ctx,norm,b),step.dt));
    return g.run(backend,{res,norm,y});
}

std::vector<float> geglu_direct_half(ggml_backend_t backend, int64_t h, int64_t rows, bool direct, bool swapped) {
    Graph g;
    auto * x = g.input(GGML_TYPE_F32, {2*h, rows});
    ggml_set_name(x, "geglu_values");
    auto * y = swapped ? ggml_geglu_swapped(g.ctx, x) : ggml_geglu(g.ctx, x);
    y = direct ? vla::retype(y, GGML_TYPE_F16) : ggml_cast(g.ctx, y, GGML_TYPE_F16);
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
    for (int repeat = 0; repeat < 3; ++repeat)
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
    if(argc==2 && std::strcmp(argv[1],"--prune-score")==0) {
        ok=compare("prune_score_exact",GGML_TYPE_F32,patch_scores(b.handle),patch_scores(b.handle,true),1e-12)&&ok;
        ok=compare("prune_score_cpu",GGML_TYPE_F32,patch_scores(cpu,true),patch_scores(b.handle,true),2e-5)&&ok;
        ggml_backend_free(cpu);ggml_backend_free(b.handle);return ok?0:1;
    }
    if(argc==2 && std::strcmp(argv[1],"--packed-norm")==0) {
        for(int offset:{0,14})
            ok=compare("small_padded_view_exact",GGML_TYPE_F32,
                small_padded_view(b.handle,offset,false),small_padded_view(b.handle,offset,true),1e-12)&&ok;
        for(int h:{256,1024})for(int rows:{1,17,50,64})
            ok=compare("packed_action_linear_exact",GGML_TYPE_F32,
                action_linear(b.handle,h,rows,10,true,false,true),
                action_linear(b.handle,h,rows,10,true,true,true),1e-12)&&ok;
        for(int h:{256,4096})for(int rows:{1,17,50,64})
            ok=compare("packed_geglu_exact",GGML_TYPE_F16,
                packed_geglu_graph(b.handle,h,rows,false),
                packed_geglu_graph(b.handle,h,rows,true),1e-12)&&ok;
        for(int h:{256,1024})for(int rows:{1,17,50,64})for(int kind:{0,1,2})
            ok=compare("packed_norm_exact",GGML_TYPE_F32,
                packed_norm_graph(b.handle,h,rows,kind,false),
                packed_norm_graph(b.handle,h,rows,kind,true),1e-12)&&ok;
        ggml_backend_free(cpu);return ok?0:1;
    }
    if(argc==2 && std::strcmp(argv[1],"--f32-patch")==0) {
        for(int nv:{1,2}) {
            const auto fused=f32_patch_graph(b.handle,nv,true);
            ok=compare("f32_patch_native",GGML_TYPE_F16,f32_patch_graph(b.handle,nv,false),fused,1e-12)&&ok;
            ok=compare("f32_patch_cpu",GGML_TYPE_F16,f32_patch_graph(cpu,nv,true),fused,1e-12)&&ok;
        }
        ggml_backend_free(cpu);ggml_backend_free(b.handle);return ok?0:1;
    }
    if (argc == 2 && std::strcmp(argv[1], "--attention256") == 0) {
        for (auto shape : {std::pair<int,int>{50,328},{50,568},{278,278},{518,518},{17,67},{1,1},{1,67},{50,640},{50,641},{640,640}})
            for (bool masked : {false,true}) {
                auto ref=attention256(cpu,shape.first,shape.second,masked,0);
                ok=compare("attention256_vs_cpu",GGML_TYPE_F16,ref,
                    attention256(b.handle,shape.first,shape.second,masked,1),5e-3)&&ok;
            }
        ggml_backend_free(cpu);ggml_backend_free(b.handle);
        return ok?0:1;
    }
    if (argc==2 && std::strcmp(argv[1],"--mapped-io")==0) {
        const auto ref=mapped_input(cpu,false);
        ok=compare("rebind_output_graph",GGML_TYPE_F32,ref,mapped_input(b.handle,false),1e-12) && ok;
        ok=compare("mapped_input_rebind_graph",GGML_TYPE_F32,ref,mapped_input(b.handle,true),1e-12) && ok;
        ggml_backend_free(cpu);ggml_backend_free(b.handle);return ok?0:1;
    }
    if (argc == 2 && std::strcmp(argv[1], "--vision-epilogue") == 0) {
        {
            Graph g;
            auto * x = g.input(GGML_TYPE_F32,{33,65536});
            auto * bias = g.input(GGML_TYPE_F32,{33});
            auto * op = vla::fused_bias_gelu(g.ctx,x,bias);
            ok = !ggml_backend_supports_op(b.handle,op) && ggml_backend_supports_op(cpu,op) && ok;
        }
        for (int64_t h : {33,1152,4304})
            for (int64_t seq : {17,256})
                for (int64_t batch : {1,2}) {
                    const auto fused = vision_epilogue(b.handle,h,seq,batch,true);
                    ok = compare("vision_epilogue_musa",GGML_TYPE_F32,
                        vision_epilogue(b.handle,h,seq,batch,false),fused,1e-12) && ok;
                    // ggml CPU GELU uses an F16 lookup table. The fused CPU
                    // callback evaluates the same F32 tanh formula as MUSA.
                    ok = compare("vision_epilogue_cpu_fallback",GGML_TYPE_F32,
                        vision_epilogue(cpu,h,seq,batch,true),fused,2e-6) && ok;
                }
        ggml_backend_free(cpu);ggml_backend_free(b.handle);
        return ok ? 0 : 1;
    }
    if (argc == 2 && std::strcmp(argv[1], "--strided-pack") == 0) {
        for (int64_t hd : {72, 256})
            for (int64_t seq : {17, 256})
                for (int64_t batch : {1, 2})
                    ok = compare("strided_pack", GGML_TYPE_F16,
                        strided_pack(cpu, hd, seq, batch), strided_pack(b.handle, hd, seq, batch), 1e-12) && ok;
        ggml_backend_free(cpu);
        ggml_backend_free(b.handle);
        return ok ? 0 : 1;
    }
    if (argc == 2 && std::strcmp(argv[1], "--f32-attention") == 0) {
        ok = compare("f32_attention_cpu", GGML_TYPE_F32,
                     vision_attention(cpu, false), vision_attention(b.handle, false), 2e-5);
        ggml_backend_free(cpu);
        ggml_backend_free(b.handle);
        return ok ? 0 : 1;
    }
    if (argc == 2 && std::strcmp(argv[1], "--dtype-switch") == 0) {
        ok = compare("graph_source_dtype_switch", GGML_TYPE_BF16,
                     graph_source_dtype(cpu), graph_source_dtype(b.handle));
        ggml_backend_free(cpu);
        ggml_backend_free(b.handle);
        return ok ? 0 : 1;
    }
    if (argc == 2 && std::strcmp(argv[1], "--vision-pack") == 0) {
        if(!vla::musa_vision_pack_enabled()) return 2;
        for(int64_t batch : {1,2})
            ok=compare("vision_pack_exact",GGML_TYPE_F16,
                vision_attention(b.handle,true,16,batch,true),
                vision_attention(b.handle,true,16,batch,false),1e-12) && ok;
        ggml_backend_free(cpu);ggml_backend_free(b.handle);
        return ok?0:1;
    }
    if (argc == 2 && std::strcmp(argv[1], "--vision-sdpa") == 0) {
        for (int64_t batch : {1,2})
            ok = compare("vision_sdpa_16heads", GGML_TYPE_F16,
                vision_attention(cpu, false, 16, batch),
                vision_attention(b.handle, true, 16, batch), 5e-3) && ok;
        ggml_backend_free(cpu); ggml_backend_free(b.handle);
        return ok ? 0 : 1;
    }
    if (argc == 2 && std::strcmp(argv[1], "--flash-attn") == 0) {
        ok = compare("flash_vision_strided_72", GGML_TYPE_F16,
                     vision_attention(b.handle, false), vision_attention(b.handle, true), 5e-3) && ok;
        for (int64_t prefix : {17, 529})
            for (bool masked : {false, true})
                ok = compare("flash_gemma_vs_unfused", GGML_TYPE_F16,
                    gemma(b.handle, GGML_TYPE_F16, prefix, true, false, masked),
                    gemma(b.handle, GGML_TYPE_F16, prefix, true, true, masked), 5e-3) && ok;
        for (int64_t prefix : {17, 278, 518})
            for (bool masked : {false, true})
                ok = compare("flash_reuse_kv", GGML_TYPE_F16,
                    gemma(b.handle, GGML_TYPE_F16, prefix, true, true, masked, false, false),
                    gemma(b.handle, GGML_TYPE_F16, prefix, true, true, masked, false, true), 1e-12) && ok;
        ok = compare("flash_gemma_recapture", GGML_TYPE_F16,
                     gemma(b.handle, GGML_TYPE_F16, 17, true, false, false, true),
                     gemma(b.handle, GGML_TYPE_F16, 17, true, true, false, true), 5e-3) && ok;
        ggml_backend_free(cpu);
        ggml_backend_free(b.handle);
        return ok ? 0 : 1;
    }

#ifdef GGML_USE_MUSA
    ok=compare("multi_layer_prefetch_graph_replay",GGML_TYPE_F32,
               prefetch_mlp_graph(b.handle,false),prefetch_mlp_graph(b.handle,true),1e-12) && ok;
#endif
    for(int nv:{1,2})
        ok=compare("u8_lut_patch_all_values",GGML_TYPE_F16,u8_patch_graph(b.handle,nv,false),
                   u8_patch_graph(b.handle,nv,true),1e-12) && ok;
#ifdef GGML_USE_MUSA
    if(vla::musa_small_gemm_supported())
#endif
    for(auto type:{GGML_TYPE_F32,GGML_TYPE_F16}) {
        for(int n:{1,17,50,64})for(int tile:{32,64})
            ok=compare("small_gemm_cpu",GGML_TYPE_F32,
                small_gemm_graph(cpu,128,64,n,type,-1),small_gemm_graph(b.handle,128,64,n,type,tile),1e-5) && ok;
        for(auto shape:{std::make_pair(1024,2560),std::make_pair(2048,1024),
                        std::make_pair(1024,8192),std::make_pair(4096,1024)})
            for(int tile:{32,64})
                ok=compare("small_gemm_real_shape",GGML_TYPE_F32,
                    small_gemm_graph(b.handle,shape.first,shape.second,50,type,0),
                    small_gemm_graph(b.handle,shape.first,shape.second,50,type,tile),1e-5) && ok;
    }
    ok=custom_parameter_replay(b.handle)&&ok;
    for(int64_t heads : {1,8})
        for(int64_t seq : {17,50,278,518})
            for(auto type : {GGML_TYPE_F32,GGML_TYPE_F16})
                ok=compare("cached_rope_strided",type,
                    cached_rope_graph(b.handle,heads,seq,type,false),
                    cached_rope_graph(b.handle,heads,seq,type,true),type==GGML_TYPE_F32?1e-6:1e-3) && ok;
    for(int64_t seq : {17,278,518})
        ok=compare("last_prefix_kv_only",GGML_TYPE_F16,
            last_prefix_kv(b.handle,seq,false),last_prefix_kv(b.handle,seq,true),1e-3) && ok;
    for(int64_t h : {33,257,1024})
        for(int64_t rows : {1,50})
            for(int steps : {3,10})
                ok=compare("action_linear_musa",GGML_TYPE_F32,
                    action_linear(b.handle,h,rows,steps,false),
                    action_linear(b.handle,h,rows,steps,true),1e-5) && ok;
    for (int64_t h : {1,32,257,1024,1025})
        for (int64_t rows : {1,50})
            for (int steps : {3,10})
                ok=compare("action_edges_exact",GGML_TYPE_F32,
                    action_edges(b.handle,h,rows,steps,false),
                    action_edges(b.handle,h,rows,steps,true),1e-12) && ok;
    for (int64_t h : {4096, 4097, 16384})
        for (int64_t rows : {3, 50})
            for (bool swapped : {false, true})
                ok = compare("geglu_direct_half_exact", GGML_TYPE_F16,
                    geglu_direct_half(b.handle, h, rows, false, swapped),
                    geglu_direct_half(b.handle, h, rows, true, swapped), 1e-12) && ok;
    for (int64_t h : {1, 33, 257, 1024, 1025})
        for (int64_t rows : {1, 50})
            for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
                const auto got = res_adarms(b.handle, h, rows, type, true);
                ok = compare("res_adarms_exact", GGML_TYPE_F32,
                    res_adarms(b.handle, h, rows, type, false), got, 1e-12) && ok;
                ok = compare("res_adarms_cpu", GGML_TYPE_F32,
                    res_adarms(cpu, h, rows, type, false), got, 3e-6) && ok;
            }

    ok = compare("graph_source_dtype_switch", GGML_TYPE_BF16,
                 graph_source_dtype(cpu), graph_source_dtype(b.handle)) && ok;
    for (int count : {17, 256})
        for (int keep : {1, count/2, count})
            ok = compare("stable_vision_keep", GGML_TYPE_F32,
                         selected_rows(cpu, count, keep), selected_rows(b.handle, count, keep), 1e-12) && ok;

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
    for (int64_t n : {7, 50, 51}) {
        ok = compare("action_input_f32", GGML_TYPE_F32,
                     action_projection(cpu,32,1024,n), action_projection(b.handle,32,1024,n), 2e-5) && ok;
        ok = compare("action_output_f32", GGML_TYPE_F32,
                     action_projection(cpu,1024,32,n), action_projection(b.handle,1024,32,n), 2e-5) && ok;
    }
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
