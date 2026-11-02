#!/usr/bin/env python3
# Copyright 2026 VinRobotics
# SPDX-License-Identifier: Apache-2.0
"""MUSA 5.1/M1000 compatibility fixes for the pinned llama.cpp (b11223).

M1000 rejects softmax's cudaFuncSetAttribute(MaxDynamicSharedMemorySize,
sharedMemPerBlockOptin) through the MUSA alias, even for tiny rows. Use the
default shared-memory budget on MUSA and retain ggml's existing low-shared-memory
kernel for wider rows. NVIDIA and HIP code paths are unchanged.

Applied at configure time, including FETCHCONTENT_SOURCE_DIR_LLAMA checkouts.
Idempotent, with exact anchors so a dependency bump cannot silently skip a fix.
"""
from pathlib import Path
import sys


def patch(root: Path) -> None:
    path = root / "ggml/src/ggml-cuda/softmax.cu"
    original = text = path.read_text()
    budget = "    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;"
    edits = [(budget, """#ifdef GGML_USE_MUSA
    // vla.cpp: M1000 cannot opt in to a larger dynamic shared-memory budget.
    const size_t smpbo = ggml_cuda_info().devices[id].smpb;
#else
""" + budget + "\n#endif", 2)]
    for kernel, indent in [("soft_max_f32<true, ncols, block, T>", "            "),
                           ("soft_max_f32<true, 0, 0, T>", "    ")]:
        anchor = f"{indent}CUDA_SET_SHARED_MEMORY_LIMIT(({kernel}), smpbo);"
        replacement = ("#ifndef GGML_USE_MUSA\n" + anchor + "\n#else\n" +
                       indent + "GGML_UNUSED(smpbo);\n#endif")
        edits.append((anchor, replacement, 1))
    for anchor, replacement, count in edits:
        if text.count(replacement) == count:
            continue
        if text.count(anchor) != count:
            raise SystemExit(f"{path}: expected {count} anchors for {anchor!r}; check llama.cpp version")
        text = text.replace(anchor, replacement)
    if text != original:
        path.write_text(text)
        print(f"MUSA: patched {path}")

    path = root / "ggml/src/ggml-cuda/ggml-cuda.cu"
    original = text = path.read_text()
    anchor = "static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {"
    replacement = """#ifdef GGML_USE_MUSA
// M1000/5.1 aliases offsets 4 GiB apart in a single allocation. Split model buffers.
static size_t ggml_backend_musa_buffer_type_get_max_size(ggml_backend_buffer_type_t) {
    return (size_t) 2 * 1024 * 1024 * 1024;
}
#endif

""" + anchor
    if replacement not in text:
        if text.count(anchor) != 1:
            raise SystemExit(f"{path}: missing device buffer interface anchor")
        text = text.replace(anchor, replacement)
    anchor = "    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX\n    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,"
    replacement = """#ifdef GGML_USE_MUSA
    /* .get_max_size     = */ ggml_backend_musa_buffer_type_get_max_size,
#else
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
#endif
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,"""
    if replacement not in text:
        if text.count(anchor) != 1:
            raise SystemExit(f"{path}: missing device max-size anchor")
        text = text.replace(anchor, replacement)
    if text != original:
        path.write_text(text)
        print(f"MUSA: patched {path}")


    # The in-tree MUSA kernels claim only their own CUSTOM nodes. Keep the
    # pointers null unless the model registers them; unknown ops use ggml.
    original = text = path.read_text()
    text = text.replace('(*ggml_musa_ext_forward)(ggml_tensor *, void *)',
                        '(*ggml_musa_ext_forward)(ggml_tensor *, void *, void *)')
    text = text.replace('ggml_musa_ext_forward(dst, (void *) ctx.stream())',
                        'ggml_musa_ext_forward(dst, (void *) ctx.stream(), dst->op == GGML_OP_MUL_MAT ? (void *) ctx.cublas_handle() : nullptr)')
    text = text.replace('    if (op->op == GGML_OP_CUSTOM && ggml_musa_ext_supports) return ggml_musa_ext_supports(op);',
                        '    if (ggml_musa_ext_supports && ggml_musa_ext_supports(op)) return true;\n    if (op->op == GGML_OP_CUSTOM) return false;')
    anchor = "static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {"
    replacement = """#ifdef GGML_USE_MUSA
extern "C" {
__attribute__((visibility("default"))) bool (*ggml_musa_ext_forward)(ggml_tensor *, void *, void *) = nullptr;
__attribute__((visibility("default"))) bool (*ggml_musa_ext_supports)(const ggml_tensor *) = nullptr;
}
#endif

""" + anchor + """
#ifdef GGML_USE_MUSA
    if (ggml_musa_ext_forward && ggml_musa_ext_forward(dst, (void *) ctx.stream(), dst->op == GGML_OP_MUL_MAT ? (void *) ctx.cublas_handle() : nullptr)) return true;
#endif"""
    if "if (ggml_musa_ext_forward && ggml_musa_ext_forward(dst," not in text:
        if text.count(anchor) != 1:
            raise SystemExit(f"{path}: missing MUSA compute hook anchor")
        text = text.replace(anchor, replacement)
    anchor = """    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {"""
    replacement = """#ifdef GGML_USE_MUSA
    if (ggml_musa_ext_supports && ggml_musa_ext_supports(op)) return true;
    if (op->op == GGML_OP_CUSTOM) return false;
#endif
""" + anchor
    if replacement not in text:
        if text.count(anchor) != 1:
            raise SystemExit(f"{path}: missing MUSA supports hook anchor")
        text = text.replace(anchor, replacement)
    if text != original:
        path.write_text(text)
        print(f"MUSA: patched extension hooks in {path}")

    # Tensor-core F16 inputs with F32 accumulation/output: avoids rounding each
    # GEMM result to F16 and the subsequent F16->F32 conversion kernel.
    original = text = path.read_text()
    anchor = "        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);"
    replacement = """#ifdef GGML_USE_MUSA
        prefer_f32_output = true;
#else
""" + anchor + "\n#endif"
    if replacement not in text:
        if text.count(anchor) != 1:
            raise SystemExit(f"{path}: missing MUSA F16 accumulation anchor")
        text = text.replace(anchor, replacement)
    if text != original:
        path.write_text(text)
        print(f"MUSA: F16 GEMMs use F32 accumulation/output in {path}")

    original = text = path.read_text()
    anchor = """        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);"""
    replacement = """#ifdef GGML_USE_MUSA
        if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32) return false;
#else
""" + anchor + "\n#endif"
    if text.count(replacement) != 2:
        if text.count(anchor) != 2:
            raise SystemExit(f"{path}: missing MUSA RMS fusion guard anchors")
        text = text.replace(anchor, replacement)
    if text != original:
        path.write_text(text)
        print(f"MUSA: F16 RMSNorm bypasses F32-only fusion in {path}")

    # Host compilation has no __MUSA_ARCH__. Do not reject kernels compiled for mp_22.
    path = root / "ggml/src/ggml-cuda/fattn.cu"
    original = text = path.read_text()
    anchor = "#ifndef FLASH_ATTN_AVAILABLE\n    GGML_UNUSED(device); GGML_UNUSED(dst);"
    replacement = """#if !defined(FLASH_ATTN_AVAILABLE) && !(defined(GGML_USE_MUSA) && !defined(__MUSA_ARCH__) && !defined(GGML_CUDA_NO_FA))
    GGML_UNUSED(device); GGML_UNUSED(dst);"""
    if replacement not in text:
        if text.count(anchor) != 1:
            raise SystemExit(f"{path}: missing MUSA host FA dispatch anchor")
        text = text.replace(anchor, replacement)
    if text != original:
        path.write_text(text)
        print(f"MUSA: fixed host FA capability check in {path}")

    # Preserve the pre-mp22 rejection when the host compiler has no arch macro.
    original = text = path.read_text()
    anchor = "    const ggml_tensor * KQV   = dst;"
    replacement = """#ifdef GGML_USE_MUSA
    if (ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_QY2) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif
""" + anchor
    if replacement not in text:
        if text.count(anchor) != 1:
            raise SystemExit(f"{path}: missing MUSA runtime FA architecture anchor")
        text = text.replace(anchor, replacement)
    if text != original:
        path.write_text(text)
        print(f"MUSA: retained pre-mp22 FA rejection in {path}")

    # Uniform head strides avoid device pointer-array construction for MQA/GQA.
    path = root / "ggml/src/ggml-cuda/ggml-cuda.cu"
    original = text = path.read_text()
    anchor = "    } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {"
    replacement = """#ifdef GGML_USE_MUSA
    } else if (ne13 == 1 && ne03 == 1 && (ne02 == 1 || r2 == 1)) {
        const int64_t stride_a = ne02 == 1 ? 0 : s02;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, s01, stride_a,
                       src1_ptr, cu_data_type_b, s11, s12,
                beta, dst_ptr, cu_data_type, ne0, ne1*ne0,
                ne12, cu_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
""" + anchor
    if replacement not in text:
        if text.count(anchor) != 1:
            raise SystemExit(f"{path}: missing MUSA strided batch anchor")
        text = text.replace(anchor, replacement)
    if text != original:
        path.write_text(text)
        print(f"MUSA: strided broadcast GEMM in {path}")

    # Avoid muBLAS host-scalar copies and stream sync for small F32 projections.
    path = root / "ggml/src/ggml-cuda/ggml-cuda.cu"
    original = text = path.read_text()
    anchor = "static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {"
    helper = """#ifdef GGML_USE_MUSA
static bool ggml_musa_small_f32_matmul(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * dst) {
    return a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           a->ne[2] == 1 && a->ne[3] == 1 && b->ne[2] == 1 && b->ne[3] == 1 &&
           (a->ne[0] <= 32 || a->ne[1] <= 32) && a->ne[0] % 2 == 0 &&
           ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(dst);
}
#endif

"""
    replacement = helper + anchor + """
#ifdef GGML_USE_MUSA
    if (ggml_musa_small_f32_matmul(src0, src1, dst)) {
        // Reuse the F32 vector kernel in batches of at most eight columns.
        for (int64_t col = 0; col < src1->ne[1]; col += MMVF_MAX_BATCH_SIZE) {
            ggml_tensor b = *src1, d = *dst;
            b.ne[1] = d.ne[1] = std::min<int64_t>(MMVF_MAX_BATCH_SIZE, src1->ne[1] - col);
            b.data = (char *) src1->data + col * src1->nb[1];
            d.data = (char *) dst->data + col * dst->nb[1];
            b.nb[2] = b.nb[3] = b.nb[1] * b.ne[1];
            d.nb[2] = d.nb[3] = d.nb[1] * d.ne[1];
            ggml_cuda_mul_mat_vec_f(ctx, src0, &b, nullptr, &d);
        }
        return;
    }
#endif"""
    if replacement not in text:
        if text.count(anchor) != 1:
            raise SystemExit(f"{path}: missing MUSA F32 projection anchor")
        text = text.replace(anchor, replacement)

    # Upgrade an existing 47a75fb dependency checkout, not just a fresh fetch.
    previous_guard = """#ifdef GGML_USE_MUSA
        if (node->op == GGML_OP_FLASH_ATTN_EXT ||
            (node->op == GGML_OP_MUL_MAT && node->src[0]->type != GGML_TYPE_F16)) {
            return false;
        }
#endif
"""
    text = text.replace(previous_guard, "")
    old_guard = """#ifdef GGML_USE_MUSA
        if (node->op == GGML_OP_MUL_MAT && node->src[0]->type == GGML_TYPE_F32 &&
            !ggml_musa_small_f32_matmul(node->src[0], node->src[1], node)) {
            return false;
        }
#endif
"""
    text = text.replace(old_guard, "")
    anchor = "        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]\n"
    replacement = """#ifdef GGML_USE_MUSA
        if (node->op == GGML_OP_MUL_MAT && node->src[0]->type == GGML_TYPE_F32 &&
            !ggml_musa_small_f32_matmul(node->src[0], node->src[1], node) &&
            !(ggml_musa_ext_supports && ggml_musa_ext_supports(node))) {
            return false;
        }
#endif
""" + anchor
    if replacement not in text:
        if text.count(anchor) != 1:
            raise SystemExit(f"{path}: missing MUSA graph compatibility anchor")
        text = text.replace(anchor, replacement)

    # Source tensors can reuse addresses across graphs while their dtype changes.
    anchor = "                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;"
    replacement = anchor + "\n                prop.node_src_types[j] = cgraph->nodes[i]->src[j]->type;"
    if replacement not in text:
        if text.count(anchor) != 1:
            raise SystemExit(f"{path}: missing graph source type anchor")
        text = text.replace(anchor, replacement)

    # MUSA corrupts some captured graphs if a fresh executable updates itself.
    anchor = """        }
        if (cuda_graph_update_required) { // Update graph executable"""
    replacement = """        } else if (cuda_graph_update_required) { // Update graph executable"""
    if replacement not in text:
        if text.count(anchor) != 1:
            raise SystemExit(f"{path}: missing graph initial update anchor")
        text = text.replace(anchor, replacement)
    if text != original:
        path.write_text(text)
        print(f"MUSA: graph-safe F32 projections and executable initialization in {path}")

    path = root / "ggml/src/ggml-cuda/common.cuh"
    original = text = path.read_text()
    anchor = "        void *   node_src_data_ptrs[GGML_MAX_SRC];"
    replacement = anchor + "\n        ggml_type node_src_types[GGML_MAX_SRC];"
    if replacement not in text:
        if text.count(anchor) != 1:
            raise SystemExit(f"{path}: missing graph source types declaration")
        text = text.replace(anchor, replacement)
    if text != original:
        path.write_text(text)
        print(f"MUSA: graph cache tracks source dtypes in {path}")


    path = root / "ggml/src/ggml-cuda/ggml-cuda.cu"
    original = text = path.read_text()
    anchor = "static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {"
    diagnostic = r"""#ifdef GGML_USE_MUSA
// Diagnostic timing deliberately disables graph execution and synchronizes each op.
struct vla_musa_op_timer {
    const ggml_tensor * t;
    cudaStream_t stream;
    cudaEvent_t begin = nullptr, end = nullptr;
    vla_musa_op_timer(const ggml_tensor * tensor, cudaStream_t s): t(tensor), stream(s) {
        static const bool enabled = std::getenv("VLA_MUSA_PROFILE_OPS") != nullptr;
        if (!enabled) return;
        GGML_ASSERT(std::getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);
        CUDA_CHECK(musaEventCreate(&begin)); CUDA_CHECK(musaEventCreate(&end));
        CUDA_CHECK(cudaEventRecord(begin, stream));
    }
    ~vla_musa_op_timer() {
        if (!begin) return;
        CUDA_CHECK(cudaEventRecord(end, stream)); CUDA_CHECK(cudaEventSynchronize(end));
        float ms; CUDA_CHECK(musaEventElapsedTime(&ms, begin, end));
        std::fprintf(stderr, "VLA_MUSA_OP {\"op\":\"%s\",\"ne\":[%lld,%lld,%lld,%lld],\"ms\":%.6f}\n",
            ggml_op_name(t->op), (long long)t->ne[0], (long long)t->ne[1],
            (long long)t->ne[2], (long long)t->ne[3], ms);
        CUDA_CHECK(cudaEventDestroy(begin)); CUDA_CHECK(cudaEventDestroy(end));
    }
};
#endif

"""
    replacement = diagnostic + anchor + """
#ifdef GGML_USE_MUSA
    vla_musa_op_timer vla_timer(dst, ctx.stream());
#endif"""
    if replacement not in text:
        if text.count(anchor) != 1:
            raise SystemExit(f"{path}: missing diagnostic timer anchor")
        text = text.replace(anchor, replacement)
    if text != original:
        path.write_text(text)
        print(f"MUSA: optional device-event op diagnostics in {path}")


    path = root / "ggml/src/ggml-cuda/ggml-cuda.cu"
    original = text = path.read_text()
    anchor = '__attribute__((visibility("default"))) bool (*ggml_musa_ext_supports)(const ggml_tensor *) = nullptr;'
    replacement = anchor + '\n__attribute__((visibility("default"))) void (*ggml_musa_ext_release)(int, void *) = nullptr;'
    if replacement not in text:
        if text.count(anchor) != 1:
            raise SystemExit(f"{path}: missing MUSA release declaration anchor")
        text = text.replace(anchor, replacement)
    anchor = """static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
"""
    replacement = anchor + """
#ifdef GGML_USE_MUSA
    if (ggml_musa_ext_release) {
        for (int d=0; d<GGML_CUDA_MAX_DEVICES; ++d)
            for (int s=0; s<GGML_CUDA_MAX_STREAMS; ++s)
                if (cuda_ctx->streams[d][s]) {
                    ggml_cuda_set_device(d);
                    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->streams[d][s]));
                    ggml_musa_ext_release(d, (void *)cuda_ctx->streams[d][s]);
                }
    }
#endif
"""
    if replacement not in text:
        if text.count(anchor) != 1:
            raise SystemExit(f"{path}: missing MUSA release callback anchor")
        text = text.replace(anchor, replacement)
    if text != original:
        path.write_text(text)
        print(f"MUSA: release extension resources with backend streams in {path}")


    # Mapped input remains a genuine CUDA/MUSA buffer, with explicit host ownership.
    path = root / "ggml/src/ggml-cuda/ggml-cuda.cu"
    text = path.read_text()
    anchor = "    void * dev_ptr = nullptr;\n    std::string name;"
    replacement = "    void * dev_ptr = nullptr;\n#ifdef GGML_USE_MUSA\n    void * mapped_host = nullptr;\n#endif\n    std::string name;"
    if replacement not in text:
        assert text.count(anchor) == 1
        text = text.replace(anchor, replacement)
    anchor = "    ~ggml_backend_cuda_buffer_context() {\n        CUDA_CHECK(cudaFree(dev_ptr));"
    replacement = "    ~ggml_backend_cuda_buffer_context() {\n#ifdef GGML_USE_MUSA\n        if (mapped_host) { CUDA_CHECK(cudaFreeHost(mapped_host)); return; }\n#endif\n        CUDA_CHECK(cudaFree(dev_ptr));"
    if replacement not in text:
        assert text.count(anchor) == 1
        text = text.replace(anchor, replacement)
    anchor = "static void ggml_backend_cuda_set_tensor_async("
    addition = """#ifdef GGML_USE_MUSA
extern "C" GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_musa_mapped_buffer(
        ggml_backend_t backend, size_t size, void ** host) {
    auto * backend_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(backend_ctx->device);
    void * ptr = nullptr, * dev = nullptr;
    auto err = musaHostAlloc(&ptr, size, musaHostAllocMapped);
    if (err == musaSuccess) err = musaHostGetDevicePointer(&dev, ptr, 0);
    if (err != musaSuccess) {
        if (ptr) musaFreeHost(ptr);
        GGML_LOG_ERROR("%s: %s\\n", __func__, musaGetErrorString(err));
        return nullptr;
    }
    auto * ctx = new ggml_backend_cuda_buffer_context(backend_ctx->device, dev);
    ctx->mapped_host = ptr; *host = ptr;
    return ggml_backend_buffer_init(ggml_backend_cuda_buffer_type(backend_ctx->device),
                                   ggml_backend_cuda_buffer_interface, ctx, size);
}
#endif

"""
    if 'ggml_backend_musa_mapped_buffer(' not in text:
        assert text.count(anchor) == 1
        text = text.replace(anchor, addition + anchor)
    path.write_text(text)


    # Batch flattening is an independent opt-in, after dtype/stride normalization.
    path = root / 'ggml/src/ggml-cuda/ggml-cuda.cu'
    text = path.read_text()
    anchor = '    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.'
    old_addition = '#ifdef GGML_USE_MUSA\n    // vla.cpp: shared weights and consecutive batches can form one wider GEMM.\n    static const bool flatten_shared = std::getenv("VLA_MUSA_FLATTEN_GEMM") != nullptr;\n    if (flatten_shared && compute_type == GGML_TYPE_F16 && cu_data_type == CUDA_R_32F &&\n        ne02 == 1 && ne03 == 1 && ne12 > 1 && ne13 == 1 &&\n        s12 == s11*ne11 && ne11 <= INT32_MAX/ne12) {\n        CUBLAS_CHECK(cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,\n                ne01, ne11*ne12, ne10,\n                alpha, src0_ptr, cu_data_type_a, s01,\n                       src1_ptr, cu_data_type_b, s11,\n                beta, dst_ptr, cu_data_type, ne0,\n                cu_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));\n        return;\n    }\n#endif\n'
    addition = '#ifdef GGML_USE_MUSA\n    // vla.cpp: shared weights and consecutive batches can form one wider GEMM.\n    static const bool flatten_shared = std::getenv("VLA_MUSA_FLATTEN_GEMM") != nullptr;\n    static const bool trace_gemm = std::getenv("VLA_MUSA_TRACE_GEMM") != nullptr;\n    if (trace_gemm) std::fprintf(stderr,\n        "VLA_MUSA_GEMM {\\"m\\":%lld,\\"n\\":%lld,\\"k\\":%lld,\\"batch\\":%lld,"\n        "\\"a_batch\\":%lld,\\"b_outer\\":%lld,\\"compute_type\\":%d,\\"out_type\\":%d,"\n        "\\"lda\\":%lld,\\"ldb\\":%lld,\\"stride_b\\":%lld,\\"flatten_requested\\":%d}\\n",\n        (long long)ne01,(long long)ne11,(long long)ne10,(long long)ne12,\n        (long long)ne02,(long long)ne13,(int)compute_type,(int)cu_data_type,\n        (long long)s01,(long long)s11,(long long)s12,(int)flatten_shared);\n    if (flatten_shared && compute_type == GGML_TYPE_F16 && cu_data_type == CUDA_R_32F &&\n        ne02 == 1 && ne03 == 1 && ne12 > 1 && ne13 == 1 &&\n        s12 == s11*ne11 && ne11 <= INT32_MAX/ne12) {\n        CUBLAS_CHECK(cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,\n                ne01, ne11*ne12, ne10,\n                alpha, src0_ptr, cu_data_type_a, s01,\n                       src1_ptr, cu_data_type_b, s11,\n                beta, dst_ptr, cu_data_type, ne0,\n                cu_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));\n        return;\n    }\n#endif\n'
    if addition not in text:
        if old_addition in text:
            path.write_text(text.replace(old_addition, addition))
        else:
            assert text.count(anchor) == 1
            path.write_text(text.replace(anchor, addition + anchor))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_ggml_musa.py <llama-source-dir>")
    patch(Path(sys.argv[1]))
