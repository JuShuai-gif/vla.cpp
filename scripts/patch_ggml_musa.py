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
    if replacement not in text:
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_ggml_musa.py <llama-source-dir>")
    patch(Path(sys.argv[1]))
