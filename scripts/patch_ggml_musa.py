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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_ggml_musa.py <llama-source-dir>")
    patch(Path(sys.argv[1]))
