// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ggml-backend.h"
#ifdef GGML_USE_MUSA
extern "C" GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_musa_mapped_buffer(
    ggml_backend_t backend, size_t size, void ** host);
#endif

struct ggml_tensor;

namespace vla {
#ifdef GGML_USE_MUSA
bool musa_register_fused_ops();
bool musa_vision_pack_enabled();
bool musa_vision_raw_qkv(const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v);
void musa_vision_epilogue(ggml_tensor * t, bool gelu, void * stream);
#else
inline bool musa_register_fused_ops() { return false; }
#endif
}
