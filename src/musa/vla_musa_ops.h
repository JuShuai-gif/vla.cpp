// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
#pragma once

struct ggml_tensor;

namespace vla {
#ifdef GGML_USE_MUSA
bool musa_register_fused_ops();
void musa_vision_epilogue(ggml_tensor * t, bool gelu, void * stream);
#else
inline bool musa_register_fused_ops() { return false; }
#endif
}
