// Copyright 2026 VinRobotics
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace vla {
#ifdef GGML_USE_MUSA
bool musa_register_fused_ops();
#else
inline bool musa_register_fused_ops() { return false; }
#endif
}
