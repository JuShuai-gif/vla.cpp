#pragma once

#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace vla {

// Attention is [key, query, head]. Match VisPruner: mean(head), then mean(query).
inline ggml_tensor * vision_patch_scores(ggml_context * C, ggml_tensor * attention) {
    GGML_ASSERT(attention->ne[3] == 1);
    const int64_t keys = attention->ne[0], queries = attention->ne[1];
    auto * heads_first = ggml_cont(C, ggml_permute(C, attention, 1, 2, 0, 3));
    auto * per_query = ggml_reshape_2d(C, ggml_mean(C, heads_first), keys, queries);
    return ggml_reshape_1d(C, ggml_mean(C, ggml_cont(C, ggml_transpose(C, per_query))), keys);
}

inline std::vector<int> vision_keep_indices(const std::vector<float> & scores, int keep) {
    if (keep <= 0 || keep > (int)scores.size() ||
        !std::all_of(scores.begin(), scores.end(), [](float x) { return std::isfinite(x); }))
        throw std::invalid_argument("invalid visual token scores/count");
    std::vector<int> ids(scores.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + keep, ids.end(), [&](int a, int b) {
        return scores[a] == scores[b] ? a < b : scores[a] > scores[b];
    });
    ids.resize(keep);
    std::sort(ids.begin(), ids.end());
    return ids;
}

// Stable top-k, returned in original spatial order. GPU uses the same score/id ordering.
constexpr uint32_t VISION_KEEP_MAGIC = 0x564b4944u;
inline void vision_keep_cpu(ggml_tensor * dst, int ith, int, void *) {
    if (ith != 0) return;
    const auto * scores = static_cast<const float *>(dst->src[0]->data);
    const auto ids = vision_keep_indices({scores, scores + dst->src[0]->ne[0]}, (int)dst->ne[0]);
    std::memcpy(dst->data, ids.data(), ids.size()*sizeof(int32_t));
}
inline ggml_tensor * vision_keep_rows(ggml_context * C, ggml_tensor * emb, ggml_tensor * scores, int keep) {
    GGML_ASSERT(scores->type == GGML_TYPE_F32 && ggml_is_contiguous(scores));
    GGML_ASSERT(keep > 0 && keep <= scores->ne[0] && emb->ne[1] == scores->ne[0]);
    static const uint32_t spec = VISION_KEEP_MAGIC;
    ggml_tensor * args[] = {scores};
    auto * ids = ggml_custom_4d(C, GGML_TYPE_I32, keep, 1, 1, 1, args, 1,
                                vision_keep_cpu, 1, (void *)&spec);
    return ggml_get_rows(C, emb, ids);
}

}
