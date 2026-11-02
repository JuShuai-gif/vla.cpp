#pragma once

#include "ggml.h"
#include <algorithm>
#include <cmath>
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

}
