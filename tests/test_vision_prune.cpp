#include "modules/vision_prune.h"
#include "ggml-cpu.h"
#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <limits>

int main() {
    auto * ctx = ggml_init({1024 * 1024, nullptr, false});
    assert(ctx);
    auto * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 3, 2);
    auto * data = static_cast<float *>(a->data);
    const float base[] = {0.1f, 0.4f, 0.2f, 0.3f};
    for (int h = 0; h < 2; ++h)
        for (int q = 0; q < 3; ++q)
            for (int k = 0; k < 4; ++k)
                data[(h * 3 + q) * 4 + k] = base[k] + 0.01f * (h * 3 + q);
    auto * scores = vla::vision_patch_scores(ctx, a);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, scores);
    assert(ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS);
    const float * result = static_cast<const float *>(scores->data);
    for (int k = 0; k < 4; ++k) assert(std::fabs(result[k] - base[k] - 0.025f) < 1e-6f);
    assert((vla::vision_keep_indices({result, result + 4}, 2) == std::vector<int>{1, 3}));
    assert((vla::vision_keep_indices({4, 2, 3, 1}, 4) == std::vector<int>{0, 1, 2, 3}));
    assert((vla::vision_keep_indices({1, 1, 1}, 2) == std::vector<int>{0, 1}));
    for (auto bad : {std::vector<float>{}, std::vector<float>{std::numeric_limits<float>::quiet_NaN()}}) {
        bool rejected = false;
        try { (void)vla::vision_keep_indices(bad, 1); } catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected);
    }
    ggml_free(ctx);
    std::puts("test_vision_prune: OK (attention reduction, spatial ordering, invalid input)");
}
