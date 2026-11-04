# vla.cpp tests, enabled with -DVLA_BUILD_TESTS=ON. predict_check needs a real
# GGUF so it is built but not run by ctest; the unit tests below are pure.

# Beside the ggml DLLs in <build>/bin; Windows has no rpath (see ../CMakeLists.txt).
if(WIN32)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()
if(VLA_WERROR)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Werror>)
endif()

add_executable(vla_predict_check predict_check.cpp)
target_link_libraries(vla_predict_check PRIVATE vla_core)
target_compile_options(vla_predict_check PRIVATE -Wall -Wextra)

add_executable(test_vision_common test_vision_common.cpp)
target_include_directories(test_vision_common PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(test_vision_common PRIVATE -Wall -Wextra)
add_test(NAME vision_common COMMAND test_vision_common)

add_executable(test_vision_prune test_vision_prune.cpp)
target_include_directories(test_vision_prune PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_vision_prune PRIVATE ggml)
target_compile_options(test_vision_prune PRIVATE -Wall -Wextra)
add_test(NAME vision_prune COMMAND test_vision_prune)

add_executable(test_rope_conventions test_rope_conventions.cpp)
target_include_directories(test_rope_conventions PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_rope_conventions PRIVATE ggml)
target_compile_options(test_rope_conventions PRIVATE -Wall -Wextra)
add_test(NAME rope_conventions COMMAND test_rope_conventions)

# Built as C so the public header stays C-clean.
add_executable(test_c_api test_c_api.c)
target_link_libraries(test_c_api PRIVATE vla)
target_compile_options(test_c_api PRIVATE -Wall -Wextra)
add_test(NAME c_api COMMAND test_c_api)

add_executable(test_dit_common test_dit_common.cpp)
target_include_directories(test_dit_common PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_dit_common PRIVATE ggml)
target_compile_options(test_dit_common PRIVATE -Wall -Wextra)
add_test(NAME dit_common COMMAND test_dit_common)

# Defines GGML_USE_OPENVINO itself so the pass is exercised on any build.
add_executable(test_graph_names test_graph_names.cpp)
target_include_directories(test_graph_names PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_graph_names PRIVATE ggml)
target_compile_options(test_graph_names PRIVATE -Wall -Wextra)
add_test(NAME graph_names COMMAND test_graph_names)

add_executable(test_qwen3vl_vit test_qwen3vl_vit.cpp)
target_include_directories(test_qwen3vl_vit PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_qwen3vl_vit PRIVATE ggml)
target_compile_options(test_qwen3vl_vit PRIVATE -Wall -Wextra)
add_test(NAME qwen3vl_vit COMMAND test_qwen3vl_vit)

# Links vla_core: it calls the real config_is_sane rather than a copy.
add_executable(test_config_guard test_config_guard.cpp)
target_include_directories(test_config_guard PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_config_guard PRIVATE vla_core)
target_compile_options(test_config_guard PRIVATE -Wall -Wextra)
add_test(NAME config_guard COMMAND test_config_guard)

add_executable(test_prompt test_prompt.cpp)
target_link_libraries(test_prompt PRIVATE vla_core)
target_compile_options(test_prompt PRIVATE -Wall -Wextra)
add_test(NAME prompt COMMAND test_prompt)

# FoldQuant CPU reference ops as GGML_OP_CUSTOM nodes on the CPU backend. The
# golden checksum is shared with tests/py/test_foldquant_ref.py.
add_executable(test_foldquant_cpu_op test_foldquant_cpu_op.cpp)
target_include_directories(test_foldquant_cpu_op PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_foldquant_cpu_op PRIVATE vla_core)
target_compile_options(test_foldquant_cpu_op PRIVATE -Wall -Wextra -ffp-contract=off)
add_test(NAME foldquant_cpu_op COMMAND test_foldquant_cpu_op)

# FoldQuant sites read back as float weights (the path every backend without
# the integer kernels takes) against the dense rotation they stand for.
add_executable(test_foldquant_dequant test_foldquant_dequant.cpp)
target_include_directories(test_foldquant_dequant PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_foldquant_dequant PRIVATE vla_core)
target_compile_options(test_foldquant_dequant PRIVATE -Wall -Wextra)
add_test(NAME foldquant_dequant COMMAND test_foldquant_dequant)

# Real MUSA execution against CPU references, without a model download.
if(GGML_MUSA)
    add_executable(test_musa_ops test_musa_ops.cpp)
    target_link_libraries(test_musa_ops PRIVATE vla_core)
    target_compile_options(test_musa_ops PRIVATE -Wall -Wextra)
    add_test(NAME musa_ops COMMAND test_musa_ops)
    add_test(NAME musa_flatten_gemm COMMAND test_musa_ops)
    set_tests_properties(musa_flatten_gemm PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180
        ENVIRONMENT "VLA_MUSA_FLATTEN_GEMM=1")
    add_test(NAME musa_f32_attention COMMAND test_musa_ops --f32-attention)
    add_test(NAME musa_strided_pack COMMAND test_musa_ops --strided-pack)
    add_test(NAME musa_vision_epilogue COMMAND test_musa_ops --vision-epilogue)
    add_test(NAME musa_mapped_io COMMAND test_musa_ops --mapped-io)
    set_tests_properties(musa_mapped_io PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
    set_tests_properties(musa_strided_pack PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
    if(VLA_MUSA_MUDNN_ATTN)
        add_test(NAME musa_vision_pack COMMAND test_musa_ops --vision-pack)
        set_tests_properties(musa_vision_pack PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180
            ENVIRONMENT "VLA_MUSA_VISION_SDPA=1;VLA_MUSA_VISION_PACK=1")
        add_test(NAME musa_vision_sdpa COMMAND test_musa_ops --vision-sdpa)
        set_tests_properties(musa_vision_sdpa PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180
            ENVIRONMENT "VLA_MUSA_VISION_SDPA=1")
    endif()

    set_tests_properties(musa_f32_attention PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
    set_tests_properties(musa_ops PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
    add_test(NAME musa_f32_patch COMMAND test_musa_ops --f32-patch)
    set_tests_properties(musa_f32_patch PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)

    if(GGML_CUDA_FA)
        add_test(NAME musa_attention256_direct COMMAND test_musa_ops --attention256)
        set_tests_properties(musa_attention256_direct PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180
            ENVIRONMENT "VLA_MUSA_GEMMA_ATTN=fused-all;VLA_MUSA_ATTN_DIRECT_OUT=1")
        add_test(NAME musa_attention256 COMMAND test_musa_ops --attention256)
        set_tests_properties(musa_attention256 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180
            ENVIRONMENT "VLA_MUSA_GEMMA_ATTN=fused-all")
        add_test(NAME musa_small_gemm COMMAND test_musa_ops --flash-attn)
        set_tests_properties(musa_small_gemm PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180
            ENVIRONMENT "VLA_MUSA_SMALL_GEMM=auto;VLA_MUSA_GEGLU_F16=1;VLA_MUSA_GEMMA_ATTN=fused-all;VLA_MUSA_GEMMA_FLAT_QK=all;VLA_MUSA_GEMMA_FLAT_AV=all")
        add_test(NAME musa_rope_table COMMAND test_musa_ops --flash-attn)
        set_tests_properties(musa_rope_table PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180
            ENVIRONMENT "VLA_PI05_ROPE_TABLE=1;VLA_PI05_K_ROPE_DIRECT=1;VLA_PI05_KV_DIRECT=1;VLA_MUSA_GEMMA_ATTN=fused-all")
        add_test(NAME musa_k_rope_direct COMMAND test_musa_ops --flash-attn)
        set_tests_properties(musa_k_rope_direct PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180
            ENVIRONMENT "VLA_PI05_K_ROPE_DIRECT=1;VLA_PI05_KV_DIRECT=1;VLA_MUSA_GEMMA_ATTN=fused-all")
        add_test(NAME musa_kv_direct COMMAND test_musa_ops --flash-attn)
        set_tests_properties(musa_kv_direct PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180
            ENVIRONMENT "VLA_PI05_KV_DIRECT=1;VLA_MUSA_GEMMA_ATTN=fused-all")
        add_test(NAME musa_flash_attn COMMAND test_musa_ops --flash-attn)
        set_tests_properties(musa_flash_attn PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
    endif()
    add_test(NAME musa_cpu_selection COMMAND test_musa_ops --expect-cpu)
    set_tests_properties(musa_cpu_selection PROPERTIES ENVIRONMENT "VLA_DEVICE=cpu")
    add_test(NAME musa_invalid_device COMMAND test_musa_ops --expect-cpu)
    set_tests_properties(musa_invalid_device PROPERTIES ENVIRONMENT "VLA_DEVICE=1024")
endif()

# The same site through ggml's SYCL backend (src/sycl/vla_sycl_foldquant.cpp),
# bit-identical to the CPU reference.
if(GGML_SYCL)
    add_executable(test_foldquant_sycl_op test_foldquant_sycl_op.cpp)
    target_include_directories(test_foldquant_sycl_op PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_foldquant_sycl_op PRIVATE vla_core)
    # It runs the CPU reference inline (no contraction) and, without a GPU,
    # drives the kernels on a SYCL queue of its own.
    target_compile_options(test_foldquant_sycl_op PRIVATE -fsycl -Wall -Wextra -ffp-contract=off)
    add_test(NAME foldquant_sycl_op COMMAND test_foldquant_sycl_op)
    # sycl-tla flash attention vs a host reference; needs an Intel GPU, so not a ctest.
    if(VLA_SYCL_TLA_DIR)
        add_executable(fmha_sycl_check fmha_sycl_check.cpp)
        target_include_directories(fmha_sycl_check PRIVATE ${CMAKE_SOURCE_DIR}/src)
        target_link_libraries(fmha_sycl_check PRIVATE vla_core)
        target_compile_options(fmha_sycl_check PRIVATE -fsycl -Wall -Wextra)
    endif()
endif()

# The same site through ggml's OpenVINO backend (src/openvino/foldquant_ov.cpp).
if(GGML_OPENVINO)
    add_executable(test_foldquant_ov_op test_foldquant_ov_op.cpp)
    target_include_directories(test_foldquant_ov_op PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_foldquant_ov_op PRIVATE vla_core)
    target_compile_options(test_foldquant_ov_op PRIVATE -Wall -Wextra)
    add_test(NAME foldquant_ov_op COMMAND test_foldquant_ov_op)
endif()

# A/B harness for the two BitVLA ternary-GEMM tilings. Built so it cannot rot,
# not registered with ctest: it needs a GPU and is read by hand.
# VLA_BITVLA_NARROW_GEMM=1 selects the old one-tile-per-CTA kernel at runtime.
if(GGML_CUDA)
    add_executable(bitvla_gemm_check bitvla_gemm_check.cu)
    target_include_directories(bitvla_gemm_check PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(bitvla_gemm_check PRIVATE bitvla_cuda_kernels)
endif()

# FoldQuant kernel micro-benchmark at GR00T shapes against cuBLAS BF16. Built
# so it cannot rot, not registered with ctest: it needs a GPU and is read by hand.
if(GGML_CUDA)
    add_executable(foldquant_gemm_check foldquant_gemm_check.cu)
    target_include_directories(foldquant_gemm_check PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(foldquant_gemm_check PRIVATE vla_fq_kernels CUDA::cublas)
endif()

# FoldQuant CUDA kernels versus the CPU reference, byte-identical; run twice,
# the second time through the VLA_FQ_CPU_REF=1 staging shim. Skips (exit 0)
# without a CUDA device.
if(GGML_CUDA)
    add_executable(test_foldquant_cuda_op test_foldquant_cuda_op.cpp)
    target_include_directories(test_foldquant_cuda_op PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_foldquant_cuda_op PRIVATE vla_core vla_cuda_ops ggml)
    target_compile_options(test_foldquant_cuda_op PRIVATE -Wall -Wextra -ffp-contract=off)
    add_test(NAME foldquant_cuda_op COMMAND test_foldquant_cuda_op)
    add_test(NAME foldquant_cuda_op_ref COMMAND test_foldquant_cuda_op)
    set_tests_properties(foldquant_cuda_op_ref PROPERTIES ENVIRONMENT "VLA_FQ_CPU_REF=1;GGML_CUDA_DISABLE_GRAPHS=1")
endif()

# The fused CUDA nodes (layers/fused_ops.h) against the ggml chains they
# replace, bit for bit. Skips itself (exit 0) when no CUDA device is present.
if(GGML_CUDA)
    add_executable(test_fused_cuda_ops test_fused_cuda_ops.cpp)
    target_include_directories(test_fused_cuda_ops PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_fused_cuda_ops PRIVATE vla_core vla_cuda_ops ggml)
    target_compile_options(test_fused_cuda_ops PRIVATE -Wall -Wextra)
    add_test(NAME fused_cuda_ops COMMAND test_fused_cuda_ops)
endif()

# Regression test for the in-tree BF16 CUDA kernels and the ggml hook they ride
# on. Skips itself (exit 0) when no CUDA device is present.
if(GGML_CUDA)
    add_executable(test_bf16_cuda_ops test_bf16_cuda_ops.cpp)
    target_include_directories(test_bf16_cuda_ops PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_bf16_cuda_ops PRIVATE vla_core vla_cuda_ops ggml)
    target_compile_options(test_bf16_cuda_ops PRIVATE -Wall -Wextra)
    add_test(NAME bf16_cuda_ops COMMAND test_bf16_cuda_ops)
    add_test(NAME bf16_cuda_ops_flat COMMAND test_bf16_cuda_ops)
    set_tests_properties(bf16_cuda_ops_flat PROPERTIES ENVIRONMENT VLA_BF16_FLAT=1)
endif()

# The same for the SYCL BF16 ops. Skips when no SYCL device is present.
if(GGML_SYCL AND NOT GGML_CUDA)
    add_executable(test_bf16_sycl_ops test_bf16_sycl_ops.cpp)
    target_include_directories(test_bf16_sycl_ops PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_bf16_sycl_ops PRIVATE vla_sycl_ops ggml)
    target_compile_options(test_bf16_sycl_ops PRIVATE -fsycl -Wall -Wextra)
    target_link_options(test_bf16_sycl_ops PRIVATE -fsycl)
    add_test(NAME bf16_sycl_ops COMMAND test_bf16_sycl_ops)
endif()

if(GGML_MUSA)
    add_test(NAME musa_packed_norm COMMAND test_musa_ops --packed-norm)
    add_test(NAME musa_prune_score COMMAND test_musa_ops --prune-score)
    set_tests_properties(musa_packed_norm musa_prune_score PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
endif()


add_executable(test_weight_loading test_weight_loading.cpp)
target_include_directories(test_weight_loading PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_weight_loading PRIVATE vla_core)
add_test(NAME weight_loading COMMAND test_weight_loading)
set_tests_properties(weight_loading PROPERTIES TIMEOUT 180)

if(GGML_MUSA)
    add_executable(bench_musa_quant bench_musa_quant.cpp)
    target_link_libraries(bench_musa_quant PRIVATE vla_core)
    target_compile_options(bench_musa_quant PRIVATE -Wall -Wextra)
    add_test(NAME musa_quant_q8 COMMAND bench_musa_quant)
    foreach(qtype IN ITEMS q8_0 q4_0 q5_0)
        add_test(NAME musa_quant_blas_${qtype} COMMAND bench_musa_quant 256 64 50 5 ${qtype})
        set_tests_properties(musa_quant_blas_${qtype} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180
            ENVIRONMENT "GGML_MUSA_QUANT_ROUTE=blas;GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16")
    endforeach()
    set_tests_properties(musa_quant_q8 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
endif()

if(GGML_MUSA)
    add_test(NAME musa_quant_q4 COMMAND bench_musa_quant 256 64 50 5 q4_0)
    add_test(NAME musa_quant_q5 COMMAND bench_musa_quant 256 64 50 5 q5_0)
    set_tests_properties(musa_quant_q4 musa_quant_q5 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
    add_executable(test_quant_calibration test_quant_calibration.cpp)
    target_include_directories(test_quant_calibration PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_quant_calibration PRIVATE vla_core)
    add_test(NAME quant_calibration COMMAND test_quant_calibration ${CMAKE_CURRENT_BINARY_DIR}/quant-calibration-test)
endif()
