# 在仓库根目录 source 本文件，参数为 pruned 或 full。
# 保留完整开关组合；不是仅打开本轮三个选项。
case "${1:-}" in
  pruned)
    export VLA_MODEL=/home/skysi/VLA/models/pi05_cube_merge_cksize50_step3_0929/gguf/model.gguf
    export VLA_MUSA_GEMMA_ATTN=fused-denoise
    ;;
  full)
    export VLA_MODEL=/home/skysi/VLA/models/cup_cube_joint_061_a800_50k_20261022/gguf/model.gguf
    export VLA_MUSA_GEMMA_ATTN=fused-all
    ;;
  *)
    printf '%s\n' '用法：source 推荐环境.sh pruned|full' >&2
    return 1
    ;;
esac
export VLA_LIBRARY=/home/skysi/code/vla.cpp/build-musa-official/libvla.so
export VLA_DEVICE=0
export VLA_N_THREADS=4
unset VLA_MUSA_FLATTEN_GEMM VLA_MUSA_TRACE_GEMM
export VLA_MUSA_VISION_SDPA=1
export VLA_PI05_VISION_QKV=1
export VLA_MUSA_VISION_EPILOGUE=all
export VLA_MUSA_VISION_PACK=1
export VLA_MUSA_ATTN_DIRECT_OUT=1
export VLA_PI05_HOST_INPUT=pinned
export VLA_PI05_ASYNC_VISION=1
export VLA_PI05_VISION_DIRECT=1
export VLA_PI05_KV_REUSE=1
export VLA_PI05_PIN_MAIN_INPUT=1
export VLA_PI05_KV_DIRECT=1
export VLA_MUSA_GEGLU_F16=1
export VLA_PI05_RES_ADARMS=1
