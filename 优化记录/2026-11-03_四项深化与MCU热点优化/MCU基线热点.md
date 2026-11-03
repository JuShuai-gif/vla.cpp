# MCU与MSYS基线热点

标记包围预热后的一次真实输入推理；Graph OFF以获取独立kernel活动。MCU时间受采集干预，不代替普通Graph ON wall；MSYS存在不同kernel重复时间戳/时长，仅核对调用数量；其累计时间和排序无效，不用于性能结论。


## pruned：1834次kernel，MCU累计480.764ms

|kernel|次数|累计ms|单次均值ms|寄存器|共享字节|
|---|---:|---:|---:|---|---|
|gemmex_nt_512_256x256_f16f32_epilogue_qy2|64|221.299|3.45780|[242.0]|[65676.0]|
|void flash_attn_tile<256, 256, 32, 1, false>(char const*, char const*, char const*, char const*, char const*, int const*, float*, float2*, float, float, float, float, unsigned int, float, int, uint3, int, int, int, int, int, int, int, int, int, int, int, long, int, int, long, int, int, int, int, int, long)|17|51.590|3.03471|[190.0]|[29716.0]|
|void vla::(anonymous namespace)::small_gemm_kernel<32>(__half const*, __half const*, float*, int, int, int)|162|43.441|0.26815|[49.0]|[8724.0]|
|gemmex_nt_256_256x128_f16f32_epilogue_qy2|121|33.740|0.27885|[242.0]|[24716.0]|
|void cpy_scalar<&(void cpy_1_scalar<float, float>(char const*, char*))>(char const*, char*, long, long, long, long, long, long, long, long, long, long, long, long, long, long, long)|7|21.040|3.00571|[39.0]|[20.0]|
|void vla::(anonymous namespace)::merged_geglu<float, __half>(float const*, __half*, int, bool)|71|13.832|0.19481|[16.0]|[20.0]|
|void k_bin_bcast<&(op_add(float, float)), float, float, float, float const*>(float const*, float const*, float*, unsigned int, unsigned int, unsigned int, uint3, uint3, uint3, uint3, uint3, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, float const*)|89|13.415|0.15073|[20.0]|[20.0]|
|void vla::(anonymous namespace)::small_gemm_kernel<64>(__half const*, __half const*, float*, int, int, int)|54|7.552|0.13985|[71.0]|[17428.0]|
|void convert_unary_cont_vec4<float, __half>(void const*, __half*, long)|114|6.141|0.05387|[8.0]|[20.0]|
|void vla::(anonymous namespace)::apply_rope_table<float>(float const*, float const*, float*, int, int, int, unsigned long, unsigned long)|143|5.063|0.03541|[15.0]|[20.0]|
|vla::(anonymous namespace)::pack_f16x4(float const*, __half2*, unsigned long, int, int, int, unsigned long, unsigned long, unsigned long)|126|4.948|0.03927|[27.0]|[20.0]|
|void vla::(anonymous namespace)::vision_epilogue<true>(float const*, float const*, float const*, float*, int)|15|4.606|0.30704|[37.0]|[20.0]|
|gemmex_nn_256_256x128_f16f32_epilogue_qy2|54|4.404|0.08156|[242.0]|[24716.0]|
|void rms_norm_f32<1024, true, false, false>(float const*, float*, int, long, long, long, float, float const*, long, long, long, uint3, uint3, uint3, uint3, float const*, long, long, long, uint3, uint3, uint3, uint3, float)|35|4.066|0.11616|[13.0]|[20.0]|
|void k_bin_bcast<&(op_mul(float, float)), float, float, float, float const*>(float const*, float const*, float*, unsigned int, unsigned int, unsigned int, uint3, uint3, uint3, uint3, uint3, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, float const*)|34|3.942|0.11595|[20.0]|[20.0]|
|musa_asm_hflash_atten_512_256x64x128|14|3.785|0.27038|[244.0]|[57472.0]|
|vla::(anonymous namespace)::gemma_probs(float const*, __half const*, __half*, int, int, float)|54|3.644|0.06747|[25.0]|[532.0]|
|void vla::(anonymous namespace)::res_adarms<float, false>(float const*, float const*, float const*, float const*, float*, int, int, float)|105|3.589|0.03418|[20.0]|[1044.0]|
|void vla::(anonymous namespace)::vision_epilogue<false>(float const*, float const*, float const*, float*, int)|30|3.572|0.11905|[39.0]|[20.0]|
|void vla::(anonymous namespace)::small_pack_kernel<float>(float const*, __half*, int, int)|162|3.567|0.02202|[6.0]|[20.0]|

## full：4584次kernel，MCU累计780.948ms

|kernel|次数|累计ms|单次均值ms|寄存器|共享字节|
|---|---:|---:|---:|---|---|
|gemmex_nt_512_256x256_f16f32_epilogue_qy2|88|307.993|3.49992|[242.0]|[65676.0]|
|void vla::(anonymous namespace)::small_gemm_kernel<32>(__half const*, __half const*, float*, int, int, int)|540|144.143|0.26693|[49.0]|[8724.0]|
|gemmex_nt_256_256x128_f16f32_epilogue_qy2|288|71.134|0.24699|[242.0]|[24716.0]|
|void vla::(anonymous namespace)::merged_geglu<float, __half>(float const*, __half*, int, bool)|197|27.598|0.14009|[16.0]|[20.0]|
|void vla::(anonymous namespace)::small_gemm_kernel<64>(__half const*, __half const*, float*, int, int, int)|180|25.215|0.14009|[71.0]|[17428.0]|
|void k_bin_bcast<&(op_add(float, float)), float, float, float, float const*>(float const*, float const*, float*, unsigned int, unsigned int, unsigned int, uint3, uint3, uint3, uint3, uint3, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, float const*)|119|22.707|0.19081|[20.0]|[20.0]|
|vla::(anonymous namespace)::pack_f16x4(float const*, __half2*, unsigned long, int, int, int, unsigned long, unsigned long, unsigned long)|395|22.279|0.05640|[27.0]|[20.0]|
|gemmex_nn_256_256x128_f16f32_epilogue_qy2|197|21.710|0.11020|[242.0]|[24716.0]|
|vla::(anonymous namespace)::gemma_probs(float const*, __half const*, __half*, int, int, float)|197|20.053|0.10179|[25.0]|[532.0]|
|void vla::(anonymous namespace)::apply_rope_table<float>(float const*, float const*, float*, int, int, int, unsigned long, unsigned long)|395|12.565|0.03181|[15.0]|[20.0]|
|void vla::(anonymous namespace)::small_pack_kernel<float>(float const*, __half*, int, int)|540|11.894|0.02203|[6.0]|[20.0]|
|void vla::(anonymous namespace)::res_adarms<float, false>(float const*, float const*, float const*, float const*, float*, int, int, float)|350|11.887|0.03396|[20.0]|[1044.0]|
|void convert_unary_cont_vec4<float, __half>(void const*, __half*, long)|162|10.420|0.06432|[8.0]|[20.0]|
|void vla::(anonymous namespace)::small_pack_kernel<__half>(__half const*, __half*, int, int)|180|8.512|0.04729|[6.0]|[20.0]|
|void vla::(anonymous namespace)::vision_epilogue<true>(float const*, float const*, float const*, float*, int)|27|8.302|0.30747|[37.0]|[20.0]|
|musa_asm_hflash_atten_512_256x64x128|27|7.275|0.26944|[244.0]|[57472.0]|
|void k_bin_bcast<&(op_mul(float, float)), float, float, float, float const*>(float const*, float const*, float*, unsigned int, unsigned int, unsigned int, uint3, uint3, uint3, uint3, uint3, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, float const*)|55|6.854|0.12462|[20.0]|[20.0]|
|void rms_norm_f32<1024, true, false, false>(float const*, float*, int, long, long, long, float, float const*, long, long, long, uint3, uint3, uint3, uint3, float const*, long, long, long, uint3, uint3, uint3, uint3, float)|35|6.569|0.18767|[13.0]|[20.0]|
|void vla::(anonymous namespace)::vision_epilogue<false>(float const*, float const*, float const*, float*, int)|54|6.437|0.11920|[39.0]|[20.0]|
|void norm_f32<1024>(float const*, float*, int, long, long, long, float)|55|6.265|0.11390|[11.0]|[24.0]|

优先级：大型prefix GEMM、剪枝prefix Attention、动作小GEMM，以及cpy_scalar/GeGLU/pack/softmax。后续逐项记录实际命中、精度与整网收益。
