# 模型加载与图像预处理优化
起点 d7f90ac。两个模型均验证，保持原有数值规则与3/10去噪步、50×7动作。所有记录仅在服务器，未同步Mac。

## 最终结果
同一M1000环境，A为d7f90ac脚本和保存的原始libvla/libvla_core，B为本次实现。按A/B/B/A交替，两模型各2次新进程启动；文件缓存未清空，不能理解为磁盘冷读。原始结果 entry-*.json、命令和日志已保留，/proc/self/maps核对每个子进程只加载指定版本的core。

|项目|剪枝 A→B|完整 A→B|
|---|---:|---:|
|Python GGUF元数据|47000.93 → 48.95 ms|63.02 → 62.08 ms|
|vla_cpp.load（含首次动态库初始化）|12.59 → 8.15 s|12.71 → 8.79 s|
|首次predict本身|7.66 → 7.64 s|7.92 → 8.11 s|
|脚本入口→首次predict返回|70.26 → 17.44 s|23.89 → 20.05 s|
|驻留predict均值|461.09 → 461.22 ms|754.81 → 754.63 ms|

启动值每配置/模型2个样本；驻留推理每配置/模型40个样本。首个predict仍有约7–8秒首次运行开销，本轮没有把它混入加载时间，也没有把预热后的460/753ms当作首帧。模型加载/预处理优化后，驻留推理基本持平，所有导出动作与d7f90ac参考逐元素一致。

## 图像预处理
同一进程、相同双图、A/B/C/D/D/C/B/A，对每段预热3次后测30次，每方案共60个双图样本；依赖导入成本不在这个稳态表内。所有像素逐元素一致。

|方案|双图均值ms|P95 ms|
|---|---:|---:|
|旧：文件解码+旧resize/pad|17.843|18.996|
|新：串行解码+减少复制|10.714|11.121|
|采用：双摄并行解码+新resize/pad|7.245|7.333|
|RGB相机帧+复用out，排除文件解码|2.798|2.946|

D的输入已是RGB帧，与包含文件解码的A/C口径不同，不能当成纯算法提速倍数。初始化Torch的成本另记在各阶段report.json的first_image_pair_ms。本轮像素算法仍为原F32除255、bilinear align_corners=False、clamp与中心零padding；没有换成数值不同的OpenCV/PIL resize。

## 已实施
1. Python GGUF数值数组批量读取。已安装gguf的原Reader把SPM的4,264,023字节逐元素包装成NumPy memmap对象；cProfile有约426万次_get_field_parts、总1.195亿次函数调用。独立适配器pi05_gguf.GGUFReader保留原Reader的版本、字节序、tensor解析，顶层数值数组改成一个视图，保持contents整数/切片语义，并校验截断。SPM直接取bytes；HF JSON、相机顺序和state统计保持一致。带profiler的82秒仅用于定位；正式表使用未开启profiler的A/B样本。
2. 缓存最多2个tokenizer实例，键是模型内容、类型和最大长度。支持SPM/HF JSON，task和state每次重新编码。30个不同state/task/截断情况与旧实现完全相同。
3. C++分块转换，最多按VLA_LOAD_THREADS或VLA_N_THREADS并行。65536元素块保持原行转换器及SIMD尾部规则；避免整个大张量的F32中间向量。同类型F32直接读取；F16/BF16仍保留原有转换舍入及NaN处理，Gemma norm的+1仍走原路径。
4. 合并权重按原顺序直接上传目标tensor对应偏移，移除大CPU拼接buffer；VLA_LOAD_DIRECT_FUSED=1启用，其他后端/调用默认仍可保留原拼接方式。检查总字节及越界；同步tensor_set返回后才复用源数据。默认拼接路径也预留容量。
5. 预处理减少重复F32复制、除法和clamp的额外输出分配；用NumPy输出区完成同值padding；Torch线程配置首次初始化一次。文件双摄解码用2个线程，顺序由map保持，数值resize仍串行以保持既有计算路径。
6. 新增prepare_rgb(uint8 RGB,size,out=...)用于相机帧与输出内存复用。prepare_image旧接口仍可使用，CLI自动使用prepare_images。

入口仍为 scripts/run_pi05.py --musa-optimized，两模型profile新增VLA_LOAD_THREADS=4、VLA_LOAD_DIRECT_FUSED=1。权重仍从原GGUF读取，没有改变dtype/额外量化。调试需要时可设VLA_PI05_PROFILE_LOAD=1，输出加载分段和读取/转换、传输、分配计时。

已有相机循环可以调用：
    from pi05_preprocess import prepare_rgb
    output = np.empty((224, 224, 3), dtype=np.float32)
    pixels = prepare_rgb(rgb_uint8_frame, 224, out=output)
    # pixels为out本身，下次调用会覆盖；作为F32_RGB_01交给同步predict。
输入必须为RGB；相机给BGR时由调用方先转换。两个相机分别持有输出buffer。

## 分阶段与撤回实验
各探索阶段每模型独立进程、30次预处理和6次predict；没有把它们混称严格A/B因果收益。对应 *.json / *.log、命令和阶段源码保留：
- baseline：cProfile定位，元数据82.421/0.105秒，**受profiler影响**。
- baseline-clean：未开启profiler，元数据48.818/0.061秒，CPP加载13.207/12.955秒；这是第一轮未交替的样本。
- candidate-threads-1/4：相同分块与复制调整，分别1/4线程；真实图片、token、动作均一致。
- pin-0/32：试过32MiB固定页暂存，跨模型收益不稳定；已从生产代码撤回。
- lut-0/1：BF16→F16的65536项精确转换查表，覆盖所有16-bit编码测试，无稳定整网加载收益；已撤回，没有把它用作近似或量化。
- direct-0/1：原拼接/直接按偏移写入，选direct-1进入A/B/B/A最终验收。

探索中的GGUF元数据写入夹具/JSON转义/未启动profiler的pstats调用失败日志也保留；修复后的成功测试才计入验收。历史方案已删除的开关不能用于假装重现；实验源码/loader_pinned.cpp、gguf_reader_lut.h及test_weight_loading_lut.cpp是对应原型，需在独立工作副本按注释/命令复原后编译。对照旧库在/home/skysi/VLA/startup-preprocess-20261104/baseline-lib，SHA256见baseline-library.json。

## 正确性与范围
- 完整32项CTest通过（含原有MUSA GPU回归）；诊断格式修正后额外重跑weight_loading通过。
- weight_loading以1048579元素验证F32/F16/BF16九种组合、norm开/关、1/4线程，涵盖所有16-bit编码、随机F32、NaN/Inf/次正规数和scalar尾块；与旧整向量转换逐字节相同。CPU还验证两种合并上传方式及混合源类型。
- 4项Python测试实际通过：多种图像几何/极窄尺寸/负stride、RGB/RGBA/灰度/JPEG、输出复用、相机顺序；metadata大小端、contents索引切片和tensor payload一致，截断拒绝。裸Python缺少可选依赖时跳过，不计作实际通过。
- 实际两个模型的预处理像素和token顺序与原始report及npy一致；正式入口A/B/B/A动作逐元素一致。
- 28组随机输入（每模型10个非U8 F32 +4个U8）及双模型F16 act入口，与上一提交参考逐元素一致；见verification-complete.json。
- 没有重新运行整模型CPU参考推理或真机机械臂闭环；本轮软件等价性对照为已验证的GPU基线。
- 进程峰值RSS原样记录于acceptance.json。剪枝样本下降，完整模型样本上升，故本轮不宣称全进程峰值内存下降。直接上传去除的是拼接临时buffer。

## 复现与数据
measure_pipeline.py用于分段，entry_probe.py/run_acceptance.py用于正式A/B/B/A，bench_preprocess.py用于同进程稳态；check_prompts.py、check_random.py和run_verification.py为验收。run_pi05_before.py是d7f90ac的原文件，旧库与当前库的实际载入路径均核对。无自动清空OS文件缓存、无未测量收益。
