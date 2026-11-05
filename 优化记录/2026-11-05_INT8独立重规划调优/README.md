# INT8 独立重规划调优

已新增并部署独立配置 configs/vla_piper_live_int8.json。原 FP16 配置、默认和显式 baseline 启动参数、生产/量化动态库及已有工作区文件均保持不变：37 个受保护文件 SHA256 校验通过。本轮没有发送机械臂动作。

## 改动与实测

只对新配置的 full/pruned 指定 int8-main-smooth、提高重规划阈值 27→36 / 25→38、开启现有动作诊断并清空旧检查历史。保持 20Hz/30Hz、50 步窗口和 5 步融合。设备绑定复制本轮备份时当前配置。

真实 GPU、保存观测、实际推理/队列/下发函数与内存 SDK 的离线对照，每组运行 30 秒、3 次预热。不是实时相机/真实 CAN/机械臂闭环测试；线程模拟进程，SDK JointCtrl 人为等待 4ms。详见 check_continuity.py。

| 模型 | 原计划间隔 P50 | 新计划间隔 P50 | 缩短 | 新动作提交 P50 / 最大 | 新稳态计划间隔样本数 |
|---|---:|---:|---:|---:|---:|
| 完整 | 1151.70 ms | 700.90 ms | 39.14% | 50.42 / 52.25 ms | 39 |
| 剪枝 | 835.17 ms | 412.50 ms | 50.61% | 34.88 / 37.44 ms | 69 |

计划间隔指动作队列合并新计划的时间差；稳态统计剔除首个间隔，完整原始时间戳和包括首个间隔的统计均保留。预测本身没有新增提速：完整 P50 688.82→686.67ms，剪枝 412.81→410.70ms，差异属于本次顺序测试波动。收益来自减少等待。固定图像与实际相机采集耗时不同，不能直接视作真机闭环延迟。

四组真实 GPU 测试均无队列耗尽、无动作提交间隔超过 2 倍设定周期。新配置实际模型融合前最少剩余 22/25 步；以历史最大 observation-to-prediction ×1.15 为预算的 15 秒压力测试（完整 915.46ms、剪枝 526.98ms）最少剩余 13/18 步，超过 5 步融合要求。该预算不保证未来推理耗时上限。

两组候选和两组预算测试的诊断记录均无丢弃、写入错误或截断，写线程正常退出。

## FP16 隔离与限制

check_config.py 对比原配置 full/pruned 的默认命令和显式 baseline 命令，与备份逐参数相同；候选新配置无 precision 参数与显式 int8-main-smooth 参数一致，实际部署文件及命令与候选一致。所有硬件发现均 mock，不执行真实启动命令。

未修改量化权重、补偿、推理代码或 NPU ViT。此次调度改善不代表量化精度已追平 FP16，也不保证闭环抓取/画圆问题已解决；动作偏差见相邻 INT8实机反馈与调度复核 记录。长期负载、热状态和真实运动效果未验收。

## 文件与复现

- deployment-notes.md：两模型新 INT8 命令、原 FP16 命令和恢复方式。
- before-state.json、baseline-live.json：备份路径、原始 live 配置和受保护文件哈希。
- candidate-live.json：已部署的独立配置精确副本。
- parameter-rationale.json：最近实机耗时依据及各模型预算。
- check_config.py、config-validation.json：隔离与真实 launcher 参数验证。
- check_continuity.py、run_native.py：完整离线对照脚本；按 full/pruned × before/candidate 执行，PYTHONPATH=src:tests/python，使用 runtime .venv。
- full/pruned-before/candidate.json 与同名 .log：真实 GPU 测试原始结果。
- full/pruned-budget.json 与同名 .log：模拟延迟压力测试。
- full/pruned-candidate/、full/pruned-budget/：原始动作诊断 JSONL。
- model-input-context.json、environment.json：模型策略、固定观测哈希、软件版本与设备环境。电源策略未单独固定，温度/频率全过程未记录。
- deploy_validated.py、acceptance.json：部署前断言、最终比较结果及新配置哈希。部署脚本拒绝覆盖现有独立配置。
- test-development-notes.txt：测试脚本字段名修正和连接中断说明。

完成最后一组时 SSH 外层连接报 reset，重连后确认四组 JSON 已完整产生、无残留测试进程。再次逐项检查输出中的队列、间隔、融合余量、诊断写入结果才执行部署；没有因为连接状态推断测试成功。
