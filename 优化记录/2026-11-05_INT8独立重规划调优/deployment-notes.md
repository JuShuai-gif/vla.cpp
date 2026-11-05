# 独立配置与使用方式

用户要求：调整 INT8 效果/调度，不能影响原 FP16。本次不执行任何物理机械臂动作。

仅新增 robot_runtime/configs/vla_piper_live_int8.json。原 vla_piper_live.json、Python 实现、模型权重、动态库、CAN 与摄像头绑定不改。原配置的完整/剪枝默认入口及显式 baseline 命令均逐项比较，原参数一致。

INT8 独立配置继承当前设备绑定，仅 full/pruned：
- precision 设为 int8-main-smooth；
- replan_threshold 完整 27→36，剪枝 25→38；
- diagnostics 设为 true，使用现有有界异步动作诊断记录；
- 清空复制来的 last_checks，避免把原配置的硬件检查历史误作新配置验收。
保持 execution_period 0.05 / 1/30 秒、execution_horizon=50、rtc_blend_steps=5。

原因：RTC 队列等待阈值会把更快推理换成更久等待；提高阈值以更早开始下一次观测和推理。阈值来自各自最近实机 observation-to-prediction 耗时，不改变模型输出精度，不保证消除量化偏差。

新的 INT8 命令（必须使用独立配置文件）：

完整：
    cd /home/skysi/code/robot_runtime && bash scripts/run_robot.sh --config configs/vla_piper_live_int8.json --profile full --precision int8-main-smooth

剪枝：
    cd /home/skysi/code/robot_runtime && bash scripts/run_robot.sh --config configs/vla_piper_live_int8.json --profile pruned --precision int8-main-smooth

恢复原 FP16 完整模型：
    cd /home/skysi/code/robot_runtime && bash scripts/run_robot.sh --config configs/vla_piper_live.json --profile full --precision baseline

恢复原 FP16 剪枝模型：
    cd /home/skysi/code/robot_runtime && bash scripts/run_robot.sh --config configs/vla_piper_live.json --profile pruned --precision baseline

原文件从未改动，恢复只需选回原命令。不要使用 INT8 新配置加 baseline 来代表原 FP16 调度；该组合仍会携带 INT8 新阈值。

备份路径见 before-state.json；备份覆盖原 live 配置和已有工作区 patch。独立 live 配置受 runtime Git 忽略规则影响，精确副本 candidate-live.json 保存在本实验目录。

验证边界：固定存量观测、真实 MUSA GPU/INT8 推理、真实运行循环与队列，线程模拟 I/O 进程、内存 SDK、模拟相机并拒绝 socket 创建。无真实 CAN/图像采集；不等于实机闭环成功率验收。预算测试模拟推理耗时为此前实测最大值乘 1.15，并非延迟上界保证。

量化质量：本次不改变权重、补偿参数或量化范围，也不重新宣称 7D cosine 能保证逐关节误差。之前 396 留出集逐关节偏差审计位于相邻 INT8实机反馈与调度复核 目录。更频繁更新可能改变闭环轨迹，抓取效果仍需用户真机验收。
