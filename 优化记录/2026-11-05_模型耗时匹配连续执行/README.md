# 按模型耗时设置连续执行参数

2026-11-05：生产配置已应用；原 full/pruned 启动命令不变。没有启动真实相机、CAN或机械臂。

## 问题和调整

历史运行的 pruned/full 都关闭 enable_rtc_smooth，replan_threshold=0：
执行50步之后才读取下一次观测并推理，因此每块之间等待完整推理。
最新剪枝日志观测至预测最大525.765ms，完整892.915ms；
SDK周期误差最大约513/885ms，与推理等待一致。原始证据在 prior-runs/。

只调整 configs/vla_piper_live.json 两个预设的三个调度字段：
- pruned：enable_rtc_smooth=true，replan_threshold=25，execution_period=1/30秒。
- full：enable_rtc_smooth=true，replan_threshold=27，execution_period=0.05秒（20Hz）。
- 两者保留50步窗口、已有5步融合、原浮点精度、模型/去噪步骤、设备/单位/使能配置。
- 执行侧RTC利用现有双线程提前计算，按已执行步数裁剪新计划并衔接；
  不修改推理代码，不使用模型未实现的训练侧RTC或warmstep。
- vla_piper.json、NPU配置、INT8可选入口和其它未提交工作不变。

## 参数依据

历史最大观测至预测耗时×1.15作为预算（剪枝604.629ms、完整1026.852ms）。
D=ceil(预算/周期)，提前阈值=D+5+1，且2D+5+1<=50。
剪枝D=19、阈值25；完整D=21、阈值27。预算范围内留6步旧计划，
覆盖5步交叉融合及1步调度余量；下一计划裁剪后仍能持续补给。
完整模型原30Hz下，历史893ms本身就需27步，2×27>50；
只增加阈值而不改变周期无法覆盖相邻两轮推理。完整模型因此采用20Hz，
其轨迹时间尺度会放慢；这不等于证明真实任务成功率不变。

## 验证结果

check_continuity.py 复用现有AST测试框架，加载生产 inference/dispatch、
ActionQueue 和融合函数；真实MUSA浮点模型读取保存的静态观测。
相机和SDK为内存替身、I/O为测试线程替身，SDK提交人工延迟4ms；
所有Python socket被禁止，持有离线测试主机锁，不访问CAN。
before各12秒、candidate各35秒，预热3次；首次启动无动作阶段不计发送间隔。
以下是模拟SDK发送间隔，不是模型耗时或物理到位时间。

| 模型 | 原配置最大间隔 | 新配置P99/最大间隔 | 新配置发送间隔数 | 启动后队列耗尽 |
| --- | --- | --- | --- | --- |
| pruned | 500.838ms | 35.099 / 40.987ms | 1034 | 0 |
| full | 795.994ms | 51.066 / 52.492ms | 683 | 0 |

真实模型新配置推理P50仍为约463.6/759.8ms；本次消除的是推理阻塞动作补给，
没有把发送频率当成模型推理速度提升。最少衔接旧计划余量为11/12步。
预算压力测试用605/1027ms的模拟推理各运行15秒，最大发送间隔35.619/50.797ms，
没有队列耗尽，没有超过两倍周期的间隔，最少衔接余量不少于5步。
只证明已测预算与负载内的软件连续性，不保证任意长尾延迟。

27项现有回归通过：test_rknn_diagnostics_integration、test_rknn_action_dispatch、
test_vla_cpp_backend。首次预算测试夹具把warmup_steps设为0，
被后端合法性校验拒绝；改为1后通过，失败日志fixture-warmup-error.log保留。
check_config.py 核对生产原路径的两个命令转发参数，以及除了三个调度字段之外
所有配置与备份一致；设备发现被mock。preservation-check.json确认29个其它
已有修改文件哈希不变，未覆盖原耗时日志或NPU工作。

## 范围与恢复

未测真实相机竞争、进程IPC、CAN延迟、物理轨迹平滑度或抓取成功率。
没有减少模型计算量；没有做新量化精度实验；没有发出机械臂动作。
20Hz是完整模型的保守执行参数，物理控制效果仍须现场验证；
离线结果不能宣称已经实现肉眼绝对无卡顿。

备份 /home/skysi/backups/robot_runtime/continuous_execution_20261105_090847/。
仅撤销此次配置可把两个预设 enable_rtc_smooth 改回false、
replan_threshold改回0、execution_period改回1/30秒；不要覆盖后续设备调整。
runtime-continuity-only.patch 保存本次配置和文档差异，before-state.json指向快照。
本次生产runtime仍保留原未提交状态，未推送其GitLab远端；归档在量化仓库本地保存。

复跑：在robot_runtime目录中，设置PYTHONPATH=src:tests/python，使用.venv/bin/python运行
check_continuity.py --profile full --schedule candidate --model native --seconds 35 --output /tmp/full-continuity.json。
参数pruned验证剪枝；model=budget验证固定预算；schedule=before复现原同步间隔。
测试使用保存图像、内存SDK并禁止socket；不要把此测试替换为真机启动进行无人监控验证。
