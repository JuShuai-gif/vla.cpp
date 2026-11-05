# INT8 启动入口接入

用户授权将已验证的量化档位接入现有 run_robot.sh 命令。
保留未指定精度时的浮点生产库和配置，增加 --precision int8-main-smooth 显式选择。
不启动相机、CAN 或机械臂；验证采用mock启动器、策略元数据检查和保存观测离线回放。
现有runtime耗时日志、单位契约调整和NPU工作保留。首次改动前备份详见before-state.json。

实施范围：可选精度参数传递、按精度选择动态库及优化配置、生产策略中添加量化档位、
兼容Python API指向隔离源码；不重建或覆盖生产浮点库。NPU/INT8组合未验证，明确拒绝静默忽略精度。

## 结果

接入完成。生产启动命令追加 --precision int8-main-smooth 即选择量化；
省略参数或指定 baseline 保持原浮点库。已更新 runtime/docs/VLA_CPP.md。
venv 中 vla_cpp_source.pth 指向隔离量化源码；生产浮点库及 main 67f47a0 未改变。

- 回归 119 项：108 通过、11 可选后端跳过；unit-tests.txt。
- 首轮新增测试夹具遗漏 device=0，触发预期 MUSA 配置校验；仅修正夹具后通过，失败记录保留。
- check_entrypoint.py 使用 mock 设备发现和真实命令构造，核验两模型的默认/baseline/INT8 参数与动态库，错误精度拒绝。
- 生产 venv、生产 backend、生产策略配置运行 4 组保存观测回放；每组 warmup=3、samples=5。
  各组均正常退出、输出有限 50×7；INT8 resolved_parameters 确认 native_int8/expert_prefix_gateup/int32。
- 剪枝 P50 464.760→411.271 ms（11.51% 降低）；完整 757.497→684.080 ms（9.69% 降低）。
  此为接入短测，不替代上一轮 BF16 留出精度验收与 ABBA 性能实验。
- 原有未提交工作中 24 个未涉及文件逐一 SHA256 一致；两份 live 配置未变；
  两份策略配置去掉新增 precision_profiles 后与原件完全一致。
- 未开启相机、CAN 或动作执行；未进行真机闭环测试；INT8 保留 experimental 标记。
- 未重新量化或改 C++ 内核；本次不重复旧精度实验。旧验收所定义的逐时刻 7D
  余弦阈值不等同于逐关节跨时间轨迹余弦保证，详见上一轮验收报告。

## 改动和恢复

runtime-integration-only.patch 仅包含本次接入差异（基于已有未提交工作），
没有把用户已有修改当作本次实现；量化仓库直接记录 Python API 及测试改动。
生产 runtime 工作区仍保留未提交修改，不对其内部 GitLab 远端推送。
本次接入尚未推送 GitHub；此前推送的量化实验提交保持不变。

before-state.json / preservation-check.json 分别给出备份与保护检查。
备份目录 /home/skysi/backups/robot_runtime/int8_integration_20261105_083431/。
manifest.json 的 files 将原路径映射到快照和 SHA256。仅切换浮点，去掉参数即可，
不必恢复文件；完整撤销接入须先确认没有后续修改，再逐文件从该快照恢复。
不要删除 vla.cpp-quant，因为当前 Python 接口和 INT8 产物依赖该目录。

生产验证入口（不访问硬件）：
    cd /home/skysi/code/robot_runtime
    source scripts/env.sh --config configs/vla_piper_live.json
    .venv/bin/python -m robot_runtime.replay --config configs/vla_piper_live.json --profile pruned --precision int8-main-smooth --observation validation/vla_cpp/pruned-observation.npz --samples 5 --warmup 3 --output /tmp/pruned-int8-replay.json

将 profile 和观测文件都改为 full 可验证完整模型。回放入口强制 execute_robot=False/dry_run=True，
仅在内存副本上覆盖配置，输出报告包含所选精度和 native 路径信息。
