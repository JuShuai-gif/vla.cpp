# 2026-11-02 前缀与动作 Attention 优化

本轮完成短序列Attention融合及两模型分配置验收。
30样本均值：裁剪580.923→542.174ms（下降6.67%）；完整1243.421→952.266ms（下降23.42%）。
裁剪只融合动作，完整融合前缀和动作；所有新路径仍需显式开关。

- [待做清单](待做清单.md)
- [GEMM路径探索](01_GEMM路径探索.md)：微基准、图拼接原型与捕获失败。
- [融合与正确性](02_融合与正确性.md)：实现、固定工作区、未通过方案。
- [最终复测](03_最终复测.md)：30样本统计、三部分对比、精度、版本与限制。
- [使用命令](使用命令.md)：两类模型启用与回退。
- final-summary.json / summarize.py：数据与复算。
- validation-summary.json：19项CTest、两模型动作模式、完整入口和三段Graph。
- validation-full-all-summary.json：完整模型前缀+动作专项。
- boundary-test.log：最小形状、640上限、641回退。
- reference-reuse-summary.json：当前基线与历史变输入参考一致性。

日志和source.patch以.gz无损保存，服务器保留未压缩原件；仓库下载后可用gzip -dk解压。
压缩记录清单.json提供原始内容SHA256；外部产物清单.json记录不入Git的动作/追踪文件。
