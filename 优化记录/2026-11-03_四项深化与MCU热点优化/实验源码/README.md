# 未采用实验的复现快照
相对 bca90f3 的完整实验补丁见 ../bca90f3-实验全集.patch（包含新增头文件），也保留了完整修改文件及 SHA256。
本快照对应两个真融合 Attention 版本、双缓冲及展开4/8、全部有效的打包/评分优化。原型入口是 test_musa_ops --attention256，环境 VLA_MUSA_GEMMA_TILED=all/all-parallel/denoise/denoise-parallel；真实推理环境见同目录各 *-env.json。
不要在验收通过的工作区直接覆盖；在独立 bca90f3 工作副本应用补丁后编译。实测负收益/精度失败的变体不会进入默认优化配置。
