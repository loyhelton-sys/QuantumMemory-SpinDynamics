# 示例清理记录

用户确认后已删除以下 33 个文件。下列文件多数未纳入 Git，不能依靠 git restore 恢复。
删除前已完整备份至 `Results/cleanup_backups/20260928_163104/removed_examples.tar.gz`，并逐文件核验 SHA-256；同目录包含 `sha256.json`。现有实验结果保持不变。

## 已移除（33 个文件）

- `examples/adaptive_10_20_10_mean_smooth.jl`
- `examples/adaptive_single_40us_arp.jl`
- `examples/adaptive_three_pulse_rase.jl`
- `examples/auto_20000_single_arp.jl`
- `examples/balanced_gradient_single_40us_arp.jl`
- `examples/balanced_single_40us_arp.jl`
- `examples/cleanup_five_ten_five_tail.jl`
- `examples/cleanup_three_pulse_frequency.jl`
- `examples/cleanup_three_pulse_head.jl`
- `examples/coherence_first_single_40us_arp.jl`
- `examples/coherence_population_resume.jl`
- `examples/coherence_until_plateau.jl`
- `examples/continue_10_20_10_soft_smooth.jl`
- `examples/continuous_iq_single_40us_arp.jl`
- `examples/continuous_single_40us_arp.jl`
- `examples/diagnose_double_nsub.jl`
- `examples/finalize_spike_cleanup.jl`
- `examples/finish_comparable_30us_rase.jl`
- `examples/finish_comparable_40us_rase.jl`
- `examples/inspect_adaptive_rase.jl`
- `examples/optimize_comparable_30us_rase.jl`
- `examples/optimize_comparable_40us_rase.jl`
- `examples/optimize_spike_cleanup.jl`
- `examples/plot_paused_coherence_run.jl`
- `examples/plot_selected_final_rase.jl`
- `examples/population_priority_10000_single_arp.jl`
- `examples/prepare_comparable_30us_rase.jl`
- `examples/prepare_comparable_40us_rase.jl`
- `examples/run_comparable_30us_rase.jl`
- `examples/run_comparable_40us_rase.jl`
- `examples/smooth_adaptive_rase.jl`
- `examples/validate_adaptive_rase.jl`
- `examples/performance/RaseGPUPrototype.jl`

## 判断依据与边界

- RaseGPUPrototype 的核函数已复制到正式 RaseGPUAccelerated 后端，基准已改用正式实现。
- 其余为历史结果专用的调权重、约束优化变体、固定预算续跑、对比实验、局部修复及配套绘图工具；移除后直接重跑这些历史实验需先恢复脚本，恢复需使用备份。不能据此断言这些方法数值上无效。
- 保留当前 original_three_pulse_rase_optimization.jl、通用 GRAPE/RASE 示例、Bloch/Pulse-Map、最新 smooth_edges_recovery 流程及其绘图导出工具。
- 保留 EtaLowMemoryGPU：最新恢复流程仍使用它。
- 保留 src/grape 下全部实现：基础算法仍被公开 API、测试、GPU 后端调用或用于对照验证。
- 已保留历史实验报告并增加历史记录标记；新增 examples/README.md 说明保留入口及数据依赖。
- 全部保留的 Julia 文件通过语法检查；64 处静态 include 目标均存在，无保留代码引用已删脚本。
- test/test_grape.jl：59 项 CPU/GPU 测试全部通过。
- 历史快照集成测试未能运行：原有 input_snapshot.jld2 不存在；未生成或覆盖历史输入数据。

- 更新后的 GPU 基准合成案例：12 项一致性断言全部通过（N=1/33/64，n_sub=1/4/8）；三个案例的目标函数及梯度差异均为零。
