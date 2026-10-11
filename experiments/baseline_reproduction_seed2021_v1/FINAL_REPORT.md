# LatentTSF、DAG、WorldTS Baseline 复现报告

24/24 个配置已完成：{'MATCHED': 22, 'CLOSE': 2}。21 组本轮真实训练，3 组严格验证复用。没有加入 Loop，也没有用测试集搜索超参数。

## A. 完成情况

|模型|数据|Backbone|T→H|Seed|执行类型|论文 MSE/MAE|实测 MSE/MAE|MSE 相对差异|最佳 epoch|指标复核|状态|
|---|---|---|---|---|---|---|---|---|---|---|---|
|DAG|NP|DAG|168→24|2021|fresh_run|0.202/0.237|0.209416/0.232343|3.67%|6|PASSED|MATCHED|
|DAG|NP|DAG|720→360|2021|verified_reuse|0.521/0.451|0.527720/0.456922|1.29%|1|PASSED|MATCHED|
|DAG|PJM|DAG|168→24|2021|verified_reuse|0.057/0.143|0.057442/0.143281|0.78%|9|PASSED|MATCHED|
|DAG|PJM|DAG|720→360|2021|verified_reuse|0.130/0.218|0.129724/0.217574|0.21%|4|PASSED|MATCHED|
|LatentTSF|Electricity|DLinear|720→192|42|fresh_run|0.168/0.271|0.171538/0.278723|2.11%|19|PASSED|MATCHED|
|LatentTSF|Electricity|DLinear|720→336|42|fresh_run|0.185/0.288|0.196716/0.306010|6.33%|9|PASSED|CLOSE|
|LatentTSF|Electricity|DLinear|720→720|42|fresh_run|0.218/0.317|0.239078/0.344222|9.67%|15|PASSED|CLOSE|
|LatentTSF|Electricity|DLinear|720→96|42|fresh_run|0.157/0.261|0.157547/0.267554|0.35%|16|PASSED|MATCHED|
|LatentTSF|Electricity|iTransformer|720→192|42|fresh_run|0.173/0.279|0.181223/0.292608|4.75%|5|PASSED|MATCHED|
|LatentTSF|Electricity|iTransformer|720→336|42|fresh_run|0.202/0.311|0.195709/0.307831|3.11%|13|PASSED|MATCHED|
|LatentTSF|Electricity|iTransformer|720→720|42|fresh_run|0.250/0.348|0.253541/0.352215|1.42%|17|PASSED|MATCHED|
|LatentTSF|Electricity|iTransformer|720→96|42|fresh_run|0.153/0.260|0.157465/0.267621|2.92%|6|PASSED|MATCHED|
|LatentTSF|ETTh1|DLinear|720→192|42|fresh_run|0.405/0.416|0.403051/0.414384|0.48%|20|PASSED|MATCHED|
|LatentTSF|ETTh1|DLinear|720→336|42|fresh_run|0.435/0.437|0.432642/0.434228|0.54%|19|PASSED|MATCHED|
|LatentTSF|ETTh1|DLinear|720→720|42|fresh_run|0.436/0.468|0.436837/0.467428|0.19%|7|PASSED|MATCHED|
|LatentTSF|ETTh1|DLinear|720→96|42|fresh_run|0.366/0.392|0.364044/0.390882|0.53%|15|PASSED|MATCHED|
|LatentTSF|ETTh1|iTransformer|720→192|42|fresh_run|0.420/0.441|0.429638/0.445351|2.29%|11|PASSED|MATCHED|
|LatentTSF|ETTh1|iTransformer|720→336|42|fresh_run|0.450/0.463|0.451844/0.462907|0.41%|13|PASSED|MATCHED|
|LatentTSF|ETTh1|iTransformer|720→720|42|fresh_run|0.501/0.510|0.506190/0.512377|1.04%|12|PASSED|MATCHED|
|LatentTSF|ETTh1|iTransformer|720→96|42|fresh_run|0.392/0.422|0.394322/0.420839|0.59%|16|PASSED|MATCHED|
|WorldTS|NP|PatchTST-state-transition|168→24|2021|fresh_run|0.171/0.217|0.167258/0.213351|2.19%|6|PASSED|MATCHED|
|WorldTS|NP|PatchTST-state-transition|720→360|2021|fresh_run|0.423/0.420|0.430395/0.416201|1.75%|2|PASSED|MATCHED|
|WorldTS|PJM|PatchTST-state-transition|168→24|2021|fresh_run|0.056/0.141|0.057516/0.142788|2.71%|20|PASSED|MATCHED|
|WorldTS|PJM|PatchTST-state-transition|720→360|2021|fresh_run|0.118/0.216|0.115082/0.208422|2.47%|7|PASSED|MATCHED|

验收按任务指定的 MSE 相对差异判定：≤5% MATCHED，5–10% CLOSE，>10% NOT_MATCHED。MAE 全部同时报告；这些门槛不是统计显著性检验。

## B. 环境与资源

Windows 10 build 18363 服务器、i9-14900KF、单张 RTX 5090 32GB。实际 GPU/Driver/RAM/磁盘快照见 `environment.json` 和 `hardware_report.md`。Python 3.10、PyTorch 2.8.0+cu128（验证 sm120 forward/backward）、私有 venv NumPy 1.26.4，其余实际版本见 `pip_freeze.txt`。共享已验证的 CUDA PyTorch，未修改全局环境。DAG 原文 Python 3.8 与本轮版本不同。四 CPU 线程、进程仅绑定 CPU16–31，Windows 数据 worker=0；GPU 训练顺序执行。

本轮最终采用的 fresh_run 累计训练墙钟时间 2237.19s；复用的原训练耗时 640.10s，未计成本轮新增训练。独立评测墙钟 54.08s。工程含 venv/数据/所有 retained artifacts 磁盘占用 1.865 GiB。这些训练计时包括 I/O，不是 CUDA kernel 活跃时间。

|Run|训练秒|独立评测秒|峰值 CUDA MB|参数量|显存记录范围|
|---|---:|---:|---:|---:|---|
|DAG_NP_H24_seed2021|228.34|1.69|40.38|491719|original training|
|DAG_NP_H360_seed2021|133.46|2.08|162.98|6058591|original training|
|DAG_PJM_H24_seed2021|305.48|2.10|43.20|557767|original training|
|DAG_PJM_H360_seed2021|201.16|2.08|171.11|6124639|original training|
|LatentTSF_Electricity_DLinear_H192_seed42|7.92|2.03|443.92|276864|original training|
|LatentTSF_Electricity_DLinear_H336_seed42|5.03|3.37|550.49|484512|original training|
|LatentTSF_Electricity_DLinear_H720_seed42|9.97|6.57|903.35|1038240|original training|
|LatentTSF_Electricity_DLinear_H96_seed42|5.90|1.03|385.72|138432|original training|
|LatentTSF_Electricity_iTransformer_H192_seed42|5.84|2.05|1945.09|4674240|original training|
|LatentTSF_Electricity_iTransformer_H336_seed42|10.29|3.42|2071.14|4748112|original training|
|LatentTSF_Electricity_iTransformer_H720_seed42|15.49|6.50|2449.02|4945104|original training|
|LatentTSF_Electricity_iTransformer_H96_seed42|7.45|1.10|1906.30|4624992|original training|
|LatentTSF_ETTh1_DLinear_H192_seed42|16.36|0.92|239.94|276864|original training|
|LatentTSF_ETTh1_DLinear_H336_seed42|16.87|1.44|292.60|484512|original training|
|LatentTSF_ETTh1_DLinear_H720_seed42|10.47|2.65|453.39|1038240|original training|
|LatentTSF_ETTh1_DLinear_H96_seed42|12.98|0.56|228.43|138432|original training|
|LatentTSF_ETTh1_iTransformer_H192_seed42|14.56|0.97|268.97|46560|original training|
|LatentTSF_ETTh1_iTransformer_H336_seed42|17.27|1.48|333.49|51312|original training|
|LatentTSF_ETTh1_iTransformer_H720_seed42|18.92|2.71|503.88|63984|original training|
|LatentTSF_ETTh1_iTransformer_H96_seed42|18.41|0.57|230.37|43392|original training|
|WorldTS_NP_H24_seed2021|341.55|1.17|275.64|345945|original training|
|WorldTS_NP_H360_seed2021|300.72|1.81|1352.24|1026729|original training|
|WorldTS_PJM_H24_seed2021|563.19|2.19|19.78|1087481|resumed fit before independent evaluation; initial training peak was not persisted|
|WorldTS_PJM_H360_seed2021|609.67|3.57|4928.02|2448713|original training|

WorldTS PJM→24 在完成训练后因包装器路径断言失败；首次训练峰值未持久化。该行给出独立评测开始前、续跑 fit 的真实观测峰值，不能当作初始训练峰值。两阶段模型、历史和训练时间已经保留并复用，没有重训练以掩盖失败。

## C. LatentTSF

固定作者 commit `7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2`，直接运行官方 my_train.py、my_AE.py、DLinear 和 iTransformer。16 组真实训练，官方预训练 AE 均加载成功且 encoder/decoder 冻结；每组文件 SHA 和 tensor hash 在训练前后相同。验证集 AE 重构指标见每组 ae_audit.json。官方 Element-wise latent MSE + 按时间点 cosine reduction 保留，未改成 WorldTS 的 trajectory cosine。

14 组 MATCHED；Electricity+DLinear H336、H720 为 CLOSE，MSE 差异约 6.3%、9.7%，同时 MAE 偏高。数据 SHA、train-only scaler、全部目标通道、模型、损失、sample stride、验证与 early stopping 已核实，尚未唯一定位数值差距原因，不能直接归因于 seed。没有为追论文值重跑挑 seed。

ETTh1 使用官方 8640/11520/14400 边界，原 CSV 剩余 3020 点不参与官方实验。Electricity 官方脚本 step=60，训练和测试样本均按此步长，不能把它和 stride=1 的另一种协议混为一谈。所有 321 通道按 features=M 评测。LatentTSF 参考表是 Appendix D Table 6；任务写的 Appendix E Table 7 是 AE ablation，已按原文纠正。

## D. DAG

固定作者 commit `0758990e2c73bb54138ea3e7b11a35cbc5476bcc`。三组 verified_reuse：逐文件源代码比较只允许 CRLF/LF 换行差异，checkpoint/data SHA、官方参数、标准化器、所有 origins、重新推理预测与旧结果一致。旧 NP→24 原始 stdout 不完整，因此补做 fresh_run，原 verified_reuse 证据另行归档。四组都使用原 TCM/CCM DAGModel 和完整 published DAG.sh 参数。

NP/PJM 每个数据文件都是 52416 小时 × 3 列；OT 是唯一目标，两列外生变量通过官方 future-exog 接口提供。原 runner 的两次取整边界是 36690/41932/52416。训练 drop_last=True；测试保留全部 10461/10125 滚动窗口。四组 mse_norm/mae_norm 与独立 float64 校验一致；之前 V1 的全量验证证据保留在报告中。外生预测缺少历史发布版本，本结果属于作者 benchmark future-covariate setting，不能声称已验证真实部署可用性。

## E. WorldTS：实现忠实度与数值复现

作者团队在 2026-10-09 发布官方 [WorldTS 源码](https://github.com/decisionintelligence/WorldTS/tree/269426ea6217d7152a244da0e209d922acd46d4f)，本轮使用其 DAG-bench 数值分支和固定 NP/PJM 命令，因此 source_type=official_source；任务的 independent_reproduction fallback 已不适用。没有重建替代 Transformer。

完整保留公式(6)–(21)对应的 causal TS Encoder、独立 numerical covariate encoder、LayerNorm/MLP 历史融合、PatchTST state transition、future-covariate residual、共享 causal Decoder，以及两阶段训练。因果 patch 使用 replicate 前补和真实历史边界；模型不加入循环。Stage 1 使用 L1、trajectory cosine 和 legacy_vc（VICReg variance/covariance），L1/cos target detach，VICReg target 仍有梯度；Decoder 冻结。Stage 2 只训练同一 Decoder，predicted-state MAE + 0.3×target-state reconstruction MAE，Stage 1 所有 tensor hashes 不变。

NP latent_dim=128、condition_dim=64、encoder/decoder patch=3、predictor d_model=64、d_ff=256；PJM latent/condition_dim=224、patch=5、predictor d_model=128、d_ff=512。两者 state transition patch=48、stride=24、1层4头，sinusoidal position、GELU FFN、flatten projection；实际层形状见四个 worldts_preflight 文件。全部架构/训练超参来自 OFFICIAL_SOURCE，不依赖缺失配置假设。环境、路径和观测包装器是 REPRODUCTION_ASSUMPTION，已单列。

Stage 1 最多20轮、lr=0.001，按验证 latent objective 选 checkpoint；Stage 2 最多20轮、NP lr=0.0001/PJM lr=0.0002，按 decoded validation MAE 选 checkpoint，patience=5。官方 history-only RevIN、Adam、type3 scheduler、原 shuffle/每批均值聚合保持不变。原代码会将 final_checkpoint_path 重置为实际 stage2_decoder_mix.pt，包装器据此归档。

训练前四种梯度/因果/边界测试通过；训练后在固定验证 origins 上重新检查标签扰动不改变预测、latent 标准差和有效协方差秩，见 worldts_trained_checkpoint_audit.json。两阶段损失逐轮保存在 worldts_stage1_metrics.csv/worldts_stage2_metrics.csv；原字段 mse_loss 在 mae 模式实际表示 L1，报告不混淆含义。没有依靠测试表现选择新的配置。

|任务|Stage1 最佳/实际轮|Stage2 最佳/实际轮|实测 MSE/MAE|论文差异|验收|
|---|---|---|---|---|---|
|NP→24|1/6|6/11|0.167258/0.213351|2.19%|MATCHED|
|NP→360|1/6|2/7|0.430395/0.416201|1.75%|MATCHED|
|PJM→24|1/6|20/20|0.057516/0.142788|2.71%|MATCHED|
|PJM→360|1/6|7/12|0.115082/0.208422|2.47%|MATCHED|

实现忠实度：可确认与固定官方源码一致，最小运行包装器 diff 可审查。数值复现：以表中实际验收状态为准。接近论文数字并不证明原作者未公开的硬件环境完全一致，也不能由单 seed 结果得出新方法优势。

## F. 实际问题与修复

完整原始失败日志保留；`ISSUE_LOG.md` 逐项记录命令、traceback文件、原因和验证。包括代理 TLS 配置、空目录、pandas 日期/索引、patool 导入、W&B deprecated Settings、Windows MAX_PATH、CRLF/LF 源码审核、WorldTS bash GPU 变量展开和官方最终 checkpoint 路径。模型架构、损失公式、目标列和测试样本均未为修复而改变。PJM 的两阶段已完成 checkpoint 成功恢复，训练时间/历史复用，日志串联全部尝试。

## G. 研究准备状态

1. LatentTSF：原模型和 AE 可信，可作为同一代码/数据协议下的基线。Electricity+DLinear 两个长 horizon 只达到 CLOSE，引用论文原值时必须同时标明本轮差距；加入新机制之前继续核查 AE/data/preprocessing 与作者运行环境差异，不能用测试集调参。
2. DAG：四组源代码、数据、指标与强基线协议可用，fresh/reuse 分开记录。未来协变量 benchmark 与真实部署证据必须分开。
3. WorldTS：使用已公开的作者源码，完整数值条件模型和两阶段训练已执行。各配置数值质量依上表判断；没有独立实现缺参问题。
4. 可进入后续实验：24组本轮实测 checkpoint 与完整原始预测，限定在相同 split、window stride、标准化和 future-covariate setting。论文参考值不能替代本轮基线。
5. 加入 Loop 前应关闭：Electricity+DLinear 论文数值差距、PJM 短期训练峰值记录缺口、单 seed 的统计不确定性、外生预测发布时间信息缺失。此轮未设计或运行 Loop。

## 产物位置

统一结果与逐组路径：`results/all_baselines.csv`。独立复核：`results/metric_verification.json`。配置来源：`results/config_provenance.csv` 和 `reports/WORLDTS_CONFIG_PROVENANCE.md`。全部 raw CSV、checkpoints、resume snapshots、scalers、predictions 保存在服务器 `D:\baseline_reproduction_seed2021_v1`；`reports/artifact_manifest.json` 记录相对路径、字节数和 SHA256。Git 只保存源码归档、脚本、配置、日志、报告与小型证据，未上传大型训练产物。具体从零/选择性/续跑命令见 REPRODUCE.md。

参考原文：[LatentTSF](https://arxiv.org/html/2602.00297v1)、[DAG](https://arxiv.org/html/2509.14933v1)、[WorldTS](https://arxiv.org/html/2609.31162v1)。

### DAG 与上一轮 V1 的逐组比较

|Run|执行类型|MSE 绝对差|MAE 绝对差|最大预测绝对差|
|---|---|---:|---:|---:|
|DAG_NP_H24_seed2021|fresh_run|0.00765749216|0.004464690084|1.360290051|
|DAG_NP_H360_seed2021|verified_reuse|1.607373357e-08|6.105451644e-09|0|
|DAG_PJM_H24_seed2021|verified_reuse|1.809224826e-09|6.568427025e-09|0|
|DAG_PJM_H360_seed2021|verified_reuse|3.038920143e-09|6.915113482e-09|0|

NP→24 新训练最佳第6轮、实际11轮；旧训练最佳第17轮、实际22轮。新旧 resolved config 完全一致，数据、标准化器和测试 origins 已核实；新预测与旧预测并不相同。尚未唯一定位两轮训练轨迹差异，不能据同一 seed 宣称逐位可复现，也不能把数值差异直接归因于 seed。其余三组复用 checkpoint 的重新推理与 V1 预测完全相同。原 NP→24 checkpoint 和 predictions 在 `reports/dag_verified_reuse_archive` 保留，SHA 也纳入 artifact manifest。

WorldTS 坍塌诊断采用固定的32个验证 origins：各组预测状态有效协方差秩约3.34–4.61，最小逐维标准差均大于0.11，近常量维度比例为0。它们没有触发所声明的严重坍塌标志；这不等于证明128/224维全部有效或该表示最优。完整统计见训练后审计 JSON。
