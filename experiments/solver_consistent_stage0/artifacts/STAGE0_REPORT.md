# Solver-Consistent Recurrent Reasoning — Stage 0

结论：**UNCLEAR；正式 seed=0 对照实验未完成。** 当前数据不合格，正式训练已被程序阻止。不得把未运行的模型指标、数值收敛曲线或四张主图补造出来。

## 1. Experimental setup

项目位于远程 Windows `D:\solver_consistent_stage0`，本地同路径保留源码、完整JSONL及小型结果副本。无关仓库未修改。官方生成器为 https://github.com/asaparov/prontoqa ，commit `0a6412b6fddf46324a1cb96e066dd7b3d89b87d6`，vendor源码未改。使用 fictional / ModusPonens / relevant distractors；扩展2048概念名及32个互斥四属性词族的词汇API输入，以解决深度6/8/12的有限词库耗尽。推理深度d映射为官方steps=d+1。它是词汇扩展的官方生成分布，不能声称完全等同官方默认数据。

生成135000条：train100000、validation10000、IID10000、OOD6/8/12各5000，标签精确平衡；前三个split深度1至4均匀。官方证明不进入模型监督，仅二分类CE。所有split的规范化逻辑实例ID全局不重叠，SHA256与manifest匹配；谓词重命名后的同构图仍可能重复。独立检查960条分层样本的标签和最短证明深度通过，但这不能证明模型必须执行该证明。

BPE仅训练于train，目标词表上限8192，实际2289；小写及数字跨度切分。max_len704，六个split实测截断率均为0。长度256会截断大量样本，因而在训练前按长度审计提高。当前模型宽512、8头、FF2048、GELU、prenorm、dropout0.1、stem3块、shared core2块，math SDPA。所有共同参数按相同seed初始化。

| model | params | trainable_params | core_params | completed_formal_updates |
| --- | --- | --- | --- | --- |
| vanilla_loop | 17296386 | 17296386 | 6304768 | 0 |
| step_conditioned_loop | 17313826 | 17313826 | 6322208 | 0 |
| vector_field | 17313826 | 17313826 | 6322208 | 0 |

参数最大相对差0.10083%。正式训练计划为seed0、2000次更新、有效batch128、AdamW3e-4/weight_decay0.01、5%warmup+cosine、clip1、BF16、每批同一随机K=4或8、相同样本顺序。2000是预先记录的探索性缩减，不能声称完成了建议的20000次更新。最后一次工程重试统一microbatch8、梯度累积16次、激活重计算、64-token填充桶；权重容量及有效batch不变。eval_batch8；延迟测量原计划固定batch32/20warmups/100batches，尚未执行。

硬件：RTX5090 / 32607MiB / Windows10 build18363 / NVIDIA616.56。最后重试PyTorch2.8.0+cu128及Python3.10，继承既有环境至独立 `.venv28`。环境快照见environment.json和environment_packages.txt。

最后一次length704/K8/micro8/激活重计算的实测显存探针：allocated=901.361328125MiB，reserved=1286.0MiB。**这是一次合成前后向探针，不是完整训练峰值、吞吐或方法效果。**

## 2. Does budget sensitivity actually exist?

**未能评估。** 三个模型的depth1学习门槛没有全部通过，正式更新均为0。未测得unseen budgets3/5/7/12/16的accuracy、harmful flips，也未测得baseline的nonuniform schedule影响。当前数据还存在确定性结构捷径；即使以后在该数据上得到高accuracy，也不能据此证明多跳推理能力。

结构审计脚本 `data/audit_structural_shortcut.py` 的predict函数只读取输入文本，不访问label/depth/ID/proof，不训练任何权重，不进行可达性分析或递归规则执行：定位查询属性的正反候选规则，统计每个候选规则的前件概念在全文出现的次数，选择出现次数较多者，再比较结论与查询的否定符号。

| split | n | accuracy | unique_choice_fraction |
| --- | --- | --- | --- |
| iid | 10000 | 1.0 | 1.0 |
| ood_12 | 5000 | 1.0 | 1.0 |
| ood_6 | 5000 | 1.0 | 1.0 |
| ood_8 | 5000 | 1.0 | 1.0 |
| train | 100000 | 1.0 | 1.0 |
| validation | 10000 | 1.0 | 1.0 |

全部135000条均100%，每个depth分层也均100%；远程和本地独立执行输出一致。官方run_experiment.py约591至617行给查询结论增加相反结论的dummy规则，其前件是孤立概念；真正结论的前件在原本ontology/事实中再次出现。这个差异泄露了正确结论的极性。这里验证的是**当前固定生成配置及词汇扩展数据**，并未宣称PrOntoQA所有配置都有同样缺陷。

此前query-only、unigram bag-of-words和negation parity约50%并未捕获这个捷径。这说明仅靠简单词袋接近随机及证明长度审计，不足以排除结构shortcut；旧审计的结论必须收窄。

## 3. Does the vector field reduce this sensitivity?

**未能评估。** budget8主结果、trained-budget损失、recurrent bypass贡献、不同solver的task benefit均NA。代码实现Euler/Heun/RK4及matchedNFE4/8/12/16，但未在已训练模型上运行完整对照。VectorField.forward只接收H、t及静态padding mask；dt仅由solver乘在外部，网络不能访问dt/K/NFE/solver。Derivative为Transformer residual increment。接口和求解器单元测试通过不等于方法有效。

## 4. Numerical behavior

独立已知ODE单元测试验证误差随分辨率提高而下降，NFE计数及sum(dt)=1正确。训练后的512例FP32隐藏态收敛、RK4参考32/64步、observed slopes及CheckI均未运行，因此全部NA。四张要求的主图目前不能生成真实数据图。数值收敛本身也不构成推理改进证据。

## 5. Alternative explanations and execution evidence

确定的数据替代解释是上述100%结构捷径；当前截断率0，参数差约0.1%，因此这两项不能解释或消除该捷径。更新幅度、confidence、core贡献、正常预算accuracy及训练速度对照均未获得可比较证据。

已实际运行的开发阶段证据：

| artifact | saved_update | best_validation_accuracy | last_validation_accuracy | peak_allocated_mb |
| --- | --- | --- | --- | --- |
| sanity_300_failed | 300 | 0.5 | 0.5 | 7432.8408203125 |
| sanity_v1_native_crash | 1300 | 0.5152 | 0.482 | 7446.5615234375 |
| sanity_v2_allocator_pilot | 100 | 0.5 | 0.5 | 15298.04736328125 |
| sanity_v2_torch211_native_crash | 200 | 0.5 | 0.5 | 15359.01318359375 |
| sanity_v2_torch28_native_crash | 200 | 0.5 | 0.5 | 15364.376953125 |

最初300-update depth1 vanilla gate完整跑完，但全部held-out检查在50%，没有通过65%门槛。16条训练样本的plumbing overfit达到100%只说明该模型可以记忆训练样本，不能替代held-out学习门槛。后续2000-update重试分别原生崩溃，历史checkpoint/log均保留且未作为正式训练初始化。

最新低显存重试日志最后记录：`{"update": 80, "budget": 4, "loss": 0.688790250569582, "lr": 0.000243, "elapsed_sec": 232.54423559998395}`。发现数据捷径后显式终止本项目所属sanity训练进程；不是自然完成、不是通过gate、不是最新重试再次崩溃。未杀其他项目进程。

Windows Application事件1000/1001明确记录：23:54的0xc0000005 faulting module=cublasLt64_12.dll；00:08的0x80000003 faulting module=nvcuda64.dll；System23:48:58还有nvlddmkm event153。见windows_crash_events.json。**故障模块位置不等于已证明根因**；不能断言只是PyTorch版本问题或低显存配置已经根治。记录中还存在与本项目不同python311路径的其他崩溃，不能归属为本实验。

Sanity checks：

| check | result |
| --- | --- |
| A | NOT_PASSED: completed initial300-update vanilla depth1 gate stayed at50%; no three-model passing gates |
| B | PASS: signature excludes dt/K/NFE/solver ID |
| C | PASS: inference budgets do not mutate state_dict |
| D | PASS: counted neural calls and ODE NFE |
| E | PASS: positive steps sum to1 |
| F | PASS: eval dropout off and deterministic repeat |
| G | PASS: final-config parameter spread <1% |
| H | PASS: global canonical underlying IDs and text IDs disjoint; checksums verified; no isomorphic-template exclusion claim |
| I | NOT_RUN: trained vector-field reference unavailable |
| extra_data_validity | FAIL: zero-training frequency shortcut100% on135000 examples |

B至H基于保存的单元测试与数据审计；最新测试还覆盖最终frozen配置的参数匹配以及纯文本shortcut的大小写/查询否定处理。stage0_results.csv中的accuracy等全写NA且status=NOT_RUN_INVALID_DATA；budget、solver、schedule仅为计划条件，nfe为NA、planned_nfe另列。不得把这些行当成完成实验。真实数据捷径结果在data_shortcut_results.csv。

## 6. Decision

**UNCLEAR（数据有效性失败及执行未完成）。** 没有证据证明baseline存在有害budget sensitivity，也没有证据证明vector field有效或无效。当前结论不符合科学STOP的条件，更不符合GO；不能用缺失结果否定或支持架构假设。

下一步先修复正反结论候选规则之间的结构不对称，至少平衡前件频率、局部图度数和实体连接模式；保证目标由实际前提可达性决定，不能仅由dummy规则身份决定。优先选择经审计的官方配置；若必须新增对称干扰或其他生成修复，应显式版本化并重新生成所有split、重新检查标签/最短深度/冲突/全局去重以及本文捷径。修复后所有模型及tokenizer从零开始，并重新跑相同学习gate。旧数据及旧gate不可复用。

同时修复或隔离Windows/CUDA运行故障，先完成受控稳定性检查。在有效任务和三个模型的学习门槛通过前，不运行额外seed、复杂loss或模型方法比较。`run_pipeline.py`现已加入结构shortcut硬门槛，当前数据重跑会自动产生本报告并停止，避免误训练。

交付边界：源码、配置、README、生成脚本、测试、135000条数据、运行日志、数据审计CSV、公平性表及此报告已保存。**三模型正式seed0训练、task指标、solver/latency测量和四张主图仍未完成。**
