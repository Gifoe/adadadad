# Research decision

**NO-GO for Idea A as a new complex readout-aware/Jacobian loss.**

Q2 — Performance consequences: NP A1 vs A0 full validation MAE gain=-3.079%, 360-hour descriptive95% interval=[-6.103,-0.249], positive blocks=5/14; NP A2 vs A0 full validation MAE gain=-1.727%, 360-hour descriptive95% interval=[-3.512,0.050], positive blocks=5/14; PJM A1 vs A0 full validation MAE gain=1.456%, 360-hour descriptive95% interval=[-1.032,3.997], positive blocks=9/14; PJM A2 vs A0 full validation MAE gain=1.278%, 360-hour descriptive95% interval=[-0.394,3.107], positive blocks=9/14. Also compare every method with Step0 above: beating deteriorated A0 alone does not establish improvement over the original model. This single-seed validation intervention supports only local feasibility; conflict and performance association do not prove a unique causal mechanism.

Q3 — Ordinary task supervision: A1 is ordinary MAE through a frozen Decoder; A2 adds only a train-calibrated copy of the same MAE. Thus any observed improvement is already obtained with simple supervision. A2 exceeding A1 would demonstrate a useful tradeoff in this budget, not a need for a new readout-aware loss. No Jacobian, adapters, loop or architecture changes were used.

Q5 — Novel mechanism space / Idea A: NO-GO for a new complex readout-aware loss. Ordinary supervision status=INCONCLUSIVE. The experiment has not isolated a concrete failure that simple decoded supervision cannot address. The next defensible action, if ordinary-supervision gains warrant follow-up, is a preregistered independent dataset and additional seeds with A0/A1/A2; retain time-block and Step0 controls. Do not invent a Jacobian mechanism from low cosine or residual forecast error.

A GO would require a specific remaining error pattern that simple supervision demonstrably fails to fix, stability across seeds/independent data, and a tested mechanism distinguishing Encoder, Decoder and temporal shift. Those facts are absent from this feasibility audit. This conclusion does not deny that gradients can disagree or that task supervision can improve a fixed Predictor; it limits what these observations justify.

## Concrete decision evidence

**1. 是否存在 latent/readout 不一致？** 有弱状态方向对齐，但未发现系统性反向冲突。NP/PJM 全预测窗口的完整目标状态梯度余弦均值为0.08077/0.07179，负余弦比例均为0；固定8个 batch 的 Predictor 参数梯度余弦为0.57837/0.25416，负比例也均为0。NP 最后120步切片出现2/4,883个负余弦（0.041%），其余完整目标预测段均为0。高误差四分位同样没有负余弦；NP 高误差组余弦反而更高（0.1033 vs低误差0.0698），PJM 高/低误差组为0.0682/0.0697。所有六个最佳 checkpoint 在相同8个 batch 上的参数余弦仍全部为正。不能把接近正交的状态梯度写成真实参数更新冲突。 训练模式下按步数稀疏抽样的106次参数诊断中有4次负余弦（NP A0为1/17，PJM A0为1/21、A2为2/21，其余0）。这说明部分更新存在短暂冲突；它既不是全训练batch的普查比例，也没有改变固定验证审计与跨任务干预的否定证据。原始轨迹及汇总均保留。

**2. 是否是可利用的 Predictor 瓶颈？** 没有跨任务的证据。相对A0，A1/A2在NP上MAE分别恶化3.079%/1.727%，在PJM上改善1.456%/1.278%。NP只有5/14个360小时区块改善，PJM为9/14；PJM两种改进的360/720/1680小时区块bootstrap区间均包含0。A1在NP的360小时描述性95%区间为[-6.103%,-0.249%]，PJM为[-1.032%,3.997%]。区间针对已选模验证集，不能作为无偏泛化检验。继续训练本身使NP的A0相对Step0改善4.181%，超过A1的1.230%和A2的2.527%；因此不能将任何Step0改善归功于decoded监督。

**5. 下一轮解决什么？** 本轮未找到简单监督无法解决的明确新机制瓶颈；不启动Jacobian loss或新表示机制。若继续验证普通任务监督，唯一具体问题是“PJM的小幅收益能否在未参与选择的数据集和额外预设seed上重复，同时解释NP为何A0更好”。冻结Encoder/Decoder的本轮实验无法因果区分Encoder表示不足、Decoder几何和时间分布变化，不能据此编造其中任何一个失败机制。
