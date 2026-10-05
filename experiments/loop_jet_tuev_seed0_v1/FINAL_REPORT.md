# Loop-JET TUEV seed0 v1 — current execution report

**Data provenance audit complete. Formal two-model experiment not completed; decision NEED_RAW_TUEV.**

The independent channel audit requested on 2026-10-05 is authoritative: [HF_CHANNEL_AUDIT.md](data_audit/HF_CHANNEL_AUDIT.md). Both processed variants are Level C: physical channel index/order/reference cannot be uniquely recovered. No JET16 conversion or formal training has run.

## Objective
Test iterative target-directed error correction with Naive K2 and DS K2, seed0. The preceding provenance stage asks whether defensible conversion into official JET16 exists. Current answer: NO under available evidence.

## Environment
Remote D:\loop_jet_tuev_seed0_v1; NVIDIA RTX5090 32GB; torch 2.8.0+cu128, CUDA 12.8, capability [12, 0], sm_120 included. Working torch retained. Process-local verified affinity mask 0xffff0000; this does not establish the cause of past native failures. Other GPU jobs observed; none stopped.

## Data
Actual 32-channel H5 downloaded via hf-mirror.com in 35.05 seconds and fully SHA verified. Actual 22-channel headers/labels verified with bounded HTTP Range, no full X download. Source history: 22 + 21 commits. Raw metadata: 518 recordings, 15 layouts, 100% required-electrode coverage. These raw facts do not identify processed columns. No source EDF join is possible from released X/y-only H5.

## Model architecture and sanity
Official revision 07f9e6491796f4f2c717d6259b1a2e24afce6a77. Actual 12 blocks, 768 width, 12 heads, patch_size=200; 16x5 tokens. Loop 4+[4]x2+4; same module parameters reused, common t/class condition and shared post/head. Parameter total 129,865,160 both K1 and K2; trainable 129,861,320. K1 max absolute output difference: 0 in FP32 and BF16. Both objective backward checks passed on synthetic batch2; no TUEV optimizer update performed. Lazy H5 worker tests and refusal of actual 32-channel files passed.

Official net predicts endpoint xhat1 directly. Original velocity uses denominator clamp_min(0.05); wrapping the endpoint again as xt+(1-t)*net would change the implementation. Official mix loss preserved: L1 + 1.0 statistics + 0.1 TV + 0.1 correlation, STFT weight0; DS=(L2+(1/3)L1)/(1+1/3). Official statistics/TV operate on the 200-point patch axis; no replacement whole-trace loss.

## Training protocol
Frozen settings: global256, AdamW betas(0.9,0.95), LR5e-5, WD0, five-epoch warmup then constant LR, official weighted sampler/drop_last, stochastic official x0/t/class-drop. Max200, min60, validate every5, patience6, relative threshold0.2%, held-out per-index seed20261005. These are configurations, not executed training. Full training/analysis driver remains pending the data gate; provided model/loader/audit code has been exercised.

## Naive / DS results
| Model | BestEpoch | FinalTestL1 | Corr | PSD | P(E2<E1) | ResidualCos | Median r | TS-FID |
|---|---|---|---|---|---|---|---|---|
| Naive K2 | Not run | Not run | Not run | Not run | Not run | Not run | Not run | Not run |
| DS K2 | Not run | Not run | Not run | Not run | Not run | Not run | Not run | Not run |

## Loop trajectory / residual alignment / contraction / t and class analysis
Not run. No trained checkpoint, held-out predictions, bootstrap confidence intervals or scientific figures exist. Synthetic smoke values are implementation checks, not model performance or mechanism evidence.

## Generation metrics
Not run. Current official pipeline includes FFT-feature TS-FID, no external pretrained backbone required; no silhouette implementation found. No generation metric substituted or invented.

## Comparison and interpretation
No trained-arm comparison and no paper numerical comparison is justified. All four mechanism cases remain unevaluated. The data gate is not evidence for or against iterative reasoning.

## Final decision
STOP formal training at this stage; NEED_RAW_TUEV for the provenance decision. The exact missing evidence is a release-specific processed channel-index/physical-derivation/reference mapping contract. Unit and filtering execution remain unverified. Complete code + data-gate evidence + channel report are delivered; two trained models, runtime benchmark, checkpoint-selected evaluation, bootstrap, generation and requested figures remain unexecuted.
