# Official checkpoint provenance

Selected artifact: `official_artifacts/r2_checkpoint_epoch_2765.pt` on the server at `D:/transition_grounded_competent_checkpoint_seed42_v3`.

The [official repository](https://github.com/OSU-NLP-Group/Loop-Think-Generalize/tree/bb22b192977b1b136fde985c8e7a2392d172d97a) links its [public artifact folder](https://drive.google.com/drive/folders/1AQvp8erdtRRn9_ix7E1ssmyWarwFNLVU). The selected path is `checkpoints/multi_hop/final_models/r2/checkpoint_epoch_2765.pt`, [official file](https://drive.google.com/file/d/1oLx2oeD-NpCuokw0AN5PMZNCB-5Ts_kr/view). Size: 342,301,539 bytes. SHA-256: `4e89fde1f592d7afece63a6182f8427beedc85a216adb474777830b6b56495d8`.

Architecture: the actual official `RecurrentGPT2Block`, four shared GPT2 transformer layers, width 768, 12 attention heads, NoPE, no input injection, maximum sequence length 50, 217 embedding rows after the official pad-token addition, and tied embedding/lm_head with the existing final LayerNorm. Prediction is at the last padded position (zero-based 49). The official R2 recurrence setting is fixed two. This experiment's matched fine-tuning recurrence is fixed five as requested.

## Maximum training hop and selection

`H_train=6`, with an explicit inference rather than a nonexistent embedded metadata field. [Official Figure 5](https://arxiv.org/html/2604.07822v1/main_results_accuracy_region.png) labels R2's maximum ID hop as six. The final artifact's optimizer and scheduler both record 4,214 updates in the current stage. The official trainer resets those states per cumulative hop stage and saves at complete epochs; the released atomic data contains 2,000 facts and 15,000 training examples per depth. At batch size 128 the D6 cumulative stage has 77,000 examples and 602 batches; 4,214 equals 602 times seven. D6 is the unique depth in D2–D40 whose stage batch count divides 4,214. This independently supports the Figure 5 mapping. The checkpoint has no explicit training-hop or seed field.

The public final-model inventory offers R1, R2, R3, R4, R5, R6, R7, R8 and dynamic artifacts; separate 12-hop folders are deeper artifacts. R1 is the one-pass vanilla comparator. No successful recurrent checkpoint trained through at most five hops was located in that inventory. R2 is the shallowest released genuinely recurrent final model; its held-out competence must still pass the preregistered 85% ID gate before fine-tuning. No alternative checkpoint will be selected based on downstream mechanism results.

All D2–D6 results are ID. Only D8/10/12/16/20 are called OOD. Fine-tuning uses atomic plus D2–D5 official TRAIN data, 62,000 records, identically in both arms. We do not regenerate facts from a guessed seed.

## Dataset and original training metadata

Official dataset: `data/multi_hop`, 200 entities and ten permutation relations; reconstruct the 2,000 atomic transitions directly from released TRAIN facts. Actual relation counts determine depth because released training type labels are offset by two. Select evaluation records strictly by `Dhop_test`, 750 per evaluated depth. Full source hashes and download IDs are pinned in `download_official.py` and `data/integrity_check.json`.

The stored epoch is 2,764 (filename uses 2,765), final stored loss 0.01456954681985173, original optimizer base LR 1e-4 and weight decay .01. With the currently published linear-scheduler formula and warmup 2,000, the stored LR suggests a 15,001-epoch stage horizon; it does not match the source's current 100,001 default. We do not claim exact scratch-training configuration reconstruction from these incomplete metadata. The released weights themselves are used unchanged, with strict state-dict loading and finite-parameter checks.

Original checkpoint seed: unknown in the artifact; published source default is 42 but this is not proof of the artifact's seed. This experiment's fine-tuning, batch order and intervention selection seeds are fixed at 42; shared noise directions use 4242.

Evaluation has zero effective dropout. Both fine-tuning arms use all-zero dropout identically. The published model constructor zeros embedding dropout but leaves other GPT2 dropout defaults, while its training loop enters eval mode without restoring train mode each epoch. Our explicit zero-dropout post-training is recorded as an experiment setting, not claimed to reconstruct every original scratch-training detail.

V1/V2 checkpoint/results/logs remain preserved. V2 A1 was stopped at 27,500 D2 updates (49/375 validation correct) as scratch-training-cost evidence; it was not run to its 50,000-update cap and is not reported as a completed failure of official reproduction.
