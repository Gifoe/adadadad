# Issue log

## 2026-10-11: local HTTPS proxy configuration

Command: requests.get(arxiv HTML) with inherited proxy environment. Original error: `ProxyError: Unable to connect to proxy; SSLEOFError`. The inherited proxy scheme was incompatible. Metadata downloads use `Session.trust_env=False` with explicit HTTP proxy locally; server downloads try HTTPS endpoints directly. The web tool verified paper contents independently. No model or protocol change.

## 2026-10-11: missing empty output directory

Command: `run_latenttsf.py --dataset ETTh1 --backbone DLinear --horizon 96`, attempt 1. Original traceback saved in `logs/ETTh1_DLinear_H96_attempt1.log`: `FileNotFoundError ... patches/latenttsf_audit_resume.diff`. File-only transport did not create empty directories. The wrapper now creates the patch directory before writing. Training had not started. Verified by rerunning the command; original log retained.

## 2026-10-11: audit-only timestamp parsing

Command: `download_and_verify_data.py`, original traceback in `data_download.log`: time-order assertion failed for NP/PJM. The audit used pandas pivot on string dates, which sorted by strings rather than datetimes. Parse timestamps before the audit pivot, preserving recorded source column order. Original model loaders and CSV bytes are unchanged. Both LatentTSF datasets independently verified: unique chronological hourly timestamps, ETTh1 17420x7, Electricity 26304x321.

## 2026-10-11: Windows local text decoding

After successfully creating the WorldTS source archive, an informational line-count command used platform-default GBK and failed on a UTF-8 source comment. Read source with explicit UTF-8. Source archive and Git blobs were intact; no training impact.

## Source corrections to task

LatentTSF horizon references come from Appendix D Table 6. Appendix E Table 7 is an AE ablation table. The values requested in the task are retained and MAE is read directly from Table 6.

WorldTS official numerical-covariate source was released on 2026-10-09 at commit `269426ea6217d7152a244da0e209d922acd46d4f`. Official source and dataset-specific published commands supersede the task's fallback assumptions. No independent model substitution is needed.

## 2026-10-11: LatentTSF optional dependency and W&B API

Attempt 2 failed during import: `ModuleNotFoundError: patoolib` from the unused M4 loader. Install `patool` in the private venv. Attempt 3 reached model creation but current W&B Settings rejects `start_method="thread"`. Remove only that deprecated logging option in the instrumented source. `WANDB_MODE=disabled` keeps experiment logs local. Original model and loss are untouched; both original tracebacks retained.

## 2026-10-11: Windows path length

Attempt 4 reached the correctly loaded frozen AE and forecaster. Official metadata-rich setting names caused `FileNotFoundError [WinError 3]` at checkpoint-directory creation under Windows MAX_PATH. The wrapper uses a short unique run ID for filesystem setting names and archives full arguments in JSON. No training or metric changes. Attempt 5 reruns after this fix.

## 2026-10-11: pandas audit index

The timestamp audit also used `Series[-1]`, which is label lookup under pandas 2. Convert timestamp Series to DatetimeIndex for positional first/last inspection. All four data shape/hash/time/split audits pass after this fix. Model loaders were unchanged.

## 2026-10-11: previous DAG source line endings

DAG checkpoint reuse audit initially failed bytewise source comparison at `ts_benchmark/pipeline.py`. The old source archive was taken from a Windows Git checkout (CRLF); the new archive uses exact Git blobs (LF). Audit now records both raw hashes and requires exact byte equality after only CRLF-to-LF normalization for every Python source file. It does not strip whitespace or tolerate substantive changes. Original failure logs retained; raw dataset/checkpoint hashes remain exact.
## 2026-10-11: WorldTS shell variable expansion

Command: run_all.py --phase worldts, first attempt. The official Bash command contains --gpus "${GPU}". Parsing argv directly on Windows retained the literal string; argparse rejected it as an integer. The wrapper now expands this execution-only argument to GPU 0. Model/preflight already passed. Original tracebacks remain in logs/PJM_H24_seed2021_attempt2.log and phase_worldts_attempt1.log.

## 2026-10-11: official WorldTS checkpoint filename

After real PJM Stage1/Stage2 training, the wrapper asserted that stages/final.pt existed. Official _run_stage_train resets final_checkpoint_path to stages/stage2_decoder_mix.pt. Read the actual resolved path instead of assuming the initial command string is retained. The two completed stages were loaded, stage-boundary RNG restored, and evaluation completed without repeating training. Actual MSE 0.05751596967883973, MAE 0.14278754282707617; official versus independent maximum difference 1.4476780330063832e-9. Original failure and success logs are retained. Initial training peak GPU memory was not persisted before the error, so the resumed memory observation is explicitly scoped and cannot stand in for the missing initial peak.

## 2026-10-11: fresh DAG runner preflight

The CPU-only fresh-run preflight built DAGModel before calling the official adapter's criterion initialization and hit AttributeError: Config has no attribute criterion. Call adapter._init_criterion(), as original forecast_fit does. Second preflight passed published-command parsing, training-hook compilation and [2,24,1] CPU forward. This affected only the additional runner preflight; the four verified DAG results were unaffected.

## 2026-10-11 22:24–22:30: remote connection loss

Existing SSH foreground session closed with Windows socket error 10054. Two new SSH attempts timed out at TCP connect. Read-only Tailscale status shows DESKTOP-C4FOPGI / 100.94.171.11 offline, last seen 2026-10-11T14:20:00.1Z. Local Tailscale is Running with no health errors. No remote reboot or process termination was issued. The server had completed WorldTS PJM24 and was training NP24; its last observed Stage1 epoch was 5. Server recovery is required to read its current epoch snapshot and finish the remaining three WorldTS configurations. A follow-up fresh DAG NP24 run is prepared to close the old original stdout gap. The project is not marked complete.


## 2026-10-11 22:55: remote connection recovered

SSH connected again after the user requested another check. Read-only LastBootUpTime was 2026-10-11 18:57:56, before the outage; no reboot occurred during the interruption. The original sequential job kept running: all four WorldTS metric files, histories and official TFB records were read and SHA-verified after recovery. No completed stage was repeated. The old interrupted delivery is retained as historical evidence and superseded by the final 24-configuration report.

## 2026-10-11 22:56: additional DAG epoch snapshot placement

Command: run_dag_fresh.py --dataset NP --horizon 24. Original traceback is retained in logs/DAG_fresh_NP_H24_seed2021_attempt1.log: AttributeError, DAG object has no attribute check_point. The audit wrapper inserted an epoch snapshot after a matching in-batch TST scheduler block, before the first validation initialized the best checkpoint. Anchor insertion explicitly to the end-of-epoch `if self.config.lradj != "TST"` block, after validation, early stopping and learning-rate adjustment. The exact instrumentation diff is patches/dag_audit_resume.diff. Attempt 2 records validation and resumable model/optimizer/RNG/best-checkpoint state. This fixes observation/resumption only; the original model, batch, loss, validation selection and seed remain unchanged. Verified old NP24 artifacts are preserved separately.

## 2026-10-11: clarify resumed PJM24 resource observation

The WorldTS wrapper measures peak memory at the end of forecast_fit, before independent inference. Correct the old wording `resumed fit/evaluation` to `resumed fit before independent evaluation`. Only the descriptive scope fields in metrics.json/resource_audit.json change; the measured 19.78173828125 MB, model/prediction hashes, scores and original training time are unchanged. The first training peak is unavailable and remains reported as unavailable, rather than treating the resumed measurement as its replacement.
