# Reproduce this server experiment

Windows command context:
```bat
call E:\Anaconda\condabin\conda.bat activate E:\Anaconda\envs\persist_stable_251
cd /d D:\neso_worldts_online_v1
set TASK_PYTHON=D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe
%TASK_PYTHON% -m pip install -r requirements.txt
%TASK_PYTHON% -m pytest -q
git clone --filter=blob:none --no-checkout https://github.com/decisionintelligence/WorldTS.git upstream/WorldTS
git -C upstream/WorldTS checkout 269426ea6217d7152a244da0e209d922acd46d4f
%TASK_PYTHON% -u -m neso_online.download --resources forecast_2025 forecast_2026_jan_jun forecast_2026_jun_dec demand_2025 demand_2026 --out data/raw --allow-unavailable
%TASK_PYTHON% -u -m neso_online.prepare --forecast data/raw/forecast_2025.csv data/raw/forecast_2026_jun_dec.csv --demand data/raw/demand_2025.csv data/raw/demand_2026.csv --lookback 336 --horizon 48 --revision-hours 6 --exclude-settlement-mismatch --out outputs/dataset
%TASK_PYTHON% -u scripts/audit_dataset.py
%TASK_PYTHON% -u scripts/native_smoke.py
%TASK_PYTHON% -u -m neso_online.native_worldts train --data outputs/dataset --out outputs/model --worldts-source upstream/WorldTS/DAG-bench --config configs/worldts_neso.json --device cuda --batch 16 --seed 2021 --epochs1 20 --epochs2 20 --patience 5
%TASK_PYTHON% -u scripts/audit_coverage_gap.py
%TASK_PYTHON% -u scripts/diagnose_selected_model.py
%TASK_PYTHON% -u scripts/analyze_replay.py val
%TASK_PYTHON% -u scripts/benchmark_compute.py
```

Read `outputs/model/updater_gate.json`. In this actual execution `allow_updater=false`, so updater training is correctly skipped. For an independently reproduced run whose predeclared gate is true, the only allowed extra model is the supplied updater with epochs20/patience5; analyze its Validation predictions and repeat the same compute benchmark before freezing. Do not fit extra architectures or retune on Test.

Final frozen stage:
```bat
%TASK_PYTHON% -u scripts/freeze_validation.py
%TASK_PYTHON% -u -m neso_online.native_worldts replay --data outputs/dataset --out outputs/model --worldts-source upstream/WorldTS/DAG-bench --config configs/worldts_neso.json --device cuda --batch 16 --seed 2021
%TASK_PYTHON% -u scripts/analyze_replay.py test
%TASK_PYTHON% -u scripts/finalize_artifacts.py
```

Do not rerun the final replay in the completed experiment: the driver refuses overwriting `test_replay.json`. A genuine replication uses an independent fresh directory and preserves this locked result. Training resumes from raw completed epoch state if interrupted; selected best checkpoints are separate. Existing completed native training must not be restarted to choose a different result.

Verify official Git HEAD, original ZIPSHA, raw downloaded SHA and the artifact manifest before comparison. Downloader validates cached files; current live NESO data may later change. Exact replication requires the retained server snapshot hashes, not an arbitrary later download. Source/publication caveats and missing2026summer first-period vintages are part of the benchmark definition.

Server launcher paths are environment-specific. Linux may invoke its own compatible Python directly, but retain the source/config/data hashes, seed, batches, losses and stage selection. Requirements must not downgrade RTX5090 CUDA-compatible Torch. Export only compact code/reports/CSV/logs/figures; large raw data, weights and prediction tensors remain server-side with manifest pointers.
