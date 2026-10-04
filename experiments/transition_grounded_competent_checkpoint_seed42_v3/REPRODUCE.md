# Verify or reproduce the recorded run

Run these commands from this experiment directory after checkout:

```text
git lfs pull
python -c "import zipfile; zipfile.ZipFile('archives/verified_selected_official_dataset.zip').extractall('.')"
python verify_publication.py
```

The verifier recomputes binary hashes, archive CRCs, canonical data equality, counterfactual paths, recorded rollout accuracies, paired intervention IDs and 5,000-row histories. The diagnostic archive may remain compressed: it contains raw pretrained carriers/logits, all three prototype sets and shared noise directions. Source hashes preserve the actual Windows line endings through `.gitattributes`; source files should not be reformatted before this byte-level audit.

To rerun the GPU experiment, create a separate empty directory and copy the experiment's Python source files, `official/`, specification and provenance/method documents into it. Do not copy completed `outputs/`, `data/` or fine-tuned `checkpoints/`: those are resume/completion markers and deliberately prevent repeating or overwriting the recorded run. In a compatible GPU environment execute `download_official.py`, `inspect_artifacts.py`, **`data_pipeline.py`**, then `run_experiment.py`. Explicit data preparation avoids treating checked-in integrity manifests as proof that the uncompressed arrays exist. The original official artifact is downloaded using its pinned ID and SHA-256; the training and intervention plans are reconstructed from fixed seeds.

Recorded environment: Python3.10.20, PyTorch2.8.0+cu128, transformers4.44.2 and RTX5090. The process affinity workaround in `run_server.cmd` is specific to the observed Windows server faults; use the appropriate environment on another host and record it. All scientific configuration remains fixed in `CONFIG.json` and `METHOD_DIFF.md`.
