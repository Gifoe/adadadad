# Loop-JET TUEV seed0 v1

Current result: **channel provenance audit completed; `NEED_RAW_TUEV`**. No JET16 conversion or formal training was performed. Both existing processed variants have ambiguous physical channel identities and reference schemes.

Read [data_audit/HF_CHANNEL_AUDIT.md](data_audit/HF_CHANNEL_AUDIT.md) and [data_audit/channel_audit.json](data_audit/channel_audit.json). The complete public text/history evidence and raw TUEV metadata subsets are in `data_audit/provenance_evidence.zip`, with per-file SHA-256 in its manifest. Large signal arrays and raw EDF/REC are not redistributed.

The frozen official code comes from [Y-Research-SBU/JET](https://github.com/Y-Research-SBU/JET) revision `07f9e6491796f4f2c717d6259b1a2e24afce6a77`, MIT. `SOURCE_MANIFEST.json` records source hashes. Official files have not been edited. The existing environment is torch2.8.0+cu128, h5py3.16.0 and pyarrow; preserve the working sm_120 torch build.

On the authorized server, the experiment lives at `D:\loop_jet_tuev_seed0_v1`. Activate the existing environment before Python to obtain its CUDA DLLs:

```bat
call E:\Anaconda\condabin\conda.bat activate E:\Anaconda\envs\persist_stable_251
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set PYTHONIOENCODING=utf-8
cd /d D:\loop_jet_tuev_seed0_v1
```

Executed audit stages:

```text
download_tuev.py                 # only pinned TUEV subtree, existing complete files cached
audit_data.py                    # actual 32-channel H5 shapes, labels, attrs, SHA
audit_provenance.py              # full Git text history; metadata-only raw manifest
inspect_manifest_schema.py
analyze_raw_manifest.py          # every indexed TUEV recording and layout
audit_variant22_headers.py       # bounded HTTP ranges; headers/labels only
audit_magnitude.py               # unscaled random samples; no unit/order inference
search_raw_sources.py            # bounded source search; no credentials read
package_provenance_evidence.py
write_channel_report.py
```

`write_channel_report.py` also uses `historical_tuev_mapping_fields.json` and `historical_h5_lfs_pointers.json`, preserved alongside the complete Git inventories for checking historical provenance. These identify unchanged H5 LFS OIDs and the older metadata's declared v2.0.1 source root.

Implemented model/loader checks, all executed on the RTX5090:

```text
sanity.py                        # K1 exact equivalence, K2 parameter count, both BF16 backward
check_loader.py                  # lazy worker handles, transpose, dtype, fixed held-out RNG
check_preflight.py               # exits 2 for incompatible current representation
```

The model and loader are prepared and exercised. The full training/trajectory/generation driver is pending the data gate. `naive_k2/NOT_RUN.json` and `ds_k2/NOT_RUN.json` enumerate missing experimental outputs. Synthetic smoke checks are not trained-model results or 200-step runtime estimates.

To unblock exact conversion, obtain a release-specific verified physical channel-index/reference mapping from the uploader's generation source and per-sample index, or an authorized raw TUEV EDF+REC source. Do not select the first16 channels, assume a standard order, infer channel identity from signal amplitude, or apply referencing/filtering without provenance.

Public metadata attribution: processed packages by USuCgex0122e; raw corpus manifest by tankalapavankalyan, whose dataset card identifies CC-BY-4.0. The underlying TUEV row identifies TUH-DUA. This delivery includes public metadata and audit observations, without redistributing raw clinical signals.
