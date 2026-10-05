# demo_dataset

This repository contains a curated collection of processed EEG benchmark datasets packaged for model training and evaluation.

Each dataset is stored in its own directory and follows a consistent split-based HDF5 layout:

- `train.h5`
- `val.h5`
- `test.h5`
- `croissant.json`
- `croissant_manifest.csv`
- `metadata_summary.json`

## Repository structure

Each dataset directory contains:

- `train.h5`, `val.h5`, `test.h5`
  - HDF5 files containing the packaged split data.
  - In most datasets, `X` stores EEG tensors and `y` stores labels.
  - Some blind-test releases may omit `y` in specific splits.
- `croissant.json`
  - Dataset-level Croissant metadata for discoverability and validation.
- `croissant_manifest.csv`
  - Split-level manifest describing file names, sample counts, channel counts, and timepoints.
- `metadata_summary.json`
  - Compact summary of task type, label vocabulary, channel count, sampling rate, window length, and split sizes.

## Data format

For a typical split file:

- `X.shape = (n_samples, n_channels, n_timepoints)`
- `y.shape = (n_samples,)` when labels are available

The exact sampling rate, window length, and channel count differ by dataset and are documented in:

- `metadata_summary.json`
- `croissant.json`

## Included datasets

This repository currently includes 60 packaged datasets:

- `AD65`
- `ADHD`
- `Awakening`
- `Awakening_balanced`
- `BCIC2A`
- `BCIC4_1`
- `BCI_Speech`
- `BETA_SSVEP`
- `BenchmarkSSVEP`
- `Broderick_CP`
- `Broderick_CP_128`
- `Broderick_Rev`
- `Broderick_Rev_128`
- `CHB_MIT`
- `CIRE`
- `ChineseEEG2_RA_Tone`
- `DEAP`
- `DUAL_FREQ_SSVEP`
- `Depression_rest`
- `EAV`
- `EEGMAT`
- `EEG_Mortality_PD`
- `EEG_SVRec`
- `ExoEEG_WalkStop`
- `FACED_new`
- `HBN_EEG`
- `HFO`
- `HMC`
- `ISRUC_S1`
- `ISRUC_S2`
- `ISRUC_S3`
- `LEMON_age`
- `LEMON_extraversion`
- `LEMON_gender`
- `Longitudinal_EEG_Reliability`
- `MDD`
- `MODMA`
- `MonitoringErrP`
- `MusicEEG`
- `PD31`
- `PEARL_Neuro`
- `Physionet_MI`
- `RestCog`
- `SEED`
- `SEEDIV`
- `SEED_FRA`
- `SEED_V`
- `SEED_VIG`
- `SEED_VII`
- `SHU_MI`
- `SSVEP`
- `Siena_EEG`
- `SleepEDF_full`
- `TDBRAIN`
- `TUAB`
- `TUEP`
- `TUEV`
- `TUSL`
- `Things_EEG2`
- `Workload`

## Metadata policy

The packaged metadata intentionally avoids internal mount paths, usernames, and local infrastructure identifiers.

Where possible, provenance in `croissant.json` points to public references such as:

- original dataset papers
- official dataset release pages
- PhysioNet / OpenNeuro / institutional dataset pages

## Notes

- This repository is a processed benchmark collection, not a raw-data release.
- Temporal segmentation is dataset-specific and is not forced to be uniform across all datasets.
- Some datasets in the same family may use different window lengths if their benchmark packaging differs.
- For exact task definitions, label mappings, and window parameters, use `metadata_summary.json`.

## Recommended usage

1. Read `metadata_summary.json` before training on a dataset.
2. Use `croissant_manifest.csv` to inspect split sizes and basic tensor dimensions.
3. Use `croissant.json` if you need machine-readable dataset metadata or want to run Croissant validation.

