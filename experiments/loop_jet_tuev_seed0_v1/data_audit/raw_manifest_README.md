---
license: cc-by-4.0
language:
- en
tags:
- eeg
- electroencephalography
- neuroscience
- brain-computer-interface
- foundation-model
- pretraining
- canonical-format
- zarr
- flac
- openneuro
- physionet
- tuh-eeg
task_categories:
- time-series-forecasting
- other
pretty_name: EEG Canonical Corpus Index
size_categories:
- 100K<n<1M
configs:
- config_name: datasets
  data_files: datasets.parquet
- config_name: recordings
  data_files: recordings.parquet
  default: true
- config_name: channels
  data_files: channels.parquet
- config_name: subjects
  data_files: subjects.parquet
---

# EEG Canonical Corpus Index

A queryable index over **79,313 hours of lossless EEG in a single unified
`.eegz` (Zarr v3) format**. The index ships as metadata only; the actual
signal bytes live as canonical Zarr stores on S3 and are read on demand.

| | |
|---|---:|
| Recordings (canonical, ready to load) | **185,802** |
| Hours of EEG (canonical) | **79,313.6** |
| Canonical on-disk size | **2.81 TB** |
| Subjects | **43,114** |
| Source datasets indexed | **411** |
| Apples-to-apples compression vs source EEG (rows with `canonical_size_bytes` set) | **3.05×** |

Every canonical row points at a Zarr-v3 group at
`s3://eeg-corpus-139156132535/canonical/v1/<dataset_id>/<recording_id>.eegz/`
with FLAC-compressed `int16` signal + parallel channel / events /
annotations arrays. Open any one through `manifest.eegz.open_recording`
or directly with `zarr.open(canonical_uri)` and get the same on-disk
shape regardless of source format (`.edf` / `.bdf` / `.set` / `.vhdr` /
`.fif` / `.cnt` / `.gdf`).

The original source files are kept at
`s3://eeg-datasets-139156132535/raw/` for **provenance + the 0.31%
residue** that did not round-trip (see [Non-OK recordings](#non-ok-recordings)).

## What's in here

The bundle is a star schema of four Parquet tables. The Dataset Viewer
above lets you browse each one interactively; you can also load them
programmatically.

| Table | Rows | Description |
|---|---:|---|
| `datasets`   | 411        | One row per source dataset. License, DOI, paradigm summary, modality, paradigm categories. |
| `recordings` | 186,386    | The fact table. One row per recording. `canonical_uri` points at the `.eegz` store for the 185,802 ok rows; `archival_uri` is provenance to the original source file. |
| `channels`   | 11,530,668 | One row per channel per recording. Name, type, units, electrode status, and 3D coordinates `(x, y, z)` when the dataset ships an `electrodes.tsv`. |
| `subjects`   | 43,114     | One row per hashed subject. Age, sex, clinical status, handedness. The hash is per-dataset (cross-dataset subject dedup is a separate concern). |

## Round-trip contract

The canonical writer asserts one of two contracts per recording before
upload:

| `roundtrip_class` | Source formats | Contract | Ok rows |
|---|---|---|---:|
| `bit_exact` | `.edf`, `.bdf` | Every digital sample equals the source byte-for-byte (within ≤1 LSB env-rounding tolerance). | 62,157 |
| `near_lossless` | `.set`, `.vhdr`, `.fif`, `.cnt`, `.gdf` | Per-channel reconstruction error ≤ 0.5 LSB of the digital→physical scale. Half-LSB is typically ~0.005 µV — well below the EEG noise floor. | 35,908 |
| (NULL) | (legacy) | Roundtrip class not recorded — the rows pre-date when the writer started persisting the class to the conversion ledger. The store itself passed the contract at write time. | 87,737 |

The conversion ledger persisted per-store metadata including
`canonical_sha256` (SHA-256 of `_eegz_integrity.json`) on 2,402 stores;
re-hashing those stores byte-for-byte returns 2,402 / 2,402 matches.
185,802 / 185,802 ok stores have a valid root `zarr.json` and `signal/zarr.json` on S3.

## Loading one recording

```python
from manifest.eegz import open_recording

uri = "s3://eeg-corpus-139156132535/canonical/v1/hbn_eeg/<recording_id>.eegz/"
rec = open_recording(uri)
print(rec.metadata.n_channels, rec.metadata.sampling_rate_hz, rec.metadata.duration_s)

# Lazy random-access slice — a single S3 GET fetches the relevant FLAC shard.
window = rec.signal[:, 60_000:120_000]   # samples 60k..120k across all channels
```

Or with pure Zarr:

```python
import zarr
store = zarr.open("s3://eeg-corpus-139156132535/canonical/v1/<dataset>/<rid>.eegz/", mode="r")
signal = store["signal"][:]              # (n_channels, n_samples)
sfreq  = store.attrs["sampling_rate_hz"]
chnames = store["channels/name"][:]
```

## Querying the manifest

### Pandas

```python
import pandas as pd
recordings = pd.read_parquet(
    "hf://datasets/tankalapavankalyan/eeg-corpus-manifest/recordings.parquet"
)
ok = recordings[recordings["conversion_status"] == "ok"]
hours_per_dataset = ok.groupby("dataset_id")["duration_s"].sum() / 3600
print(hours_per_dataset.sort_values(ascending=False).head())
```

### DuckDB (joins across all four tables, zero downloads beyond columns you touch)

```python
import duckdb
db = duckdb.connect()
db.sql("INSTALL httpfs; LOAD httpfs;")
BASE = "https://huggingface.co/datasets/tankalapavankalyan/eeg-corpus-manifest/resolve/main"

# Recordings ready to load, ordered by canonical store size.
db.sql(f"""
    SELECT dataset_id, recording_id, canonical_uri, roundtrip_class,
           canonical_size_bytes / 1e6 AS canonical_mb
    FROM '{BASE}/recordings.parquet'
    WHERE conversion_status = 'ok'
    ORDER BY canonical_size_bytes DESC NULLS LAST
    LIMIT 10
""").show()
```

### Polars (lazy, S3 URIs preserved)

```python
import polars as pl
rec = pl.scan_parquet(
    "hf://datasets/tankalapavankalyan/eeg-corpus-manifest/recordings.parquet"
)
(rec.filter((pl.col("conversion_status") == "ok") & (pl.col("sampling_rate_hz") >= 250))
    .group_by("dataset_id")
    .agg(pl.col("duration_s").sum().alias("seconds"),
         pl.col("recording_id").count().alias("n_rec"))
    .collect())
```

### `datasets`

```python
from datasets import load_dataset

ds = load_dataset("tankalapavankalyan/eeg-corpus-manifest", "recordings", split="train")
print(ds[0])
print(ds.features)
```

## Schema — `recordings` (the main fact table)

All identifier columns are stable across builds via UUIDv5 on the
archival URI.

| Column | Type | Notes |
|---|---|---|
| `recording_id` | string | UUIDv5 over `archival_uri`; stable across builds |
| `dataset_id` | string | FK → `datasets.dataset_id` |
| `canonical_uri` | string | `s3://.../canonical/v1/<dataset_id>/<recording_id>.eegz/` — Zarr v3 store. NULL when `conversion_status != 'ok'`. |
| `canonical_size_bytes` | int64 | On-disk size of the `.eegz` store (sum of chunk + metadata bytes). NULL on legacy stores enriched from S3 listings rather than ledgers. |
| `canonical_sha256` | string | SHA-256 of the store's `_eegz_integrity.json` blob. NULL on stores written before the integrity-blob scheme shipped. |
| `conversion_pipeline_id` | string | Pipeline version reported by the converter. |
| `conversion_status` | string, **never NULL** | `ok` (185,802) / `error_read` (568) / `error_download` (9) / `error_verify` (3) / `error_write` (1) / `not_yet_converted` (3). |
| `roundtrip_class` | string | `bit_exact` / `near_lossless` / NULL (legacy). |
| `conversion_dt_utc` | timestamp[UTC] | When the conversion finished. |
| `subject_id_in_dataset` | string | As in source (`sub-NDARxxx`, `sub-LTPxxx`, anon clinical code). |
| `subject_canonical_hash` | string | FK → `subjects.subject_hash`. |
| `session_id`, `run_id`, `task` | string | BIDS-style entities; nullable. |
| `archival_uri` | string | `s3://.../raw/...` pointer to the source file. Provenance; reading from here only makes sense for non-ok residue. |
| `archival_format` | string | `edf` / `bdf` / `set` / `vhdr` / `fif` / `cnt` / `gdf`. |
| `file_size_bytes` | int64 | Source file size from S3 listing; always populated. |
| `duration_s` | float64 | Aligned to canonical `zarr.json` for ok rows; inherited from BIDS sidecar otherwise. |
| `n_channels`, `n_eeg_channels` | int32 | Aligned to canonical `zarr.json` for ok rows. |
| `sampling_rate_hz` | float32 | Aligned to canonical `zarr.json` for ok rows. |
| `reference` | string | e.g. `Cz`, `common`. |
| `montage_name` | string | e.g. `HydroCel Geodesic Sensor Net`. |
| `channel_names` | list[string] | Inline, per recording. |
| `manufacturer` | string | EGI, BioSemi, … (sparse). |
| `power_line_frequency` | float32 | 50 / 60 Hz. |
| `recording_type` | string | `continuous` / `epoched`. |
| `modality` | string | `eeg` (room for `ieeg`, `meg` in future). |
| `bids_compliant` | bool | |
| `bids_entities`, `companion_uris` | JSON string | Stringified `map<string, string>`; parse with `json.loads`. |
| `header_source`, `header_read_status` | string | Provenance for the original BIDS-sidecar / EDF-header read. |

For converted recordings, `sampling_rate_hz` / `duration_s` /
`n_channels` / `n_eeg_channels` in `recordings.parquet` and the
corresponding attributes on the canonical store's `zarr.json` are
byte-for-byte identical across all 185,802 ok rows.

See the full schema in
[`manifest/schema.py`](https://github.com/ritivel/eegData/blob/main/manifest/schema.py)
on the companion code repo.

## Non-OK recordings

The 584 non-ok rows (0.31% of the corpus) break down as:

| Count | Dataset(s) | Reason | Recoverable? |
|---:|---|---|---|
| 226 | `openneuro_ds002181` | `.fdt` binary samples never uploaded by the publisher | ❌ permanent — nothing on S3 to convert from |
| 85  | `openneuro_ds004166` | corrupt `.set` metadata; EEGLAB reader handled 128/213 | ⚠️ partial — needs reader extension |
| 254 | scattered across 42 small datasets (1–39 each) | upstream source quirks; mostly corrupt `.set` metadata, missing companions, MNE-unreadable epoched files | ⚠️ case-by-case |
|   3 | `openneuro_ds004952` | read OK but failed signal-level verify | ⚠️ inspect why |
|   9 | various | S3 transient `error_download` | ✅ retry |
|   3 | `peers_memory` BDFs | shard worker terminated before processing them | ✅ relaunch one c6in.16xlarge for ~10 min |
|   3 | `peers_memory` (not_yet_converted) | same shard; never reached convert phase | ✅ same retry |
|   1 | scattered | `error_write` | ⚠️ inspect ledger |

For ML pretraining, filter `WHERE conversion_status = 'ok'` and you
have **79,313.6 hours of lossless EEG**.

## Upstream data-quality patterns

Documented patterns from the per-dataset conversion ledgers. None of
these are bugs in the canonical store — the stores either succeeded
under the patterns below or were excluded.

| Pattern | Example dataset | Auto-handled? |
|---|---|---|
| `.vhdr` missing `MarkerFile=` line | various | ✅ patched in-place by the writer |
| `.vhdr` references a differently-named `.eeg` companion | various | ✅ linked via the writer's internal `.vhdr` reference resolver |
| Numeric `RDA_*` BrainVision channels typed as `misc` | various | ✅ blanket-promoted to `eeg` when all channels carry that prefix |
| EEGLAB `.set` references missing `.fdt` | `ds002181`, `ds004306` | ❌ blocked — no upstream data |
| Corrupt `.set` metadata (sample-count mismatch, missing fields) | `ds003343`, `ds003645`, `ds004166` | ❌ blocked |
| EEGLAB MAT v7.3 (HDF5-backed) `.set` files | several OpenNeuro releases | ✅ handled via `pymatreader` |
| Epoched `.set` (no continuous signal) | `ds004519`, `ds004771`, `ds005946` | ✅ writer concatenates epochs into a continuous stream + per-epoch annotations |
| Mixed-modality recordings (EEG + ECG + EMG) | various PhysioNet | ✅ per-type unit map; non-µV-scalable channels recorded in `dropped_channels` |

## Corpus composition

The corpus mixes hand-curated anchor datasets, the OpenNeuro EEG bulk,
and a foundation-model benchmark surface.

### Anchor datasets

| Dataset | Format | Channels | sfreq | License |
|---|---|---|---|---|
| [HBN-EEG (Healthy Brain Network)](https://childmind.org/science/global-open-science/healthy-brain-network/) | EEGLAB `.set` | 129 (EGI HydroCel) | 500 Hz | CC-BY-SA-4.0 |
| [PEERS Memory EEG (ds004395)](https://openneuro.org/datasets/ds004395) | EDF / BDF | 129 (EGI), 137, 144, 272 (BioSemi) | 250–2048 Hz | CC0-1.0 |
| [TUH-EEG Corpus](https://isip.piconepress.com/projects/tuh_eeg/) (TUEG family) | EDF | 20–41 (clinical) | 250 / 256 / 400 / 512 / 1000 Hz | [TUH DUA](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml) |
| OpenNeuro EEG datasets (auto-ingested via `manifest.openneuro_catalog`) | mixed (`.edf` / `.set` / `.vhdr` / `.bdf` / `.fif`) | varies | varies | mostly CC0-1.0 |

### Foundation-model benchmark / downstream eval

The datasets every EEG foundation model paper reports on, so a model
trained on this corpus can be evaluated against the same tables LaBraM
/ CBraMod / REVE / DIVER-1 publish, without leaving the pipeline.

| Dataset | Source | Recordings | Subjects | Hours | License |
|---|---|---:|---:|---:|---|
| **PhysioNet Sleep-EDFx** — sleep cassette + telemetry | [PhysioNet](https://physionet.org/content/sleep-edfx/) | 197 | 197 | 3,849 | ODC-By |
| **PhysioNet HMC** — Haaglanden Medisch Centrum sleep staging | [PhysioNet](https://physionet.org/content/hmc-sleep-staging/) | 154 | 154 | 1,164 | ODC-By |
| **PhysioNet CAP** — Cyclic Alternating Pattern sleep | [PhysioNet](https://physionet.org/content/capslpdb/) | 108 | 108 | 993 | ODC-By |
| **PhysioNet CHB-MIT** — pediatric seizure | [PhysioNet](https://physionet.org/content/chbmit/) | 665 | 23 | 962 | ODC-By |
| **PhysioNet Siena** — adult scalp epilepsy | [PhysioNet](https://physionet.org/content/siena-scalp-eeg/) | 38 | 14 | 130 | ODC-By |
| **PhysioNet MMI** (EEGMMIDB) — motor imagery | [PhysioNet](https://physionet.org/content/eegmmidb/) | 1,526 | 109 | 49 | ODC-By |
| **Mumtaz** — MDD vs healthy controls | [figshare 4244171](https://figshare.com/articles/dataset/EEG_Data_New/4244171) | 180 | 64 | 20 | CC-BY-4.0 |
| **PhysioNet EEGMAT** — mental arithmetic | [PhysioNet](https://physionet.org/content/eegmat/) | 72 | 36 | 2 | ODC-By |
| **TUAB** — TUH Abnormal | TUH NEDC | 2,993 | 2,329 | 1,141 | TUH DUA |
| **TUAR** — TUH Artifact | TUH NEDC | 310 | 213 | 100 | TUH DUA |
| **TUEV** — TUH Events | TUH NEDC | 518 | 370 | 149 | TUH DUA |

PhysioNet datasets resolve via the [PhysioNet AWS Open Data mirror](https://registry.opendata.aws/physionet/)
(`s3://physionet-open/`); each is re-mirrored into the home bucket for
same-region reads.

## Licensing

The manifest itself (this Parquet bundle) is released under **CC-BY-4.0**.
The underlying EEG signals are governed by their original licenses, and
those licenses apply equally to the canonical `.eegz` stores derived
from them.

| Source family | Signal license | Access requirements |
|---|---|---|
| HBN-EEG | CC-BY-SA-4.0 | Open; cite Shirazi et al. 2024 |
| PEERS ds004395 | CC0-1.0 | Open |
| OpenNeuro bulk | almost all CC0-1.0 | Open; cite the original upload per OpenNeuro convention |
| TUH-EEG family (TUEG, TUAB, TUAR, TUEV, TUSZ, TUEP, TUSL) | TUH DUA | **Signed Data Use Agreement required** via [NEDC](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml); the manifest only indexes metadata. |
| PhysioNet (MMI, Sleep-EDFx, CHB-MIT, Siena, CAP, EEGMAT, HMC) | ODC-By | Open; cite each PhysioNet record per Citation file |
| Mumtaz (figshare 4244171) | CC-BY-4.0 | Open; cite Mumtaz et al. 2017 |

In particular, **the manifest does not redistribute TUH signal data**
through the canonical store either. Those rows record file-level
metadata only; `canonical_uri` points at a same-bucket store gated by
the same DUA.

## Companion code

The full pipeline that produces this corpus — schema definitions, S3
walkers, BIDS sidecar parser, EDF/BDF direct header parser, OpenNeuro
mass-sync, canonical `.eegz` writer + reader + verifier, EC2
orchestrator — lives at **[ritivel/eegData](https://github.com/ritivel/eegData)**.

```bash
# Convert one recording end-to-end (no S3 upload).
python -m manifest.eegz convert-one \
    --recording-id sub-001-ses-1 \
    --dataset-id   physionet_eegmat \
    --source-uri   s3://eeg-datasets-139156132535/raw/.../Subject00_2.edf \
    --source-format edf \
    --subject-id-in-dataset Subject00 \
    --no-upload

# Build the published manifest from the per-dataset conversion ledgers + canonical zarr.json.
python -m manifest.eegz_to_manifest \
    --input  s3://eeg-corpus-139156132535/manifest/latest/recordings.parquet \
    --output s3://eeg-corpus-139156132535/manifest/latest/recordings.parquet \
    --also-publish-latest

# Push the manifest to this Hugging Face dataset repo.
python -m manifest.publish_hf
```

Architectural notes and on-disk layout details are in
[`docs/DESIGN.md`](https://github.com/ritivel/eegData/blob/main/docs/DESIGN.md);
operator firefighting is in
[`manifest/eegz/PLAYBOOK.md`](https://github.com/ritivel/eegData/blob/main/manifest/eegz/PLAYBOOK.md).

## Citation

If this corpus accelerates your work, a one-line acknowledgment is
appreciated and **please also cite the underlying source datasets**:

```bibtex
@misc{eeg_canonical_corpus,
  author = {Pavan Kalyan Tankala},
  title  = {EEG Canonical Corpus Index: A unified .eegz store over open-access EEG pretraining corpora},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/tankalapavankalyan/eeg-corpus-manifest}},
}
```
