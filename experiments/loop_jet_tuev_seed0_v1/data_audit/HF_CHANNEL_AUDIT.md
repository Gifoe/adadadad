# HF TUEV Channel Provenance Audit

**结论：NO。当前公开证据不能唯一恢复两个 HF 张量的物理通道身份、顺序及 reference；决定为 `NEED_RAW_TUEV`。**

审计已执行，严格 JET conversion 和正式 Loop-JET 训练均未执行。该结论是数据可识别性结论，不是对 Loop-JET 研究假设的否定。审计日期：2026-10-05。

## 1. Processed HF variants found

| Variant | Repository revision | Actual X shapes (train / val / test) | Confidence |
|---|---|---|---|
| 22 channels | `USuCgex0122e/demo_dataset_seed20260726@41602cf9363d83204e6b7145b04efaeafebd9ae4` | [8370, 22, 1000] / [2496, 22, 1000] / [1684, 22, 1000] | C — Ambiguous |
| 32 channels | `USuCgex0122e/demo_dataset@db96e77affa66398c553427045ee040f06230c2a` | [16409, 32, 1000] / [13206, 32, 1000] / [12908, 32, 1000] | C — Ambiguous |

两个版本的真实 H5 均只有 X/y，文件与数据集 attrs 为空。32 通道全文件下载并以发布的 LFS SHA-256 校验；22 通道只读取 HTTP Range 返回的 headers 和 labels，共 317976 字节，没有下载 X 信号。

## 2. 22-channel variant

- 源仓库 `USuCgex0122e/demo_dataset_seed20260726`；revision `41602cf9363d83204e6b7145b04efaeafebd9ae4`。
- train/val/test = 8,370 / 2,496 / 1,684；X float32，y int32，六类 0–5。metadata 声称 subject-disjoint，split seed=20260726；未获得每样本 subject ID，无法独立验证此声明。
- 源原始数据版本未知；全部 21 个可达 commit 没有任何 Python preprocessing source。
- channel 0–21 的物理身份全部 unknown；reference / montage / unit 全部 unknown。不能将其称为 referential 或 bipolar。
- metadata 声明 200 Hz、5 s、1000 点、0.3–75 Hz、60 Hz notch；没有实际处理代码证明滤波执行。`filtering execution provenance unverified`，不重复滤波。
- `export_summary.json` 当前内容涉及另外六个数据集，没有 TUEV 的 channel mapping。

## 3. 32-channel variant

- 源仓库 `USuCgex0122e/demo_dataset`；revision `db96e77affa66398c553427045ee040f06230c2a`。
- train/val/test = 16,409 / 13,206 / 12,908；X float32，y int32，六类 0–5。三个 H5 共 925,966,698 字节，完整 SHA 校验记录见 `h5_audit.json`。
- 历史 commit `281c4515d354ffe86a2e2f6bf00159c9df7c1002` 的 metadata / Croissant 指向 `/vePFS-0x0d/eeg-h5/raw/TUEV/v2.0.1/edf`；这是源版本 v2.0.1 的历史声明。三份 H5 的 LFS OID 在完整历史中没有变化，因此这不是另一个 H5 版本。它仍不能证明每列的物理通道身份。
- channel 0–31 的物理身份全部 unknown；reference / montage / unit 全部 unknown。
- 唯一 Python 文件 `generate_croissant.py` 生成 metadata，不读取 EDF 生成 H5。其第 470–481 行从 `dataset_config.json.channel_names` 或 `sample_index.parquet.channel_names_json` 取名字，第 639–643 行还读取 conversion/validation summaries。
- 这些逐样本索引和生成配置在全部公开 commit 的文件树中均不存在。H5 的原始 EDF / REC 文件名、subject、recording ID、channel-index mapping 也均未保存。
- 200 Hz / 5 s / 1000 点及 filtering 参数是 metadata 声明；filter 执行代码仍未验证。

### 完整 provenance 搜索范围

| Repository | Commits | Current text files | Unique historical text blobs | Historical Python paths |
|---|---:|---:|---:|---|
| USuCgex0122e/demo_dataset | 22 | 187 | 303 | generate_croissant.py |
| USuCgex0122e/demo_dataset_seed20260726 | 21 | 75 | 76 | none |

搜索全部历史与当前 `.py/.json/.yaml/.yml/.csv/.txt/.md`；完整 inventory、原始文本和 keyword hits 保存于 `provenance_evidence.zip`，各文件 SHA 见 `provenance_evidence_manifest.json`。没有只检查 README 或单一 commit。两个 Git bare clone 与 manifest metadata 下载均通过 hf-mirror.com 完成；镜像 commit 与官方当前 commit 一致，完整祖先历史可达。

## 4. Raw TUEV metadata from eeg-corpus-manifest

来源 `tankalapavankalyan/eeg-corpus-manifest@e4c2d62a9230d053c948a6681a63d1ca0d3f353e`，严格筛选 `dataset_id=tuh_eeg_events`。只下载 metadata Parquet，不读取原始 EDF 或 S3 signal。

- recording IDs：518；subjects：370；有序 channel layouts：15。
- n_channels 分布：`{"24": 20, "25": 14, "27": 122, "28": 47, "30": 2, "31": 102, "32": 24, "33": 187}`。
- sampling rate：`{"250.0": 518}`，全部原始 recording 为 250 Hz。
- reference 字段：`{"<missing>": 518}`；montage 字段：`{"<missing>": 518}`，均缺失。
- 名字多为 `EEG FP1-REF` 等，表现为原始 reference 标签；仅凭此不推断 HF 输出的 reference 类型。
- 必需电极覆盖：518/518 = 100.0%；只做 upper-case、EEG prefix、末尾 -REF/-LE 清理。没有合并 bipolar 名字或使用 T7/T3 等别名猜测。
- channel 表与 recording 表的 order mismatch：0；normalization collision：0。
- 原始 channel units：`{"uV": 15600, "bpm": 1}`；这些单位不能迁移为 HF processed unit。
- 至少前 20 条 recording 的 ID、subject、channel names、rate、reference、montage 已保存于 `raw_tuev_first20.json/.csv`。全部 518 recording 和 15,601 channel rows 在 evidence archive 中。

| Layout SHA prefix | Frequency | n_channels | First two channel labels | Required electrodes |
|---|---:|---:|---|---|
| 31e365da4d35 | 185 | 33 | EEG FP1-REF / EEG FP2-REF | all |
| cf9b40d718fd | 122 | 27 | EEG FP1-REF / EEG FP2-REF | all |
| ce58bd666cda | 81 | 31 | EEG FP1-REF / EEG FP2-REF | all |
| c3513262b401 | 40 | 28 | EEG FP1-REF / EEG FP2-REF | all |
| 4c4e65d10bae | 21 | 31 | EEG FP1-REF / EEG FP2-REF | all |
| ee766ad33f14 | 19 | 24 | EEG FP1-REF / EEG FP2-REF | all |
| 7ecfe65b8b44 | 17 | 32 | EEG FP1-REF / EEG FP2-REF | all |
| 097cf294c7fe | 9 | 25 | EEG FP1-REF / EEG FP2-REF | all |
| 442ee4fddcef | 7 | 28 | EEG FP1-REF / EEG FP2-REF | all |
| 6e8486291c62 | 6 | 32 | EEG FP1-REF / EEG FP2-REF | all |
| aaf7d116dcc4 | 5 | 25 | EEG FP1-REF / EEG FP2-REF | all |
| caba04304b0f | 2 | 30 | EEG FP1-REF / EEG FP2-REF | all |
| e7bf5c1dd19d | 2 | 33 | EEG FP2-REF / EEG FP1-REF | all |
| b9ee11543658 | 1 | 24 | EEG FP1-REF / EEG FP2-REF | all |
| f7a63f931e15 | 1 | 32 | EEG FP1-REF / EEG FP2-REF | all |

**两个 recording 的 layout `e7bf5c1dd19...` 从 `FP2, FP1, F4, F3, ...` 开始，常见 layout 从 `FP1, FP2, F3, F4, ...` 开始。** 因此即使原始 recording 全部具备所需电极，也不能把常见 raw order 当作通用 processed order。

## 5. Can JET 16 bipolar channels be reconstructed?

**当前不能。** 缺的是 release-specific 的 `processed tensor index → physical channel/derivation + reference` 证据链。原始 EDF channel names 已知，但 processed sample 到 EDF 没有唯一对应；也没有 preprocessing source 证明保留了原始 channel order。两个版本均为 Level C，不是已证明缺少电极的 Level D。

32 与 22 通道、sample counts、subjects 均不同，不能认定只是同一张量的重新 split，也不能用某一版本的潜在 order 代替另一版本。

## 6. Selected source variant

**None。** 未生成 `data/tuev_jet16/`，未选择前 16 通道，未假设 10–20 或 EDF 顺序，未重复 referencing / filtering。

## 7. Exact conversion equations

目标顺序如下。它们是 JET 的目标定义，尚未绑定任何 HF index，因此没有执行减法或 selection：

```text
00: FP1-F7
01: F7-T3
02: T3-T5
03: T5-O1
04: FP2-F8
05: F8-T4
06: T4-T6
07: T6-O2
08: FP1-F3
09: F3-C3
10: C3-P3
11: P3-O1
12: FP2-F4
13: F4-C4
14: C4-P4
15: P4-O2
```

若后续 A/B 证据确认 referential/raw electrode channels，按上列同源 electrode 做差；若明确为 bipolar，只按已验证的完整 derivation 名称选取。当前 reference 类型为 D. unknown，两个分支均不满足。

## 8. Remaining uncertainties

关键缺失证据只有一个：**与当前发布的 H5 绑定的物理 channel-index/reference 契约**。可通过精确 HDF5 generation source + explicit channel list/order，或 aligned source EDF index + explicit order-preservation source补齐；不能以 raw corpus 常见顺序替代。
单位 / normalization、filtering execution、22-channel raw version 和 paper split equivalence 也尚未验证。JET 官方 EDF preprocessing 明确 `get_data(units="uV")`，官方 loader 再 `/100.0`；HF H5 和公开 metadata 没有 signal unit 字段。
32-channel 固定随机 10 samples/split 的未缩放 min/max/mean/std、percentiles、channel-wise std 见 `unscaled_magnitude32.json`。看到全零 sample-channel 对不等于证明 padding；幅度不用于猜单位或通道。22-channel 本轮只读 headers/labels，未额外读取 X，unit 同样 unknown。
服务器源文件搜索访问 17337 个目录；范围为 `D/E/F directories to depth 4; to depth 12 only inside TUH/TUEV/event paths. Skip system/software/cache trees. No file contents read.`。发现的 TUH/TUEV/event raw EDF+REC 为 0，processed 文件只有本次下载的 32 通道 H5。该搜索有明确深度边界，不声称证明整机没有 raw 数据。
conversion sanity（100 样本、5 张转换图）不适用，因为禁止执行尚无 provenance 的转换；没有制作伪转换数据或伪检查图。

## 9. Final decision

**`NEED_RAW_TUEV`。NO under currently available defensible provenance。**

channel audit 已完成；正式长训练保持停止。恢复可由 uploader 提供上述映射契约完成，也可在具有授权的 raw EDF+REC 上走官方 preprocessing。当前没有把 22/32 通道输入当作严格 JET 16 双极导联。

### Primary sources / attribution

- [Processed 32-channel repository](https://huggingface.co/datasets/USuCgex0122e/demo_dataset/tree/db96e77affa66398c553427045ee040f06230c2a)
- [Historical metadata with v2.0.1 source root](https://huggingface.co/datasets/USuCgex0122e/demo_dataset/blob/281c4515d354ffe86a2e2f6bf00159c9df7c1002/TUEV/metadata_summary.json)
- [Metadata generator, not H5 preprocessing](https://huggingface.co/datasets/USuCgex0122e/demo_dataset/blob/db96e77affa66398c553427045ee040f06230c2a/generate_croissant.py)
- [Processed 22-channel repository](https://huggingface.co/datasets/USuCgex0122e/demo_dataset_seed20260726/tree/41602cf9363d83204e6b7145b04efaeafebd9ae4)
- [EEG corpus manifest](https://huggingface.co/datasets/tankalapavankalyan/eeg-corpus-manifest/tree/e4c2d62a9230d053c948a6681a63d1ca0d3f353e), publisher tankalapavankalyan, metadata dataset card CC-BY-4.0; raw TUEV row identifies TUH-DUA. Only public metadata subsets are redistributed.
- [Official JET preprocessing](https://github.com/Y-Research-SBU/JET/blob/07f9e6491796f4f2c717d6259b1a2e24afce6a77/data/preprocess_tuev.py), MIT; snapshot retained under `official/`.
