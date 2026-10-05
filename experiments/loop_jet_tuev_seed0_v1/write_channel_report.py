"""Render the observed channel audit; missing provenance never becomes a mapping."""
import csv
import hashlib
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT/'data_audit'

def load(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))

def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

def main():
    raw = load('raw_tuev_summary.json')
    history = load('provenance_search_summary.json')
    h32 = load('h5_audit.json')
    h22 = load('variant22_h5_header_audit.json')
    source_search = load('raw_source_file_search.json')
    magnitude = load('unscaled_magnitude32.json')
    historic = load('historical_tuev_mapping_fields.json')
    pointer_history = load('historical_h5_lfs_pointers.json')
    variants = []
    for key, n, h5 in [('alternate_22', 22, h22), ('original_32', 32, h32)]:
        meta_path = OUT/'processed_provenance'/key/'current'/'TUEV'/'metadata_summary.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        assert meta['channel_names'] == []
        variants.append({
            'variant': key, 'source_repository': history[key]['repo'],
            'source_commit': history[key]['pinned_revision'], 'n_channels': n,
            'sampling_rate': meta['sampling_rate_hz'], 'window_seconds': meta['window_length_seconds'],
            'timepoints': meta['timepoints_per_sample'], 'source_preprocessing_script': None,
            'source_raw_dataset_version': 'v2.0.1, declared by unchanged-H5 historical metadata' if n == 32 else 'unknown',
            'filtering': {'metadata_claim': '0.3-75 Hz bandpass; 60 Hz notch', 'source_code_verified': False,
                          'status': 'metadata-declared; filtering execution provenance unverified; do not refilter'},
            'channel_order': [],
            'channel_index_identity': [{'index': i, 'physical_channel_or_derivation': None} for i in range(n)],
            'reference_type': 'D. unknown', 'montage': 'unknown', 'signal_unit': 'unknown',
            'jet_required_electrodes_available': False, 'availability_status': 'unverified in processed tensors',
            'jet16_reconstructable': False, 'confidence_level': 'C',
            'processed_sample_to_raw_edf': 'not recoverable from released X/y-only files',
            'split_counts': meta['split_counts'],
            'split_provenance': meta.get('split_policy', 'not specified') + '; original JET split equivalence unverified',
            'actual_hdf5_shapes': {s: entry['datasets']['X']['shape'] for s,entry in h5['splits'].items()},
            'decision': 'NEED_RAW_TUEV',
        })
    pairs = [('FP1','F7'),('F7','T3'),('T3','T5'),('T5','O1'),('FP2','F8'),('F8','T4'),('T4','T6'),('T6','O2'),
             ('FP1','F3'),('F3','C3'),('C3','P3'),('P3','O1'),('FP2','F4'),('F4','C4'),('C4','P4'),('P4','O2')]
    result = {'audit_complete': True, 'variant': 'both existing public variants',
              'variants': variants, 'raw_manifest_recordings': raw['n_recordings'],
              'raw_manifest_unique_layouts': raw['n_unique_ordered_layouts'],
              'raw_required_electrode_coverage': raw['fraction_with_required_electrodes'],
              'jet16_reconstructable': False, 'confidence_level': 'C', 'selected_source_variant': None,
              'conversion_performed': False, 'formal_training_started': False,
              'decision': 'NEED_RAW_TUEV', 'answer': 'NO under currently available defensible provenance',
              'single_decisive_missing_evidence': 'A release-specific processed channel-index to physical channel/derivation and reference contract, backed by the exact generation source or an aligned per-sample index plus explicit order-preservation source.',
              'additional_unverified_items': ['processed signal units/normalization', 'filtering execution source',
                                             'JET split equivalence', '22-channel raw dataset version'],
              'raw_source_search': {'matched_raw_files': source_search['matched_raw_files'],
                                    'policy': source_search['policy']},
              'target_derivations': [a+'-'+b for a,b in pairs],
              'boolean_semantics': 'jet_required_electrodes_available=false means not verified available in processed tensors; it does not prove physical absence.'}
    dump(OUT/'channel_audit.json', result)
    for variant in variants:
        dump(OUT/(variant['variant']+'_channel_audit.json'), variant)
    first20 = load('raw_tuev_first20.json')
    with (OUT/'raw_tuev_first20.csv').open('w', encoding='utf-8-sig', newline='') as f:
        fields = ['recording_id','subject_id_in_dataset','n_channels','sampling_rate_hz','channel_names','reference','montage_name','archival_uri']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in first20:
            value = {k:row[k] for k in fields}
            value['channel_names'] = json.dumps(value['channel_names'])
            writer.writerow(value)
    evidence = OUT/'evidence'
    evidence.mkdir(exist_ok=True)
    shutil.copyfile(OUT/'processed_provenance'/'original_32'/'current'/'generate_croissant.py', evidence/'generate_croissant.py')
    for key in ('original_32', 'alternate_22'):
        shutil.copyfile(OUT/'processed_provenance'/key/'commit_history.tsv', evidence/(key+'_commit_history.tsv'))
    old_blob = next(v['blob'] for v in historic['hf-original.git'] if v['path'].endswith('metadata_summary.json') and v['source_root'])
    shutil.copyfile(OUT/'processed_provenance'/'original_32'/'historical_blobs'/(old_blob+'.txt'), evidence/'original32_historical_metadata.json')
    sections = [
        '# HF TUEV Channel Provenance Audit\n',
        '**结论：NO。当前公开证据不能唯一恢复两个 HF 张量的物理通道身份、顺序及 reference；决定为 `NEED_RAW_TUEV`。**\n',
        '审计已执行，严格 JET conversion 和正式 Loop-JET 训练均未执行。该结论是数据可识别性结论，不是对 Loop-JET 研究假设的否定。审计日期：2026-10-05。\n',
        '## 1. Processed HF variants found\n',
        '| Variant | Repository revision | Actual X shapes (train / val / test) | Confidence |\n|---|---|---|---|',
    ]
    for v in variants:
        shapes = ' / '.join(str(s) for s in v['actual_hdf5_shapes'].values())
        sections.append(f"| {v['n_channels']} channels | `{v['source_repository']}@{v['source_commit']}` | {shapes} | C — Ambiguous |")
    sections += ['\n两个版本的真实 H5 均只有 X/y，文件与数据集 attrs 为空。32 通道全文件下载并以发布的 LFS SHA-256 校验；22 通道只读取 HTTP Range 返回的 headers 和 labels，共 '+str(sum(e['http_bytes_transferred'] for e in h22['splits'].values()))+' 字节，没有下载 X 信号。\n',
                 '## 2. 22-channel variant\n',
                 '- 源仓库 `USuCgex0122e/demo_dataset_seed20260726`；revision `41602cf9363d83204e6b7145b04efaeafebd9ae4`。',
                 '- train/val/test = 8,370 / 2,496 / 1,684；X float32，y int32，六类 0–5。metadata 声称 subject-disjoint，split seed=20260726；未获得每样本 subject ID，无法独立验证此声明。',
                 '- 源原始数据版本未知；全部 21 个可达 commit 没有任何 Python preprocessing source。',
                 '- channel 0–21 的物理身份全部 unknown；reference / montage / unit 全部 unknown。不能将其称为 referential 或 bipolar。',
                 '- metadata 声明 200 Hz、5 s、1000 点、0.3–75 Hz、60 Hz notch；没有实际处理代码证明滤波执行。`filtering execution provenance unverified`，不重复滤波。',
                 '- `export_summary.json` 当前内容涉及另外六个数据集，没有 TUEV 的 channel mapping。\n',
                 '## 3. 32-channel variant\n',
                 '- 源仓库 `USuCgex0122e/demo_dataset`；revision `db96e77affa66398c553427045ee040f06230c2a`。',
                 '- train/val/test = 16,409 / 13,206 / 12,908；X float32，y int32，六类 0–5。三个 H5 共 925,966,698 字节，完整 SHA 校验记录见 `h5_audit.json`。',
                 '- 历史 commit `281c4515d354ffe86a2e2f6bf00159c9df7c1002` 的 metadata / Croissant 指向 `/vePFS-0x0d/eeg-h5/raw/TUEV/v2.0.1/edf`；这是源版本 v2.0.1 的历史声明。三份 H5 的 LFS OID 在完整历史中没有变化，因此这不是另一个 H5 版本。它仍不能证明每列的物理通道身份。',
                 '- channel 0–31 的物理身份全部 unknown；reference / montage / unit 全部 unknown。',
                 '- 唯一 Python 文件 `generate_croissant.py` 生成 metadata，不读取 EDF 生成 H5。其第 470–481 行从 `dataset_config.json.channel_names` 或 `sample_index.parquet.channel_names_json` 取名字，第 639–643 行还读取 conversion/validation summaries。',
                 '- 这些逐样本索引和生成配置在全部公开 commit 的文件树中均不存在。H5 的原始 EDF / REC 文件名、subject、recording ID、channel-index mapping 也均未保存。',
                 '- 200 Hz / 5 s / 1000 点及 filtering 参数是 metadata 声明；filter 执行代码仍未验证。\n',
                 '### 完整 provenance 搜索范围\n',
                 '| Repository | Commits | Current text files | Unique historical text blobs | Historical Python paths |\n|---|---:|---:|---:|---|',
    ]
    for key,v in history.items():
        sections.append(f"| {v['repo']} | {v['commits_examined']} | {len(v['current_text_files'])} | {v['unique_text_blobs_examined']} | {', '.join(v['all_historical_python_paths']) or 'none'} |")
    sections += ['\n搜索全部历史与当前 `.py/.json/.yaml/.yml/.csv/.txt/.md`；完整 inventory、原始文本和 keyword hits 保存于 `provenance_evidence.zip`，各文件 SHA 见 `provenance_evidence_manifest.json`。没有只检查 README 或单一 commit。两个 Git bare clone 与 manifest metadata 下载均通过 hf-mirror.com 完成；镜像 commit 与官方当前 commit 一致，完整祖先历史可达。\n',
                 '## 4. Raw TUEV metadata from eeg-corpus-manifest\n',
                 '来源 `tankalapavankalyan/eeg-corpus-manifest@e4c2d62a9230d053c948a6681a63d1ca0d3f353e`，严格筛选 `dataset_id=tuh_eeg_events`。只下载 metadata Parquet，不读取原始 EDF 或 S3 signal。\n',
                 f"- recording IDs：{raw['n_unique_recording_ids']}；subjects：{raw['n_subjects']}；有序 channel layouts：{raw['n_unique_ordered_layouts']}。",
                 f"- n_channels 分布：`{json.dumps(raw['n_channels_distribution'],sort_keys=True)}`。",
                 f"- sampling rate：`{json.dumps(raw['sampling_rate_distribution'])}`，全部原始 recording 为 250 Hz。",
                 f"- reference 字段：`{json.dumps(raw['reference_distribution'])}`；montage 字段：`{json.dumps(raw['montage_distribution'])}`，均缺失。",
                 '- 名字多为 `EEG FP1-REF` 等，表现为原始 reference 标签；仅凭此不推断 HF 输出的 reference 类型。',
                 f"- 必需电极覆盖：{raw['recordings_with_required_electrodes']}/{raw['n_recordings']} = {raw['fraction_with_required_electrodes']:.1%}；只做 upper-case、EEG prefix、末尾 -REF/-LE 清理。没有合并 bipolar 名字或使用 T7/T3 等别名猜测。",
                 f"- channel 表与 recording 表的 order mismatch：{len(raw['channel_table_vs_recording_order_mismatches'])}；normalization collision：{len(raw['normalization_collisions'])}。",
                 f"- 原始 channel units：`{json.dumps(raw['channels_table_units_distribution'])}`；这些单位不能迁移为 HF processed unit。",
                 '- 至少前 20 条 recording 的 ID、subject、channel names、rate、reference、montage 已保存于 `raw_tuev_first20.json/.csv`。全部 518 recording 和 15,601 channel rows 在 evidence archive 中。\n',
                 '| Layout SHA prefix | Frequency | n_channels | First two channel labels | Required electrodes |\n|---|---:|---:|---|---|',
    ]
    for layout in raw['layouts']:
        first = ' / '.join(layout['channel_names'][:2])
        sections.append(f"| {layout['layout_sha256'][:12]} | {layout['frequency']} | {len(layout['channel_names'])} | {first} | {'all' if layout['all_jet_electrodes_present'] else 'missing'} |")
    sections += ['\n**两个 recording 的 layout `e7bf5c1dd19...` 从 `FP2, FP1, F4, F3, ...` 开始，常见 layout 从 `FP1, FP2, F3, F4, ...` 开始。** 因此即使原始 recording 全部具备所需电极，也不能把常见 raw order 当作通用 processed order。\n',
                 '## 5. Can JET 16 bipolar channels be reconstructed?\n',
                 '**当前不能。** 缺的是 release-specific 的 `processed tensor index → physical channel/derivation + reference` 证据链。原始 EDF channel names 已知，但 processed sample 到 EDF 没有唯一对应；也没有 preprocessing source 证明保留了原始 channel order。两个版本均为 Level C，不是已证明缺少电极的 Level D。\n',
                 '32 与 22 通道、sample counts、subjects 均不同，不能认定只是同一张量的重新 split，也不能用某一版本的潜在 order 代替另一版本。\n',
                 '## 6. Selected source variant\n',
                 '**None。** 未生成 `data/tuev_jet16/`，未选择前 16 通道，未假设 10–20 或 EDF 顺序，未重复 referencing / filtering。\n',
                 '## 7. Exact conversion equations\n',
                 '目标顺序如下。它们是 JET 的目标定义，尚未绑定任何 HF index，因此没有执行减法或 selection：\n',
                 '```text\n'+'\n'.join(f'{i:02d}: {a}-{b}' for i,(a,b) in enumerate(pairs))+'\n```\n',
                 '若后续 A/B 证据确认 referential/raw electrode channels，按上列同源 electrode 做差；若明确为 bipolar，只按已验证的完整 derivation 名称选取。当前 reference 类型为 D. unknown，两个分支均不满足。\n',
                 '## 8. Remaining uncertainties\n',
                 '关键缺失证据只有一个：**与当前发布的 H5 绑定的物理 channel-index/reference 契约**。可通过精确 HDF5 generation source + explicit channel list/order，或 aligned source EDF index + explicit order-preservation source补齐；不能以 raw corpus 常见顺序替代。',
                 '单位 / normalization、filtering execution、22-channel raw version 和 paper split equivalence 也尚未验证。JET 官方 EDF preprocessing 明确 `get_data(units="uV")`，官方 loader 再 `/100.0`；HF H5 和公开 metadata 没有 signal unit 字段。',
                 '32-channel 固定随机 10 samples/split 的未缩放 min/max/mean/std、percentiles、channel-wise std 见 `unscaled_magnitude32.json`。看到全零 sample-channel 对不等于证明 padding；幅度不用于猜单位或通道。22-channel 本轮只读 headers/labels，未额外读取 X，unit 同样 unknown。',
                 f"服务器源文件搜索访问 {source_search['directories_visited']} 个目录；范围为 `{source_search['policy']}`。发现的 TUH/TUEV/event raw EDF+REC 为 {len(source_search['matched_raw_files'])}，processed 文件只有本次下载的 32 通道 H5。该搜索有明确深度边界，不声称证明整机没有 raw 数据。",
                 'conversion sanity（100 样本、5 张转换图）不适用，因为禁止执行尚无 provenance 的转换；没有制作伪转换数据或伪检查图。\n',
                 '## 9. Final decision\n',
                 '**`NEED_RAW_TUEV`。NO under currently available defensible provenance。**\n',
                 'channel audit 已完成；正式长训练保持停止。恢复可由 uploader 提供上述映射契约完成，也可在具有授权的 raw EDF+REC 上走官方 preprocessing。当前没有把 22/32 通道输入当作严格 JET 16 双极导联。\n',
                 '### Primary sources / attribution\n',
                 '- [Processed 32-channel repository](https://huggingface.co/datasets/USuCgex0122e/demo_dataset/tree/db96e77affa66398c553427045ee040f06230c2a)',
                 '- [Historical metadata with v2.0.1 source root](https://huggingface.co/datasets/USuCgex0122e/demo_dataset/blob/281c4515d354ffe86a2e2f6bf00159c9df7c1002/TUEV/metadata_summary.json)',
                 '- [Metadata generator, not H5 preprocessing](https://huggingface.co/datasets/USuCgex0122e/demo_dataset/blob/db96e77affa66398c553427045ee040f06230c2a/generate_croissant.py)',
                 '- [Processed 22-channel repository](https://huggingface.co/datasets/USuCgex0122e/demo_dataset_seed20260726/tree/41602cf9363d83204e6b7145b04efaeafebd9ae4)',
                 '- [EEG corpus manifest](https://huggingface.co/datasets/tankalapavankalyan/eeg-corpus-manifest/tree/e4c2d62a9230d053c948a6681a63d1ca0d3f353e), publisher tankalapavankalyan, metadata dataset card CC-BY-4.0; raw TUEV row identifies TUH-DUA. Only public metadata subsets are redistributed.',
                 '- [Official JET preprocessing](https://github.com/Y-Research-SBU/JET/blob/07f9e6491796f4f2c717d6259b1a2e24afce6a77/data/preprocess_tuev.py), MIT; snapshot retained under `official/`.\n',
    ]
    (OUT/'HF_CHANNEL_AUDIT.md').write_text('\n'.join(sections), encoding='utf-8')
    for arm in ('naive_k2','ds_k2'):
        config = json.loads((ROOT/'configs'/f'{arm}.json').read_text())
        config['status'] = 'blocked_by_channel_provenance'
        dump(ROOT/'configs'/f'{arm}.json',config)
        (ROOT/arm).mkdir(exist_ok=True)
        dump(ROOT/arm/'config.json', config)
        dump(ROOT/arm/'NOT_RUN.json', {'formal_optimizer_steps':0,'trained':False,
            'blocker':'Level C channel identity/order/reference; unit also unverified',
            'not_produced':['200-step benchmark','train_log.csv','val_log.csv','best_val_checkpoint.pt',
                            'last_checkpoint.pt','trajectory metrics','generation metrics','training figures']})
    dump(ROOT/'analysis'/'runtime_estimate.json', {'status':'not_measured_due_to_data_gate',
        'naive_k2_seconds_per_step':None,'ds_k2_seconds_per_step':None,
        'estimated_200_epoch_hours':None,'reason':'No incompatible-data benchmark or training executed. Synthetic smoke is not a runtime estimate.'})
    dump(ROOT/'STATUS.json', {'channel_audit':'complete','data_conversion':'not_performed',
        'formal_training':'not_started','decision':'NEED_RAW_TUEV','scientific_hypothesis':'not_evaluated'})
    sanity = json.loads((ROOT/'analysis'/'sanity.json').read_text())
    final = f'''# Loop-JET TUEV seed0 v1 — current execution report

**Data provenance audit complete. Formal two-model experiment not completed; decision NEED_RAW_TUEV.**

The independent channel audit requested on 2026-10-05 is authoritative: [HF_CHANNEL_AUDIT.md](data_audit/HF_CHANNEL_AUDIT.md). Both processed variants are Level C: physical channel index/order/reference cannot be uniquely recovered. No JET16 conversion or formal training has run.

## Objective
Test iterative target-directed error correction with Naive K2 and DS K2, seed0. The preceding provenance stage asks whether defensible conversion into official JET16 exists. Current answer: NO under available evidence.

## Environment
Remote D:\\loop_jet_tuev_seed0_v1; NVIDIA RTX5090 32GB; torch {sanity['torch']}, CUDA {sanity['cuda']}, capability {sanity['capability']}, sm_120 included. Working torch retained. Process-local verified affinity mask 0xffff0000; this does not establish the cause of past native failures. Other GPU jobs observed; none stopped.

## Data
Actual 32-channel H5 downloaded via hf-mirror.com in 35.05 seconds and fully SHA verified. Actual 22-channel headers/labels verified with bounded HTTP Range, no full X download. Source history: 22 + 21 commits. Raw metadata: 518 recordings, 15 layouts, 100% required-electrode coverage. These raw facts do not identify processed columns. No source EDF join is possible from released X/y-only H5.

## Model architecture and sanity
Official revision 07f9e6491796f4f2c717d6259b1a2e24afce6a77. Actual 12 blocks, 768 width, 12 heads, patch_size=200; 16x5 tokens. Loop 4+[4]x2+4; same module parameters reused, common t/class condition and shared post/head. Parameter total {sanity['official_parameter_count']:,} both K1 and K2; trainable {sanity['official_trainable_parameter_count']:,}. K1 max absolute output difference: 0 in FP32 and BF16. Both objective backward checks passed on synthetic batch2; no TUEV optimizer update performed. Lazy H5 worker tests and refusal of actual 32-channel files passed.

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
'''
    (ROOT/'FINAL_REPORT.md').write_text(final, encoding='utf-8')
    print('CHANNEL_AUDIT_COMPLETE', result['decision'])

if __name__ == '__main__':
    main()
