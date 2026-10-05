"""All indexed TUEV recordings, exact ordered layouts, conservative normalization."""
import collections
import hashlib
import json
import pathlib
import re
from process_affinity import configure_process_affinity
configure_process_affinity()
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT/'data_audit'
DATA = OUT/'raw_manifest'
PAIRS = [('FP1','F7'),('F7','T3'),('T3','T5'),('T5','O1'),
         ('FP2','F8'),('F8','T4'),('T4','T6'),('T6','O2'),
         ('FP1','F3'),('F3','C3'),('C3','P3'),('P3','O1'),
         ('FP2','F4'),('F4','C4'),('C4','P4'),('P4','O2')]
REQUIRED = set(v for pair in PAIRS for v in pair)

def normalize(name):
    name = name.strip().upper()
    name = re.sub(r'^EEG\s+', '', name)
    name = re.sub(r'-(REF|LE)$', '', name)
    return name.strip()

def counts(rows, key):
    return dict(collections.Counter('<missing>' if r.get(key) is None else str(r[key]) for r in rows))

def main():
    datasets = pq.read_table(DATA/'datasets.parquet').to_pylist()
    relevant = [r for r in datasets if r['dataset_id'] == 'tuh_eeg_events' or str(r.get('short_name')).upper() == 'TUEV']
    ids = [r['dataset_id'] for r in relevant]
    assert ids, 'No uniquely identified TUEV dataset rows'
    recordings = pq.read_table(DATA/'recordings.parquet', filters=[('dataset_id', 'in', ids)]).to_pylist()
    assert recordings, 'No TUEV recordings'
    recordings.sort(key=lambda r: (r['archival_uri'], r['recording_id']))
    layouts = {}
    covered, collisions = 0, []
    for row in recordings:
        names = row.get('channel_names') or []
        encoded = json.dumps(names, separators=(',', ':')).encode()
        key = hashlib.sha256(encoded).hexdigest()
        layout = layouts.setdefault(key, {'layout_sha256': key, 'frequency': 0, 'channel_names': names,
                                          'normalized_names': [normalize(s) for s in names]})
        layout['frequency'] += 1
        norm = layout['normalized_names']
        present = REQUIRED.issubset(set(norm))
        row['jet_required_electrodes_present_after_label_normalization'] = present
        row['layout_sha256'] = key
        if present:
            covered += 1
        duplicates = [k for k, n in collections.Counter(norm).items() if n > 1]
        if duplicates:
            collisions.append({'recording_id': row['recording_id'], 'normalized_duplicate_names': duplicates})
    for layout in layouts.values():
        layout['missing_jet_electrodes'] = sorted(REQUIRED-set(layout['normalized_names']))
        layout['all_jet_electrodes_present'] = not layout['missing_jet_electrodes']
    layout_list = sorted(layouts.values(), key=lambda l: (-l['frequency'], l['layout_sha256']))
    record_ids = [r['recording_id'] for r in recordings]
    channels = pq.read_table(DATA/'channels.parquet', filters=[('recording_id', 'in', record_ids)]).to_pylist()
    grouped = collections.defaultdict(list)
    for row in channels:
        grouped[row['recording_id']].append(row)
    mismatches = []
    for row in recordings:
        listed = sorted(grouped[row['recording_id']], key=lambda r: r['channel_index'])
        if listed and [r['channel_name'] for r in listed] != row['channel_names']:
            mismatches.append(row['recording_id'])
    file_hashes = {}
    for path in DATA.glob('*'):
        if path.is_file():
            file_hashes[path.name] = {'bytes': path.stat().st_size,
                                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    result = {'source_repo': 'tankalapavankalyan/eeg-corpus-manifest',
              'source_revision': 'e4c2d62a9230d053c948a6681a63d1ca0d3f353e',
              'dataset_rows': relevant, 'n_recordings': len(recordings),
              'n_unique_recording_ids': len(set(record_ids)),
              'n_subjects': len({r['subject_id_in_dataset'] for r in recordings}),
              'n_unique_ordered_layouts': len(layouts), 'layouts': layout_list,
              'n_channels_distribution': counts(recordings, 'n_channels'),
              'sampling_rate_distribution': counts(recordings, 'sampling_rate_hz'),
              'reference_distribution': counts(recordings, 'reference'),
              'montage_distribution': counts(recordings, 'montage_name'),
              'header_read_status_distribution': counts(recordings, 'header_read_status'),
              'required_electrodes': sorted(REQUIRED),
              'recordings_with_required_electrodes': covered,
              'fraction_with_required_electrodes': covered/len(recordings),
              'normalization_policy': 'Uppercase; strip whitespace; remove EEG prefix and terminal -REF/-LE only. No T7/T8/P7/P8 aliases, no bipolar collapse.',
              'normalization_collisions': collisions,
              'channels_table_rows_for_tuev': len(channels),
              'channels_table_units_distribution': counts(channels, 'units'),
              'channels_table_type_distribution': counts(channels, 'channel_type'),
              'channel_table_vs_recording_order_mismatches': mismatches,
              'file_checksums': file_hashes,
              'processed_to_raw_alignment': 'NOT ESTABLISHED: raw recording identity/order is not evidence of processed row/channel identity/order.'}
    for filename, value in [('raw_tuev_summary.json', result), ('raw_tuev_recordings.json', recordings),
                            ('raw_tuev_first20.json', recordings[:20]), ('raw_tuev_channels.json', channels)]:
        (OUT/filename).write_text(json.dumps(value, indent=2, default=str, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('layouts', 'file_checksums')}, indent=2, default=str), flush=True)

if __name__ == '__main__':
    main()
