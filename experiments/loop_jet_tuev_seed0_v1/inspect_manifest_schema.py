import json
import pathlib
from process_affinity import configure_process_affinity
configure_process_affinity()
import pyarrow.parquet as pq

root = pathlib.Path(__file__).resolve().parent/'data_audit'/'raw_manifest'
report = {}
for name in ('datasets', 'recordings', 'channels'):
    f = pq.ParquetFile(root/f'{name}.parquet')
    report[name] = {'rows': f.metadata.num_rows, 'row_groups': f.metadata.num_row_groups,
                    'schema': str(f.schema_arrow)}
    if name == 'datasets':
        records = f.read().to_pylist()
        report[name]['tuev_rows'] = [row for row in records if row['dataset_id'] == 'tuh_eeg_events' or str(row.get('short_name')).upper() == 'TUEV']
(root.parent/'manifest_schema.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
print(json.dumps(report, indent=2, default=str), flush=True)
