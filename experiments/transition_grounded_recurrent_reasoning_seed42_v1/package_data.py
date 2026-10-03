"""Archive the actual verified data and frozen sample order for delivery."""
import zipfile,pathlib,hashlib,json
from data_pipeline import ROOT,dump
target=ROOT/'data/verified_dataset_seed42.zip'
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in sorted((ROOT/'data').rglob('*')):
        if p.is_file() and p.suffix in ('.json','.npz','.npy'):
            info=zipfile.ZipInfo(p.relative_to(ROOT).as_posix(),date_time=(2000,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=3)
dump(ROOT/'data_archive_manifest.json',dict(path=str(target),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),includes='all actual generated train/test JSON, array caches, relation table, vocabulary, data audit, and shared 20000-step sample plan'))
print(target.stat().st_size,flush=True)
