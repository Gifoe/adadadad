"""Package recorded data/diagnostics without altering scientific artifacts."""
import ctypes,sys
if sys.platform=='win32':
 api=ctypes.windll.kernel32.SetProcessAffinityMask;api.argtypes=[ctypes.c_void_p,ctypes.c_size_t];api.restype=ctypes.c_int;assert api(ctypes.c_void_p(-1),0xffff0000)
import pathlib,zipfile,hashlib,json
ROOT=pathlib.Path(__file__).resolve().parent;ARCH=ROOT/'archives';ARCH.mkdir(exist_ok=True)
def package(name,paths,compression):
 p=ARCH/name;tmp=p.with_suffix('.zip.tmp')
 with zipfile.ZipFile(tmp,'w',compression=compression,compresslevel=1 if compression else None,allowZip64=True) as z:
  for src in paths:z.write(src,src.relative_to(ROOT).as_posix())
 tmp.replace(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=h.hexdigest(),members=len(paths))
rows=[package('verified_selected_official_dataset.zip',list((ROOT/'data').glob('*')),zipfile.ZIP_DEFLATED)]
if len(list((ROOT/'outputs').glob('prototypes_*.pt')))==3:
 paths=list((ROOT/'carriers').glob('*.npz'))+list((ROOT/'outputs').glob('prototypes_*.pt'))+[ROOT/'outputs'/n for n in ['intervention_plan.json','intervention_noise_directions.npy','intervention_plan_manifest.json','pretrained_carrier_manifest.json']]
 rows.append(package('recorded_carriers_prototypes_and_controls.zip',paths,zipfile.ZIP_STORED))
(ROOT/'outputs/archive_manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(rows),flush=True)
