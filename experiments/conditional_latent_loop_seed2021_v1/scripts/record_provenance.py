from common import *
import importlib.metadata,platform

sources={}
for repo in ('DAG','LatentTSF'):
    sources[repo]={str(p.relative_to(ROOT/'upstream'/repo)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'upstream'/repo).rglob('*') if p.is_file() and not any(v in p.parts for v in ('__pycache__','dataset','result'))}
own={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('scripts','models','configs') for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
json_write(ROOT/'source_manifest.json',{'DAG_commit':'0758990e2c73bb54138ea3e7b11a35cbc5476bcc','LatentTSF_commit':'7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2','upstream_file_hashes':sources,'experiment_file_hashes':own})
packages={}
for name in ('torch','numpy','pandas','scikit-learn','scipy','einops','ray','dash','matplotlib','reformer-pytorch','psutil'):
    packages[name]=importlib.metadata.version(name)
json_write(ROOT/'environment.json',{'platform':platform.platform(),'python':sys.version,'packages':packages,'GPU':torch.cuda.get_device_name(0),'CUDA':torch.version.cuda,'cpu_threads':torch.get_num_threads(),'CPU_affinity_mask':'0xffff0000','venv':'system-site-packages plus private overrides','seed':2021})
print('PROVENANCE_RECORDED',packages,flush=True)
