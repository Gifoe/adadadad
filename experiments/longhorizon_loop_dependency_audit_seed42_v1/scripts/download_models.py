"""Immutable model provenance and resumable official model download."""
import argparse,concurrent.futures,datetime,hashlib,json,os,pathlib,requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
REPOS=['nics-efc/TaH2-1.7B-max2','nics-efc/TaH2-1.7B-Standard']
PINNED={'nics-efc/TaH2-1.7B-max2':'e965dff2bd539b5be8087126b7d26bcb47e1b159','nics-efc/TaH2-1.7B-Standard':'bbf0d82946c8e242ab2dc1f3be20408e317562ec'}
def metadata(repo):
    failures=[];session=requests.Session();session.trust_env=False
    for endpoint in ['https://hf-mirror.com','https://huggingface.co']:
        try:
            response=session.get(endpoint+'/api/models/'+repo+'/revision/'+PINNED[repo],timeout=(15,45));response.raise_for_status();m=response.json()
            assert m['sha']==PINNED[repo],'Pinned model revision did not match metadata response'
            info={'repo':repo,'revision':m['sha'],'endpoint':endpoint,'files':[x['rfilename'] for x in m['siblings']],
                  'probed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'failed_endpoints':failures}
            for name in ['config.json','tah_config.json','tokenizer_config.json']:
                if name in info['files']:
                    result=session.get(f'{endpoint}/{repo}/resolve/{info["revision"]}/{name}',timeout=(15,45));result.raise_for_status();info[name]=result.json()
            dest=ROOT/'outputs/source_metadata';dest.mkdir(parents=True,exist_ok=True)
            (dest/(repo.split('/')[-1]+'.json')).write_text(json.dumps(info,indent=2),encoding='utf8')
            print('MODEL_PROVENANCE',json.dumps({k:v for k,v in info.items() if k!='tokenizer_config.json'}),flush=True)
            return info
        except Exception as error:failures.append({'endpoint':endpoint,'type':type(error).__name__,'message':str(error)[:300]})
    raise RuntimeError(json.dumps(failures))
def main():
    p=argparse.ArgumentParser();p.add_argument('--probe-only',action='store_true');a=p.parse_args()
    for repo in REPOS:
        info=metadata(repo)
        if a.probe_only:continue
        from huggingface_hub import snapshot_download
        dest=ROOT/'models'/repo.split('/')[-1]
        snapshot_download(repo,revision=info['revision'],endpoint=info['endpoint'],local_dir=dest,
                          allow_patterns=['*.json','*.safetensors','*.bin','*.txt','*.model','*.jinja'],max_workers=2)
        files=[]
        for path in sorted(dest.rglob('*')):
            if not path.is_file() or '.cache' in path.parts:continue
            h=hashlib.sha256()
            with path.open('rb') as f:
                while block:=f.read(1048576):h.update(block)
            files.append({'path':path.relative_to(ROOT).as_posix(),'size_bytes':path.stat().st_size,'sha256':h.hexdigest()})
        (ROOT/'outputs/source_metadata'/(repo.split('/')[-1]+'.download.json')).write_text(json.dumps({'repo':repo,'revision':info['revision'],'files':files},indent=2),encoding='utf8')
        print('MODEL_DOWNLOAD_VERIFIED',repo,sum(x['size_bytes'] for x in files),flush=True)
if __name__=='__main__':main()
