"""Download official public artifacts on the execution host; verify pinned bytes."""
import hashlib,os,pathlib,time
import requests
ROOT=pathlib.Path(__file__).resolve().parent/'official_artifacts';ROOT.mkdir(exist_ok=True)
FILES=[
 ('1oLx2oeD-NpCuokw0AN5PMZNCB-5Ts_kr','r2_checkpoint_epoch_2765.pt',342301539,'4e89fde1f592d7afece63a6182f8427beedc85a216adb474777830b6b56495d8'),
 ('1-ChSYN2b8EjJahxdD6a9LF1NJzV35PdV','vocab.json',2229,'54beefa3f1d310ad96376c80f2bbfbc689703d42188e41d211a1bafa416cbd93'),
 ('12TSeReTnXBSet5XzCD9wSs6mO8Us9i2G','test_small.json',2941757,'a286cb2537bda10784f55c65c029735a6cfe68eb1b9277153f95590de72a84d9'),
 ('18AZXc1dFWHy3iEoSpbXqMybOqmxF75nb','train.json',172811881,'7f56438ed63fdd66761848607c283a1f534815300896c466cb1d249858205aef'),
 ('1CuoajnhvCXT5iEstqaKPaR5IeMi4JcC0','test.json',162183305,'75facaf41bb9d15236a2d2b7e6fab8d606d884f187bf3c9b5440d9d5c2967d87')]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 s=requests.Session();proxy=os.getenv('PUBLIC_ARTIFACT_PROXY')
 if proxy:s.trust_env=False;s.proxies={'http':proxy,'https':proxy}
 for fid,name,size,digest in FILES:
  if os.getenv('ARTIFACT_ONLY') and name!=os.getenv('ARTIFACT_ONLY'):continue
  p=ROOT/name
  if p.exists() and p.stat().st_size==size and sha(p)==digest:print('VERIFIED_EXISTING',name,flush=True);continue
  print('DOWNLOAD_START',name,flush=True);started=time.time()
  temp=p.with_suffix(p.suffix+'.download.part');offset=temp.stat().st_size if temp.exists() else 0
  if offset==size and sha(temp)==digest:temp.replace(p);print('VERIFIED_RESUMED_FILE',name,flush=True);continue
  headers={'Range':f'bytes={offset}-'} if offset else {}
  r=s.get('https://drive.usercontent.google.com/download',params={'id':fid,'export':'download','confirm':'t'},headers=headers,stream=True,timeout=(30,90));r.raise_for_status()
  if offset and r.status_code==206:assert r.headers.get('Content-Range','').startswith(f'bytes {offset}-')
  elif offset:offset=0
  if 'text/html' in r.headers.get('Content-Type',''):raise RuntimeError('Unexpected HTML instead of official artifact '+name)
  n=offset;last=time.time();print('RESUME_OFFSET',name,offset,flush=True)
  with temp.open('ab' if offset else 'wb') as f:
   for b in r.iter_content(1024*1024):
    f.write(b);n+=len(b)
    if time.time()-last>=10:print('DOWNLOAD_BYTES',name,n,size,flush=True);last=time.time()
  assert n==size,(name,n,size)
  assert sha(temp)==digest,('SHA256_MISMATCH',name)
  temp.replace(p);print('DOWNLOAD_VERIFIED',name,n,digest,'seconds',time.time()-started,flush=True)
if __name__=='__main__':main()
