import requests,urllib.parse,json,time
url='https://drive.usercontent.google.com/download?id=1oLx2oeD-NpCuokw0AN5PMZNCB-5Ts_kr&export=download&confirm=t'
p=requests.utils.get_environ_proxies(url)
print('PROXY',json.dumps({k:(urllib.parse.urlsplit(v).scheme,urllib.parse.urlsplit(v).hostname,urllib.parse.urlsplit(v).port) for k,v in p.items()}),flush=True)
for mode in ['system_proxy','direct']:
 s=requests.Session();s.trust_env=False
 if mode=='system_proxy':s.proxies={k:('http://'+v.split('://',1)[-1]) if k=='https' and v.startswith('https://') else v for k,v in p.items()}
 t=time.time()
 try:
  r=s.get(url,stream=True,timeout=(12,20));print(mode,r.status_code,r.headers.get('Content-Type'),r.headers.get('Content-Length'),'seconds',time.time()-t,flush=True)
  if r.status_code==200:
   b=next(r.iter_content(65536));print('FIRST_BYTES',b[:4].hex(),'received',len(b),flush=True)
   if b[:2]==b'PK':r.close();break
  r.close()
 except Exception as e:print(mode,type(e).__name__,str(e)[:180],flush=True)
