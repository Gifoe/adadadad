import concurrent.futures,hashlib,importlib.metadata,json,os,pathlib,platform,shutil,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import requests
from neso_online.download import RESOURCES

def request_resource(item):
    name,rid=item;result={'name':name,'resource_id':rid}
    for endpoint,params in [('resource_show',{'id':rid}),('datastore_search',{'resource_id':rid,'limit':5})]:
        url='https://api.neso.energy/api/3/action/'+endpoint
        started=time.perf_counter()
        try:
            response=requests.get(url,params=params,timeout=(15,45))
            result[endpoint]={'url':response.url,'status':response.status_code,'elapsed_seconds':time.perf_counter()-started,
                              'body':response.json() if 'json' in response.headers.get('Content-Type','') else response.text[:2000]}
        except Exception as e:result[endpoint]={'error':repr(e),'elapsed_seconds':time.perf_counter()-started}
    (ROOT/'outputs/source_metadata'/f'{name}.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    return result

def main():
    (ROOT/'outputs/source_metadata').mkdir(parents=True,exist_ok=True)
    info={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'platform':platform.platform(),
          'python':sys.version,'executable':sys.executable,'disk_free_bytes':shutil.disk_usage(ROOT).free,
          'package_versions':{}}
    for name in ['torch','numpy','pandas','requests','pytest','scipy','scikit-learn','einops','reformer-pytorch','matplotlib','psutil','tzdata']:
        try:info['package_versions'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:info['package_versions'][name]=None
    import torch
    info['torch_cuda']=torch.version.cuda;info['cuda_available']=torch.cuda.is_available()
    info['gpu']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    try:
        import psutil
        info['ram']=psutil.virtual_memory()._asdict()
    except ImportError:pass
    info['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu',
                                               '--format=csv,noheader'],text=True).strip()
    info['official_source_commit']=subprocess.check_output(['git','-C',str(ROOT/'upstream/WorldTS'),'rev-parse','HEAD'],text=True).strip()
    assert info['official_source_commit']=='269426ea6217d7152a244da0e209d922acd46d4f'
    (ROOT/'outputs/environment.json').write_text(json.dumps(info,indent=2,default=str),encoding='utf-8')
    print('ENVIRONMENT',info['gpu'],info['package_versions'],flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        records=list(executor.map(request_resource,RESOURCES.items()))
    (ROOT/'outputs/source_probe.json').write_text(json.dumps(records,indent=2,default=str),encoding='utf-8')
    for record in records:
        sample=record['datastore_search'];meta=record['resource_show']
        sb=sample.get('body',{});mb=meta.get('body',{})
        print('SOURCE',record['name'],'count',sb.get('result',{}).get('total') if isinstance(sb,dict) else None,
              'csv',mb.get('result',{}).get('url') if isinstance(mb,dict) else None,flush=True)

if __name__=='__main__':main()
