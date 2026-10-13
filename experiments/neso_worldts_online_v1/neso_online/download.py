"""Download official NESO CKAN resources with streaming, retry, SHA256 logging.

Archives are very large. Use `--resources forecast_2025 demand_2025 ...` to
obtain official full-year CSVs. Data redistribution is avoided; files stay local.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
from urllib.parse import quote
import requests

# Resource IDs are taken from the official NESO resource documentation.
RESOURCES={
 'forecast_2025':'fc13df13-2dad-4a1c-b9e3-4569efba4955',
 'forecast_2026_jan_jun':'d6375700-69c2-4c25-8bde-883a205d742e',
 'forecast_2026_jun_dec':'31861619-0b86-47ba-bac2-d008a760af54',
 'demand_2025':'b2bde559-3455-4021-b179-dfe60c0337b0',
 'demand_2026':'8a4a771c-3929-4e56-93ad-cdf13219dea5',
}


def fetch(name,outdir,session=None,chunk_mb=2):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    if name not in RESOURCES:raise ValueError(f'Unknown resource {name}')
    rid=RESOURCES[name]
    api_url=f'https://api.neso.energy/datastore/dump/{rid}?format=csv'
    filename=outdir/f'{name}.csv'
    sess=session or requests.Session()
    if filename.exists() and filename.with_suffix('.metadata.json').is_file():
        saved=json.loads(filename.with_suffix('.metadata.json').read_text())
        h=hashlib.sha256()
        with filename.open('rb') as f:
            for block in iter(lambda:f.read(2**20),b''):h.update(block)
        if saved['sha256']!=h.hexdigest():raise ValueError(f'Existing CSV hash mismatch: {filename}')
        print(f'EXISTING_SHA_VERIFIED {filename}',flush=True);return filename
    meta_response=sess.get('https://api.neso.energy/api/3/action/resource_show',params={'id':rid},timeout=(20,45))
    if meta_response.status_code==404:
        info={'name':name,'resource_id':rid,'status':'OFFICIAL_RESOURCE_REMOVED_OR_UNAVAILABLE',
              'checked_url':meta_response.url,'body':meta_response.json()}
        (outdir/f'{name}.unavailable.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
        raise LookupError(f'Official resource no longer exists: {name} ({rid})')
    meta_response.raise_for_status();metadata=meta_response.json()['result']
    urls=list(dict.fromkeys([metadata['url'],api_url]))
    part=filename.with_suffix('.csv.part')
    failures=[]
    for attempt in range(4):
        url=urls[min(attempt,len(urls)-1)]
        try:
            print(f'DOWNLOAD_START {name} attempt={attempt+1} {url}',flush=True)
            with sess.get(url,stream=True,timeout=(20,60),headers={'User-Agent':'NESO-WorldTS-Research/0.1'}) as resp:
                resp.raise_for_status()
                ctype=resp.headers.get('Content-Type','')
                if 'application/json' in ctype.lower():
                    raise ValueError(f'Unexpected JSON instead of CSV: {resp.text[:300]}')
                with part.open('wb') as f:
                    received=0;last_print=0
                    for block in resp.iter_content(chunk_size=chunk_mb*1024*1024):
                        if block:
                            f.write(block);received+=len(block)
                            if received-last_print>=32*1024*1024:
                                print(f'DOWNLOAD_PROGRESS {name} bytes={received}',flush=True);last_print=received
            with part.open('rb') as f:
                head=f.read(500).decode('utf-8-sig',errors='replace')
            if 'SETTLEMENT' not in head and 'DATE_GMT' not in head:
                raise ValueError(f'Response does not look like NESO CSV: {head[:200]}')
            part.replace(filename)
            h=hashlib.sha256()
            with filename.open('rb') as f:
                for block in iter(lambda:f.read(2**20),b''):
                    h.update(block)
            info={'resource_id':rid,'name':name,'source_url':url,'file':str(filename),
                  'size_bytes':filename.stat().st_size,'sha256':h.hexdigest(),
                  'resource_metadata':metadata,'download_failures':failures,
                  'downloaded_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            filename.with_suffix('.metadata.json').write_text(json.dumps(info,indent=2))
            print(json.dumps(info,indent=2),flush=True);return filename
        except Exception as e:
            failures.append({'attempt':attempt+1,'url':url,'error':repr(e)})
            print(f'{name}: attempt={attempt+1} error={e}',flush=True)
            if attempt==3:raise
            time.sleep(2**attempt)
    raise RuntimeError('unreachable')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--resources',nargs='+',choices=list(RESOURCES),default=list(RESOURCES))
    ap.add_argument('--out',default='data/raw')
    ap.add_argument('--allow-unavailable',action='store_true',help='Record a removed official ID and continue remaining resources; does not create a substitute CSV')
    a=ap.parse_args()
    for n in a.resources:
        try:fetch(n,a.out)
        except LookupError:
            if not a.allow_unavailable:raise
            print(f'RESOURCE_UNAVAILABLE_RECORDED {n}',flush=True)

if __name__=='__main__':main()
