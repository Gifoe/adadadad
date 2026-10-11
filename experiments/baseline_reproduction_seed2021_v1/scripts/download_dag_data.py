"""Retrieve only NP/PJM and metadata from the authors' ZIP, verifying CRC and SHA."""
from common import ROOT,DATA_SHA,sha,write_json
import io,os,pathlib,struct,zipfile,zlib,requests
URL='https://drive.usercontent.google.com/download?id=1K2AvogpOpSz1PiQ53dPchzGv_PqlCWAK&export=download&confirm=t'
def download():
 s=requests.Session();s.trust_env=False
 proxy=os.getenv('AUDIT_DOWNLOAD_PROXY','')
 if proxy:s.proxies={'http':proxy,'https':proxy}
 transferred=0
 def get(start,end):
  nonlocal transferred
  r=s.get(URL,headers={'Range':f'bytes={start}-{end}'},timeout=(20,120));r.raise_for_status()
  assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {start}-'),'Require partial download support'
  transferred+=len(r.content);return r
 first=get(0,0);size=int(first.headers['Content-Range'].split('/')[-1]);tail=get(size-65536,size-1).content
 eocd=tail.rfind(b'PK\x05\x06');assert eocd>=0
 fields=struct.unpack_from('<4s4H2LH',tail,eocd);cd_size,cd_offset=fields[5:7]
 cd=get(cd_offset,cd_offset+cd_size-1).content;archive=zipfile.ZipFile(io.BytesIO(cd+tail[eocd:]))
 selected=[v for v in archive.infolist() if pathlib.PurePosixPath(v.filename).name in ['NP.csv','PJM.csv','FORECAST_META.csv']]
 assert {'NP.csv','PJM.csv','FORECAST_META.csv'}=={pathlib.PurePosixPath(v.filename).name for v in selected}
 out=ROOT/'data';out.mkdir(exist_ok=True);evidence=[]
 for v in selected:
  offset=v.header_offset+cd_offset;header=get(offset,offset+29).content;h=struct.unpack('<4s5H3L2H',header);assert h[0]==b'PK\x03\x04'
  start=offset+30+h[-2]+h[-1];compressed=get(start,start+v.compress_size-1).content
  raw=zlib.decompress(compressed,-15) if v.compress_type==8 else compressed;assert len(raw)==v.file_size and zlib.crc32(raw)==v.CRC
  dst=out/pathlib.PurePosixPath(v.filename).name;part=dst.with_suffix('.part');part.write_bytes(raw)
  if dst.stem in DATA_SHA:assert sha(part)==DATA_SHA[dst.stem]
  part.replace(dst);evidence.append({'archive_member':v.filename,'bytes':len(raw),'sha256':sha(dst)});print('DOWNLOAD_VERIFIED',dst.name,flush=True)
 write_json(out/'source_manifest.json',{'url':URL,'archive_bytes':size,'downloaded_bytes':transferred,'files':evidence})
if __name__=='__main__':download()
