"""Compact reproducibility record without local usernames or credentials."""
import datetime,hashlib,importlib.metadata,json,pathlib,platform,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]

def main():
    versions={n:importlib.metadata.version(n) for n in ['torch','transformers','accelerate','numpy','PyYAML','safetensors','huggingface-hub','requests','pytest','matplotlib','scipy','msgpack','pyzmq','fastapi','uvicorn','termcolor','tqdm']}
    upstream={}
    for name in ['TaH','NESTFUL']:
        folder=ROOT/'upstream'/name
        upstream[name]={'commit':subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip(),
                        'tracked_diff_sha256':hashlib.sha256(subprocess.check_output(['git','-C',str(folder),'diff','HEAD'])).hexdigest()}
    configs={}
    for name in ['TaH2-1.7B-max2','TaH2-1.7B-Standard']:
        folder=ROOT/'models'/name;items={}
        for filename in ['config.json','tah_config.json','tokenizer_config.json','generation_config.json','chat_template.jinja']:
            p=folder/filename
            if p.exists():items[filename]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}
        configs[name]=items
    script_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for d in ['scripts','tests'] for p in (ROOT/d).iterdir() if p.is_file() and p.suffix in ['.py','.ps1']}
    report={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'platform':platform.platform(),'python':platform.python_version(),
            'gpu_driver_memory':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
            'versions':versions,'upstream':upstream,'model_config_hashes':configs,'script_hashes':script_hashes,
            'manifest_sha256':hashlib.sha256((ROOT/'outputs/sample_manifest.jsonl').read_bytes()).hexdigest(),
            'config_sha256':hashlib.sha256((ROOT/'config.yaml').read_bytes()).hexdigest()}
    (ROOT/'outputs/execution_provenance.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
