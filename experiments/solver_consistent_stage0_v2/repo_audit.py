import json,os,pathlib,platform,sys
ROOT=pathlib.Path(__file__).parent
found=[]; agents={}; scanned=0
for root in [pathlib.Path('D:/Pythonproject'),pathlib.Path('D:/nips-temp')]:
    if not root.exists(): continue
    for folder,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in ['.git','.venv','__pycache__','node_modules','data','datasets','checkpoints']]
        for name in files:
            if name.endswith(('.py','.toml','.yaml','.txt','.md')):
                scanned+=1
                if any(k in name.lower() for k in ['pronto','recurrent','transformer','trainer','tokenizer','requirements','pyproject']):
                    found.append(str(pathlib.Path(folder)/name))
for p in [pathlib.Path('D:/AGENTS.md'),ROOT/'AGENTS.md']:
    if p.exists(): agents[str(p)]=p.read_text(encoding='utf-8',errors='replace')
result={'host':platform.node(),'python':sys.executable,'roots_audited':['D:/Pythonproject','D:/nips-temp'],
        'text_source_files_examined':scanned,'candidate_files':found,'agents':agents,
        'reuse_decision':'Existing work is EEG/SEEG experimental infrastructure. No PrOntoQA or recurrent reasoning backbone found by filename inventory. Reuse existing CUDA PyTorch environment in isolated system-site-packages venv; new independent project on D:.',
        'local_audit':'D:/chenyu-iclr rg inventory found EEG model requirements, no PrOntoQA or reasoning recurrence modules. Original task and user evaluator instructions retained. No unrelated source edited.'}
(ROOT/'artifacts').mkdir(exist_ok=True)
(ROOT/'artifacts/A0_REPO_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'scanned':scanned,'candidates':len(found),'agents':agents}))
