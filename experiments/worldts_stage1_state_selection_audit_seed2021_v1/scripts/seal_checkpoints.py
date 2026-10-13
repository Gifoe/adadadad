"""Hash all server-only checkpoints and reverify read-only baseline inputs."""
import hashlib, json, pathlib, sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from ops import *

def seal():
    record_path=ROOT/'outputs/checkpoint_hashes.json'
    record=json.loads(record_path.read_text(encoding='utf-8'))
    assert len(record['protected_inputs'])==16
    for protected in record['protected_inputs']:
        path=pathlib.Path(protected['path'])
        assert path.is_file() and sha(path)==protected['sha256'],str(path)
    manifest=json.loads((ROOT/'configs/source_manifest.json').read_text())
    assert len(manifest)==312
    for item in manifest:
        assert sha(ROOT/item['path'])==item['sha256'],item['path']
    additions=[]
    output_dir=ROOT/'outputs/checkpoints'
    for path in sorted(output_dir.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in ['.pt','.npz']:
            continue
        rel=path.relative_to(ROOT).as_posix()
        addition={'status':'EXPERIMENT_ARTIFACT','relative_path':rel,
                  'absolute_path':str(path),'sha256':sha(path),
                  'bytes':path.stat().st_size,'kind':path.suffix.lower()}
        parts=path.relative_to(output_dir).parts
        if parts:addition['dataset']=parts[0]
        if len(parts)>1 and parts[1].startswith('decoder_from_stage1_epoch'):
            addition['stage']='stage2_decoder_mix'
            addition['stage1_epoch']=int(parts[1].split('epoch')[-1])
            if path.name.startswith('decoder_epoch') and path.suffix=='.pt':
                addition['decoder_epoch']=int(path.stem.split('epoch')[-1])
        elif path.name.startswith('replay_stage1_epoch'):
            addition['stage']='stage1_latent_raw'
            addition['stage1_epoch']=int(path.stem.split('epoch')[-1])
        additions.append(addition)
    expected=sum(1 for r in additions if r.get('stage')=='stage1_latent_raw')
    decoder=sum(1 for r in additions if r.get('stage')=='stage2_decoder_mix' and 'decoder_epoch' in r)
    assert expected==12 and decoder==126,(expected,decoder)
    record['new_artifacts']=additions
    record['protected_inputs_unchanged']=True
    record['source_manifest_entries_verified']=len(manifest)
    record['source_zip_sha256']=sha(ROOT/'official_worldts_source.zip')
    record['raw_stage1_snapshots']=expected
    record['decoder_epoch_checkpoints']=decoder
    write_json(record_path,record)
    rows=[]
    for historic in record['inventory']:
        rows.append({'status':historic['status'],'dataset':historic['dataset'],
                     'stage':historic['stage'],'stage1_epoch':historic['epoch'],
                     'absolute_path':historic['path'],'sha256':historic['sha256'],
                     'bytes':historic.get('bytes')})
    rows+=additions
    write_csv(ROOT/'outputs/checkpoint_inventory.csv',rows)
    print('CHECKPOINTS_SEALED',len(additions),'rawstage1',expected,
          'decoder_epoch_ckpt',decoder,'protected_sha_ok',flush=True)

if __name__=='__main__':seal()
