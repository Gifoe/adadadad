"""Capture execution environment without touching model or dataset state."""
import json,os,pathlib,platform,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ops import *
import sklearn,matplotlib

if __name__=='__main__':
    info={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
          'torch':torch.__version__,'torch_cuda':torch.version.cuda,
          'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
          'gpu_total_mib':torch.cuda.get_device_properties(0).total_memory/1048576 if torch.cuda.is_available() else None,
          'numpy':np.__version__,'pandas':pd.__version__,
          'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__,
          'cpu_affinity_mask_hex':'0xffff0000' if os.name=='nt' and (os.cpu_count() or 0)>=32 else None,
          'omp_num_threads':os.environ.get('OMP_NUM_THREADS'),
          'mkl_num_threads':os.environ.get('MKL_NUM_THREADS'),
          'data_sha256':{ds:sha(BASE/'data'/(ds+'.csv')) for ds in ['NP','PJM']},
          'source_zip_sha256':sha(ROOT/'official_worldts_source.zip'),
          'protocol_sha256':sha(ROOT/'configs/protocol.json')}
    assert info['data_sha256']==DATA_SHA and info['gpu_name'] is not None
    write_json(ROOT/'outputs/environment.json',info)
    print('ENVIRONMENT_CAPTURED',info['gpu_name'],info['torch'],info['torch_cuda'])
