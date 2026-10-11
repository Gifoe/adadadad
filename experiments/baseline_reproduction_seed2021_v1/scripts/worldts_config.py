from common import *
import shlex
def published_config(dataset,horizon):
 src=ROOT/'upstream/WorldTS/DAG-bench';lines=(src/'scripts/w_future/worldts.sh').read_text(encoding='utf-8').splitlines()
 for line in lines:
  if not line.startswith('run ') or f'--data-name-list {dataset}.csv ' not in line:continue
  argv=shlex.split(line)[2:];hp=json.loads(argv[argv.index('--model-hyper-params')+1])
  if hp['horizon']==horizon:return line,argv,hp
 raise ValueError((dataset,horizon))
