import json,pathlib,psutil
root=pathlib.Path(__file__).resolve().parent
state=json.loads((root/'artifacts/status.json').read_text()); pid=state['pid']
for p in psutil.Process(pid).children():
    args=p.cmdline()
    assert pathlib.Path(args[0]).resolve().is_relative_to(root)
    if 'train.py' in args and '--sanity' in args:
        print('Stopping owned development-only sanity PID',p.pid,flush=True); p.kill()
