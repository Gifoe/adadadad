import json,pathlib,time,psutil
root=pathlib.Path(__file__).resolve().parent
state=json.loads((root/'artifacts/status.json').read_text()); pid=state['pid']
meta=root/'artifacts/sanity/vanilla_loop/metadata.json'
initial=json.loads(meta.read_text())['update'] if meta.exists() else 0
print('Waiting for next saved checkpoint after',initial,flush=True)
while True:
    try: current=json.loads(meta.read_text())['update']
    except (OSError,json.JSONDecodeError): current=initial
    if current>initial: break
    if not psutil.pid_exists(pid): raise RuntimeError('Pipeline already exited')
    time.sleep(.5)
for process in psutil.Process(pid).children():
    args=process.cmdline()
    assert pathlib.Path(args[0]).resolve().is_relative_to(root)
    if 'train.py' in args and '--sanity' in args:
        print('Stopping owned sanity process',process.pid,'after saved checkpoint',current,flush=True); process.kill(); process.wait(timeout=30)
