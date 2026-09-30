import pathlib,shutil
root=pathlib.Path(__file__).resolve().parent
for src,dst in [('artifacts/sanity','artifacts/sanity_v1_native_crash'),('data/tokenized','data/tokenized_v1'),('configs/frozen.json','configs/frozen_v1.json'),('artifacts/frozen_length_memory.json','artifacts/frozen_length_memory_v1.json')]:
    source=root/src; destination=root/dst
    assert source.resolve().is_relative_to(root) and destination.resolve().is_relative_to(root)
    if source.exists() and not destination.exists(): source.rename(destination)
archive=root/'artifacts/sanity_v1_native_crash'
archive.mkdir(exist_ok=True)
for name in ['pipeline.stdout.log','pipeline.stderr.log','status.json']:
    source=root/'artifacts'/name
    if source.exists(): shutil.copy2(source,archive/name)
print('V1 artifacts retained; fresh v2 tokenizer/config/models before any formal training')
