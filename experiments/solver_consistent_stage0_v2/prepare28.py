import pathlib,shutil
root=pathlib.Path(__file__).resolve().parent
src=root/'artifacts/sanity'; dst=root/'artifacts/sanity_v2_torch211_native_crash'
assert src.resolve().is_relative_to(root) and dst.resolve().is_relative_to(root)
if src.exists() and not dst.exists(): src.rename(dst)
dst.mkdir(exist_ok=True)
for name in ['pipeline.stdout.log','pipeline.stderr.log','status.json','environment.json','environment_packages.txt']:
    p=root/'artifacts'/name
    if p.exists(): shutil.copy2(p,dst/name)
with (root/'EXPERIMENT_CHANGELOG.md').open('a') as f:
    f.write('\n- 2026-10-01: native0xC0000005 recurred under math attention and allocator cap, so neither fused attention nor paging was established as its cause. Survey found installed PyTorch2.8.0+cu128 in persist_stable_251. Create separate .venv28 inheriting this environment; explicitly activate its Conda DLL PATH and use one CPU thread. All neural/data/optimizer/budget/seed settings remain V2-frozen; restart ALL gates/models from common initialization in the same software environment. This is a runtime compatibility attempt, not a method result. Retain all PyTorch2.11 failures.\n')
print('Torch2.11 failures archived; fresh gates under2.8 with matching Conda DLL path')
