import json,pathlib,shutil
root=pathlib.Path(__file__).resolve().parent
src=root/'artifacts/sanity'; dst=root/'artifacts/sanity_v2_allocator_pilot'
assert src.resolve().is_relative_to(root) and dst.resolve().is_relative_to(root)
if src.exists() and not dst.exists(): src.rename(dst)
dst.mkdir(exist_ok=True)
for name in ['pipeline.stdout.log','pipeline.stderr.log','status.json']:
    shutil.copy2(root/'artifacts'/name,dst/name)
with (root/'EXPERIMENT_CHANGELOG.md').open('a') as f:
    f.write('\n- Before formal training: math-attention WDDM allocator cache filled31846/32607MB while max live allocation was much lower, causing severe slowdown. Preserve sanity_v2_allocator_pilot and restart all V2 learning gates from original initialization with a fixed per-process CUDA allocator cap65% for all models. Frozen architecture/tokenizer/data/optimizer/update budgets unchanged. This is an engineering memory policy, not a neural method change.\n')
print('Allocator pilot archived; identical fresh gates with65% CUDA memory cap')
