import pathlib
r=pathlib.Path(__file__).parent
p=r/'EXPERIMENT_CHANGELOG.md'
marker='Structural shortcut hard gate, 2026-10-01'
s=p.read_text(encoding='utf-8')
if marker not in s:
    p.write_text(s+'\n- '+marker+': pure-text, no-training query-goal antecedent-frequency heuristic achieves100% on all135000 final examples, all depth strata, independently reproduced locally/remotely. Do not treat current data as a valid multihop reasoning test. Terminated only this project owned sanity child on discovery; latest checkpointing retry was explicitly interrupted, not complete/not passed/not another diagnosed crash. Added mandatory structural audit before any formal training; pipeline now writes STAGE0_REPORT.md/NA result CSV/fairness/data-shortcut CSV and exits blocked with UNCLEAR. Formal updates remain0 for all models. Latest remote tests14passed,960-sample semantic/minimum-hop audit passed. Preserve original dataset/config/checkpoints, request no extra seeds/losses until structural data repair and runtime diagnosis.\n',encoding='utf-8')
print('changelog saved')
