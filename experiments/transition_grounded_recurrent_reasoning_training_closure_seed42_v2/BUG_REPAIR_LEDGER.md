# Runtime and bookkeeping ledger

## Native Windows failure after the 10,500-update save point

The A1 Python process terminated with Windows native access violation (0xc0000005), not a Python assertion, nonfinite-loss error or curriculum failure. Fault-handler frames include transformers GPT2 MLP/Conv1D torch.addmm. Windows Application event1000 (2026-10-04 15:11:51 server local time) names E:/Anaconda/envs/persist_stable_251/VCRUNTIME140.dll, offset0x121ec. Other earlier Python processes also have access-violation events in c10.dll/python310.dll. No matching Display/nvlddmkm/WHEA events were returned by the queried system-log window. This evidence does not identify the root cause.

The saved A1 curriculum state is D2 at10,500 updates, validation7/375=1.8667%. Restore model, AdamW, scheduler, Torch/CUDA/NumPy/Python RNG, shuffled epoch order/cursor and accumulated learning curves from official_baseline_resume.pt. No architecture, data, recurrence policy, threshold, optimizer or budget changes. The interrupted log is preserved as execution_attempt1.log. Uncheckpointed work after10,500 is not counted as effective optimizer updates; up to499 updates may have been lost, with no exact count available. Stage timing includes persisted checkpoint-to-checkpoint work; full wall time also includes interruption/recovery overhead.

## Resume and reporting repairs before completion

Handle a checkpoint saved at an epoch boundary by regenerating the next shuffled epoch before slicing a batch. A stage-failure checkpoint cannot advance into a later stage. Active-stage checkpoint state is retained when an already-passed stage is resumed. Preserve the original run start across process restarts and list each process start separately.

Evaluation writes a separate evaluation_runtime.json so final stage bookkeeping cannot erase measured OOD evaluation time. Reports distinguish staged workload time from full process wall time (the latter includes recovery overhead). These changes repair execution/accounting, not the research method. A1 success depends on completing current stages at the official threshold; independent endpoint retention is reported without an additional unrequested A1 selection gate.
