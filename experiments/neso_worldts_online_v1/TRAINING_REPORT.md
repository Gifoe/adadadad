# Actual native WorldTS training

| Stage | Executed epochs | Selected epoch | Selected Validation objective | Optimizer steps | Stage seconds | Trainable parameters | Peak allocated GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11 | 6 | 1.640980629 | 968 | 20.055 | 347376 | 827.646 |
| 2 | 20 | 20 | 0.215500087 | 1760 | 22.633 | 148097 | 297.403 |

TRAIN1406windows →88optimizer steps/epoch; batch16 with final14-window batch retained. Stage1 early-stopped at11 after5non-improving epochs; Stage2 completed its20epoch budget and selected20. Its curve was still improving at the boundary; this fixed-budget result does not establish convergence or SOTA. Combined measured stage time42.688s excludes import/download/preparation and later calibration/replay.

The exact native losses, gradients and frozen tensor checks passed. Selected Stage1 modules are preserved in selected Stage2 checkpoint (SHA:74f86a4db8a022efa912a2d136cec4c4b1f95b1a9a3ff52fa85d7f5300535565). Optimizer/RNG/generator resume state is saved after every completed epoch; retained checkpoints are the raw last state per stage, latest resume state, initial model and best state per stage. Earlier overwritten raw epoch states are not retained. Every objective remains in the full training history.

Complete Validation diagnostics: latent predicted feature std=0.384599, target feature std=0.434443, predicted temporal std=0.365837. Latents are finite and not identically constant under these checks; these diagnostics do not prove representation quality. Future-label foil difference=0.0; fresh checkpoint prediction difference=0.0; fresh restored saved Validation MW prediction difference=0.0. Checkpoint tensors load strictly with matching stage flags.

Updater training: skipped by predeclared Validation gate; no updater score or checkpoint is invented. Original7engineering tests and repaired13tests passed; real-model forward/backward evidence is independent. Curves:`outputs/figures/training_curves.png`. Raw evidence:`train_history.json`, training log, native smoke, selected model diagnostics and checkpoint SHA manifest.
