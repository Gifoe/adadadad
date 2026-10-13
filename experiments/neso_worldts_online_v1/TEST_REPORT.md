# Current executed test evidence

Server: NVIDIA RTX 5090, Python 3.10.20, Torch 2.8.0+cu128. The supplied engineering suite passed 7 tests on this server; the repaired suite passed 13 tests. Raw logs are in outputs/logs/tests_original.log and tests_final.log.

Those tests include dummy/synthetic fixtures and are not the scientific results. Separate real evidence includes an official native-model forward/backward smoke on 16 actual TRAIN windows, finite nonzero E/C/P gradients, Decoder-only Stage 2 updates, finite [16,48,1] predictions, target-label foil, strict state hash and matching checkpoint restore. The complete selected-model Validation predictions reproduce exactly with matching stage flags (observed maximum difference 0.0 MW).

Actual native Stage 1 and Stage 2 were trained on 1406 genuine windows. Validation 354 and the locked eligible Test 303 windows were evaluated; all MW/MW² metrics independently match preserved predictions and masks. See TRAINING_REPORT.md, VALIDATION_RESULTS.csv, TEST_RESULTS.csv and FINAL_REPORT.md. The final content/lock verification is outputs/artifacts_verification.json.

The original supplied CPU/dummy report, which preceded server execution and stated that real results were unverified, is preserved as historical evidence in outputs/original_docs/TEST_REPORT.md and original_package.zip. Its synthetic sample counts must not be used as NESO experimental results. Remaining limitations are source publication timing and incomplete strict April-August coverage; these remain explicitly unresolved.
