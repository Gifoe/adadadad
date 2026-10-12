# Phase0 preflight

All four complete original Stage2 checkpoints strictly loaded and freshly evaluated at every official test origin; no baseline retraining. Models, targets, origins and training-only scalers match archived outputs. Test verification belongs only to these original checkpoints.

| dataset   |   horizon |   test_mse |   test_mae |   validation_mae |   max_prediction_difference |   stage1_best_epoch |
|:----------|----------:|-----------:|-----------:|-----------------:|----------------------------:|--------------------:|
| NP        |       360 |   0.430395 |   0.416201 |         0.204109 |                           0 |                   1 |
| PJM       |       360 |   0.115082 |   0.208422 |         0.166145 |                           0 |                   1 |
| NP        |        24 |   0.167258 |   0.213351 |         0.107484 |                           0 |                   1 |
| PJM       |        24 |   0.057516 |   0.142788 |         0.110261 |                           0 |                   1 |

All structural, Predictor-only freezing, decoded-gradient, future-label invariance, original decoder/RevIN, checkpoint roundtrip, artificial cosine, float64 metrics and no_grad guard tests passed. Parameter hashes and per-module counts are in preflight_metrics.json.

`h_hist` is the original history encoder state; `h_target` encodes true future labels with the historical patch prefix and historical RevIN statistics. `h_pred` depends on history and permitted covariates only. The causal patch decoder receives the same historical state prefix for both predicted and target states. `_decode_model_states` detaches frozen history, but does not detach supplied future states. The official deployment `_predict_only` intentionally uses no_grad and detaches h_pred; it is used as a reference and must not be reused as the A1 training path.

Frozen Encoder/Decoder parameters use requires_grad=False and eval(). Predictor training enables only its parameter gradients and train() mode. Decoder autograd remains enabled with respect to predicted states. Target reconstruction is a label-based diagnostic, not a forecast lower bound.

Source pin:269426ea6217d7152a244da0e209d922acd46d4f. Exact frozen objective is10*latent L1+15*trajectory cosine+0.05*prediction legacy VC+0.05*target legacy VC. Frozen target regularization is constant for Predictor; prediction regularization remains batch/time coupled and differentiable.

## Numerical execution and validation-order correction

Preflight attempts1–4 are preserved in outputs/logs. Attempt1 used batch32 for independent test prediction while the original artifact used batch64; its maximum difference2.384185791015625e-6 failed the original1e-6 tolerance. Returning to the original batch64 yields bitwise identical original test forecasts for all four tasks. Attempts2–4 then exposed the shuffled/unweighted validation issue. The official validation loader used shuffle=True and an unweighted mean across batches, including a short tail batch; historical ordering was not archived. All new comparisons use the same chronological batch32 partition with the unchanged official criterion. The current validator matches the official validator exactly on the same batches. Historical randomized scores fall inside the mathematically possible tail-partition range, which is a consistency check, not a proof of the lost historical order. No prediction tolerance was relaxed and no candidate test forecasts were generated.
