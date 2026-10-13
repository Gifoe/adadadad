# Build and evidence status (2026-10-13)

| Work item | Current status | Evidence |
| --- | --- | --- |
| NESO source discovery | Verified URLs & CKAN IDs | Official NESO dataset pages |
| Vintage + ND raw CSV downloader | Implemented, unrun | `neso_online/download.py` |
| Period end / DST normalization | Implemented + tested | `tests/test_pipeline.py` |
| Forecast A/B pair construction | Implemented + synthetic pass | 142 synthetic pairs; **not NESO observations** |
| Disjoint target-time train/val/test split | Implemented + tested | `tests/test_pipeline.py` |
| Native WorldTS stages 1/2 adapter | Implemented, native smoke pending | `neso_online/native_worldts.py`; generic mock driver test only |
| Old / full-recompute forecasting | Implemented, native smoke pending | `replay` |
| Output correction / one-step latent updater | Implemented, native smoke pending | unit/integration test with dummy core |
| Real 2025–2026 NESO data downloaded | **Not done** | Container network/DNS restricted |
| RTX 5090 WorldTS training and valid/test scores | **Not done** | No CUDA GPU in current runtime |
| Fully causal online ND availability | **Not established** | NESO Historic Demand posted ~21 days late, no issue-level ND history |
| Forecast archive precise public availability | **Not established** | `Forecast_Datetime` is production timestamp, not explicit publication timestamp |

Do not confuse source-interface checks and synthetic-dataset sanity with a native
GPU training run. No numerical WorldTS test results have been generated here.
