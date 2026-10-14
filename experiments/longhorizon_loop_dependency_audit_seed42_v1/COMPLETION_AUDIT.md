# Requirement-to-evidence audit

This ledger distinguishes executed work from blocked requirements. The research
decision may be INCONCLUSIVE; it is not a claim that the official engine was
fully reproduced. Machine checks are in outputs/artifact_verification.json.

| Requirement | Authoritative evidence | Scope / status |
| --- | --- | --- |
| Official source and checkpoints | execution_provenance.json, source_metadata/, model configuration hashes | Exact TaH/NESTFUL commits and both HF revisions pinned |
| Official serving reproduction | official_engine_example.json, resolver log retained on server | BLOCKED: Linux-only Triton stack; actual Windows Gloo exception; parity unverified |
| Frozen model, BF16, seed42, one GPU | environment.json, per-phase model hashes, native_backend.py | Native backend only; no training/backward/optimizer |
| 20 representative baseline prompts × 4 methods | pilot_sample_ids.json, baseline_pilot_records.jsonl, baseline_results.json | Different Standard weights distinguished from same-max2 comparisons |
| Output-length sensitivity | length_probe_plan.json, baseline_length_isolated_records.jsonl, baseline_length_probe.json | Separate 4-prompt development sensitivity; validated per prompt; failed legacy runs not used for accuracy |
| Forced depths and unchanged readout | native_backend.py, preflight_native.json | Native stop_prob_mix preserved; exact manual-mixture check |
| Snapshot independence, replay, depth/time causality | tests/, preflight_native.json, strict aligned nulls | Exact branch independence/repeat/null checks; BF16 full/incremental target-CE limits disclosed |
| NESTFUL provenance, valid execution and no False=0 | data_audit_tokenized.json, sample_manifest.jsonl, tests/test_data_semantics.py | Official basic functions, exact references, structured DAGs and exclusions |
| Controlled generator / splits / renaming | prepare_data.py, fixed manifest, data semantics tests | 16 operations, 4 depths, matched histograms/character lengths; not a public benchmark |
| Locked primary 2x2 | case_plan_audit*.json, intervention_results.jsonl | 253 pairs / 1012 core cells per layout, 128 independent parent units |
| Actual loop/cache/cost information | raw branch rows, core_runtime_*.json, runtime_profile.json | Native forward wall times and loop counts; FLOPs proxy limits explicit |
| Nondependent, mask, shallow and local-window controls | raw rows, analysis_*.json, NUMERIC_REPAIR.md | Coarse event-role matching; unavailable H16/nonmatching controls recorded |
| Low-local threshold fixed before audit | development_thresholds.json, development_records_frozen.jsonl | Exact-byte frozen development hash; no audit retuning |
| Problem-cluster statistics / strata | analyze_results.py, analysis_audit*.json | Original problems or matched groups, not independent tokens |
| Complete H2/H4/H8/H16 coverage | primary and depths_aligned plans/analysis | H4/H16 primary; H2/H8 separate locked-manifest extension; no extra independent groups |
| Equal-budget Stage4 | research_decision.json, FINAL_REPORT section G | NOT OPENED: positive dependency-specific low-local/native-K1 gate did not pass |
| Full-task versus teacher-forced claims | baseline records, FINAL_REPORT sections B–I | No free-generation improvement or deployable-oracle claim |
| Report, figures and reproducibility | FINAL_REPORT.md, figures/, README.md, requirements-native.txt | Actual measured results and unresolved platform/identifiability limits |
| Research ideas | Separate local IDEA_CANDIDATES.md delivered to user | Conditional hypotheses, nearest prior work, minimum tests and falsification; no unverified novelty claim |
| Protect previous experiments | Final Git diff against base 96756f06e07219390dc0c48638683299673cfc3e | Changes must be restricted to this new experiment prefix |

Full-vocabulary anchor NPZ files and weights remain on the execution server.
The public raw records include all measured branch utilities, target CE, top-8
per-loop logits, stop scores, policies, hashes and costs needed to inspect the
reported comparisons. No missing experiment is replaced by simulated data.
