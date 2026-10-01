# EXPERIMENT_CHANGELOG

All entries below precede formal training. No formal model has produced a non-zero update.

- 2026-10-02: Created a new V2 branch and directory. Old data, tokenizer, checkpoints, and logs are excluded from all V2 inputs.
- 2026-10-02: Pinned official PrOntoQA source at commit `0a6412b6fddf46324a1cb96e066dd7b3d89b87d6`; vendored source remains unmodified.
- 2026-10-02: Added versioned post-generation adapter. It removes the isolated opposite-goal rule and adds a predicate-renamed copy of the connected proof component. The decoy proof axiom is placed on the queried entity with opposite polarity, so unsigned fact occurrence is matched while the signed decoy chain remains unreachable. Generator version is `prontoqa-symmetric-component-v2`.
- 2026-10-02: Added global exact-text, canonical-FOL/entity-normalized, and predicate-renaming-invariant graph deduplication.
- 2026-10-02: Added an all-row parser/forward-chaining/minimum-depth/contradiction/checksum verifier and a pre-registered shortcut suite.
- 2026-10-02: Development pilot v1 failed because the opposite antecedent did not occur as a queried-entity fact at depth 1; `last_hop_selected_fact` reached 100%. The pilot was never eligible for training.
- 2026-10-02: Development pilot v2 passed semantic verification on all 1,080 pilot rows. Its small held-out strata are too small to decide shortcut thresholds; full 135,000-row generation and audit are required.
- 2026-10-02: Formal training budget corrected to the prompt default of 20,000 updates for every model. Microbatch and max length remain subject to the mandated pre-training tokenizer/VRAM gates.
