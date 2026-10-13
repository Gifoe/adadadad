# Local test evidence

Runtime: CPU-only PyTorch 2.10.0 in a container with no GPU and no access to
NESO bulk archive download URLs. The native authors' repository was inspected
through GitHub, but its full dependency tree and official training checkpoints
were not installed here.

`python -m pytest -q`: **7 passed**.

Tests include:
- UK spring/fall DST half-hour settlement-day counts (46 / 50) and ordinary 48.
- Real NESO forecast schema/period-end convention (`2025-01-01 00:30`, SP 1).
- UK day-first `03/04/2026` parsing instead of US month-first.
- End-to-end pairing, contiguity, late forecast replacement mask, and version order.
- Disjoint target-label temporal splits, including exclusion of boundary straddles.
- Exactly zero initial correction from zero-initialized latent updater.
- Generic (NON-OFFICIAL) dummy model two-stage train, updater train,
  output-space calibrator, and baseline replay.

One synthetic run produced: 90,432 forecast rows, 8,016 demand rows,
**142 valid version-pairs**, train/validation/test **43/87/12**. These numbers
are **synthetic test data, not NESO dataset statistics**.

Unverified: end-to-end native WorldTS import/actual training, 2026 split archive
pairing, real data coverage, 5090 memory/time, model generalization, and
real-time ND release timestamps. All real model metric fields remain empty.
