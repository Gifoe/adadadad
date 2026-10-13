# Real NESO data audit

retrospective forecast-vintage replay; instantaneous historicND availability unverified; production timestamp is not verified public-release time

| Split | Genuine windows | Unique target half-hours | First target UTC | Last target UTC | Revised windows |
|---|---:|---:|---|---|---:|
| train | 1406 | 16994 | 2025-01-08 00:30:00+00:00 | 2025-12-31 18:00:00+00:00 | 100.00% |
| val | 354 | 4308 | 2026-01-01 00:30:00+00:00 | 2026-03-31 18:00:00+00:00 | 100.00% |
| test | 303 | 3888 | 2026-04-01 00:30:00+00:00 | 2026-07-19 01:00:00+00:00 | 100.00% |

d6375700-69c2-4c25-8bde-883a205d742e returns official404 and is absent from current package resources; no substitute CSV fabricated

The requested alias `forecast_2026_jun_dec` now points to the official resource named "Embedded Solar and Wind Forecast Archive 2026". Actual per-file coverage is recorded below; it is not assumed from the alias.

{
  "forecast_raw_rows": 9658114,
  "forecast_valid_rows": 9643219,
  "archive_duplicate_rows": 0,
  "settlement_time_mismatches": 1268,
  "timestamp_conflict_policy": "exclude ambiguous rows without choosing either endpoint",
  "distinct_forecast_issues": 15353,
  "distinct_valid_times": 31824,
  "file_coverage": [
    {
      "path": "data\\raw\\forecast_2025.csv",
      "raw_rows": 5369698,
      "retained_future_finite_rows": 5359861,
      "excluded_future_timestamp_conflicts": 0,
      "raw_settlement_time_mismatches": 0,
      "raw_missing_issue": 0,
      "raw_missing_gmt_endpoint": 0,
      "raw_missing_wind_or_solar": 0,
      "issue_min": "2025-01-01 00:12:00+00:00",
      "issue_max": "2025-12-31 23:12:17+00:00",
      "target_min": "2025-01-01 00:30:00+00:00",
      "target_max": "2026-01-14 00:00:00+00:00"
    },
    {
      "path": "data\\raw\\forecast_2026_jun_dec.csv",
      "raw_rows": 4288416,
      "retained_future_finite_rows": 4283358,
      "excluded_future_timestamp_conflicts": 1268,
      "raw_settlement_time_mismatches": 1268,
      "raw_missing_issue": 0,
      "raw_missing_gmt_endpoint": 0,
      "raw_missing_wind_or_solar": 48,
      "issue_min": "2026-01-01 00:12:00+00:00",
      "issue_max": "2026-10-13 15:52:45+00:00",
      "target_min": "2026-01-01 00:30:00+00:00",
      "target_max": "2026-10-26 00:00:00+00:00"
    }
  ]
}

All retained rows have matching GMT and local-settlement endpoints. The complete raw2025/2026 pass identified1268 still-future2026 rows with conflicting endpoint keys, concentrated in late-April/early-May target dates; those ambiguous rows are excluded without choosing an endpoint. Raw missingness and conflict counts are in the manifest and timestamp_failure_audit.json. All arrays are finite. A/B use exactly the same target timestamps; completed periods are unchanged; all histories end by A. Future target supports are disjoint across splits. Train-only normalization is independently recorded.

Actual DST settlement-day counts: [{"date": "2025-03-30", "count": 46, "min": 1, "max": 46}, {"date": "2025-10-26", "count": 50, "min": 1, "max": 50}, {"date": "2026-03-29", "count": 46, "min": 1, "max": 46}]

Seasonal naive uses a fixed UTC lag of336 half-hour slots. Around DST this differs from matching a local-clock weekday; that is the supplied V1 definition.
