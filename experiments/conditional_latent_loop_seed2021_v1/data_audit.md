# Data and protocol audit

Official DAG archive; only NP/PJM extracted with CRC32 and SHA256 verification.
Target is the last channel OT after the official TFB long-format reader; the DAG paper defines NP/PJM OT as market electricity price. The CSV label itself does not spell out price.
Hourly chronological split: tv=floor(N*0.8), train=floor(tv*0.875). Validation/test may use prior history; future labels remain inside their own split.
All windows use stride 1; train windows have complete labels inside training. No test-driven tuning.
MSE_norm=mean(((prediction-target)/training_target_std)^2); MAE_norm=mean(abs(prediction-target)/training_target_std). Equal complete windows, no dropped test windows.
Covariates are released grid/load/wind forecasts. Their issue time/vintage is not encoded in the archive. This is benchmark-available future information, not a verified operational availability claim.
DAG scripts use MAE validation and drop_last=True for multivariate training; preserved in B0. B1-B5 select decoded MSE and share full-window training (drop_last=False), 50 epochs/patience 5.
Frozen-AE representation is a pointwise scalar feature expansion, not evidence of physical latent-state identification.

## NP

```json
{
  "shape": [
    52416,
    3
  ],
  "columns": [
    " Grid load forecast",
    " Wind power forecast",
    "OT"
  ],
  "target": "OT",
  "raw_channel_timestamp_alignment_verified": true,
  "regular_hourly_spacing_verified": true,
  "boundaries": {
    "train": [
      0,
      36690
    ],
    "validation": [
      36690,
      41932
    ],
    "test": [
      41932,
      52416
    ]
  },
  "boundary_dates": {
    "train_first": "2013-01-01 00:00:00",
    "train_last": "2017-03-09 17:00:00",
    "val_first": "2017-03-09 18:00:00",
    "val_last": "2017-10-14 03:00:00",
    "test_first": "2017-10-14 04:00:00",
    "test_last": "2018-12-24 23:00:00"
  },
  "target_scaler": {
    "mean": [
      29.0189341782502
    ],
    "std": [
      9.487614364166628
    ],
    "n_fit": 36690
  },
  "covariate_scaler": {
    "mean": [
      43985.083442354866,
      1443.3550149904606
    ],
    "std": [
      8709.006813103179,
      1114.3319004599693
    ],
    "n_fit": 36690
  },
  "missing": {
    " Grid load forecast": 0,
    " Wind power forecast": 0,
    "OT": 0
  },
  "sample_sha256": "29e142d895fa3254bf138e3bec32051e081794a7393a642db36a5d21859d36e1",
  "tasks": {
    "24": {
      "history": 168,
      "horizon": 24,
      "windows": [
        36499,
        5219,
        10461
      ],
      "first_origins": [
        168,
        36690,
        41932
      ],
      "last_origins": [
        36666,
        41908,
        52392
      ],
      "input_shapes": {
        "y_hist": [
          "B",
          168,
          1
        ],
        "c_hist": [
          "B",
          168,
          2
        ],
        "c_future": [
          "B",
          24,
          2
        ]
      },
      "loss_only_future_target_shape": [
        "B",
        24,
        1
      ],
      "future_latent_shape": [
        "B",
        24,
        32
      ]
    },
    "360": {
      "history": 720,
      "horizon": 360,
      "windows": [
        35611,
        4883,
        10125
      ],
      "first_origins": [
        720,
        36690,
        41932
      ],
      "last_origins": [
        36330,
        41572,
        52056
      ],
      "input_shapes": {
        "y_hist": [
          "B",
          720,
          1
        ],
        "c_hist": [
          "B",
          720,
          2
        ],
        "c_future": [
          "B",
          360,
          2
        ]
      },
      "loss_only_future_target_shape": [
        "B",
        360,
        1
      ],
      "future_latent_shape": [
        "B",
        360,
        32
      ]
    }
  }
}
```

## PJM

```json
{
  "shape": [
    52416,
    3
  ],
  "columns": [
    " System load forecast",
    " Zonal COMED load foecast",
    "OT"
  ],
  "target": "OT",
  "raw_channel_timestamp_alignment_verified": true,
  "regular_hourly_spacing_verified": true,
  "boundaries": {
    "train": [
      0,
      36690
    ],
    "validation": [
      36690,
      41932
    ],
    "test": [
      41932,
      52416
    ]
  },
  "boundary_dates": {
    "train_first": "2013-01-01 00:00:00",
    "train_last": "2017-03-09 17:00:00",
    "val_first": "2017-03-09 18:00:00",
    "val_last": "2017-10-14 03:00:00",
    "test_first": "2017-10-14 04:00:00",
    "test_last": "2018-12-24 23:00:00"
  },
  "target_scaler": {
    "mean": [
      31.53727002737804
    ],
    "std": [
      21.829201912662867
    ],
    "n_fit": 36690
  },
  "covariate_scaler": {
    "mean": [
      91147.5884982284,
      11402.336154265467
    ],
    "std": [
      16181.055018553472,
      2180.3368639388077
    ],
    "n_fit": 36690
  },
  "missing": {
    " System load forecast": 0,
    " Zonal COMED load foecast": 0,
    "OT": 0
  },
  "sample_sha256": "14feec9daa86e14ba4ed062c690468fa615af6fb18ba56006149b087e26058e5",
  "tasks": {
    "24": {
      "history": 168,
      "horizon": 24,
      "windows": [
        36499,
        5219,
        10461
      ],
      "first_origins": [
        168,
        36690,
        41932
      ],
      "last_origins": [
        36666,
        41908,
        52392
      ],
      "input_shapes": {
        "y_hist": [
          "B",
          168,
          1
        ],
        "c_hist": [
          "B",
          168,
          2
        ],
        "c_future": [
          "B",
          24,
          2
        ]
      },
      "loss_only_future_target_shape": [
        "B",
        24,
        1
      ],
      "future_latent_shape": [
        "B",
        24,
        32
      ]
    },
    "360": {
      "history": 720,
      "horizon": 360,
      "windows": [
        35611,
        4883,
        10125
      ],
      "first_origins": [
        720,
        36690,
        41932
      ],
      "last_origins": [
        36330,
        41572,
        52056
      ],
      "input_shapes": {
        "y_hist": [
          "B",
          720,
          1
        ],
        "c_hist": [
          "B",
          720,
          2
        ],
        "c_future": [
          "B",
          360,
          2
        ]
      },
      "loss_only_future_target_shape": [
        "B",
        360,
        1
      ],
      "future_latent_shape": [
        "B",
        360,
        32
      ]
    }
  }
}
```
