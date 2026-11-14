# btc-cent-forecast-01 entry

Forecast Bitcoin's price 240 minutes ahead from the last 72 hours of
1-minute closes. One program, `main.py` (Python 3.13, standard library
only), run once per case: it reads every `<id>.in` in the input
directory and writes `<id>.out` as
`{"forecasts": [{"case": "<id>", "price": <number>}]}`.

## Method

The score is mean absolute error in cents, so the forecast targets
the conditional median of the price 4 hours out. It is the checkpoint
close adjusted by the average of two small log-signals, capped at
+/-2%:

1. **Mean reversion to the trailing 6-hour mean** — half the
   log-distance back: `-0.5 * log(last / SMA360)`. Over the practice
   window, 4-hour-ahead returns were negatively correlated with
   recent returns (about -0.36 vs the last 4h, -0.53 vs the last 24h),
   so chasing momentum loses and fading the short-term move wins.
2. **Ridge regression** over six short-horizon log features
   (deviations from the 3h / 6h / 12h / 24h trailing means and the
   1h / 4h log returns), fitted on 64 hourly pseudo-cases cut from
   the kit's practice window (Nov 7-11 2026), penalty 1e-4, chosen
   by 4-block time-series cross-validation. Weights are hard-coded
   in `main.py`.

Momentum extrapolation, volatility scaling, anchor smoothing and
trend-gated reversion were all tried and rejected: each either lost
to persistence or only won in-sample.

## Local scores (mean absolute error, cents; lower is better)

| evaluation | this entry | persistence baseline |
|---|---|---|
| 4-block CV, 64 hourly pseudo-cases | 39,097 | 42,871 |
| practice cases p00-p04 (scoreable) | 46,513 | 47,055 |
| 4h-spaced pseudo-cases (in-sample ridge) | 38,057 | 41,768 |

p05's actual is not in the kit (its target is the end of the
practice window), so no 6-case local score exists; the published
6-case practice baseline is 44,521.58 cents. Per-case runtime is
about 10 ms (limit: 8 s). Two runs on the same case produce
byte-identical output.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=../btc-cent-forecast-01-kit/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Files

- `main.py` — the entry (only program at the repo root)
- `LICENSE` — MIT, from the kit starter
- `.makefun/owner.json` — payout wallet claim file
