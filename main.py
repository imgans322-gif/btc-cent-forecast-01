#!/usr/bin/env python3
"""btc-cent-forecast-01 entry: forecast BTC's price 240 minutes ahead.

Method (all stdlib, deterministic, offline). The score is mean absolute
error in cents, so the right target is the conditional *median* of the
price 4 hours out, not a bold directional bet. Two signals, averaged:

1. Short-horizon mean reversion: the price tends to drift back toward
   its trailing 6-hour mean over the next 4 hours, so the forecast pulls
   halfway back:  adj = -0.5 * log(last / SMA360).

2. A small ridge regression over six short-horizon log features
   (deviations from the 3h / 6h / 12h / 24h trailing means and the
   1h / 4h log returns), fitted in-session on 64 hourly pseudo-cases
   cut from the kit's practice window (Nov 7-11 2026, each checkpoint
   with >= 24h of history and a known +240m actual). Ridge penalty
   1e-4, chosen by 4-block time-series cross-validation.

The two log-adjustments are averaged (equal weight: the 50/50 blend
was the most consistent across slices; heavier ridge weight won
in-sample but not on the practice cases) and capped at +/-2%, so the
forecast never strays far from the checkpoint close (the persistence
baseline). Local evaluation, all in cents of mean absolute error:

- 4-block time-series cross-validation on 64 hourly pseudo-cases
  (ridge refit on the other three blocks each time): 39,097 vs
  42,871 for persistence (-8.8%).
- The 5 scoreable practice cases (p00-p04; each actual is the next
  case's last price, so p05's actual is not in the kit): 46,513 vs
  47,055 for persistence (-1.2%). Small, and the two cases in the
  steep downtrend (p01, p02) actually lose to persistence there;
  the edge comes from the bounce cases (p03, p04).
- 4h-spaced pseudo-cases (ridge weights fitted on the same window,
  so optimistic): 38,057 vs 41,768 (-8.9%).

The published 6-case practice baseline is 44,521.58 cents; no
6-case score of this entry can be computed locally because p05's
actual (+240m = the end of the practice window) is not in the kit.

I/O follows the kit exactly: read every <id>.in in the input
directory (MAKEFUN_INPUT_DIR, else /input), write <id>.out to the
output directory (MAKEFUN_OUTPUT_DIR, else /output) as
{"forecasts": [{"case": "<id>", "price": <number>}]}.
"""

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999
CAP = 0.02  # max |log adjustment| vs the checkpoint close

# Ridge weights, fitted in-session (see module docstring). Feature order:
# [log(P/SMA180), log(P/SMA360), log(P/SMA720), r60, r240, log(P/SMA1440)]
RIDGE_W = [-0.504776, -0.247269, 0.819939, 0.088480, -0.228885, -0.406490]
SMA_REV_C = 0.5  # weight on the 6-hour mean-reversion signal


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _price5(x):
    """Keep a price on the grid's 5 decimals and inside (0, 10_000_000)."""
    return round(_clamp(x, MIN_PRICE, MAX_PRICE) * 1e5) / 1e5


def _mean_tail(prices, n):
    """Mean of the last n closes (or all of them if fewer)."""
    k = min(n, len(prices))
    return sum(prices[-k:]) / k


def _log_ratio(a, b):
    return math.log(a / b)


def forecast(prices):
    """Forecast the close 240 minutes after the last price in `prices`."""
    last = prices[-1]
    n = len(prices)

    # Signal 1: halfway back toward the trailing 6-hour mean.
    adj_sma = -SMA_REV_C * _log_ratio(last, _mean_tail(prices, 360))

    # Signal 2: ridge over six short-horizon log features.
    feats = [
        _log_ratio(last, _mean_tail(prices, 180)),
        _log_ratio(last, _mean_tail(prices, 360)),
        _log_ratio(last, _mean_tail(prices, 720)),
        _log_ratio(last, prices[-61]) if n >= 61 else 0.0,
        _log_ratio(last, prices[-241]) if n >= 241 else 0.0,
        _log_ratio(last, _mean_tail(prices, 1440)),
    ]
    adj_ridge = sum(w * f for w, f in zip(RIDGE_W, feats))

    # Ensemble average, capped so the forecast stays near the anchor.
    adj = _clamp(0.5 * (adj_sma + adj_ridge), -CAP, CAP)
    return _price5(last * math.exp(adj))


def forecast_case(case):
    markets = case["markets"]
    if not isinstance(markets, list) or len(markets) < 1:
        raise ValueError("case holds no market")
    prices = markets[0]["prices"]
    if not isinstance(prices, list) or len(prices) == 0:
        raise ValueError("market has no prices")
    last = prices[-1]
    if not isinstance(last, (int, float)) or not math.isfinite(last) or last <= 0:
        raise ValueError("last price is not a positive finite number")
    try:
        return forecast(prices)
    except (ValueError, ZeroDivisionError, OverflowError):
        # Never crash a case: fall back to the persistence guess, which
        # is always a valid forecast.
        return _price5(last)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in inputs:
        case_id = name[: -len(".in")]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        price = forecast_case(case)
        out = {"forecasts": [{"case": case_id, "price": price}]}
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
