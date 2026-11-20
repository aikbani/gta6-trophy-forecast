# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/aikbani/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/aikbani/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one — every PS5 trophy's Exophase earn rate at 2026-11-26 00:00 UTC, as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) on the logit scale. Entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) (`gta6-trophy-rarity-forecast-01`, $250, 20 places).

## How it works

The earn rate is `earners / players_tracked`, and the two sides move for different reasons, so `main.py` forecasts them separately and divides at the end. Pure standard library, deterministic, no network, ~50 ms per case.

**Earners are recovered as integers.** Exophase prints rates to 2 decimals, so `earners = round(rate * players / 100)` is exact in practice — on launch night every trophy implies 4–13 earners, and the implied total (943) matches the page's own `trophies_earned` count.

**Two regimes, detected from the case history:**

1. **Flat history** (the practice cases p00/p01 — every read identical). Exophase refreshes a game's counts on its own schedule, not with every player's trophy, so a flat page means "no refresh yet", not "nothing happened". The kit rules publish the first refresh: 638 players tracked from launch night, refreshed to 3,868 by the 04:30 UTC read, every rate but the one at 100% lower. The forecast is therefore a dilution forecast:
   `median = (1.13 * earners + 0.8) / (players_last * 3868/638) * 100`, spread σ = 0.18 logit. The earners growth (1.13× + 0.8) is calibrated so the house baseline scores ~1.51 on the practice under this answer, as the rules state it does (typical miss ~0.68 points). The trophy at 100% stays at 100%.

2. **Evolving history** (the final case f00 — ~36 h of reads, many refreshes). Per trophy, a weighted least-squares line of `logit(rate)` against `ln(1 + hours since release)`, anchored at the last read, slope shrunk toward the global median slope (empirical-Bayes, weights = binomial information ≈ earners). A second view fits `log(earners)` against `log(players)` per trophy (elasticity, also shrunk) and divides extrapolated earners by extrapolated players; players themselves follow a power law in time since release (a linear-in-time fit of log players was tested and rejected — it overshoots the player count 20×). The two logit forecasts are averaged, the projected move is damped 20% toward no-change, and medians pass through a partial isotonic (PAVA) pass ordered by last observed rate so the difficulty hierarchy stays monotone up to noise. Spread is per trophy (residual scatter + slope uncertainty + horizon term, clamped to [0.15, 0.60] logit).

## Local backtest (replay driver, `dev/` in the workdir — not shipped)

The scorer was validated against the rules' worked example exactly. The practice answers are not in the kit (the server holds them), so the dev backtest scores against the hypothesised refreshed page — old integer earners, grown modestly, divided by 3,868 players, trophy #2 at 100% — under several growth scenarios:

| Truth scenario | Dev loss |
| --- | --- |
| Central (median exactly right) | **0.039** |
| Earners ×1.2 / ×1.25 / ×1.3 / ×1.4 | 0.047 / 0.043 / 0.043 / 0.052 |
| Earners ×1.1 | 0.086 |
| Earners unchanged (×1.0) | 0.136 |
| Earners ×1.25 + per-trophy noise 0.10 logit | 0.064 |
| Pure additive (+1.5 earners/trophy) | 0.040 |

Leaderboard bars (19 Nov 12:35 UTC): #1 0.0591 · #3 0.0836 · #20 0.127 · #50 0.144 · house baseline ~1.51. The central scenarios beat #1; only the earners-unchanged scenario (which contradicts the rules' stated baseline loss of ~1.51) falls to ~rank 30.

The final-case model was also tested on six synthetic final histories (cohort simulation: power-law player growth, saturating earn probabilities, Exophase-style refresh steps every 2–7 h, target 4.5 days out): **mean loss 0.123** (per-world 0.10–0.18, coverage 0.98). The synthetic worlds are the author's invention, so this is a robustness check, not a promise.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input dir, writes one `<id>.out` per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, ≤ 65,536 bytes, deterministic byte-for-byte across runs.

## Files

- `main.py` — the whole forecaster (the entrypoint).
- `LICENSE` — MIT.
- `.makefun/owner.json` — claim wallet.
