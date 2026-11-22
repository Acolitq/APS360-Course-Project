# Predicting Human Chess Moves by Skill Level with a CNN

APS360 course project. Given a chess position and a rating band (1100–2100 Lichess rapid, five 200-point bands),
a CNN predicts the move a human in that band would play, as a probability distribution over legal moves.

## Layout

| Path | Contents |
| --- | --- |
| `notebooks/aps360_progress.ipynb` | Full progress-report pipeline: stream a Lichess monthly dump, filter, encode, split by game, baselines, CNN, figures, new-month test. Run top to bottom on a Colab GPU (~20–30 min). |
| `results/progress/results.json` | Every number from the progress run (config, data stats, test metrics, learning curves). |
| `results/progress/numbers.tex` | The same numbers as LaTeX macros for the report. |
| `figures/progress/` | Data statistics, learning curves / per-band accuracy, and qualitative examples. |

## Progress-run summary (train month 2025-01, final test on 2025-02)

| Model | Test top-1 | Test top-3 | Test CE | New-month top-1 |
| --- | --- | --- | --- | --- |
| Uniform random legal move | 5.3% | – | – | – |
| Capture heuristic (no learning) | 20.0% | – | – | 20.1% |
| Linear (from + to scores) | 29.2% | 53.7% | 2.49 | 29.6% |
| CNN, rating band input | 31.4% | 56.0% | 2.39 | 31.7% |
| CNN, no band input (ablation) | 32.5% | 57.7% | 2.35 | 32.6% |

349,209 positions from 30,000 games (6,000 per band), split 80/10/10 by game.

## Data

[Lichess open database](https://database.lichess.org) (CC0). The dataset is not committed; the notebook rebuilds it.
