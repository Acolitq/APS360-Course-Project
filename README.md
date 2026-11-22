# Predicting Human Chess Moves by Skill Level with a CNN

APS360 course project. Given a chess position and a rating band (1100–2100 Lichess rapid, five 200-point bands),
a CNN predicts the move a human in that band would play, as a probability distribution over legal moves.

## Layout

| Path | Contents |
| --- | --- |
| `notebooks/aps360_progress.ipynb` | **v2** progress pipeline (current). Run top to bottom on a Colab T4 GPU, about 60–80 min. Writes figures, `results.json`, `numbers.tex` and model weights to `progress_outputs.zip`. |
| `notebooks/aps360_progress_v1.ipynb` | v1 pipeline that produced `results/progress_v1` and `figures/progress_v1`. |
| `results/progress_v1/` | v1 run: every number (`results.json`) and LaTeX macros (`numbers.tex`). |
| `figures/progress_v1/` | v1 figures. |
| `report/diagrams.tex` | TikZ diagrams of the baselines and the primary model. |
| `report/snippets.tex` | Paste-ready report text, table and figure blocks. All numbers are macros from `numbers.tex`. |

## v2 changes (in response to the v1 results)

| v1 finding | v2 change |
| --- | --- |
| CNN overfit after epoch 3 (val loss 2.39 → 3.51); 97% of its 8.6M parameters sat in one 2048→4096 FC layer | Residual CNN (6 blocks, 128 channels) with a fully convolutional from–to policy head, about 2.0M parameters; cosine LR decay; v1 still trained for comparison |
| Band input did not help (31.4% with band vs 32.5% without; shuffled band 31.2%), single seed | 3 seeds per variant, paired game-bootstrap CI on the band gain, band-X-tested-on-band-Y matrix, mean (not only max) band effect |
| 349k positions vs ~2M planned; all games from the first hours of 1 Jan | 12,000 games per band, each eligible game kept with probability 0.25 to widen the time window; time span recorded |
| Split by game only | Hashed player IDs; accuracy reported separately for test games whose players never appear in training |
| Figures 11 in wide (fonts ~4 pt in the report), legend over bars | Figures drawn at 5.5 in (ICLR text width), 7–8 pt fonts, legends outside, kept epoch marked; PDF and PNG |

## v1 results (train month 2025-01, final test on 2025-02)

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
Usernames are never stored, only one-way hashes used for the player-overlap check.
