# Predicting Human Chess Moves by Skill Level with a CNN

APS360 course project. Given a chess position and a rating band (1100–2100 Lichess rapid, five 200-point bands),
a CNN predicts the move a human in that band would play, as a probability distribution over legal moves.

## Progress report (v1)

Everything in the progress report comes from `notebooks/aps360_progress_v1.ipynb`.
Run it top to bottom on a Colab T4 GPU (about 30 min). It streams the data, trains the baselines and CNNs, evaluates
on a new month, and writes the figures and `results.json`.

| Path | Contents |
| --- | --- |
| `notebooks/aps360_progress_v1.ipynb` | Pipeline for the progress report: data, baselines, CNN, evaluation. |
| `results/progress_v1/results.json` | Every number in the report: data statistics, per-band accuracy, learning curves. |
| `results/progress_v1/numbers.tex` | The same numbers as LaTeX macros. |
| `figures/progress_v1/` | Report figures: data statistics, learning curves, qualitative examples. |

### Pipeline

1. **Data.** Stream the January 2025 Lichess rated-games dump. Keep rated rapid games between two humans that ended
   normally, with player ratings within 200 points. Assign each game to a band by mean rating, and stop once every band
   has 6,000 games (485,452 games scanned).
2. **Positions.** Sample up to 12 positions per game, skipping:
   - the first 10 plies,
   - moves made with under 30 s on the clock,
   - forced moves.

   Encode each position from the mover's view as 17 board planes plus 5 one-hot band planes (22×8×8). The label is
   the played move as from×64+to.
3. **Split.** 349,209 positions, split 80/10/10 by game (279,549 / 34,899 / 34,761).
4. **Models.**
   - Capture heuristic (no learning).
   - Linear model: score = a[from] + b[to], 180k parameters.
   - CNN: 7 conv-BN-ReLU layers at 64 channels, 1×1 conv, then an FC layer to 4096 logits; 8.6M parameters.

   Illegal moves are masked before the softmax. The models are trained with cross-entropy and Adam, and the epoch with
   the lowest validation loss is kept.
5. **New-month test.** February 2025 games (46,653 positions), never used for training or tuning.

### Results

| Model | Test top-1 | Test top-3 | Test CE | New-month top-1 |
| --- | --- | --- | --- | --- |
| Uniform random legal move | 5.3% | – | – | – |
| Capture heuristic (no learning) | 20.0% | – | – | 20.1% |
| Linear (from + to scores) | 29.2% | 53.7% | 2.49 | 29.6% |
| CNN, rating band input | 31.4% | 56.0% | 2.39 | 31.7% |
| **CNN, no band input (ablation)** | **32.5%** | **57.7%** | **2.35** | **32.6%** |

- **Overfitting:** the CNN overfits after epoch 3; validation loss rises from 2.39 to 3.51 while training loss falls
  to 1.0.
- **Band input:** it does not help yet. Shuffling the band at test time gives 31.2% vs 31.4%. Each CNN variant was
  trained once, so the 1-point gap may be run-to-run noise.

## Work in progress for the final report (v2)

These files are **not** part of the progress report.

| Path | Contents |
| --- | --- |
| `notebooks/aps360_progress.ipynb` | v2 pipeline. It saves checkpoints to Google Drive, so a Colab disconnect can resume. Changes from v1: <ul><li>residual CNN with a fully convolutional policy head (about 2.0M parameters instead of 8.6M), to fix the overfitting</li><li>more data</li><li>3 seeds with a bootstrap CI on the band effect, and a cross-band matrix</li><li>a check for players who appear in both train and test</li></ul> |
| `report/diagrams.tex`, `report/snippets.tex` | Draft LaTeX for the final report, driven by the v2 `numbers.tex`. |

## Data

[Lichess open database](https://database.lichess.org) (CC0). The dataset is not committed; the notebooks rebuild it.
