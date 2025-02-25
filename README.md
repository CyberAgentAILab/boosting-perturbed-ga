<div align="center">

<h2>Boosting Perturbed Gradient Ascent for Last-Iterate Convergence in Games</h2>

<a href='https://openreview.net/forum?id=Jrt9iWalFy'><img src='https://img.shields.io/badge/OpenReview-Paper-blue'></a>

</div>

This repository contains the code for the paper **"Boosting Perturbed Gradient Ascent for Last-Iterate Convergence in Games"**.


## Installation
```bash
pip install -r requirements.txt
```

## Run Experiments

To run experiments, use the following command:

```bash
python main.py --multirun +experiment=<SETTING>/<GAME>/<METHOD>
```

In the above command, please select `<SETTING>`, `<GAME>`, and `<METHOD>` from the options below:

- `<SETTING>`: `full` (for full feedback) or `noisy` (for noisy feedback)
- `<GAME>`: `random_payoff` or `hard_concave_convex`
- `<METHOD>`: `gabp`, `apga`, `og`, or `aog`

Specifically, to reproduce the results in our paper, run the following commands:
```bash
# Random Payoff Game with full feedback
python main.py --multirun +experiment=full/random_payoff/gabp,full/random_payoff/apga,full/random_payoff/og,full/random_payoff/aog

# Hard Concave-Convex Game with full feedback
python main.py --multirun +experiment=full/hard_concave_convex/gabp,full/hard_concave_convex/apga,full/hard_concave_convex/og,full/hard_concave_convex/aog

# Random Payoff Game with noisy feedback
python main.py --multirun +experiment=noisy/random_payoff/gabp,noisy/random_payoff/apga,noisy/random_payoff/og,noisy/random_payoff/aog

# Hard Concave-Convex Game with noisy feedback
python main.py --multirun +experiment=noisy/hard_concave_convex/gabp,noisy/hard_concave_convex/apga,noisy/hard_concave_convex/og,noisy/hard_concave_convex/aog
```

## Citation
Kenshi Abe, Mitsuki Sakamoto, Kaito Ariu, and Atsushi Iwasaki. Boosting Perturbed Gradient Ascent for Last-Iterate Convergence in Games. In ICLR, 2025

Bibtex:
```
@inproceedings{abe2025boosting,
  title={Boosting Perturbed Gradient Ascent for Last-Iterate Convergence in Games},
  author={Kenshi Abe and Mitsuki Sakamoto and Kaito Ariu and Atsushi Iwasaki},
  booktitle={ICLR},
  year={2025},
  url={https://openreview.net/forum?id=Jrt9iWalFy}
}
```