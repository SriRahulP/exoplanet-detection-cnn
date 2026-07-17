#  Exoplanet Detection Using Deep Learning

**A CNN that classifies exoplanets from NASA Kepler light curves — and honestly confronts what happens when you train on 37 positive examples out of 5,087 stars.**

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

📫 **[Sri Rahul P]** · [LinkedIn](https://www.linkedin.com/in/srirahulp2005) · [Email](mailto:rahulnjr111@gmail.com) · [Resume](https://your-resume-link.com)

---

## At a Glance

| | |
|---|---|
| **Task** | Binary classification — exoplanet vs. non-exoplanet, from time-series brightness data |
| **Data** | 5,087 NASA Kepler stars, 3,197 time-steps each, 0.73% positive class |
| **Model** | 1D CNN (Conv1D + MaxPooling), TensorFlow/Keras, ~210K params |
| **Headline result** | **0.95 AUC-ROC** |
| **The twist** | Default threshold gave 0% recall despite high AUC — diagnosed and fixed via threshold tuning |

## Table of Contents
- [Why This Project Is Different](#why-this-project-is-different)
- [Problem](#problem)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [How to Run](#how-to-run)
- [Repo Structure](#repo-structure)
- [Roadmap](#roadmap)

## Why This Project Is Different

Most beginner exoplanet-detection projects report a single accuracy number and
stop. This one doesn't — because on this dataset, **accuracy is actively
misleading**. A model that predicts "no exoplanet" for every star scores
99.27% accuracy while catching zero real planets.

Instead, this project:
- Surfaces that failure mode instead of hiding it
- Diagnoses *why* it happens (extreme 137:1 class imbalance)
- Fixes it with a documented, threshold-tuning trade-off, not just a rerun with different hyperparameters
- Evaluates using metrics that actually make sense for imbalanced, high-stakes classification (AUC-ROC, precision/recall — not accuracy)

If you're reviewing this as a recruiter or interviewer: the interesting part
of this project isn't the CNN architecture (fairly standard) — it's the
diagnostic process around evaluating it honestly. Happy to walk through it.

## Problem
When a planet passes in front of its star (a "transit"), it blocks a tiny
fraction of the star's light, causing a small periodic dip in brightness.
This project trains a CNN to recognize that pattern automatically from raw
Kepler telescope data — the same approach used in Google/UT Austin's
**AstroNet** research (Shallue & Vanderburg, 2018), which discovered two real
exoplanets missed by earlier detection methods.

![Light curve comparison](assets/light_curve_comparison.png)

## Dataset
[Kaggle: Exoplanet Hunting in Deep Space](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data)
(NASA Kepler mission data)

| | Rows | Confirmed exoplanets | Imbalance |
|---|---|---|---|
| Train | 5,087 | 37 | 0.73% |
| Test | 570 | 5 | 0.88% |

Each row = one star's brightness measured at 3,197 points in time.

## Approach
1. **Preprocessing** — Detrended each light curve with a Savitzky-Golay
   filter to remove slow stellar-variability drift, then z-score normalized,
   so the model learns dip *shape* rather than absolute brightness.
   ![Preprocessing](assets/preprocessing_before_after.png)
2. **Model** — 3-block Conv1D + MaxPooling1D CNN → dense layers → sigmoid
   output. ~210K parameters, built in TensorFlow/Keras.
3. **Imbalance handling** — `class_weight="balanced"` during training,
   penalizing a missed exoplanet ~137x more heavily than a misclassified
   non-planet.
4. **Training** — Adam optimizer, binary cross-entropy loss, early stopping
   on validation loss to avoid overfitting on a tiny positive class.

## Results

| Metric | Score |
|---|---|
| **AUC-ROC** | **0.95** |
| Recall @ threshold 0.5 (default) | 0% |
| Recall @ threshold 0.05 (tuned) | 40% |
| Recall @ threshold 0.01 (aggressive) | 100% |

![Confusion Matrix](assets/confusion_matrix.png)
![Threshold Trade-off](assets/threshold_tradeoff.png)

**Reading these numbers honestly:** the test set contains only 5 real
planets, so single-digit swings in "planets caught" move recall by 20
points — these results should be read as a directional trade-off, not a
precise benchmark. The 0.95 AUC-ROC is the more statistically stable
number here, and the one I'd stand behind confidently in an interview.

## How to Run
```bash
git clone https://github.com/SriRahulP/exoplanet-detection-cnn.git
cd exoplanet-detection-cnn
pip install -r requirements.txt
```
Download `exoTrain.csv` and `exoTest.csv` from the
[Kaggle dataset link above](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data)
and place them in `data/`.

```bash
python src/preprocessing.py   # builds and saves processed arrays
python src/train.py           # trains and saves the model
python src/evaluate.py        # evaluates across thresholds
```

## Repo Structure
```
exoplanet-detection-cnn/
├── data/                 # place downloaded CSVs here (not committed)
├── src/
│   ├── explore_data.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── assets/               # plots used in this README
├── requirements.txt
└── README.md
```

## Roadmap
- [ ] Phase-fold light curves using a detected orbital period (Box Least
      Squares algorithm) — the full AstroNet approach — to further separate
      signal from noise
- [ ] Dual-branch CNN using both a global and local view of the folded curve
- [ ] SMOTE / synthetic oversampling to address imbalance directly during
      training, rather than only via loss weighting

## Tech Stack
Python · TensorFlow/Keras · NumPy · Pandas · SciPy · scikit-learn ·
Matplotlib · Seaborn

---

📫 **Interested in discussing this project?** Reach out — [LinkedIn](https://www.linkedin.com/in/srirahulp2005) · [Email](mailto:rahulnjr111@gmail.com)
