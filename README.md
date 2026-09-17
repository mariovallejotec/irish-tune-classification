# Irish Tune Classification

Classifying Irish traditional tune types (reel, jig, hornpipe, etc.) directly from raw ABC-notation text with a character-level LSTM, compared against a support-vector-machine baseline built on hand-crafted symbolic features.

Submitted to ISMIR 2026 as a Late-Breaking Demo. Project page: https://mariovallejotec.github.io/irish-tune-classification/

## Data

Tunes and type labels come from [The Session](https://github.com/adactio/TheSession-data) (`csv/tunes.csv`), not redistributed in this repo. Download it separately and place `tunes.csv` in the repo root before running `preprocess.py`.

## Pipeline

```
preprocess.py                  # parses tunes.csv, splits train/val/test at the tune level
build_features.py              # music21 features for the SVM baseline
build_features_augmented.py    # features for the augmented training set
augment_abc.py                 # pitch-transposition augmentation
build_augmented_train.py       # builds the augmented training CSV
train_svm_baseline.py          # SVM (RBF) baseline
train_svm_augmented.py         # SVM trained on augmented data
train_lstm.py                  # character-level bidirectional LSTM
```

## Results

| Model | Accuracy | Macro-F1 |
|---|---|---|
| SVM (RBF) | 82.7% | **0.739** |
| LSTM (class-weighted) | 78.4% | 0.653 |
| LSTM (unweighted) | **83.9%** | 0.692 |

Full results, ablation, and figures: `paper/main.pdf` or the project page above.
