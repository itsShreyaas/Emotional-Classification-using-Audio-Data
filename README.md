# Emotional Classification using Audio Data

---

## What this project does

This project classifies human speech audio into one of six emotions using deep learning. MFCC features are extracted from audio files and fed into a CNN to learn patterns linked to each emotion.

Emotions: angry, calm, fearful, happy, neutral, sad

---

## Dataset

RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song
Livingstone and Russo (2018), PLOS ONE. https://doi.org/10.1371/journal.pone.0196391

- 24 actors (12 male, 12 female)
- Around 2068 audio samples across 6 emotion classes
- Files are organised into per-emotion folders inside dataset/

The dataset folder is not included in this repo because of file size. Download the dataset from https://drive.google.com/drive/folders/1kAZr-x6Bo7ZjcYIoZaiV0WriIERQCxnj?usp=drive_link.

---

## Method

- Feature extraction: MFCC + delta + delta-delta (40 coefficients, stacked as 3 channels)
- Augmentation: gaussian noise, time stretching, pitch shifting applied to training data only
- Model: 4-block CNN with batch normalisation and dropout
- Train/test split: 80% training, 20% testing (stratified)
- Optimiser: Adam with ReduceLROnPlateau and EarlyStopping

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 92.07% |
| Model | MFCC + CNN |
| Dataset | RAVDESS (combined) |

---

## How to run

1. Clone the repo
2. Install dependencies
3. Download RAVDESS and set up the dataset folder
4. Open emotion_classification_v2.ipynb in VS Code or Google Colab
5. Run all cells top to bottom

---

## Dependencies

```
pip install librosa tensorflow scikit-learn seaborn matplotlib numpy
```

---

## References

Livingstone S, Russo F (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLOS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391

---

## Authors

Yogesh R Mehta

Shreyas Gupta
