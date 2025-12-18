Grammar-Scoring-Engine-from-Voice-Samples-2025

Speech-to-score pipeline using Hugging Face Transformers and PyTorch.
Includes audio preprocessing with Librosa, feature extraction via Wav2Vec2, and regression-based grammar scoring.
Designed for scalable spoken language evaluation in high-volume recruitment environments.

📌 Project Overview

This project automates the evaluation of spoken language proficiency.
It analyzes raw speech signals and predicts objective grammar scores using deep learning–based speech representations.

A pretrained Wav2Vec2 model is used to extract high-level speech embeddings, which are then passed to a regression-based machine learning model for scoring.

🚀 Features

Audio preprocessing and feature extraction from raw .wav files

Deep speech embeddings using Wav2Vec2 (Transformer-based model)

Grammar score prediction using Random Forest Regressor

Evaluation using Mean Absolute Error (MAE)

Fully CPU-compatible (no GPU required)

Kaggle-ready submission pipeline

🧠 Model Architecture

Feature Extractor: Wav2Vec2 (pretrained, Hugging Face Transformers)

Embedding Size: 768

Regression Model: Random Forest Regressor

Evaluation Metric: MAE

📦 Pretrained Model

This project uses a pretrained Wav2Vec2 model from Hugging Face.

Due to size constraints, the model and processor files are not included in this repository.

They can be loaded as follows:

from transformers import Wav2Vec2Processor, Wav2Vec2Model

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

📊 Results

Achieved ~0.53 MAE on the validation set

Generated predictions for unseen test audio samples

🛠️ Tech Stack

Python

PyTorch

Hugging Face Transformers

Librosa

Scikit-learn

NumPy, Pandas

📁 Project Structure
Grammar-Scoring-Engine/
│
├── Grammar_Scoring.ipynb
│
├── data/                     # Download from Kaggle
│   ├── audios/
│   │   ├── train/
│   │   └── test/
│   │
│   └── csvs/
│       ├── train.csv
│       └── test.csv
│
├── submission.csv
├── test_predictions.csv
└── README.md

📥 Dataset

The dataset used in this project is sourced from Kaggle:

SHL Intern Hiring Assessment 2025 – Grammar Scoring from Voice Samples

Due to size and licensing constraints, audio files are not included in this repository and must be downloaded separately from Kaggle.

🔮 Future Enhancements

Fine-tuning Wav2Vec2 on domain-specific speech data

Experimenting with XGBoost / LightGBM regressors

Adding prosodic and acoustic features

Model ensembling for improved performance

👤 Author

Vinay Shivaji Vyankatkar
B.Tech (IT) | Machine Learning & AI Enthusiast
