# Grammar-Scoring-Engine-from-Voice-Samples-2025
Speech-to-Score pipeline using Hugging Face Transformers and PyTorch. Features include audio preprocessing with Librosa, feature extraction via Wav2Vec2, and a regression head for automated grammar scoring. Designed to scale spoken language evaluations in a high-volume recruitment environment.

## 📌 Project Overview
Developed this project automates the evaluation of spoken language proficiency. By utilizing deep learning, the system analyzes raw audio features to predict objective grammar and fluency scores.

The system extracts high-level speech representations using a pretrained Wav2Vec2 model and predicts grammar scores using a regression-based machine learning approach.

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


           ## Pretrained Model

          This project uses the pretrained Wav2Vec2 model from Hugging Face.
          Due to size constraints, the model and processor files are not included
          in this repository.
          Model can be loaded using:
          
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
            ├── data/ --------Download Kaggle
            │   ├── audios/
            │   │   ├── train/
            │   │   └── test/
            │   │
            │   └── csvs/
            │       ├── train.csv    
            │       └── test.csv
            │
            ├── submission.csv
            │
            ├── test_predictions.csv
            │
            └── README.md

🔮 Future Enhancements

      Fine-tuning Wav2Vec2 on domain-specific speech data
      Experimenting with XGBoost / LightGBM regressors
      Adding prosodic and acoustic features
      Model ensembling for improved performance
      
👤 Author
Vinay Shivaji Vyankatkar
B.Tech (IT) | Machine Learning & AI Enthusiast
