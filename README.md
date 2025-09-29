# 🏦 Smart-Premium

This project predicts **insurance premiums** using customer and policy details such as age, income, health score, claim history, etc.  
The model has been trained with **Stacking Ensemble (Random Forest + Gradient Boosting + Linear Regression)** and integrates with **MLflow** for experiment tracking.  
Finally, a **Streamlit web app** provides real-time predictions.

---

## 🚀 Features
- Data preprocessing and feature engineering
- Log-transformed target training for stability
- Ensemble model (stacking) with cross-validation
- MLflow experiment tracking
- Streamlit app for user-friendly predictions
- Outputs predictions as CSV for test data

---

## 📂 Project Structure
.
├── app/
│ └── streamlit_app.py # Streamlit app for deployment
├── artifacts/ # Trained models and saved artifacts
├── data/
│ ├── train.csv
│ ├── test.csv
│ └── sample_submission.csv
├── eda_reports
├── src/
│ ├── init.py
│ ├── features.py # Feature engineering
│ ├── train.py # Model training with MLflow
│ ├── evaluate.py # Model evaluation
│ └── predict.py # Generate predictions on test set
├── submission.csv # Final output predictions
├── README.md # Project documentation
└── .gitignore # Ignore unnecessary files

---

## 🛠 Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/sreevaidhy2907/Smart-Premium.git
   cd Smart-Premium

2. Create a virtual environment and install dependencies:
   ```bash
    python -m venv .venv
    .venv\Scripts\activate   # Windows
    source .venv/bin/activate  # Linux/Mac

    pip install -r requirements.txt

📊 Training & Evaluation

1. Train the model:
   ```bash
   python -m src.train

2. Evaluate performance:
   ```bash
   python -m src.evaluate --mode train
   python -m src.evaluate --mode test

📦 Predictions

1. Generate predictions on the test set:
   ```bash
      python -m src.predict
   
Predictions will be saved to submission.csv

🌐 Streamlit App

1. Launch the web app:
    ```bash
      cd app
      streamlit run streamlit_app.py
Enter customer details and get premium predictions instantly.

📈 MLflow Tracking

1. Start MLflow UI to explore training runs:
    ```bash
     mlflow ui
Then open http://127.0.0.1:5000 in your browser.

✅ Results

Best CV R²: ~0.129 on log scale
Final deployment model: Stacking Ensemble
Predictions clipped between 100–1300
   
