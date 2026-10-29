# Spam Message Detection

## Project Overview
A machine learning project that classifies SMS messages as **SPAM** or **HAM** (legitimate messages).

## Features
- Text preprocessing
- TF-IDF text vectorization
- Logistic Regression classification
- Spam probability estimation
- Interactive message input in the terminal
- Saved model and vectorizer for reuse

## Technologies Used
- Python
- scikit-learn
- TF-IDF
- Logistic Regression
- Joblib

## Project Files
- `spam_detector.py` — runs the spam detector
- `spam_model.pkl` — saved machine learning model
- `tfidf_vectorizer.pkl` — saved TF-IDF vectorizer
- `requirements.txt` — Python dependencies

## How to Run

1. Install Python 3.
2. Install dependencies:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

3. Run the detector:

   ```bash
   python3 spam_detector.py
   ```

4. Enter a message and view the prediction.

## Example Results

| Message | Prediction |
|---|---|
| Congratulations! You won a free iPhone! Claim now! | SPAM |
| Hey, are you coming to college tomorrow? | HAM |
| URGENT! Claim your cash prize now! | SPAM |

## Learning Outcomes
This project demonstrates basic text preprocessing, feature extraction, model inference, and evaluation concepts in machine learning.

## Future Improvements
- Evaluate the model on a held-out test dataset
- Compare multiple classification algorithms
- Build a web interface using Streamlit 
