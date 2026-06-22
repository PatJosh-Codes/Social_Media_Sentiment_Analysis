# Social Media Sentiment Analysis Using Machine Learning

A Flask web application that predicts the sentiment (Positive / Negative / Neutral) of social media posts using classical machine learning models trained with TF-IDF features.

---

## Features

- **Four ML models** trained and compared: Logistic Regression, Support Vector Machine, Random Forest, and Multinomial Naïve Bayes
- **Best model auto-selected** and saved to `model.pkl`
- **Full text preprocessing pipeline**: lowercasing, URL/mention removal, emoji conversion, stop-word removal, lemmatisation
- **TF-IDF vectorisation** with up to 5 000 features
- **Confidence scores** and per-class probability breakdown
- **Modern responsive UI** — dark theme, animated bars, example texts
- **Keyboard shortcut**: `Ctrl + Enter` to analyse

---

## Project Structure

```
Social_Media_Sentiment_Analysis/
│
├── app.py                 # Flask application entry-point
├── prediction.py          # Load model + predict sentiment
├── preprocess.py          # Text preprocessing pipeline
├── train_model.py         # Train, evaluate, and save best model
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html         # Main UI page
│
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   └── images/
│
├── dataset/
│   └── sentiment_dataset.csv   # ← place your dataset here
│
└── notebooks/
    └── Social_Media_Sentiment_Analysis.ipynb
```

---



