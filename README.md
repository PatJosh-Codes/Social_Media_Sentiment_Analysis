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

## Dataset

Place your CSV file at `dataset/sentiment_dataset.csv`.

The file must contain at least two columns:

| Column name (any of these)             | Description              |
|----------------------------------------|--------------------------|
| `text` / `tweet` / `post` / `content` | Raw social-media text    |
| `label` / `sentiment` / `target`      | Sentiment class          |

Supported label values: `positive`, `negative`, `neutral` (case-insensitive).

---

## Installation

### 1. Clone / download the project

```bash
git clone <repo-url>
cd Social_Media_Sentiment_Analysis
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\\Scripts\\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training the Model

Make sure your dataset is at `dataset/sentiment_dataset.csv`, then run:

```bash
python train_model.py
```

The script will:

1. Load and clean the dataset
2. Apply the full preprocessing pipeline
3. Fit a TF-IDF vectoriser (5 000 features)
4. Train all four models and print a comparison table
5. Save the best model → `model.pkl`
6. Save the vectoriser → `vectorizer.pkl`
7. Save the label encoder → `label_encoder.pkl`

Sample output:

```
[INFO] Loaded dataset: 1000 rows, 6 columns
[INFO] Classes: ['negative', 'neutral', 'positive']

=======================================================
  Training: Logistic Regression
=======================================================
  Accuracy : 0.8650
  ...

=================================================================
  MODEL COMPARISON
=================================================================
 Model                    Accuracy  Precision  Recall  F1-Score
 Logistic Regression        0.865      0.867   0.865     0.864
 Support Vector Machine     0.870      0.872   0.870     0.869
 Random Forest              0.845      0.848   0.845     0.844
 Multinomial Naïve Bayes    0.820      0.825   0.820     0.819

[INFO] Best model: Support Vector Machine (Accuracy = 0.8700)
```

---

## Running the Flask Application

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

> **Note:** `model.pkl`, `vectorizer.pkl`, and `label_encoder.pkl` must exist before running the app. Run `train_model.py` first if they are missing.

---

## API

The app exposes a single REST endpoint:

### `POST /predict`

**Request body (JSON)**

```json
{ "text": "I absolutely love this product!" }
```

**Response (JSON)**

```json
{
  "sentiment":  "Positive",
  "confidence": 94.3,
  "all_probs":  { "Positive": 0.943, "Negative": 0.032, "Neutral": 0.025 },
  "error":      null
}
```

---

## Example Screenshots

| Input & Analysis | Result with Confidence |
|-----------------|------------------------|
| *(add screenshot here)* | *(add screenshot here)* |

---

## Acknowledgements

- [Scikit-learn](https://scikit-learn.org/) — machine learning
- [NLTK](https://www.nltk.org/) — natural language processing
- [Flask](https://flask.palletsprojects.com/) — web framework
- [emoji](https://pypi.org/project/emoji/) — emoji handling
