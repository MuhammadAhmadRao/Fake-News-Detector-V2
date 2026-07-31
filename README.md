# Fake News Detection

NLP-based ML pipeline to classify news articles as Fake or Real. Built for an AI/ML internship assessment.

## Dataset
- **Source**: ISOT Fake News Dataset (`fake.csv`, `true.csv`)
- **Total**: 44,898 articles (23,481 Fake, 21,417 Real)
- **Columns**: `title`, `text`, `subject`, `date`

## Pipeline
1. **Preprocessing**: lowercase, remove URLs/HTML/punctuation/numbers, remove stopwords, filter short words
2. **Features**: TF-IDF (5,000 features, unigrams + bigrams)
3. **Split**: 80/20 train-test with stratification
4. **Models**: Logistic Regression, Random Forest, Naive Bayes, Linear SVM

## Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 97.98% | 97.28% | 98.53% | 97.90% | 99.79% |
| Random Forest | 96.47% | 96.79% | 95.77% | 96.28% | 99.44% |
| Naive Bayes | 94.02% | 93.62% | 93.86% | 93.74% | 98.27% |
| **SVM (Linear)** | **98.65%** | **98.33%** | **98.86%** | **98.59%** | **99.84%** |

- **Best Model**: SVM (Linear)
- **Confusion Matrix**: 4,623 TN / 72 FP / 49 FN / 4,234 TP

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/Fake-News-Detector.git
cd Fake-News-Detector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Train the model
```bash
python fake_news_detector.py
```

### CLI prediction
```bash
python predict.py "Your headline here"
```

### Web UI (Streamlit)
```bash
streamlit run app.py
```

## Files
- `fake_news_detector.py` — main training pipeline
- `predict.py` — CLI prediction tool
- `app.py` — Streamlit web UI
- `requirements.txt` — dependencies
- `fake_news_model.pkl` — saved SVM model
- `tfidf_vectorizer.pkl` — saved vectorizer
- `model_comparison.png` — metrics chart
- `confusion_matrix.png` — heatmap
- `roc_curves.png` — ROC curves

## Tech Stack
Python, scikit-learn, pandas, numpy, matplotlib, seaborn, joblib, streamlit

## Limitations
- The ISOT dataset contains primarily political news (Reuters political coverage vs. political conspiracy blogs). The model learns political vocabulary patterns rather than general semantic understanding.
- Predictions are most accurate for political headlines (>80% confidence).
- Non-political topics (science, sports, entertainment) may yield lower confidence scores because they fall outside the training distribution.
- For a production system, retraining on a more diverse multi-domain dataset is recommended.