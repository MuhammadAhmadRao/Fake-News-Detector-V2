import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import re
import string
import warnings
import joblib
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("FAKE NEWS DETECTION PIPELINE")
print("Assessment for AI/ML Internship")
print("=" * 60)

# ============================================================
# STEP 1: TEXT PREPROCESSING (FIXED - source words removed)
# ============================================================

STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
    'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
    'just', 'don', 'should', 'now'
])

# SOURCE-SPECIFIC WORDS - removes Reuters/fake-news formatting bias
SOURCE_WORDS = {
    'reuters', 'washington', 'tuesday', 'wednesday', 'thursday', 'friday',
    'monday', 'saturday', 'sunday', 'january', 'february', 'march', 'april',
    'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
    'video', 'via', 'image', 'images', 'watch', 'photo', 'photos', 'source',
    'breaking', 'update', 'report', 'reports', 'said', 'says', 'say', 'share',
    'tweet', 'twitter', 'facebook', 'youtube', 'click', 'read', 'story', 'stories',
    'published', 'copyright', 'reserved', 'subscribe', 'newsletter', 'email',
    'follow', 'like', 'comment', 'post', 'posted', 'latest', 'trending'
}

STOPWORDS = STOPWORDS.union(SOURCE_WORDS)

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return ' '.join(words)

# ============================================================
# STEP 2: LOAD DATASET
# ============================================================

df_fake = pd.read_csv("dataset/fake.csv")
df_true = pd.read_csv("dataset/true.csv")

df_fake['label'] = 0
df_true['label'] = 1

df = pd.concat([df_fake, df_true], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nDataset loaded successfully!")
print(f"Total samples: {len(df)}")
print(f"Fake news: {len(df[df['label']==0])}")
print(f"Real news: {len(df[df['label']==1])}")

# ============================================================
# STEP 3: EDA
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nMissing values:")
print(df.isnull().sum())

print(f"\nLabel distribution:")
print(df['label'].value_counts())

df['text_length'] = df['text'].fillna('').apply(len)
df['title_length'] = df['title'].fillna('').apply(len)

print(f"\nText length stats:")
print(df.groupby('label')['text_length'].describe())

# ============================================================
# STEP 4: TF-IDF
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: TF-IDF VECTORIZATION")
print("=" * 60)

df['combined_text'] = df['title'].fillna('') + " " + df['text'].fillna('')

print("Preprocessing text...")
df['processed_text'] = df['combined_text'].apply(preprocess_text)

df = df[df['processed_text'].str.len() > 0].reset_index(drop=True)

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X = tfidf.fit_transform(df['processed_text'])
y = df['label'].values

print(f"TF-IDF matrix shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================
# STEP 5: MODEL TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: MODEL TRAINING & COMPARISON")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'SVM (Linear)': LinearSVC(C=1.0, random_state=42, max_iter=5000)
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        'Model': name,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4),
        'AUC-ROC': round(auc, 4)
    })

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")

results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
print(results_df.to_string(index=False))

# ============================================================
# STEP 6: BEST MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: DETAILED EVALUATION")
print("=" * 60)

best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name}")

y_pred_best = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Fake', 'Real']))

cm = confusion_matrix(y_test, y_pred_best)
print("\nConfusion Matrix:")
print(cm)

# ============================================================
# STEP 7: PLOTS
# ============================================================

plt.figure(figsize=(12, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
x = np.arange(len(results_df))
width = 0.15

for i, metric in enumerate(metrics):
    plt.bar(x + i*width, results_df[metric], width, label=metric)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Comparison - Fake News Detection')
plt.xticks(x + width*2, results_df['Model'], rotation=15)
plt.legend()
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - All Models')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150)
plt.close()

# ============================================================
# STEP 8: FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: TOP DISCRIMINATIVE FEATURES")
print("=" * 60)

lr_model = models['Logistic Regression']
feature_names = tfidf.get_feature_names_out()
coefficients = lr_model.coef_[0]

top_real_idx = np.argsort(coefficients)[-15:]
print("\nTop words indicating REAL news:")
for idx in reversed(top_real_idx):
    print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")

top_fake_idx = np.argsort(coefficients)[:15]
print("\nTop words indicating FAKE news:")
for idx in top_fake_idx:
    print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")

# ============================================================
# STEP 9: CUSTOM PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: PREDICTIONS ON CUSTOM HEADLINES")
print("=" * 60)

custom_headlines = [
    "Donald Trump signs executive order on immigration policy",
    "Hillary Clinton emails prove conspiracy to rig election",
    "Republican senators propose new tax reform bill",
    "Obama birth certificate proven fake by forensic experts",
    "White House spokesman confirms new foreign policy strategy",
    "Secret documents reveal GOP plot to steal election",
    "Democratic lawmakers introduce healthcare reform legislation",
    "Breaking: Government hiding truth about economic collapse",
    "Minister announces new trade agreement with European Union",
    "Shocking: Political elites running secret child trafficking ring"
]

processed_headlines = [preprocess_text(h) for h in custom_headlines]
headline_vectors = tfidf.transform(processed_headlines)
predictions = best_model.predict(headline_vectors)

if hasattr(best_model, "predict_proba"):
    probabilities = best_model.predict_proba(headline_vectors)
else:
    decisions = best_model.decision_function(headline_vectors)
    probabilities = np.column_stack([
        1 / (1 + np.exp(decisions)),
        1 / (1 + np.exp(-decisions))
    ])

print(f"\nPredictions using {best_model_name}:\n")
print("-" * 80)
for i, headline in enumerate(custom_headlines):
    pred_label = "REAL" if predictions[i] == 1 else "FAKE"
    confidence = probabilities[i][predictions[i]] * 100
    print(f"Headline: {headline}")
    print(f"Prediction: {pred_label} (Confidence: {confidence:.1f}%)")
    print("-" * 80)

# ============================================================
# STEP 10: SAVE MODEL
# ============================================================

joblib.dump(best_model, 'fake_news_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print("Saved files:")
print("  - fake_news_model.pkl")
print("  - tfidf_vectorizer.pkl")
print("  - model_comparison.png")
print("  - confusion_matrix.png")
print("  - roc_curves.png")