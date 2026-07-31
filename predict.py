import joblib
import re
import string
import sys

model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
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

def preprocess(text):
    if not text:
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

def predict(text):
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    decision = model.decision_function(vec)[0]
    confidence = (1 / (1 + 2.718 ** (-abs(decision)))) * 100
    label = "REAL" if pred == 1 else "FAKE"
    return label, confidence

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        label, conf = predict(text)
        print(f"Prediction: {label} ({conf:.1f}% confidence)")
    else:
        tests = [
            "NASA confirms water discovery on Mars surface",
            "SHOCKING: Government hiding alien bodies in Area 51"
        ]
        for t in tests:
            label, conf = predict(t)
            print(f"[{label}] {t} ({conf:.1f}%)")