import streamlit as st
import joblib
import re
import string

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

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("Fake News Detector")
st.write("Enter a political news headline or article to check if it's real or fake.")

# Example buttons
st.subheader("Try an example:")
cols = st.columns(2)
examples = {
    "Real: Tax Reform Bill": "Republican senators propose new tax reform bill to lower corporate rates",
    "Real: Foreign Policy": "White House spokesman confirms new foreign policy strategy for Middle East",
    "Real: Healthcare": "Democratic lawmakers introduce healthcare reform legislation in Congress",
    "Fake: Conspiracy": "Hillary Clinton emails prove conspiracy to rig election against opponents",
    "Fake: Birth Certificate": "Obama birth certificate proven fake by forensic experts in new report",
    "Fake: Trafficking": "Shocking evidence reveals political elites running secret child trafficking ring"
}

with cols[0]:
    for name, text in list(examples.items())[:3]:
        if st.button(name, use_container_width=True):
            st.session_state["text"] = text

with cols[1]:
    for name, text in list(examples.items())[3:]:
        if st.button(name, use_container_width=True):
            st.session_state["text"] = text

default_text = st.session_state.get("text", "")
user_input = st.text_area("Or paste your own news text here:", value=default_text, height=150)

if st.button("Analyze", type="primary"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            label, confidence = predict(user_input)
        
        if label == "REAL":
            st.success(f"REAL NEWS ({confidence:.1f}% confidence)")
        else:
            st.error(f"FAKE NEWS ({confidence:.1f}% confidence)")
        
        st.info("Note: This model is trained on political news. Results are most accurate for political headlines.")
    else:
        st.warning("Please enter some text.")