import streamlit as st
import requests
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
st.set_page_config(page_title="News App", layout="wide")

API_KEY = "9e69b274d6394fb6aca50d1569b8542d"

# ---------------- USERS ----------------
USERS = {
    "Nikethana": "Nikethana",
    "Prathiksha": "Prathiksha",
    "Jahnavi": "Jahnavi"
}

# ---------------- SESSION INIT ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = ""

if "page" not in st.session_state:
    st.session_state.page = "home"

if "favorites" not in st.session_state:
    st.session_state.favorites = []

if "history" not in st.session_state:
    st.session_state.history = []

if "current_query" not in st.session_state:
    st.session_state.current_query = ""

if "results" not in st.session_state:
    st.session_state.results = None

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {background: linear-gradient(to right, #dbeafe, #bfdbfe);}
.card {
    background:white;
    padding:20px;
    border-radius:15px;
    margin-bottom:15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
if not st.session_state.logged_in:
    st.title("🔐 Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u in USERS and USERS[u] == p:
            st.session_state.logged_in = True
            st.session_state.user = u
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# ---------------- LOGOUT ----------------
def logout():
    st.session_state.logged_in = False
    st.session_state.user = ""
    st.session_state.favorites = []
    st.session_state.history = []
    st.session_state.results = None
    st.rerun()

st.sidebar.button("🚪 Logout", on_click=logout)
st.sidebar.write(f"👤 {st.session_state.user}")

# ---------------- HOME ----------------
if st.session_state.page == "home":
    st.title("📰 Smart News AI")

    if st.button("🚀 Enter App"):
        st.session_state.page = "app"
        st.rerun()

    st.stop()

# ---------------- MAIN ----------------
st.title("📰 Personalized News Aggregator")

category = st.sidebar.selectbox(
    "Category",
    ["general", "business", "technology", "sports", "health", "science", "entertainment"]
)

num_results = st.sidebar.slider("Articles", 1, 10, 5)

# ---------------- HISTORY ----------------
st.sidebar.subheader("🕒 History")
selected_history = st.sidebar.selectbox(
    "Load search",
    [""] + st.session_state.history[::-1]
)

# ---------------- INPUT ----------------
user_input = st.text_input("🔍 Enter interest:", value=st.session_state.current_query)

# ---------------- FETCH ----------------
def fetch_news(query, category):
    if query.strip():
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={API_KEY}"
    else:
        url = f"https://newsapi.org/v2/top-headlines?country=us&category={category}&apiKey={API_KEY}"

    data = requests.get(url).json()

    return pd.DataFrame([
        {
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "url": a.get("url", "")
        }
        for a in data.get("articles", [])
    ])

# ---------------- NLP ----------------
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    return " ".join(w for w in text.split() if w not in stop_words)

# ---------------- RECOMMENDER ----------------
def generate(query, category):
    df = fetch_news(query, category)

    if df.empty:
        return []

    df["content"] = (df["title"] + " " + df["description"]).apply(preprocess)

    vec = TfidfVectorizer(max_features=5000)
    X = vec.fit_transform(df["content"])

    user_vec = vec.transform([preprocess(query if query else category)])
    scores = cosine_similarity(user_vec, X)[0]

    idx = scores.argsort()[::-1]

    return [
        {
            "title": df.iloc[i]["title"],
            "description": df.iloc[i]["description"],
            "url": df.iloc[i]["url"],
            "score": scores[i],
            "id": str(hash(df.iloc[i]["title"]))
        }
        for i in idx[:num_results]
    ]

# ---------------- SEARCH ----------------
if st.button("🚀 Get News"):
    st.session_state.current_query = user_input

    if user_input and user_input not in st.session_state.history:
        st.session_state.history.append(user_input)

    st.session_state.results = generate(user_input, category)
    st.rerun()

# ---------------- LOAD HISTORY ----------------
if selected_history:
    if selected_history != st.session_state.current_query:
        st.session_state.current_query = selected_history
        st.session_state.results = generate(selected_history, category)
        st.rerun()

# ---------------- DISPLAY ----------------
if st.session_state.results is not None:

    found = False

    for article in st.session_state.results:

        if article["score"] > 0:
            found = True

            st.markdown(f"""
            <div class="card">
                <h3>{article['title']}</h3>
                <p>{article['description']}</p>
                <a href="{article['url']}">Read</a>
            </div>
            """, unsafe_allow_html=True)

            article_id = article["id"]

            # ✅ PRE-CHECK IF ALREADY SAVED
            is_saved = article_id in [a["id"] for a in st.session_state.favorites]

            saved = st.checkbox("⭐ Save", key=f"chk_{article_id}", value=is_saved)

            if saved and not is_saved:
                st.session_state.favorites.append(article)

    # ✅ NO RESULTS MESSAGE
    if not found:
        st.warning("⚠️ No matching records found.")

# ---------------- FAVORITES ----------------
st.sidebar.subheader("⭐ Favorites")

if st.session_state.favorites:
    for fav in st.session_state.favorites:
        st.sidebar.markdown(f"[✔ {fav['title']}]({fav['url']})")
else:
    st.sidebar.write("No favorites yet")
