import requests
import pandas as pd

API_KEY = "9e69b274d6394fb6aca50d1569b8542d"

def fetch_news():
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    articles = []
    #print(data)
    for article in data["articles"]:
        articles.append({
            "title": article["title"],
            "description": article["description"]
        })
    
    return pd.DataFrame(articles)

df = fetch_news()
#print(df.head())

import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
df["title"] = df["title"].fillna("")
df["description"] = df["description"].fillna("")
def preprocess(text):
    if not isinstance(text, str):   # handles NaN, None, float
        return ""
    
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)
#df["content"] = df["title"] + " " + df["description"]
#df["content"] = df["content"].apply(preprocess)
df["content"] = (df["title"].fillna("") + " " + df["description"].fillna("")).apply(preprocess)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["content"])
# Example: user likes tech-related articles
#user_interest = "technology artificial intelligence programming"
user_interest = "ai tech software computer innovation Iran Japan Trump Obesity"
user_vec = vectorizer.transform([user_interest])
from sklearn.metrics.pairwise import cosine_similarity

# Compare user interest with all news articles
similarity_scores = cosine_similarity(user_vec, X)

# Get top 5 matching articles
top_indices = similarity_scores[0].argsort()[-5:][::-1]

print("\n🔍 User Interest:", user_interest)
print("\n📰 Recommended Articles:\n")

for i in top_indices:
    print("Title:", df.iloc[i]["title"])
    #print("Score:", similarity_scores[0][i])
    print("-" * 50)
