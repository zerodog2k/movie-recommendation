import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load reviews dataset
df = pd.read_csv("datasets/reviews.txt", sep="\t", header=None, names=["label", "text"])

df["label"] = df["label"].astype(int)
df["text"] = df["text"].astype(str)

print(df.head())
print(f"Loaded {len(df)} rows.")

# --- train TF-IDF vectorizer ---
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_vec = vectorizer.fit_transform(df["text"])
y = df["label"]

# --- train classifier ---
clf = LogisticRegression()
clf.fit(X_vec, y)

# --- save both ---
pickle.dump(clf, open("nlp_model.pkl", "wb"))
pickle.dump(vectorizer, open("tranform.pkl", "wb"))

print("Model + Vectorizer saved successfully!")
