import pandas as pd
import neattext.functions as nfx
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load IMDb dataset
df = pd.read_csv("IMDB Dataset.csv")  # Ensure this file is in backend/

# Preprocessing
df["clean_text"] = df["review"].apply(nfx.remove_special_characters)
df["clean_text"] = df["clean_text"].apply(nfx.remove_stopwords)
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Convert Text to Features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save Model
joblib.dump(model, "models/classifier.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
