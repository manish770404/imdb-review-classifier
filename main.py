from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel

app = FastAPI()

# ✅ Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL for security (e.g., "http://localhost:3000")
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load the trained model and vectorizer
try:
    model = joblib.load("models/classifier.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
except Exception as e:
    print("Error loading model:", e)

# ✅ Define expected request format
class ReviewRequest(BaseModel):
    review: str  # 🔥 The frontend must send JSON with this key

# ✅ Define Prediction Endpoint
@app.post("/predict/")
async def predict_sentiment(request: ReviewRequest):
    try:
        text = request.review
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)
        sentiment = "positive" if prediction[0] == 1 else "negative"
        return {"sentiment": sentiment}
    except Exception as e:
        return {"error": str(e)}
