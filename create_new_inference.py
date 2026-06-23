"""
DEPRECATED / ALTERNATE ENTRY POINT (BE-12)
-------------------------------------------
This file is an orphaned alternate FastAPI application and is NOT the production entry
point. The canonical server is `app.py`. This file is kept for reference only.
If you intend to run this, ensure it stays in sync with app.py's schemas.
"""
from fastapi import FastAPI  # pyrefly: ignore[missing-import]
from fastapi.middleware.cors import CORSMiddleware  # pyrefly: ignore[missing-import]
from pydantic import BaseModel  # pyrefly: ignore[missing-import]
from datetime import datetime, timezone
from typing import Optional
import os
import pymongo  # pyrefly: ignore[missing-import]
import joblib  # pyrefly: ignore[missing-import]
import pandas as pd  # pyrefly: ignore[missing-import]
import numpy as np  # pyrefly: ignore[missing-import]

from dotenv import load_dotenv  # pyrefly: ignore[missing-import]
load_dotenv()

# Import your custom modules
from models import InferenceModel
from inference_utils import infer_user, pd_to_tier

# MongoDB connection
client = pymongo.MongoClient(os.getenv("MONGO_URI"))
db = client["crednova"]
users_coll = db["users"]

# Load model artifacts
try:
    bundle = joblib.load("artifacts/bharatscore_pipeline_bundle.pkl")
    explainer = bundle["explainer"]
    feature_names = bundle["feature_names"]
    print("Bundle loaded successfully!")
    
    # Try to load the new inference wrapper first
    try:
        inference = joblib.load("artifacts/new_inference_wrapper.pkl")
        print("New inference wrapper loaded successfully!")
    except Exception:  # BE-10: fix bare except
        # If new one doesn't exist, try to create it
        print("Creating new inference wrapper...")
        preprocessor = bundle["preprocessor"]
        calibrated_clf = bundle["calibrated_clf"]
        inference = InferenceModel(preprocessor, calibrated_clf)
        joblib.dump(inference, "artifacts/new_inference_wrapper.pkl")
        print("New inference wrapper created and saved!")
        
except Exception as e:
    print(f"Error loading models: {e}")
    import traceback
    traceback.print_exc()
    bundle = inference = explainer = feature_names = None

app = FastAPI(title="CredNova API", description="API for CredNova score prediction", version="2.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- MODELS --------------------
class ProfileRequest(BaseModel):
    clerk_user_id: str
    name: str
    gender: str
    state: str
    occupation: str

class OnboardRequest(BaseModel):
    clerk_user_id: str
    user_type: str
    region: str
    sms_count: float
    bill_on_time_ratio: Optional[float] = None
    recharge_freq: float
    sim_tenure: float
    location_stability: float
    income_signal: float
    coop_score: float
    land_verified: int
    age_group: str
    loan_amount_requested: float
    recharge_pattern: str       # BE-2: was missing
    loan_category: str          # BE-2: was missing
    psychometric_score: float   # BE-2: was missing
    consent: bool = True

class InputData(BaseModel):
    user_type: str
    region: str
    sms_count: float
    bill_on_time_ratio: float
    recharge_freq: float
    sim_tenure: float
    location_stability: float
    income_signal: float
    coop_score: float
    land_verified: int
    age_group: str
    loan_amount_requested: float
    recharge_pattern: str       # BE-2: was missing
    loan_category: str          # BE-2: was missing
    psychometric_score: float   # BE-2: was missing

# -------------------- HELPERS --------------------
TIER_BINS = [(0.00, 0.05, "A+"), (0.05, 0.10, "A"), (0.10, 0.20, "B"), (0.20, 0.35, "C"), (0.35, 1.00, "D")]

# -------------------- ENDPOINTS --------------------
@app.post("/profile")
def create_or_update_profile(req: ProfileRequest):
    doc = {
        "clerk_user_id": req.clerk_user_id,
        "profile": {
            "name": req.name,
            "gender": req.gender,
            "state": req.state,
            "occupation": req.occupation,
        },
        "has_profile": True,
        "profile_updated_at": datetime.now(timezone.utc),  # BE-9
    }
    users_coll.update_one({"clerk_user_id": req.clerk_user_id}, {"$set": doc}, upsert=True)
    return {"status": "stored", "clerk_user_id": req.clerk_user_id}

@app.get("/profile")
def get_profile(clerk_user_id: str):
    proj = {"_id": 0, "profile": 1, "has_profile": 1, "clerk_user_id": 1}
    user = users_coll.find_one({"clerk_user_id": clerk_user_id}, proj)
    if user and user.get("profile"):
        return {"profile": user["profile"], "has_profile": True}
    return {"profile": None, "has_profile": False}

@app.post("/onboard")
def onboard(req: OnboardRequest):
    if req.bill_on_time_ratio is None:
        req.bill_on_time_ratio = 0.0
    doc = {
        "clerk_user_id": req.clerk_user_id,
        "raw": req.model_dump(),  # BE-7/8: was req.dict()
        "created": datetime.now(timezone.utc),  # BE-9: was datetime.utcnow()
        "status": "received"
    }
    inserted_id = users_coll.insert_one(doc).inserted_id
    return {"mongo_id": str(inserted_id), "clerk_user_id": req.clerk_user_id, "status": "stored"}

@app.post("/predict")
def predict(data: InputData):
    if inference is None:
        return {"error": "Model not loaded"}
    
    try:
        df = pd.DataFrame([data.model_dump()])  # BE-7/8: was data.dict()
        result = infer_user(df, inference, explainer, feature_names, top_k_shap=5)
        return result
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.get("/predict/{user_id}")
def predict_existing_user(user_id: str):
    """Predict risk for an existing user from database"""
    if inference is None:
        return {"error": "Model not loaded"}
    
    try:
        from bson import ObjectId
        user = users_coll.find_one({"_id": ObjectId(user_id)})
        if not user:
            return {"error": "User not found"}
        
        raw_data = user["raw"]
        df = pd.DataFrame([raw_data])
        result = infer_user(df, inference, explainer, feature_names, top_k_shap=5)
        
        # Optionally save prediction to MongoDB
        users_coll.update_one({"_id": ObjectId(user_id)}, {"$set": {"prediction": result, "status": "predicted"}})
        return result
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.get("/stats")
def get_stats():
    total_users = users_coll.count_documents({})
    predicted_users = users_coll.count_documents({"prediction": {"$exists": True}})
    return {
        "total_users": total_users,
        "predicted_users": predicted_users,
        "pending_predictions": total_users - predicted_users
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": inference is not None,
        "message": "CredNova API v2 is running!"
    }

@app.get("/")
def root():
    return {"message": "CredNova API v2 is running!"}