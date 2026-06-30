import joblib
import pandas as pd
import numpy as np

model           = joblib.load("artifacts/unified_credit_model_updated_final.pkl")
scaler          = joblib.load("artifacts/unified_credit_scaler_updated_final.pkl")
feature_columns = joblib.load("artifacts/unified_feature_columns_updated_final.pkl")
label_encoder   = joblib.load("artifacts/unified_label_encoder_updated_final.pkl")

def run_inference(payload: dict) -> dict:
    df = pd.DataFrame([payload])[feature_columns]
    scaled = scaler.transform(df)
    pred_encoded = model.predict(scaled)[0]
    pred_proba = model.predict_proba(scaled)[0]
    label = label_encoder.inverse_transform([pred_encoded])[0]
    confidence = float(np.max(pred_proba))
    approval_prob = float(pred_proba[1]) if len(pred_proba) > 1 else confidence

    # risk_probability = probability of being HIGH-RISK (1 - approval)
    risk_probability = round(1.0 - approval_prob, 4)

    # Derive a CIBIL-range credit score (300–900) from approval probability
    credit_score = int(300 + (approval_prob * 600))

    # Map to a simple risk tier
    if approval_prob >= 0.80:
        risk_level = "A+"
    elif approval_prob >= 0.65:
        risk_level = "A"
    elif approval_prob >= 0.50:
        risk_level = "B"
    elif approval_prob >= 0.35:
        risk_level = "C"
    else:
        risk_level = "D"

    return {
        # Fields the frontend reads directly
        "credit_score": credit_score,
        "risk_probability": risk_probability,
        "risk_level": risk_level,
        # Extra fields kept for backward compat / admin view
        "approval_probability": round(approval_prob, 4),
        "confidence": round(confidence, 4),
        "credit_tier": str(label),
        "alt_cibil_score": credit_score,
        "tier": risk_level,
        "pd": risk_probability,
    }