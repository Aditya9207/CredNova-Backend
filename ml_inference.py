import joblib
import pandas as pd
import numpy as np

from system_logger import get_logger

logger = get_logger("ml")

logger.info("Loading ML artifacts from artifacts/ …")
try:
    model = joblib.load("artifacts/unified_credit_model_updated_final.pkl")
    scaler = joblib.load("artifacts/unified_credit_scaler_updated_final.pkl")
    feature_columns = joblib.load("artifacts/unified_feature_columns_updated_final.pkl")
    label_encoder = joblib.load("artifacts/unified_label_encoder_updated_final.pkl")
    logger.info(
        "ML artifacts loaded — features=%s model=%s scaler=%s encoder=%s",
        len(feature_columns),
        type(model).__name__,
        type(scaler).__name__,
        type(label_encoder).__name__,
    )
except Exception as exc:
    logger.exception("Failed to load ML artifacts: %s", exc)
    raise


def run_inference(payload: dict) -> dict:
    logger.info(
        "ML inference started — CIBIL=%s age=%s income=%s upi_monthly=%s cash_ratio=%s",
        payload.get("CIBIL_score"),
        payload.get("age"),
        payload.get("annual_income"),
        payload.get("upi_transactions_monthly"),
        payload.get("cash_transaction_ratio"),
    )
    try:
        df = pd.DataFrame([payload])[feature_columns]
        scaled = scaler.transform(df)
        pred_encoded = model.predict(scaled)[0]
        pred_proba = model.predict_proba(scaled)[0]
        label = label_encoder.inverse_transform([pred_encoded])[0]
        confidence = float(np.max(pred_proba))
        approval_prob = float(pred_proba[1]) if len(pred_proba) > 1 else confidence

        risk_probability = round(1.0 - approval_prob, 4)
        credit_score = int(300 + (approval_prob * 600))

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

        result = {
            "credit_score": credit_score,
            "risk_probability": risk_probability,
            "risk_level": risk_level,
            "approval_probability": round(approval_prob, 4),
            "confidence": round(confidence, 4),
            "credit_tier": str(label),
            "alt_cibil_score": credit_score,
            "tier": risk_level,
            "pd": risk_probability,
        }
        logger.info(
            "ML inference complete — credit_score=%s risk_level=%s approval_prob=%s tier=%s",
            credit_score,
            risk_level,
            round(approval_prob, 4),
            label,
        )
        return result
    except Exception as exc:
        logger.exception("ML inference failed: %s", exc)
        raise
