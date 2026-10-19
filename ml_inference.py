import joblib
import pandas as pd
import numpy as np

from system_logger import get_logger

logger = get_logger("ml")

_model = None
_scaler = None
_feature_columns = None
_label_encoder = None

def load_artifacts():
    global _model, _scaler, _feature_columns, _label_encoder
    if _model is None:
        logger.info("Loading ML artifacts on-demand from artifacts/ …")
        try:
            _model = joblib.load("artifacts/unified_credit_model_updated_final.pkl")
            _scaler = joblib.load("artifacts/unified_credit_scaler_updated_final.pkl")
            _feature_columns = joblib.load("artifacts/unified_feature_columns_updated_final.pkl")
            _label_encoder = joblib.load("artifacts/unified_label_encoder_updated_final.pkl")
            logger.info(
                "ML artifacts loaded — features=%s model=%s scaler=%s encoder=%s",
                len(_feature_columns),
                type(_model).__name__,
                type(_scaler).__name__,
                type(_label_encoder).__name__,
            )
        except Exception as exc:
            logger.exception("Failed to load ML artifacts: %s", exc)
            raise
    return _model, _scaler, _feature_columns, _label_encoder

def __getattr__(name: str):
    if name in ("model", "scaler", "feature_columns", "label_encoder"):
        m, s, f, l = load_artifacts()
        mapping = {"model": m, "scaler": s, "feature_columns": f, "label_encoder": l}
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
        m, s, f, l = load_artifacts()
        df = pd.DataFrame([payload])[f]
        scaled = s.transform(df)
        pred_encoded = m.predict(scaled)[0]
        pred_proba = m.predict_proba(scaled)[0]
        label = l.inverse_transform([pred_encoded])[0]
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
