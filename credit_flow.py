"""
Credit evaluation pipeline: optional bank PDF → features → external ML on Render → MongoDB.
Admin asset verification can update has_home / has_gold and trigger re-scoring.
"""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import pandas as pd
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Header
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from insights_service import (
    aggregate_spending_from_csv,
    build_insights_response,
    llm_spending_and_credit_advice,
)
from ml_features import statement_branch_snapshot
from statement_service import (
    calculate_statement_metrics,
    categorize_dataframe,
    clean_bank_statement,
    compute_statement_analysis,
    dataframe_to_csv_string,
    process_bank_pdf,
)

logger = logging.getLogger("crednova.credit_flow")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s [credit-flow] %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False

CREDIT_MODEL_URL = os.getenv(
    "CREDIT_MODEL_URL", "https://credit-scoring-model-7q1s.onrender.com/predict"
)
ADMIN_API_KEY = os.getenv("CREDIT_ADMIN_API_KEY", "")


_mongo_client = None


def _credit_collection():
    global _mongo_client
    import pymongo

    if _mongo_client is None:
        _mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI"))
    return _mongo_client["crednova"]["credit_applications"]


def normalize_business_type(value: str) -> str:
    if not value:
        return "self_employed"
    t = value.strip().lower().replace(" ", "_").replace("-", "_")
    allowed = {"self_employed", "salaried", "business", "farmer", "professional"}
    if t in allowed:
        return t
    if "self" in t or "employ" in t:
        return "self_employed"
    if "salar" in t:
        return "salaried"
    if "farm" in t:
        return "farmer"
    if "bus" in t:
        return "business"
    return "self_employed"


class CreditApplyForm(BaseModel):
    """JSON body fields (also accepted as multipart form field `data` JSON string)."""

    clerk_user_id: Optional[str] = None
    full_name: str
    phone_number: Optional[str] = None
    age: int = Field(ge=18, le=100)
    # Gig / thin-file applicants may have no bureau history
    no_cibil_score: bool = False
    CIBIL_score: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("cibil_score", "CIBIL_score"),
    )
    annual_income: float = Field(ge=0)
    existing_loans: int = Field(ge=0, default=0)
    late_payments: int = Field(ge=0, default=0)
    credit_utilization: float = Field(ge=0, le=1, default=0.0)
    business_vintage_years: float = Field(ge=0, default=0.0)
    business_type: str = "self_employed"
    has_home: int = Field(ge=0, le=1, default=0)
    has_gold: int = Field(ge=0, le=1, default=0)
    # Optional overrides if bank PDF is skipped or parsing fails
    upi_transactions_monthly: Optional[float] = None
    cash_transaction_ratio: Optional[float] = None
    # If user claims high-value collateral, flag for physical verification
    request_physical_asset_verification: bool = False
    # KYC (stored on application; not sent to external ML payload)
    email: Optional[str] = None
    date_of_birth: Optional[str] = None  # ISO date YYYY-MM-DD
    pan_number: Optional[str] = None
    current_address: Optional[str] = None
    asset_location_address: Optional[str] = None

    @field_validator("business_type", mode="before")
    @classmethod
    def coerce_business(cls, v):
        if v is None:
            return "self_employed"
        return str(v)

    @model_validator(mode="after")
    def validate_cibil(self):
        if self.no_cibil_score:
            return self
        if self.CIBIL_score is None:
            raise ValueError(
                "CIBIL score (300–900) is required unless no_cibil_score is true "
                "(e.g. no credit history / gig workers)."
            )
        if not (300 <= int(self.CIBIL_score) <= 900):
            raise ValueError("CIBIL score must be between 300 and 900")
        return self


class AssetVerificationBody(BaseModel):
    has_home: int = Field(ge=0, le=1)
    has_gold: int = Field(ge=0, le=1)
    home_assessed_value_inr: Optional[float] = None
    gold_assessed_value_inr: Optional[float] = None
    inspector_notes: str = ""


class BankEmployeeCsvMeta(BaseModel):
    """Multipart companion JSON for POST /credit/bank-employee/analyze-csv"""

    pan_number: str
    aadhaar: str
    account_number: Optional[str] = None


BANK_EMPLOYEE_CSV_PAN = "ISAPD7498P"
BANK_EMPLOYEE_CSV_AADHAAR = "360467541335"


def _normalize_pan_bank(pan: str) -> str:
    return pan.strip().upper().replace(" ", "")


def _normalize_aadhaar_digits(aadhaar: str) -> str:
    return "".join(c for c in str(aadhaar) if c.isdigit())


router = APIRouter()


def _require_admin(x_admin_key: str | None):
    if not ADMIN_API_KEY:
        return
    if not x_admin_key or x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing admin API key")


def _int_upi_monthly(raw: Any) -> int:
    """Remote /predict API expects int; PDF metrics use floats (e.g. 13.2)."""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0
    return max(0, int(round(v)))


def build_model_payload(form: CreditApplyForm, statement: dict[str, Any]) -> dict[str, Any]:
    upi = form.upi_transactions_monthly
    if upi is None:
        upi = statement.get("monthly_upi", 0.0)
    cash = form.cash_transaction_ratio
    if cash is None:
        cash = statement.get("cash_transaction_ratio", 0.0)

    upi_int = _int_upi_monthly(upi)

    # External model expects a scalar; use range floor as thin-file proxy when undeclared
    cibil_for_model = (
        300
        if (form.no_cibil_score or form.CIBIL_score is None)
        else int(form.CIBIL_score)
    )
    return {
        "CIBIL_score": cibil_for_model,
        "age": int(form.age),
        "existing_loans": int(form.existing_loans),
        "late_payments": int(form.late_payments),
        "credit_utilization": float(form.credit_utilization),
        "annual_income": float(form.annual_income),
        "upi_transactions_monthly": upi_int,
        "cash_transaction_ratio": float(cash),
        "business_vintage_years": float(form.business_vintage_years),
        "has_home": int(form.has_home),
        "has_gold": int(form.has_gold),
        "business_type": normalize_business_type(form.business_type),
    }


async def call_credit_model(payload: dict[str, Any]) -> dict[str, Any]:
    # Render cold starts can exceed 60s; remote schema expects ints for some fields (see build_model_payload).
    timeout = httpx.Timeout(120.0, connect=30.0)
    logger.info(
        "Step 5/8: Sending payload to online ML model (POST %s) upi_transactions_monthly=%s",
        CREDIT_MODEL_URL,
        payload.get("upi_transactions_monthly"),
    )
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(CREDIT_MODEL_URL, json=payload)
    except httpx.ReadTimeout as e:
        logger.exception("Credit model HTTP read timeout (try again; Render may be cold)")
        raise HTTPException(
            status_code=504,
            detail="Credit model service timed out. Wait a minute and retry (remote service may be waking up).",
        ) from e
    except httpx.RequestError as e:
        logger.exception("Credit model request failed: %s", e)
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach credit model service: {e!s}",
        ) from e
    if r.status_code != 200:
        logger.error(
            "Credit model HTTP %s: %s",
            r.status_code,
            (r.text or "")[:800],
        )
        raise HTTPException(
            status_code=502,
            detail=f"Credit model error {r.status_code}: {r.text[:500]}",
        )
    logger.info("Step 5/8: HTTP 200 — response body received from online ML model; parsing JSON")
    return r.json()


async def _run_credit_apply(
    form: CreditApplyForm,
    statement_pdf: UploadFile | None,
    pdf_password: str | None,
    pan_scan: UploadFile | None = None,
    *,
    statement_df: Optional[pd.DataFrame] = None,
    csv_parse_message: str = "",
) -> dict[str, Any]:
    logger.info(
        "Step 1/8: Start apply clerk_user_id=%s no_cibil=%s has_pdf=%s has_csv_df=%s",
        form.clerk_user_id or "(none)",
        form.no_cibil_score,
        bool(statement_pdf and getattr(statement_pdf, "filename", None)),
        statement_df is not None,
    )
    statement_metrics: dict[str, Any] = {}
    parse_message = ""
    transactions_csv: Optional[str] = None
    row_count = 0
    parsed_df: Any = None

    if statement_df is not None:
        cleaned = clean_bank_statement(statement_df.copy())
        final = categorize_dataframe(cleaned)
        parse_message = csv_parse_message or "CSV_UPLOAD_OK"
        transactions_csv = dataframe_to_csv_string(final)
        row_count = len(final)
        parsed_df = final
        statement_metrics = calculate_statement_metrics(final)
        logger.info(
            "Step 2-3/8: Preloaded CSV/DataFrame rows=%s parse=%s monthly_upi=%s",
            row_count,
            parse_message,
            statement_metrics.get("monthly_upi"),
        )
    elif statement_pdf is not None and getattr(statement_pdf, "filename", None):
        raw = await statement_pdf.read()
        logger.info("Step 2/8: PDF uploaded size_bytes=%s", len(raw))
        if len(raw) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="PDF too large (max 5MB to prevent memory limits)")
        df, parse_message, statement_metrics = process_bank_pdf(raw, pdf_password or None)
        if df is None:
            if parse_message == "PASSWORD_REQUIRED":
                raise HTTPException(status_code=400, detail="PDF is password protected; send pdf_password")
            raise HTTPException(status_code=400, detail=parse_message)
        transactions_csv = dataframe_to_csv_string(df)
        row_count = len(df)
        parsed_df = df
        raw_upi = statement_metrics.get("monthly_upi")
        logger.info(
            "Step 3/8: PDF pipeline finished — transaction CSV built rows=%s parse_message=%s monthly_upi_raw=%s",
            row_count,
            parse_message,
            raw_upi,
        )
        logger.info(
            "Step 3/8: PDF data extraction complete; ready to merge with form for local scoring payload"
        )
    else:
        logger.info("Step 2-3/8: No bank PDF; using form/defaults for statement metrics")

    statement_analysis = compute_statement_analysis(parsed_df)
    logger.info(
        "Step 3c/8: Statement analysis available=%s months=%s",
        statement_analysis.get("available"),
        len(statement_analysis.get("monthly") or []),
    )

    needs_physical = bool(form.request_physical_asset_verification) or (
        form.has_home == 1 or form.has_gold == 1
    )

    payload = build_model_payload(form, statement_metrics)
    logger.info(
        "Step 4/8: Local ML branch — statement-derived snapshot=%s; merged payload keys for remote /predict",
        statement_branch_snapshot(statement_metrics),
    )
    logger.info(
        "Step 4/8: Local processing — merged applicant form + statement metrics into model payload "
        "(CIBIL_score=%s age=%s annual_income=%s upi_transactions_monthly=%s)",
        payload.get("CIBIL_score"),
        payload.get("age"),
        payload.get("annual_income"),
        payload.get("upi_transactions_monthly"),
    )
    model_out = await call_credit_model(payload)
    if isinstance(model_out, dict):
        logger.info(
            "Step 5b/8: Online ML model response summary keys=%s preview=%s",
            list(model_out.keys()),
            {k: model_out[k] for k in list(model_out)[:6]},
        )
    else:
        logger.info("Step 5b/8: Online ML model returned non-dict type=%s", type(model_out).__name__)

    doc = {
        "clerk_user_id": form.clerk_user_id,
        "applicant": {
            "full_name": form.full_name,
            "phone_number": form.phone_number,
            "email": form.email,
            "date_of_birth": form.date_of_birth,
            "pan_number": form.pan_number,
            "current_address": form.current_address,
            "asset_location_address": form.asset_location_address,
        },
        "form": form.model_dump(),
        "statement": {
            "parse_message": parse_message,
            "metrics": statement_metrics,
            "row_count": row_count,
            "analysis": statement_analysis,
        },
        "model_payload": payload,
        "model_output": model_out,
        "transactions_csv": transactions_csv,
        "asset_verification": {
            "status": "pending_visit" if needs_physical else "not_required",
            "home_assessed_value_inr": None,
            "gold_assessed_value_inr": None,
            "inspector_notes": "",
            "verified_at": None,
        },
        "created": datetime.now(timezone.utc),
        "status": "scored",
        "bank_employee_csv": bool(statement_df is not None),
    }

    coll = _credit_collection()
    ins = coll.insert_one(doc)
    app_id = str(ins.inserted_id)
    logger.info(
        "Step 6/8: Persisted scored application to DB (MongoDB) application_id=%s — portfolio/API can fetch this record",
        app_id,
    )

    # Optional PAN scan (image/PDF) — stored for admin review; max 8MB
    if pan_scan is not None and getattr(pan_scan, "filename", None):
        pan_saved = False
        try:
            raw_pan = await pan_scan.read()
            if raw_pan and len(raw_pan) <= 8 * 1024 * 1024:
                upload_dir = Path(__file__).resolve().parent / "uploads" / "pan"
                upload_dir.mkdir(parents=True, exist_ok=True)
                suffix = Path(pan_scan.filename or "scan").suffix or ".bin"
                if suffix.lower() not in {".jpg", ".jpeg", ".png", ".pdf", ".webp", ".bin"}:
                    suffix = ".bin"
                out_path = upload_dir / f"{app_id}{suffix}"
                out_path.write_bytes(raw_pan)
                coll.update_one(
                    {"_id": ins.inserted_id},
                    {
                        "$set": {
                            "applicant.pan_scan_path": str(out_path.as_posix()),
                            "applicant.pan_scan_original_name": pan_scan.filename,
                        }
                    },
                )
                pan_saved = True
            else:
                logger.warning(
                    "Step 7/8: PAN file missing or over 8MB; skipped application_id=%s",
                    app_id,
                )
        except Exception as e:
            logger.warning("Step 7/8: PAN scan save failed application_id=%s err=%s", app_id, e)
        if pan_saved:
            logger.info("Step 7/8: PAN scan saved application_id=%s", app_id)
    else:
        logger.info("Step 7/8: No PAN scan upload")

    logger.info(
        "Step 8/8: Apply complete — returning JSON to client (frontend can store + show portfolio) application_id=%s",
        app_id,
    )
    return {
        "application_id": app_id,
        "model_output": model_out,
        "model_payload": payload,
        "statement_metrics": statement_metrics,
        "statement_analysis": statement_analysis,
        "parse_message": parse_message,
        "asset_verification": doc["asset_verification"],
        "transactions_csv_available": bool(transactions_csv),
    }


@router.post("/apply")
async def credit_apply(
    data: str = Form(..., description="JSON string: CreditApplyForm"),
    statement_pdf: UploadFile | None = File(default=None),
    pan_scan: UploadFile | None = File(default=None, description="Optional PAN card image or PDF"),
    pdf_password: str | None = Form(default=None),
):
    """
    Multipart submit: field ``data`` = JSON string (CreditApplyForm), optional ``statement_pdf``, optional ``pan_scan``.
    """
    try:
        form = CreditApplyForm.model_validate_json(data)
    except Exception as e:
        logger.warning("Invalid apply JSON: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid JSON in `data`: {e}")
    logger.info("POST /credit/apply: form JSON validated, entering pipeline")
    return await _run_credit_apply(form, statement_pdf, pdf_password, pan_scan)


@router.post("/apply-json")
async def credit_apply_json(body: CreditApplyForm):
    """JSON-only application (no bank PDF). Uses overrides or zeros for statement-derived fields."""
    logger.info("POST /credit/apply-json: entering pipeline")
    return await _run_credit_apply(body, None, None)


@router.post("/bank-employee/analyze-csv")
async def bank_employee_analyze_csv(
    data: str = Form(..., description="JSON: BankEmployeeCsvMeta (pan, aadhaar, account_number)"),
    statement_csv: UploadFile = File(..., description="Bank statement transactions CSV"),
):
    """
    Bank-employee-only path: upload a statement CSV for the registered demo identity.
    Runs the same pipeline as /credit/apply (metrics → remote ML → MongoDB) using CSV rows as the statement source.
    """
    try:
        meta = BankEmployeeCsvMeta.model_validate_json(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in `data`: {e}")
    pan = _normalize_pan_bank(meta.pan_number)
    ad = _normalize_aadhaar_digits(meta.aadhaar)
    if pan != BANK_EMPLOYEE_CSV_PAN or ad != BANK_EMPLOYEE_CSV_AADHAAR:
        raise HTTPException(
            status_code=403,
            detail="CSV analysis is only enabled for the demo customer "
            f"(PAN {BANK_EMPLOYEE_CSV_PAN} and matching Aadhaar on file).",
        )
    raw = await statement_csv.read()
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="CSV too large (max 5MB)")
    if not raw:
        raise HTTPException(status_code=400, detail="Empty CSV file")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV has no data rows")

    form = CreditApplyForm(
        clerk_user_id=None,
        full_name="Adarsh Dhawale",
        phone_number="9800000000",
        age=32,
        no_cibil_score=False,
        CIBIL_score=720,
        annual_income=900000.0,
        existing_loans=0,
        late_payments=0,
        credit_utilization=0.25,
        business_vintage_years=3.0,
        business_type="salaried",
        has_home=0,
        has_gold=0,
        pan_number=pan,
        email="demo@crednova.local",
    )
    logger.info("POST /credit/bank-employee/analyze-csv: validated demo identity; running pipeline on CSV rows=%s", len(df))
    return await _run_credit_apply(
        form,
        None,
        None,
        None,
        statement_df=df,
        csv_parse_message="BANK_EMPLOYEE_CSV",
    )


@router.get("/application/{application_id}")
def get_application(application_id: str):
    try:
        oid = ObjectId(application_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid application_id")

    coll = _credit_collection()
    doc = coll.find_one({"_id": oid}, {"transactions_csv": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    doc["_id"] = str(doc["_id"])
    return doc


@router.get("/application/{application_id}/insights")
async def get_application_insights(application_id: str):
    """
    Spending breakdown (keyword rules on statement CSV debits) + credit tips.
    Optional OpenAI ``gpt-4o-mini`` narrative when ``OPENAI_API_KEY`` is set.
    """
    try:
        oid = ObjectId(application_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid application_id")

    coll = _credit_collection()
    doc = coll.find_one({"_id": oid}, {"transactions_csv": 1, "model_output": 1, "form": 1})
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    csv_text = doc.get("transactions_csv")
    model_output = doc.get("model_output") or {}
    form = doc.get("form") or {}
    spending = aggregate_spending_from_csv(csv_text or "")
    llm_narrative, llm_tips, llm_used = await llm_spending_and_credit_advice(
        spending, model_output, form
    )
    return build_insights_response(
        csv_text if isinstance(csv_text, str) else None,
        model_output,
        form,
        llm_narrative,
        llm_tips,
        llm_used,
    )


@router.get("/application/{application_id}/transactions.csv")
def download_transactions_csv(application_id: str):
    try:
        oid = ObjectId(application_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid application_id")

    coll = _credit_collection()
    doc = coll.find_one({"_id": oid}, {"transactions_csv": 1, "applicant.full_name": 1})
    if not doc or not doc.get("transactions_csv"):
        raise HTTPException(status_code=404, detail="No transaction export for this application")

    from fastapi.responses import Response

    name = (doc.get("applicant") or {}).get("full_name") or "applicant"
    safe = "".join(c if c.isalnum() else "_" for c in name)[:40]
    return Response(
        content=doc["transactions_csv"],
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{safe}_transactions.csv"'},
    )


@router.patch("/admin/application/{application_id}/assets")
async def admin_update_assets(
    application_id: str,
    body: AssetVerificationBody,
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
):
    _require_admin(x_admin_key)

    try:
        oid = ObjectId(application_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid application_id")

    coll = _credit_collection()
    doc = coll.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    form_data = doc.get("form") or {}
    # Rebuild CreditApplyForm subset from stored form
    merged = {**form_data, "has_home": body.has_home, "has_gold": body.has_gold}
    try:
        form = CreditApplyForm.model_validate(merged)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Stored form incompatible: {e}")

    statement_metrics = (doc.get("statement") or {}).get("metrics") or {}
    payload = build_model_payload(form, statement_metrics)
    model_out = await call_credit_model(payload)

    coll.update_one(
        {"_id": oid},
        {
            "$set": {
                "form.has_home": body.has_home,
                "form.has_gold": body.has_gold,
                "model_payload": payload,
                "model_output": model_out,
                "asset_verification": {
                    "status": "verified",
                    "home_assessed_value_inr": body.home_assessed_value_inr,
                    "gold_assessed_value_inr": body.gold_assessed_value_inr,
                    "inspector_notes": body.inspector_notes,
                    "verified_at": datetime.now(timezone.utc),
                },
                "rescored_at": datetime.now(timezone.utc),
            }
        },
    )

    return {
        "application_id": application_id,
        "model_output": model_out,
        "model_payload": payload,
        "asset_verification": {
            "status": "verified",
            "home_assessed_value_inr": body.home_assessed_value_inr,
            "gold_assessed_value_inr": body.gold_assessed_value_inr,
        },
    }


@router.get("/admin/applications/recent")
def admin_list_recent(limit: int = 30, x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")):
    _require_admin(x_admin_key)
    coll = _credit_collection()
    cursor = (
        coll.find(
            {},
            {
                "transactions_csv": 0,
            },
        )
        .sort("created", -1)
        .limit(min(limit, 100))
    )
    out = []
    for d in cursor:
        d["_id"] = str(d["_id"])
        out.append(d)
    return {"applications": out}
