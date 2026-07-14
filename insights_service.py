"""
Spending breakdown + credit tips from parsed statement CSV.

Hybrid design (industry-aligned):
- Rule/keyword aggregation for category totals (fast, no API cost, works offline).
- Optional OpenAI ``gpt-4o-mini`` for short narrative + tailored tips (cheap JSON mode).

Refs: hybrid categorization + LLM for narrative; GPT-4o-mini for structured extraction cost/quality balance.
"""
from __future__ import annotations

import io
import json
import os
import re
from typing import Any, Optional

import httpx
import pandas as pd

from system_logger import get_logger

logger = get_logger("insights")

OPENAI_MODEL = os.getenv("OPENAI_INSIGHTS_MODEL", "gpt-4o-mini")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _openai_key() -> str:
    """Read at call time so keys work after `.env` reload without importing module again."""
    return (os.getenv("OPENAI_API_KEY") or "").strip()

# (label, regex) — first match wins; debits attributed to spending
_SPEND_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "Food & dining",
        re.compile(
            r"zomato|swiggy|restaurant|food|domino|kfc|mcdonald|burger|pizza|"
            r"grocery|bigbasket|blinkit|dunzo|eat\.|uber\s*eats|faasos|box8",
            re.I,
        ),
    ),
    (
        "Entertainment",
        re.compile(
            r"netflix|hotstar|prime\s*video|spotify|cinema|bookmyshow|pvr|inox|"
            r"steam|playstation|sony\s*liv|jio\s*cinema|entertainment",
            re.I,
        ),
    ),
    (
        "Travel & transport",
        re.compile(
            r"uber|ola|irctc|makemytrip|goibibo|indigo|spicejet|airasia|cleartrip|"
            r"fuel|petrol|diesel|shell|bpcl|hp\s*petrol|indian\s*oil|metro|rapido|redbus",
            re.I,
        ),
    ),
    (
        "Shopping",
        re.compile(
            r"amazon|flipkart|myntra|ajio|nykaa|meesho|decathlon|reliance\s*digital|"
            r"croma|vijay\s*sales",
            re.I,
        ),
    ),
    (
        "Utilities & bills",
        re.compile(
            r"electric|water|gas|broadband|jio|airtel|vi\s*bill|act\s*fibernet|"
            r"bescom|mseb|tata\s*power|rent|lease|maintenance",
            re.I,
        ),
    ),
    (
        "Health",
        re.compile(
            r"pharmacy|apollo|medplus|hospital|diagnostic|practo|1mg|netmeds|"
            r"health|doctor|lab",
            re.I,
        ),
    ),
    (
        "Investments",
        re.compile(
            r"zerodha|groww|upstox|angel\s*one|et\s*money|paytm\s*money|"
            r"mutual\s*fund|\bmf\b|sip\b|demat|nse|bse|kite|smallcase|"
            r"investments?|\bstocks?\b|equity|\bnps\b|\bppf\b|epf|lic\s*policy|"
            r"camskf|karvy|nsdl|cdsl",
            re.I,
        ),
    ),
]


def _classify_remark(remark: str) -> str:
    t = str(remark or "")
    for label, pat in _SPEND_RULES:
        if pat.search(t):
            return label
    return "General & other"


def aggregate_spending_from_csv(csv_text: str) -> dict[str, Any]:
    """Sum debit amounts by spending category from exported statement CSV."""
    if not csv_text or len(csv_text.strip()) < 10:
        logger.info("Spending aggregation skipped — empty or missing CSV")
        return {"categories": [], "total_debit_inr": 0.0, "row_count": 0}

    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    if "remarks" not in df.columns:
        return {"categories": [], "total_debit_inr": 0.0, "row_count": len(df)}

    if "debits" not in df.columns:
        df["debits"] = 0.0
    else:
        df["debits"] = pd.to_numeric(df["debits"], errors="coerce").fillna(0.0)  # type: ignore

    spend = df[df["debits"] > 0].copy()
    if spend.empty:
        return {
            "categories": [],
            "total_debit_inr": 0.0,
            "row_count": len(df),
        }

    spend["category"] = spend["remarks"].astype(str).map(_classify_remark)  # type: ignore
    grouped = spend.groupby("category", sort=False)["debits"].sum()
    total = float(grouped.sum()) or 1.0

    cats: list[dict[str, Any]] = []
    for name, amt in grouped.items():
        amt_f = float(amt)
        cats.append(
            {
                "category": name,
                "debits_inr": round(amt_f, 2),
                "pct_of_debit_spend": round(100.0 * amt_f / total, 1),
            }
        )
    cats.sort(key=lambda x: x["debits_inr"], reverse=True)

    logger.info(
        "Spending aggregated — rows=%s categories=%s total_debit_inr=%s",
        len(df),
        len(cats),
        round(float(grouped.sum()), 2),
    )

    return {
        "categories": cats,
        "total_debit_inr": round(float(grouped.sum()), 2),
        "row_count": len(df),
    }


def _pct_map(categories: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in categories:
        name = str(c.get("category") or "")
        try:
            out[name] = float(c.get("pct_of_debit_spend") or 0)
        except (TypeError, ValueError):
            out[name] = 0.0
    return out


def rule_based_spending_narrative(
    spending: dict[str, Any],
    model_output: dict[str, Any],
    form: dict[str, Any],
) -> str:
    """Readable summary when OpenAI narrative is unavailable — still feels personal."""
    cs = float(model_output.get("credit_score") or model_output.get("final_cibil_score") or 0)
    rl = str(model_output.get("risk_level") or "—")
    cats = spending.get("categories") or []
    total_d = float(spending.get("total_debit_inr") or 0)
    if not cats and total_d <= 0:
        return (
            f"Your model assessment shows a score around {int(cs)} with a {rl} risk band. "
            "Once debit lines are categorised from your statement export, we can tie suggestions to how you spend."
        )
    top_names = [str(c.get("category")) for c in cats[:3]]
    tail = ", ".join(top_names[:2]) + ("…" if len(top_names) > 2 else "")
    return (
        f"Based on your statement debits (about ₹{total_d:,.0f} tracked), spending leans toward {tail}. "
        f"Together with your model score near {int(cs)} ({rl} risk), the cards below are tailored to typical Indian credit behaviour — "
        "not generic advice."
    )


def rule_based_credit_tips(
    model_output: dict[str, Any],
    spending: dict[str, Any],
    form: dict[str, Any],
) -> list[str]:
    """
    Data-driven tips from model score, risk, spending % by category, and form flags.
    Designed to read as realistic for demos even without an LLM.
    """
    tips: list[str] = []
    rp = float(model_output.get("risk_probability") or 0.3)
    cs = float(model_output.get("credit_score") or model_output.get("final_cibil_score") or 0)
    rl = str(model_output.get("risk_level") or "").upper()

    no_cibil = bool(form.get("no_cibil_score"))
    existing_loans = int(form.get("existing_loans") or 0)
    util = float(form.get("credit_utilization") or 0)

    # Score band — reference the number when we have it
    if cs > 0:
        if cs >= 780:
            tips.append(
                f"At ~{int(cs)}, you are in a strong band — keep every EMI and card payment on autopay so one missed date does not erase years of history."
            )
        elif cs >= 700:
            tips.append(
                f"With a model score near {int(cs)}, staying below ~30% card utilisation and avoiding new hard enquiries for 3–6 months helps nudge CIBIL upward."
            )
        elif cs >= 650:
            tips.append(
                f"A score around {int(cs)} has room to climb: pay at least the minimum before the due date, then chip away principal when cashflow allows."
            )
        else:
            tips.append(
                "Scores under ~650 recover fastest when you fix payment delays first — settle or restructure toxic dues before taking fresh unsecured loans."
            )

    # Risk probability
    if rp < 0.28:
        tips.append(
            "Your estimated risk is on the lower side — preserve that by not maxing credit lines before a large loan application."
        )
    elif rp > 0.38:
        tips.append(
            "Higher estimated risk usually improves when you reduce revolving balances and space out new credit applications by several months."
        )

    if "HIGH" in rl and cs > 0:
        tips.append(
            "A higher risk band with lenders often reflects utilisation or recent enquiries — pull a free CIBIL report and dispute any account you do not recognise."
        )

    if no_cibil:
        tips.append(
            "Thin-file applicants: a secured card or gold loan with perfect repayment builds bureau history faster than many small unsecured enquiries."
        )

    if existing_loans >= 3:
        tips.append(
            f"You declared {existing_loans} active facilities — consolidating high-APR unsecured debt into one structured EMI can simplify payments and protect your score."
        )

    if util > 0.45:
        tips.append(
            f"Credit utilisation around {util:.0%} on declared limits hurts scores — pay down before the statement date so reported utilisation drops."
        )

    # Category mix — use actual %
    cats = spending.get("categories") or []
    pm = _pct_map(cats)

    def add_cat(cat: str, pct: float, msg: str) -> None:
        if pm.get(cat, 0) >= pct:
            tips.append(msg)

    add_cat(
        "Food & dining",
        22.0,
        "Food and delivery are a large share of debits — meal-planning even a few days a week frees cash for predictable EMI discipline.",
    )
    add_cat(
        "Entertainment",
        12.0,
        "Entertainment is material in your spend — set a monthly cap and route “fun” money through UPI so cashflow stays visible to future underwriters.",
    )
    add_cat(
        "Travel & transport",
        15.0,
        "Travel and commute spend is prominent — if fuel/UPI is high, a single consolidated fuel card payment can simplify tracking.",
    )
    add_cat(
        "Shopping",
        22.0,
        "Discretionary shopping shows up strongly — stagger large purchases so card utilisation does not spike in the month lenders see.",
    )
    add_cat(
        "Investments",
        8.0,
        "Investments are visible in your debit mix — SIP continuity helps discipline; avoid redeeming ELSS or long-term funds for short-term card rollovers.",
    )
    add_cat(
        "Utilities & bills",
        8.0,
        "Recurring utility and rent payments on time build a steady repayment pattern — ideal for bureau “thin” files.",
    )
    add_cat(
        "Health",
        10.0,
        "Health spend is non-trivial — keep invoices; some lenders accept recurring medical insurance as stability context.",
    )

    # Top category explicit
    if cats:
        top = cats[0]
        tn = str(top.get("category") or "")
        tp = float(top.get("pct_of_debit_spend") or 0)
        if tn and tp >= 35:
            tips.append(
                f"Your largest bucket is {tn} (~{tp:.0f}% of debits) — that is the first place to rebalance if you need room for loan eligibility."
            )

    # Universal high-value India tips (only if list still thin)
    defaults = [
        "Pay card bills in full when possible; interest charges do not help your score and eat into savings.",
        "Avoid being guarantor on multiple loans — defaults there hit your bureau like your own debt.",
        "Keep old credit cards open (no annual fee) — longer average age of accounts supports the score.",
        "Space unsecured personal loans and BNPL sign-ups; each enquiry can dent CIBIL in the short term.",
    ]
    for d in defaults:
        if len(tips) >= 8:
            break
        if d not in tips:
            tips.append(d)

    # Dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in tips:
        t = str(t).strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)

    return out[:10]


async def llm_spending_and_credit_advice(
    aggregates: dict[str, Any],
    model_output: dict[str, Any],
    form: dict[str, Any],
) -> tuple[Optional[str], Optional[list[str]], bool]:
    """
    Optional GPT-4o-mini call. Returns (spending_narrative, extra_credit_tips, used_llm).

    Uses aggregated JSON only — not raw transaction rows (privacy + cost).
    """
    if not _openai_key():
        logger.info("LLM insights skipped — OPENAI_API_KEY not set; using rule-based tips")
        return None, None, False

    logger.info("LLM insights request — model=%s categories=%s", OPENAI_MODEL, len(aggregates.get("categories") or []))

    payload = {
        "spending": aggregates,
        "model_output": {
            "credit_score": model_output.get("credit_score") or model_output.get("final_cibil_score"),
            "risk_probability": model_output.get("risk_probability"),
            "risk_level": model_output.get("risk_level"),
        },
        "applicant": {
            "annual_income": form.get("annual_income"),
            "no_cibil_score": form.get("no_cibil_score"),
        },
    }

    system = (
        "You are CredNova, an India-focused credit assistant. "
        "Given ONLY aggregated spending by category (INR debits) and a credit model summary, "
        "respond with valid JSON only, no markdown."
    )
    user = (
        "Data:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON with this exact shape:\n"
        '{\n'
        '  "spending_narrative": "2-3 sentences on spending pattern, neutral tone",\n'
        '  "credit_tips": ["tip1", "tip2", "tip3", "tip4", "tip5"]\n'
        "}\n"
        "credit_tips: exactly 5 items. Each must be ONE concrete action the borrower can take to improve "
        "creditworthiness or CIBIL score in India (on-time EMIs, utilisation, enquiry spacing, secured mix, "
        "dispute errors, UPI/digital footprint where relevant). "
        "Write for flash-card style reading: one sentence each, under 220 characters, no numbering prefix. "
        "Tie to the data when possible (e.g. high food share → budget for EMIs); never invent rupee amounts not given."
    )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                OPENAI_URL,
                headers={
                    "Authorization": f"Bearer {_openai_key()}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.4,
                    "response_format": {"type": "json_object"},
                },
            )
        if r.status_code != 200:
            logger.warning("LLM insights failed — HTTP %s", r.status_code)
            return None, None, False
        data = r.json()
        txt = data["choices"][0]["message"]["content"]
        parsed = json.loads(txt)
        narrative = parsed.get("spending_narrative")
        tips_raw = parsed.get("credit_tips")
        tips = [str(t) for t in tips_raw] if isinstance(tips_raw, list) else None
        logger.info("LLM insights success — tips=%s narrative_len=%s", len(tips or []), len(narrative or ""))
        return (
            narrative if isinstance(narrative, str) else None,
            tips,
            True,
        )
    except Exception as exc:
        logger.warning("LLM insights error: %s", exc)
        return None, None, False


def build_insights_response(
    csv_text: Optional[str],
    model_output: dict[str, Any],
    form: dict[str, Any],
    llm_narrative: Optional[str],
    llm_tips: Optional[list[str]],
    llm_used: bool,
) -> dict[str, Any]:
    spending = aggregate_spending_from_csv(csv_text or "")
    base_tips = rule_based_credit_tips(model_output, spending, form)
    merged_tips = list(dict.fromkeys((llm_tips or []) + base_tips))[:10]
    narrative = (
        llm_narrative
        if (llm_narrative and str(llm_narrative).strip())
        else rule_based_spending_narrative(spending, model_output, form)
    )

    return {
        "spending_by_category": spending["categories"],
        "total_debit_tracked_inr": spending["total_debit_inr"],
        "statement_rows": spending["row_count"],
        "credit_tips": merged_tips,
        "spending_narrative": narrative,
        "llm_used": llm_used,
        "rule_based_tips": base_tips,
    }
