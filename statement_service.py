"""
Bank statement PDF → cleaned dataframe → lightweight categorization → metrics
for the deployed credit model (UPI monthly count, cash ratio). Self-contained
to avoid heavy optional deps (transformers/spacy) from the Streamlit stack.
"""
from __future__ import annotations

import io
import logging
import re
from typing import Any, Optional, Tuple

_stmt_log = logging.getLogger("crednova.statement")
if not _stmt_log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s [statement] %(message)s"))
    _stmt_log.addHandler(_h)
    _stmt_log.setLevel(logging.INFO)
    _stmt_log.propagate = False

import numpy as np
import pandas as pd
import pdfplumber


def _extract_all_tables(
    file_bytes: bytes, password: str | None, table_settings: dict | None = None
) -> list:
    """Collect table rows from every page; reset buffer each call."""
    bio = io.BytesIO(file_bytes)
    all_data: list = []
    with pdfplumber.open(bio, password=password or None) as pdf:
        for page in pdf.pages:
            if table_settings:
                tables = page.extract_tables(table_settings=table_settings)
            else:
                tables = page.extract_tables()
            for table in tables or []:
                if table:
                    all_data.extend(table)
            page.flush_cache()
    return all_data


def _extract_full_text(file_bytes: bytes, password: str | None) -> str:
    bio = io.BytesIO(file_bytes)
    chunks: list[str] = []
    with pdfplumber.open(bio, password=password or None) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                chunks.append(t)
            page.flush_cache()
    return "\n".join(chunks)


def fallback_dataframe_from_statement_text(text: str) -> Tuple[pd.DataFrame, str]:
    """
    When pdfplumber finds no tables (common for HDFC/ICICI-style exports and image-heavy PDFs),
    build a minimal transaction-like dataframe from text lines so metrics + ML pipeline can run.
    """
    date_re = re.compile(
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b|\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
    )
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and len(ln.strip()) > 3]
    rows: list[dict[str, Any]] = []
    base = pd.Timestamp.now().normalize()

    txn_hint = re.compile(
        r"upi|imps|neft|rtgs|atm|withdraw|deposit|transfer|debit|credit|txn|ref\s*no|inr|rs\.?",
        re.I,
    )
    for i, line in enumerate(lines[:800]):
        if not txn_hint.search(line) and not date_re.search(line):
            continue
        m = date_re.search(line)
        td: Any = pd.NaT
        if m:
            raw = m.group(0).strip()
            td = pd.to_datetime(raw, dayfirst=True, errors="coerce")
        if pd.isna(td):
            td = base - pd.Timedelta(days=(i % 180) + 1)
        rows.append(
            {
                "Trans. Date": td,
                "Remarks": line[:900],
                "Debits": 0.0,
                "Credits": 0.0,
                "Balance": 0.0,
            }
        )

    if not rows:
        for i, line in enumerate(lines[:400]):
            if re.search(r"\d", line):
                rows.append(
                    {
                        "Trans. Date": base - pd.Timedelta(days=(i % 120) + 1),
                        "Remarks": line[:900],
                        "Debits": 0.0,
                        "Credits": 0.0,
                        "Balance": 0.0,
                    }
                )

    if not rows:
        rows.append(
            {
                "Trans. Date": base,
                "Remarks": (text[:1500] if text else "Empty statement text."),
                "Debits": 0.0,
                "Credits": 0.0,
                "Balance": 0.0,
            }
        )

    df = pd.DataFrame(rows)
    return df, "Text fallback (no tables — inferred rows from statement text)"


def detect_and_parse_pdf(file_bytes: bytes, password: str | None = None) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        all_data = _extract_all_tables(file_bytes, password, None)

        if not all_data:
            _stmt_log.info("PDF: no tables with default settings — trying relaxed table strategies")
            for settings in (
                {
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "intersection_tolerance": 8,
                },
                {"vertical_strategy": "text", "horizontal_strategy": "text"},
            ):
                try:
                    all_data = _extract_all_tables(file_bytes, password, settings)
                except Exception as ex:
                    _stmt_log.warning("PDF: table extract with settings failed: %s", ex)
                    all_data = []
                if all_data:
                    break

        if not all_data:
            plain = _extract_full_text(file_bytes, password)
            if plain and len(plain.strip()) >= 15:
                _stmt_log.info(
                    "PDF: still no tables — using text fallback (%s chars)",
                    len(plain),
                )
                return fallback_dataframe_from_statement_text(plain)
            return None, (
                "Could not read transactions from this PDF. "
                "Try: (1) export from net banking as PDF with text, not a photo scan, "
                "(2) password-protected PDFs need the correct password field."
            )

        df = pd.DataFrame(all_data)
        if len(df) < 2:
            plain = _extract_full_text(file_bytes, password)
            if plain and len(plain.strip()) >= 15:
                _stmt_log.info("PDF: extracted table too small — text fallback")
                return fallback_dataframe_from_statement_text(plain)
            return None, "Parsed table had insufficient rows."

        if any("S.No" in str(row) for row in all_data[:10]):
            header_idx = 0
            for i, row in enumerate(all_data):
                if "S.No" in str(row):
                    header_idx = i
                    break
            df = pd.DataFrame(all_data[header_idx + 1:], columns=all_data[header_idx])
            mapping = {
                "Txn Date": "Trans. Date",
                "Value Date": "Value. Date",
                "Description": "Remarks",
                "Withdrawals (Dr)": "Debits",
                "Deposits (Cr)": "Credits",
                "Balance (INR)": "Balance",
                "Cheque No": "Reference",
            }
            df.rename(columns=mapping, inplace=True)
            return df, "IDBI Bank Detected"

        try:
            df.columns = df.iloc[0]
            df = df[1:]
        except Exception as ex:
            _stmt_log.warning("PDF: could not interpret first row as header: %s", ex)
            plain = _extract_full_text(file_bytes, password)
            if plain and len(plain.strip()) >= 15:
                return fallback_dataframe_from_statement_text(plain)
            return None, "Could not interpret bank statement table headers."

        if len(df) < 1:
            plain = _extract_full_text(file_bytes, password)
            if plain and len(plain.strip()) >= 15:
                return fallback_dataframe_from_statement_text(plain)
            return None, "Statement table had no data rows."

        current_cols = [str(c).lower() for c in df.columns if c is not None]

        def find_col(keywords):
            for i, col in enumerate(df.columns):
                if any(kw in str(col).lower() for kw in keywords):
                    return col
            return None

        mapping = {}
        if not any(kw in current_cols for kw in ["debits", "withdrawals"]):
            found = find_col(["debit", "withdrawal", "out", "paid out", "payment"])
            if found:
                mapping[found] = "Debits"
        if not any(kw in current_cols for kw in ["credits", "deposits"]):
            found = find_col(["credit", "deposit", "in", "received"])
            if found:
                mapping[found] = "Credits"
        if not any(kw in current_cols for kw in ["balance"]):
            found = find_col(["balance", "bal"])
            if found:
                mapping[found] = "Balance"
        if not any(kw in current_cols for kw in ["remarks", "description"]):
            found = find_col(["remarks", "description", "narration", "particulars"])
            if found:
                mapping[found] = "Remarks"
        if mapping:
            df.rename(columns=mapping, inplace=True)

        return df, "Generic Format (Smart Mapped) Detected"
    except Exception as e:
        if "password" in str(e).lower():
            return None, "PASSWORD_REQUIRED"
        return None, f"Error: {str(e)}"


_DIGITAL_TXN_RE = re.compile(
    r"upi|imps|neft|rtgs|vpa|phonepe|paytm|gpay|google\s*pay|amazon\s*pay|ybl|oksbi|okaxis|okhdfc|@",
    re.I,
)


def normalize_bank_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map typical bank export headers → trans_date, credits, debits, remarks.
    Without this, generic PDFs keep a column named `date` and compute_statement_analysis exits early.
    """
    df = df.copy()

    def _lower_map() -> dict[str, str]:
        return {str(c).strip().lower().replace(" ", "_").replace(".", ""): c for c in df.columns}

    lm = _lower_map()
    if "trans_date" not in lm:
        for key in (
            "txn_date",
            "transaction_date",
            "tran_date",
            "posting_date",
            "book_date",
            "value_date",
            "date",
        ):
            if key in lm:
                df = df.rename(columns={lm[key]: "trans_date"})
                break
    lm = _lower_map()
    if "credits" not in lm and "credit" in lm:
        df = df.rename(columns={lm["credit"]: "credits"})
    lm = _lower_map()
    if "credits" not in lm and "deposits" in lm:
        df = df.rename(columns={lm["deposits"]: "credits"})
    lm = _lower_map()
    if "debits" not in lm and "debit" in lm:
        df = df.rename(columns={lm["debit"]: "debits"})
    lm = _lower_map()
    if "debits" not in lm and "withdrawals" in lm:
        df = df.rename(columns={lm["withdrawals"]: "debits"})
    lm = _lower_map()
    if "debits" not in lm and "withdrawal" in lm:
        df = df.rename(columns={lm["withdrawal"]: "debits"})
    lm = _lower_map()
    if "remarks" not in lm:
        for key in ("narration", "description", "particulars", "details", "remarks"):
            if key in lm:
                df = df.rename(columns={lm[key]: "remarks"})
                break
    return df


def _fill_dates_from_remarks(series: pd.Series) -> pd.Series:
    def _one(txt: Any) -> Any:
        m = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", str(txt))
        if not m:
            return pd.NaT
        return pd.to_datetime(m.group(1), dayfirst=True, errors="coerce")

    return series.map(_one)


def clean_bank_statement(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for i, col in enumerate(df.columns):
        if col is None or pd.isna(col):
            new_col = f"column_{i}"
        else:
            new_col = str(col).strip().replace(".", "").replace(" ", "_").lower()
        cols.append(new_col)

    unique_cols = []
    counts: dict[str, int] = {}
    for col in cols:
        if col in counts:
            counts[col] += 1
            unique_cols.append(f"{col}_{counts[col]}")
        else:
            counts[col] = 0
            unique_cols.append(col)

    df = df.copy()
    df.columns = unique_cols
    df = normalize_bank_columns(df)

    for c in ["value_date", "reference"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    df = df.reset_index(drop=True)

    for col in ["debits", "credits", "balance", "amount"]:
        if col in df.columns and df[col].dtype == "object":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("R", "", regex=False)
                .str.replace(" ", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "trans_date" in df.columns:
        s = df["trans_date"]
        try:
            df["trans_date"] = pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")
        except (ValueError, TypeError):
            df["trans_date"] = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if "remarks" in df.columns:
            miss = df["trans_date"].isna()
            if miss.any():
                parsed = _fill_dates_from_remarks(df.loc[miss, "remarks"])
                df.loc[miss, "trans_date"] = parsed
        if len(df) > 0:
            if df["trans_date"].isna().all():
                base = pd.Timestamp.now().normalize()
                df["trans_date"] = [base - pd.Timedelta(days=min(i, 365)) for i in range(len(df))]
            else:
                fill_ts = pd.Timestamp.now().normalize()
                df["trans_date"] = df["trans_date"].ffill().bfill().fillna(fill_ts)
        df = df.dropna(subset=["trans_date"])
        df = df.reset_index(drop=True)

    for col in ["debits", "credits", "balance", "amount"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if "remarks" in df.columns:
        df["remarks"] = df["remarks"].fillna("")

    return df


_MERCHANT_MAP = {
    "zomato": "Food & Dining",
    "swiggy": "Food & Dining",
    "amazon": "Shopping",
    "flipkart": "Shopping",
    "kreditbee": "Lending",
    "navi": "Lending",
    "ring ": "Lending",
    "cashe": "Lending",
    "mpokket": "Lending",
}


def _extract_merchant(text: str) -> str:
    text = str(text).lower()
    m = re.search(r"upi/([a-zA-Z0-9.\-_]+)/", text)
    if m:
        return m.group(1)
    clean_text = re.sub(
        r"(vpa|upi|ref|txn|transfer to|pmt to|imps|neft|rtgs)", "", text, flags=re.I
    ).strip()
    clean_text = re.sub(r"(dr|cr|net|at|from|to|to self|on)", "", clean_text).strip()
    return clean_text.split()[0] if clean_text else "unknown"


def _keyword_category(text: str) -> Optional[str]:
    t = str(text).lower()
    if any(x in t for x in ["failed", "declined"]):
        return "Failed Transaction"
    if "reversed" in t or "reversal" in t:
        return "Reversal"
    if "transfer to self" in t or "own account" in t or "self transfer" in t:
        return "Self Transfer"
    if any(x in t for x in ["atm", "cash wdl", "cash withdrawal", "cwdr"]):
        return "Cash Activity"
    if "salary" in t or "payroll" in t:
        return "Salary"
    if "interest" in t and ("cr" in t or "credit" in t):
        return "Income"
    for kw, cat in _MERCHANT_MAP.items():
        if kw in t:
            return cat
    if "upi" in t or "phonepe" in t or "paytm" in t or "gpay" in t or "google pay" in t:
        return "UPI Digital"
    return None


def categorize_row(remarks: Any) -> str:
    text = str(remarks or "")
    cat = _keyword_category(text)
    if cat:
        return cat
    merchant = _extract_merchant(text)
    m = str(merchant).lower()
    for key, val in _MERCHANT_MAP.items():
        if key in m:
            return val
    return "Other"


def categorize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    remark_col = "remarks" if "remarks" in out.columns else None
    if remark_col is None:
        out["category"] = "Other"
        return out
    out["category"] = out[remark_col].apply(categorize_row)
    return out


def calculate_statement_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Aligns with credit_engine-style outputs used for the Render model."""
    if df.empty:
        return {
            "monthly_upi": 0.0,
            "cash_transaction_ratio": 0.0,
            "months_analyzed": 1.0,
            "statement_row_count": 0,
        }

    if "trans_date" in df.columns:
        td = pd.to_datetime(df["trans_date"], errors="coerce")
        date_range = (td.max() - td.min()).days if td.notna().any() else 0
        months = max(1.0, round(date_range / 30.44, 1))
        # Same-day or broken dates: spread rows across ~estimated months for rate sanity
        if months <= 1.0 and len(df) > 15:
            est = max(1.0, min(18.0, round(len(df) / 22.0, 1)))
            months = est
    else:
        months = 1.0

    noise_cats = ["Self Transfer", "Failed Transaction", "Reversal"]
    cat_col = "category" if "category" in df.columns else None
    if cat_col:
        clean_df = df[~df[cat_col].isin(noise_cats)].copy()
    else:
        clean_df = df.copy()

    total_txns = len(clean_df)
    if "remarks" in clean_df.columns:
        ser = clean_df["remarks"].astype(str)
        digital_mask = ser.str.contains(_DIGITAL_TXN_RE, na=False)
        upi_count = int(digital_mask.sum())
    else:
        upi_count = 0
    monthly_upi = round(upi_count / months, 1) if months else 0.0

    if cat_col and total_txns > 0:
        cash_txns = len(clean_df[clean_df[cat_col] == "Cash Activity"])
        cash_ratio = round(cash_txns / total_txns, 4)
    elif "remarks" in clean_df.columns and total_txns > 0:
        ser = clean_df["remarks"].astype(str)
        cash_like = ser.str.contains(
            r"atm|cash\s*wdl|cash\s*withdraw|cwdr|cash\s*dep", case=False, na=False
        )
        cash_ratio = round(float(cash_like.sum()) / total_txns, 4)
    else:
        cash_ratio = 0.0

    return {
        "monthly_upi": float(monthly_upi),
        "cash_transaction_ratio": float(cash_ratio),
        "months_analyzed": float(months),
        "statement_row_count": int(len(df)),
    }


def compute_statement_analysis(df: Optional[pd.DataFrame]) -> dict[str, Any]:
    """
    Monthly credits/debits, totals, balance peaks, rent heuristic — JSON-serializable for dashboard.
    """
    empty: dict[str, Any] = {
        "available": False,
        "monthly": [],
        "totals": {
            "total_credits_inr": 0.0,
            "total_debits_inr": 0.0,
            "net_savings_inr": 0.0,
            "months_in_chart": 0,
        },
        "balance": {"peak_inr": None, "peak_month_label": None, "closing_inr": None},
        "rental": {"credits_inr": 0.0, "months_with_rent": 0, "avg_monthly_inr": 0.0},
        "insights": [],
    }
    if df is None or df.empty:
        return empty

    d = normalize_bank_columns(df.copy())
    if "trans_date" not in d.columns:
        return empty

    try:
        d["trans_date"] = pd.to_datetime(
            d["trans_date"], errors="coerce", dayfirst=True, format="mixed"
        )
    except (ValueError, TypeError):
        d["trans_date"] = pd.to_datetime(d["trans_date"], errors="coerce", dayfirst=True)
    if "remarks" in d.columns:
        miss = d["trans_date"].isna()
        if miss.any():
            d.loc[miss, "trans_date"] = _fill_dates_from_remarks(d.loc[miss, "remarks"])
    d = d.dropna(subset=["trans_date"])
    if d.empty:
        return empty

    if "credits" not in d.columns:
        if "credit" in d.columns:
            d["credits"] = pd.to_numeric(d["credit"], errors="coerce").fillna(0.0)
        else:
            d["credits"] = 0.0
    else:
        d["credits"] = pd.to_numeric(d["credits"], errors="coerce").fillna(0.0)
    if "debits" not in d.columns:
        if "debit" in d.columns:
            d["debits"] = pd.to_numeric(d["debit"], errors="coerce").fillna(0.0)
        else:
            d["debits"] = 0.0
    else:
        d["debits"] = pd.to_numeric(d["debits"], errors="coerce").fillna(0.0)

    d["_ym"] = d["trans_date"].dt.to_period("M")
    agg = d.groupby("_ym", sort=True).agg(credits=("credits", "sum"), debits=("debits", "sum"))
    monthly: list[dict[str, Any]] = []
    for period, row in agg.iterrows():
        ts = period.to_timestamp()
        monthly.append(
            {
                "key": str(period),
                "label": ts.strftime("%b %y"),
                "credits": float(row["credits"]),
                "debits": float(row["debits"]),
            }
        )

    total_credits = float(d["credits"].sum())
    total_debits = float(d["debits"].sum())
    net = total_credits - total_debits

    peak_inr = None
    peak_label = None
    closing_inr = None
    if "balance" in d.columns:
        d["balance"] = pd.to_numeric(d["balance"], errors="coerce")
        if d["balance"].notna().any():
            idx = d["balance"].idxmax()
            peak_inr = float(d.loc[idx, "balance"])
            peak_label = pd.to_datetime(d.loc[idx, "trans_date"]).strftime("%b %Y")
            d_sorted = d.sort_values("trans_date")
            last_bal = pd.to_numeric(d_sorted["balance"], errors="coerce").iloc[-1]
            if pd.notna(last_bal):
                closing_inr = float(last_bal)

    rental_credits = 0.0
    rent_months: set[str] = set()
    if "remarks" in d.columns:
        mask = d["remarks"].astype(str).str.contains(
            r"rent|lease|landlord|tenant|housing", case=False, na=False
        )
        rental_credits = float(d.loc[mask, "credits"].sum())
        if mask.any():
            rent_months = set(d.loc[mask, "_ym"].astype(str).unique())
    rental_months_n = len(rent_months)
    avg_rent = rental_credits / rental_months_n if rental_months_n else 0.0

    insights: list[str] = []
    if net >= 0:
        insights.append(
            f"Net inflow of ₹{net:,.0f} over the statement window — positive savings signal for underwriting."
        )
    else:
        insights.append(
            f"Net outflow of ₹{abs(net):,.0f} — review discretionary spend vs. declared income."
        )
    if total_credits > 0 and total_debits > 0:
        ratio = total_debits / max(total_credits, 1.0)
        if ratio > 0.95:
            insights.append("Debit volume is close to credit volume — monitor liquidity buffers.")
        elif ratio < 0.5:
            insights.append("Strong credit-heavy flow — healthy surplus vs. withdrawals.")
    if rental_credits > 0:
        insights.append(
            "Recurring rent-related credits detected in narration — use as supporting income context where valid."
        )
    if "remarks" in d.columns and len(d) > 0:
        dig_share = float(
            len(d[d["remarks"].astype(str).str.contains(_DIGITAL_TXN_RE, na=False)]) / len(d)
        )
        if dig_share > 0.05:
            insights.append(
                f"About {dig_share * 100:.0f}% of rows match digital-rail patterns (UPI/IMPS/NEFT/VPA, etc.)."
            )
        else:
            insights.append(
                "Few digital keywords in text — narration may use abbreviations your bank hides; "
                "prefer ‘download as PDF with text’ from net banking."
            )
    if not insights:
        insights.append("Statement parsed successfully — factors blended with ML model score on the dashboard.")

    return {
        "available": True,
        "monthly": monthly,
        "totals": {
            "total_credits_inr": total_credits,
            "total_debits_inr": total_debits,
            "net_savings_inr": net,
            "months_in_chart": len(monthly),
        },
        "balance": {
            "peak_inr": peak_inr,
            "peak_month_label": peak_label,
            "closing_inr": closing_inr,
        },
        "rental": {
            "credits_inr": rental_credits,
            "months_with_rent": rental_months_n,
            "avg_monthly_inr": float(avg_rent),
        },
        "insights": insights[:6],
    }


def process_bank_pdf(
    file_bytes: bytes, password: str | None = None
) -> Tuple[Optional[pd.DataFrame], str, dict[str, Any]]:
    _stmt_log.info("PDF: starting extraction (detect tables → clean → categorize → metrics)")
    raw, message = detect_and_parse_pdf(file_bytes, password)
    if raw is None:
        _stmt_log.warning("PDF: extraction failed: %s", message)
        return None, message, {}
    _stmt_log.info("PDF: raw tables parsed into dataframe rows=%s", len(raw))
    cleaned = clean_bank_statement(raw.copy())
    final = categorize_dataframe(cleaned)
    _stmt_log.info("PDF: cleaned & categorized rows=%s", len(final))
    metrics = calculate_statement_metrics(final)
    _stmt_log.info(
        "PDF data extraction complete: metrics monthly_upi=%s cash_ratio=%s months=%s row_count=%s",
        metrics.get("monthly_upi"),
        metrics.get("cash_transaction_ratio"),
        metrics.get("months_analyzed"),
        metrics.get("statement_row_count"),
    )
    return final, message, metrics


def dataframe_to_csv_string(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)
