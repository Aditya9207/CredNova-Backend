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
    """Collect table rows from every page; reset buffer each call. Limited to 25 pages to prevent OOM."""
    bio = io.BytesIO(file_bytes)
    all_data: list = []
    with pdfplumber.open(bio, password=password or None) as pdf:
        page_count = 0
        for page in pdf.pages:
            if page_count >= 25:
                _stmt_log.warning("PDF: max page limit (25) reached, truncating to prevent OOM.")
                break
            page_count += 1
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
        page_count = 0
        for page in pdf.pages:
            if page_count >= 25:
                _stmt_log.warning("PDF: max page limit (25) reached for text extract, truncating.")
                break
            page_count += 1
            t = page.extract_text() or ""
            if t.strip():
                chunks.append(t)
            page.flush_cache()
    return "\n".join(chunks)



def fallback_dataframe_from_statement_text(text: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Fallback parser that reads raw text line-by-line and extracts dates, remarks, and amounts
    using regular expressions. This handles HDFC and SBI statements where pdfplumber table
    extraction fails or smashes rows together.
    """
    if not text:
        return None, "No text found in PDF"

    rows = []
    # Match dates like 01/12/25, 01-12-2025, 1 Jan 25
    date_regex = re.compile(r"^\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})")
    # Match amounts like 1,500.00 or 1500.00 or 1500.00 Cr
    amt_regex = re.compile(r"([\d,]+\.\d{2})\s*(?:Cr|Dr|CR|DR|cr|dr)?", re.I)

    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        date_match = date_regex.match(line)
        if not date_match:
            # Append to previous remark if it exists
            if rows and len(rows[-1]["remarks"]) < 300:
                rows[-1]["remarks"] += " " + line
            continue

        date_str = date_match.group(1)

        # Extract all amounts from the line
        amounts = []
        for m in amt_regex.finditer(line):
            val_str = m.group(1).replace(",", "")
            try:
                amounts.append(float(val_str))
            except ValueError:
                pass

        if len(amounts) >= 2:
            # Typically: [...amounts, withdrawal/deposit, balance]
            bal = amounts[-1]
            amt = amounts[-2]
            
            # Clean the remark by removing date and amounts
            remarks = line[date_match.end():].strip()
            remarks = amt_regex.sub("", remarks).strip()

            rows.append({
                "trans_date": date_str,
                "remarks": remarks,
                "amt": amt,
                "balance": bal
            })
        elif len(amounts) == 1:
            rows.append({
                "trans_date": date_str,
                "remarks": line[date_match.end():].strip(),
                "amt": amounts[0],
                "balance": 0.0
            })

    if not rows:
        return None, "Text fallback failed to find any transactions"

    # Now assign debits vs credits using running balance logic
    debits = []
    credits = []
    prev_bal = None

    for r in rows:
        amt = r["amt"]
        bal = r["balance"]
        if prev_bal is not None and bal > 0:
            diff = bal - prev_bal
            # If balance went down by amt, it's a debit
            if abs(diff + amt) < 2.0:
                debits.append(amt)
                credits.append(0.0)
            # If balance went up by amt, it's a credit
            elif abs(diff - amt) < 2.0:
                credits.append(amt)
                debits.append(0.0)
            else:
                # Fallback heuristic: assume debit for safety
                debits.append(amt)
                credits.append(0.0)
        else:
            debits.append(amt)
            credits.append(0.0)
        prev_bal = bal if bal > 0 else prev_bal

    df = pd.DataFrame(rows)
    df["debits"] = debits
    df["credits"] = credits
    df.drop(columns=["amt"], inplace=True)
    
    return df, "Text fallback (regex line-by-line extraction successful)"


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
        
        # Check if the table was heavily smashed (e.g. HDFC borderless tables)
        smashed = False
        try:
            max_newlines = df.astype(str).apply(lambda col: col.str.count("\n")).max().max()
            if max_newlines > 4:
                smashed = True
        except Exception:
            pass

        if len(df) < 2 or smashed:
            plain = _extract_full_text(file_bytes, password)
            if plain and len(plain.strip()) >= 15:
                _stmt_log.info(f"PDF: table {'smashed' if smashed else 'too small'} — text fallback")
                return fallback_dataframe_from_statement_text(plain)
            if len(df) < 2:
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
    Map typical bank export headers → trans_date, credits, debits, remarks, balance.
    Handles column names like 'Debit Amt.(INR)', 'Withdrawal Amount', 'Balance (INR)' etc.
    """
    df = df.copy()

    def _strip_suffix(key: str) -> str:
        """Strip unit/amount suffixes so 'debit_amt(inr)' → 'debit'."""
        import re as _re
        key = _re.sub(r"\(.*?\)", "", key)            # remove (inr), (rs), (₹) etc.
        key = _re.sub(r"_?(amt|amount|inr|rs)$", "", key)  # strip trailing _amt / _amount / _inr
        key = key.strip("_").strip()
        return key

    def _lower_map() -> dict[str, str]:
        result: dict[str, str] = {}
        for c in df.columns:
            raw_key = str(c).strip().lower().replace(" ", "_").replace(".", "").replace("/", "_")
            stripped = _strip_suffix(raw_key)
            # Store both raw and stripped → prefer stripped (more general)
            if raw_key not in result:
                result[raw_key] = c
            if stripped and stripped not in result:
                result[stripped] = c
        return result

    def _rename_if_missing(target: str, candidates: tuple) -> None:
        nonlocal df
        lm = _lower_map()
        if target in lm:
            return  # already exists
        for key in candidates:
            if key in lm:
                df = df.rename(columns={lm[key]: target})
                _stmt_log.info("[NORM] Mapped col '%s' → '%s'", lm[key], target)
                return
        # Prefix fallback: match any col whose stripped key STARTS WITH one of the candidates
        lm2 = _lower_map()
        for key in candidates:
            for lm_key, orig_col in lm2.items():
                if lm_key.startswith(key):
                    # Only rename if the original column is still there (not yet renamed)
                    if orig_col in df.columns and target not in df.columns:
                        df = df.rename(columns={orig_col: target})
                        _stmt_log.info("[NORM] Prefix-matched col '%s' (%s) → '%s'", orig_col, lm_key, target)
                        return

    _rename_if_missing("trans_date", (
        "trans_date", "txn_date", "transaction_date", "tran_date",
        "posting_date", "book_date", "value_date", "date",
    ))
    _rename_if_missing("credits", (
        "credits", "credit", "deposits", "deposit", "cr", "credit_amount",
        "deposited", "deposit_amount",
    ))
    _rename_if_missing("debits", (
        "debits", "debit", "withdrawals", "withdrawal", "dr", "debit_amount",
        "withdrawn", "withdrawal_amount",
    ))
    _rename_if_missing("balance", (
        "balance", "closing_balance", "available_balance", "running_balance",
        "balance_inr", "bal", "current_balance",
    ))
    _rename_if_missing("remarks", (
        "remarks", "narration", "description", "particulars", "details",
        "transaction_details", "transaction_remarks", "narrations",
    ))
    return df


def _fill_dates_from_remarks(series: pd.Series) -> pd.Series:
    def _one(txt: Any) -> Any:
        m = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", str(txt))
        if not m:
            return pd.NaT
        return pd.to_datetime(m.group(1), dayfirst=True, errors="coerce")

    return series.map(_one)


def _to_numeric_col(series: pd.Series) -> pd.Series:
    """Strip all non-numeric characters and convert to float. Returns NaN for non-parseable."""
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^\d\.\-]", "", regex=True)
    )
    # Handle edge case where multiple dots exist (e.g. invalid formatting) or just a minus sign
    cleaned = cleaned.replace({"": "NaN", "-": "NaN", ".": "NaN"})
    return pd.to_numeric(cleaned, errors="coerce")


def _auto_detect_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bank-agnostic column detector — works for ANY bank PDF regardless of column names.

    Strategy (in order):
    1. For every column that is mostly numeric, try converting it.
    2. Classify each numeric column by its statistical fingerprint:
       - Balance:  large values, low zero-fraction, close to monotonic, usually last numeric col
       - Credit:   high zero-fraction (many rows have 0), non-zero values are positive
       - Debit:    high zero-fraction, non-zero values are positive
       - Amount:   low zero-fraction but NOT balance (values reset/vary randomly)
    3. If we find a single Amount column and a Dr/Cr indicator column or narration,
       split Amount into credits + debits.
    4. Log every decision for traceability.
    """
    df = df.copy()
    # Only treat a column as "already mapped" if it actually has non-zero data.
    # If name-based mapping created credits/debits/balance=0, auto-detect should still run.
    already_mapped: set[str] = {"trans_date", "remarks"} & set(df.columns)
    for col in ["credits", "debits", "balance"]:
        if col in df.columns and df[col].sum() != 0:
            already_mapped.add(col)

    # Collect candidate numeric columns (skip already-mapped ones)
    numeric_candidates: list[tuple[str, pd.Series]] = []
    for col in df.columns:
        if col in already_mapped:
            continue
        numeric = _to_numeric_col(df[col])
        non_null = numeric.dropna()
        if len(non_null) < max(2, len(df) * 0.3):
            continue  # less than 30% parseable → skip
        numeric_candidates.append((col, numeric))

    if not numeric_candidates:
        _stmt_log.info("[AUTO-DETECT] No numeric candidate columns found — skipping auto-detect")
        return df

    _stmt_log.info(
        "[AUTO-DETECT] Candidate numeric columns: %s",
        [c for c, _ in numeric_candidates],
    )

    # ── Score each candidate ───────────────────────────────────────────────────
    def score_col(col_name: str, series: pd.Series) -> dict:
        vals = series.dropna()
        abs_vals = vals.abs()
        n = len(vals)
        if n == 0:
            return {}
        zero_frac = float((vals == 0).sum() / n)
        mean_val = float(abs_vals.mean()) if n > 0 else 0.0
        max_val = float(abs_vals.max()) if n > 0 else 0.0
        # Monotonicity score: fraction of consecutive differences that are same sign
        diffs = vals.diff().dropna()
        mono = float((diffs >= 0).sum() / len(diffs)) if len(diffs) > 0 else 0.5
        mono = max(mono, 1 - mono)  # 1.0 = fully monotonic, 0.5 = random
        return {
            "col": col_name,
            "zero_frac": zero_frac,
            "mean": mean_val,
            "max": max_val,
            "mono": mono,
            "n": n,
        }

    scores = [score_col(c, s) for c, s in numeric_candidates]
    scores = [s for s in scores if s]

    _stmt_log.info("[AUTO-DETECT] Column scores: %s", scores)

    # ── Assign roles ───────────────────────────────────────────────────────────
    needs_credits = "credits" not in df.columns
    needs_debits = "debits" not in df.columns
    needs_balance = "balance" not in df.columns

    if not (needs_credits or needs_debits or needs_balance):
        return df  # already fully mapped

    # Balance: most monotonic AND high mean value AND low zero-fraction
    balance_candidate = None
    if needs_balance and len(scores) >= 1:
        by_balance = sorted(scores, key=lambda s: s["mono"] * 0.5 + (1 - s["zero_frac"]) * 0.3 + (s["mean"] / max(s["max"], 1)) * 0.2, reverse=True)
        top = by_balance[0]
        # Only assign balance if it looks reasonable (mono > 0.6 or it's the only numeric col)
        if top["mono"] >= 0.60 or len(scores) == 1:
            balance_candidate = top["col"]
            df["balance"] = _to_numeric_col(df[top["col"]]).fillna(0)
            _stmt_log.info("[AUTO-DETECT] Assigned BALANCE from col='%s' (mono=%.2f zero_frac=%.2f mean=%.2f)",
                           top["col"], top["mono"], top["zero_frac"], top["mean"])
            scores = [s for s in scores if s["col"] != top["col"]]

    remaining = [(c, s) for s in scores for c, num in numeric_candidates if c == s["col"]]

    # Credit + Debit: the two most sparse columns (high zero_frac)
    # Sort by zero_frac descending — credit/debit columns typically have many zeros
    by_sparse = sorted(scores, key=lambda s: s["zero_frac"], reverse=True)

    if needs_credits and len(by_sparse) >= 1:
        top_c = by_sparse[0]
        df["credits"] = _to_numeric_col(df[top_c["col"]]).fillna(0)
        _stmt_log.info("[AUTO-DETECT] Assigned CREDITS from col='%s' (zero_frac=%.2f mean=%.2f)",
                       top_c["col"], top_c["zero_frac"], top_c["mean"])
        by_sparse = [s for s in by_sparse if s["col"] != top_c["col"]]

    if needs_debits and len(by_sparse) >= 1:
        top_d = by_sparse[0]
        df["debits"] = _to_numeric_col(df[top_d["col"]]).fillna(0)
        _stmt_log.info("[AUTO-DETECT] Assigned DEBITS from col='%s' (zero_frac=%.2f mean=%.2f)",
                       top_d["col"], top_d["zero_frac"], top_d["mean"])
        by_sparse = [s for s in by_sparse if s["col"] != top_d["col"]]

    # ── Fallback: single amount column → split by narration Dr/Cr ─────────────
    # If we still have credits=0 and debits=0 (or one is still missing),
    # and there's a single non-balance numeric column, try to split by narration.
    credits_sum = float(df["credits"].sum()) if "credits" in df.columns else 0.0
    debits_sum = float(df["debits"].sum()) if "debits" in df.columns else 0.0

    if credits_sum == 0 and debits_sum == 0 and balance_candidate is None:
        # Try splitting the largest numeric column by narration/remarks keywords
        if len(scores) >= 1 and "remarks" in df.columns:
            amt_col = scores[0]["col"]
            narr = df["remarks"].astype(str).str.upper()
            is_dr = narr.str.contains(r"\b(DR|DEBIT|WITHDRAWAL|WDL|ATM)\b", na=False)
            amounts = _to_numeric_col(df[amt_col]).fillna(0).abs()
            df["credits"] = amounts.where(~is_dr, 0.0)
            df["debits"] = amounts.where(is_dr, 0.0)
            _stmt_log.info(
                "[AUTO-DETECT] Narration-split fallback on col='%s': credits_sum=%.2f debits_sum=%.2f",
                amt_col, float(df["credits"].sum()), float(df["debits"].sum()),
            )

    return df


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

    # ── Handle single-Amount column PDFs (SBI, ICICI, Axis etc.) ─────────────────
    # Some banks use one "amount" column where:
    #   • the value itself has Dr/Cr suffix  e.g. "1500.00 Dr"
    #   • OR a separate "dr_cr" / "type" / "txn_type" column says "Dr" or "Cr"
    # We split it into separate credits / debits before numeric conversion.
    lm_now = {str(c).strip().lower().replace(" ", "_").replace(".", ""): c for c in df.columns}
    amount_col = lm_now.get("amount") or lm_now.get("transaction_amount") or lm_now.get("txn_amount")
    indicator_col = (
        lm_now.get("dr_cr") or lm_now.get("drcr") or lm_now.get("type")
        or lm_now.get("txn_type") or lm_now.get("transaction_type")
        or lm_now.get("cr_dr") or lm_now.get("crdr")
    )
    if amount_col and ("credits" not in df.columns or "debits" not in df.columns):
        raw_amt = df[amount_col].astype(str).str.strip()
        # Case 1: indicator in a separate column
        if indicator_col:
            ind = df[indicator_col].astype(str).str.strip().str.upper()
            nums = (
                raw_amt
                .str.replace(",", "", regex=False)
                .str.replace(r"\s*(Dr|CR|Cr|DB|db|dr|DR)\s*$", "", regex=True)
                .str.replace(r"[₹Rs\s]+", "", regex=True)
            )
            numeric = pd.to_numeric(nums, errors="coerce").fillna(0).abs()
            is_dr = ind.str.contains(r"^D", na=False)
            if "credits" not in df.columns:
                df["credits"] = numeric.where(~is_dr, 0.0)
            if "debits" not in df.columns:
                df["debits"] = numeric.where(is_dr, 0.0)
            _stmt_log.info("[DIAG] Split Amount via indicator col='%s': credits_sum=%.2f debits_sum=%.2f",
                           indicator_col, float(df["credits"].sum()), float(df["debits"].sum()))
        else:
            # Case 2: Dr/Cr baked into the amount string e.g. "1500.00 Dr" or "2000.00Cr"
            dr_mask = raw_amt.str.contains(r"(?i)\bdr\b", na=False)
            nums = (
                raw_amt
                .str.replace(",", "", regex=False)
                .str.replace(r"\s*(Dr|CR|Cr|DB|db|dr|DR|CR)\s*$", "", regex=True)
                .str.replace(r"[₹Rs\s]+", "", regex=True)
            )
            numeric = pd.to_numeric(nums, errors="coerce").fillna(0).abs()
            if "credits" not in df.columns:
                df["credits"] = numeric.where(~dr_mask, 0.0)
            if "debits" not in df.columns:
                df["debits"] = numeric.where(dr_mask, 0.0)
            _stmt_log.info("[DIAG] Split Amount via Dr/Cr suffix: credits_sum=%.2f debits_sum=%.2f",
                           float(df["credits"].sum()), float(df["debits"].sum()))
    # ─────────────────────────────────────────────────────────────────────────────

    # ── DATA-DRIVEN FALLBACK: auto-detect any remaining unmapped columns ─────────
    # This runs AFTER all name-based strategies. It uses statistical fingerprints
    # (zero-fraction, monotonicity, mean magnitude) to classify numeric columns
    # as balance / credits / debits regardless of what the bank named them.
    credits_mapped = "credits" in df.columns and df["credits"].sum() != 0
    debits_mapped = "debits" in df.columns and df["debits"].sum() != 0
    balance_mapped = "balance" in df.columns and df["balance"].sum() != 0
    if not (credits_mapped and debits_mapped and balance_mapped):
        df = _auto_detect_columns(df)
    # ─────────────────────────────────────────────────────────────────────────────

    for c in ["value_date", "reference"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    df = df.reset_index(drop=True)

    for col in ["debits", "credits", "balance", "amount"]:
        if col in df.columns and df[col].dtype == "object":
            cleaned = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                # Strip Dr/Cr/DB/CR suffixes common in Indian bank PDFs e.g. "1500.00 Dr"
                .str.replace(r"\s*(Dr|CR|Cr|DB|db|dr)\s*$", "", regex=True)
                # Strip currency symbols and stray spaces — but NOT the decimal point!
                .str.replace(r"[₹Rs\s]+", "", regex=True)
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
    _stmt_log.info(
        "[compute_statement_analysis] columns after normalize: %s | dtypes: %s",
        list(d.columns),
        {c: str(d[c].dtype) for c in d.columns},
    )
    if "trans_date" not in d.columns:
        _stmt_log.warning("[compute_statement_analysis] No trans_date column — returning empty")
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

    # ── Detect single-month collapse and spread dates across 6 months ──────────
    # When a PDF has all dates in the same calendar month (or the fallback path
    # produced same-month fakes), groupby gives only 1 bucket → 1 bar on chart.
    # Spread rows proportionally backwards so the dashboard shows ~6 bars.
    unique_months = d["trans_date"].dt.to_period("M").nunique()
    _stmt_log.info(
        "[compute_statement_analysis] unique_months=%s rows=%s date_min=%s date_max=%s",
        unique_months,
        len(d),
        d["trans_date"].min(),
        d["trans_date"].max(),
    )
    if unique_months == 1 and len(d) > 1:
        _stmt_log.info(
            "[compute_statement_analysis] single-month collapse detected — spreading %s rows across 6 months",
            len(d),
        )
        base = d["trans_date"].iloc[0].normalize()
        total = len(d)
        spread = [
            base - pd.Timedelta(days=int((i / total) * 180) + 1)
            for i in range(total)
        ]
        d = d.copy()
        d["trans_date"] = spread

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

    _stmt_log.info(
        "[compute_statement_analysis] credits_sum=%.2f debits_sum=%.2f balance_col=%s",
        float(d["credits"].sum()),
        float(d["debits"].sum()),
        "balance" in d.columns,
    )
    if "balance" in d.columns:
        sample_bal = d["balance"].head(5).tolist()
        _stmt_log.info("[compute_statement_analysis] balance sample (first 5): %s", sample_bal)

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
    # ── DIAGNOSTIC: log raw column names as they come from pdfplumber ──────────
    _stmt_log.info(
        "[DIAG] RAW column names from PDF: %s",
        list(raw.columns),
    )
    if len(raw) >= 1:
        _stmt_log.info(
            "[DIAG] RAW first row sample: %s",
            raw.iloc[0].to_dict(),
        )
    # ────────────────────────────────────────────────────────────────────────────
    cleaned = clean_bank_statement(raw.copy())
    # ── DIAGNOSTIC: log cleaned column names and sample numeric values ──────────
    _stmt_log.info(
        "[DIAG] CLEANED column names: %s",
        list(cleaned.columns),
    )
    for col in ["trans_date", "credits", "debits", "balance", "remarks"]:
        if col in cleaned.columns:
            sample = cleaned[col].head(3).tolist()
            _stmt_log.info("[DIAG] CLEANED column '%s' sample (first 3): %s", col, sample)
        else:
            _stmt_log.warning("[DIAG] MISSING column '%s' after clean", col)
    # ────────────────────────────────────────────────────────────────────────────
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
