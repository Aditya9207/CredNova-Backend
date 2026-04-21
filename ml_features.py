"""
Local ML branch (feature engineering only — no weights learned here).

Bank PDF → ``statement_service`` extracts transaction rows and **statement_metrics**
(``monthly_upi``, ``cash_transaction_ratio``, …). The applicant **form** is merged with
those metrics in ``credit_flow.build_model_payload`` → JSON body for the **remote**
``CREDIT_MODEL_URL`` /predict service.

Parallel branch: **LLM insights** (``insights_service``) consume the same exported CSV
for spending categories + tips; they do not feed the online ML model.
"""
from __future__ import annotations

from typing import Any

# Keys produced by ``calculate_statement_metrics`` in ``statement_service``
STATEMENT_METRIC_KEYS = (
    "monthly_upi",
    "cash_transaction_ratio",
    "months_analyzed",
    "statement_row_count",
)


def statement_branch_snapshot(statement_metrics: dict[str, Any]) -> dict[str, Any]:
    """Serializable view of PDF-derived inputs used when building the remote model payload."""
    if not statement_metrics:
        return {}
    return {k: statement_metrics.get(k) for k in STATEMENT_METRIC_KEYS}
