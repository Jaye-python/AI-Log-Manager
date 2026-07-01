import re
from typing import Any, Optional


def _extract_key_detail(log_message: str) -> str:
    """Pulls out the most useful part of a log message — the actual description
    of what went wrong, stripping away the log level prefix and bracket labels
    that appear at the start."""
    match = re.search(r"\]\s*(.+)", log_message)
    detail = match.group(1).strip() if match else log_message.strip()
    if len(detail) > 160:
        detail = detail[:157].rstrip() + "..."
    return detail


def generate_summary(
    log_message: str,
    service: str,
    severity: str,
    predicted_label_id: str,
    predicted_label_name: str,
    confidence: Optional[float] = None,
    typical_resolution: Optional[str] = None,
) -> dict:
    """Builds a plain-English breakdown of a single log entry for an on-call engineer.
    Takes the predicted root cause and looks up what it means, how serious it is,
    and what action to take — then packages it all into a readable response."""
    detail = _extract_key_detail(log_message or "")

    issue = f"{detail}" if detail else "No message detail available."

    summary: dict[str, Any] = {
        "issue": issue,
        "predicted_root_cause_id": predicted_label_id,
        "predicted_root_cause_label": predicted_label_name,
        "affected_service": service or "unknown",
        "severity": severity or "unknown",
        "recommended_action": typical_resolution or "No recommended action on file.",
    }
    if confidence is not None:
        summary["confidence"] = round(float(confidence), 4)
    return summary
