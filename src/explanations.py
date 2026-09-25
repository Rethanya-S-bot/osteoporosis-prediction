"""Plain-language explanation fallback for model predictions."""
from __future__ import annotations


def build_explanation(risk_label: str, contributing_factors: list[dict]) -> str:
    """Create a cautious explanation when no external LLM is configured."""
    factor_names = [str(item["feature"]).replace("_", " ") for item in contributing_factors[:3]]
    factors = ", ".join(factor_names) if factor_names else "the submitted clinical factors"
    guidance = {
        "Normal": "Continue bone-supportive activity, adequate nutrition, and routine preventive care.",
        "Osteopenia": "Discuss prevention and whether follow-up bone-density testing is appropriate.",
        "Osteoporosis": "Discuss the result promptly with a qualified clinician and ask whether confirmatory evaluation is needed.",
    }.get(risk_label, "Discuss this result with a qualified clinician.")
    return (
        f"The model classified this profile as {risk_label}. The strongest model signals were {factors}. "
        f"{guidance} This is a risk estimate, not a diagnosis or a substitute for medical advice."
    )