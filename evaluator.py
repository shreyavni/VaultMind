from typing import Dict, Union

from rapidfuzz import fuzz


def calculate_groundedness(answer: str, context: str) -> float:
    """
    Calculates a lexical groundedness score (0-100).

    The score estimates how closely the generated answer's wording
    matches the retrieved document context.
    """
    if not answer or not context:
        return 0.0

    answer = answer.lower().strip()
    context = context.lower().strip()

    return fuzz.partial_ratio(answer, context)


def evaluate_answer(answer: str, context: str) -> Dict[str, Union[float, str]]:
    """
    Returns an evaluation result (score + status label) based on
    groundedness score.
    """
    score = calculate_groundedness(answer, context)

    if score >= 80:
        status = "Highly Grounded"
    elif score >= 60:
        status = "Moderately Grounded"
    elif score >= 40:
        status = "Weakly Grounded"
    else:
        status = "Potentially Ungrounded"

    return {"score": round(score, 1), "status": status}