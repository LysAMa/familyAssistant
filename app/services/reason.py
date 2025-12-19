
from ..config import get_settings

def explain_recommendation(legs: list[dict], weather: dict | None = None) -> dict:
    # Simple heuristic if no LLM is configured
    penalties = {"bicycling": 0, "walking": 0, "driving": 0, "transit": 0}
    cond = (weather or {}).get("current", {}).get("condition", "").lower()
    if any(k in cond for k in ["rain","pluie","neige","snow"]):
        penalties["walking"] += 12
        penalties["bicycling"] += 10

    ranked = []
    for leg in legs:
        base = leg.get("etaMinutes") or 9999
        score = base + penalties.get(leg.get("mode",""), 0)
        ranked.append((score, leg))
    ranked.sort(key=lambda x:x[0])
    best_leg = ranked[0][1] if ranked else None

    explanation = "Choix basé sur ETA + météo: "
    if best_leg:
        explanation += f"{best_leg['mode']} ({best_leg['etaMinutes']} min)"
        if any(penalties.values()):
            explanation += ", modes pénalisés par météo."
    return {
        "bestMode": best_leg.get("mode") if best_leg else None,
        "ranked": [r[1] for r in ranked],
        "explanation": explanation
    }
