"""LangChain-powered family transport planner.

This module introduces a simple multi-agent setup:
- Memory agent: stores roster/schedule/rules for the family.
- Logistics agent: proposes driver/mode/packing advice per event.
- Orchestrator: walks upcoming events and assembles guidance.

All state is in-memory; callers can supply family data to override the
defaults. If LangChain/OpenAI are unavailable the module falls back to a
rule-based planner so the API still responds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable

try:  # Prefer the modern LangChain import path
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:  # pragma: no cover - runtime fallback for older installs
    try:
        from langchain.chat_models import ChatOpenAI  # type: ignore
        from langchain.schema import SystemMessage, HumanMessage  # type: ignore
    except Exception:  # noqa: BLE001 - broad to keep optional dependency soft
        ChatOpenAI = None  # type: ignore
        SystemMessage = None  # type: ignore
        HumanMessage = None  # type: ignore


# Minimal family seed data to keep memory populated even without user input.
DEFAULT_FAMILY = {
    "members": [
        {"name": "Alex", "role": "Father", "can_drive": True},
        {"name": "Sam", "role": "mother", "can_drive": True},
        {"name": "Liam", "role": "kid", "age": 5, "can_drive": False},
        {"name": "Zoe", "role": "kid", "age": 14, "can_drive": False},
    ],
    "rules": [
        "Kids never travel alone to school or activities.",
        "Adults drive themselves unless stated otherwise.",
        "The family only has one car available.",
        "If an adult has already taken the car the other one takes public transport.",
    ],
    "schedule": [
        {
            "person": "Liam",
            "travel_method" : "None",
            # "event": "School drop-off",
            # "start_time": "07:50",
            # "location": "Elm Street School",
            "needs_driver": True,
        },
        {
            "person": "Zoe",
            "travel_method" : "walk",
            # "event": "Dance practice",
            # "start_time": "17:00",
            # "location": "Studio A",
            "needs_driver": True,
        },
        {
            "person": "Alex",
            "travel_method" : "None",
            # "event": "Work commute",
            # "start_time": "08:30",
            # "location": "Downtown",
            "needs_driver": False,
        },
         {
            "person": "Sam",
            "travel_method" : "public transport",
            # "event": "Work commute",
            # "start_time": "08:30",
            # "location": "Downtown",
            "needs_driver": False,
        },
    ],
}


@dataclass
class FamilyState:
    members: list[dict] = field(default_factory=list)
    schedule: list[dict] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        return (
            f"Members: {self.members}\n"
            f"Rules: {self.rules}\n"
            f"Schedule: {self.schedule}\n"
        )


@lru_cache(maxsize=1)
def _family_state() -> FamilyState:
    return FamilyState(
        members=list(DEFAULT_FAMILY["members"]),
        schedule=list(DEFAULT_FAMILY["schedule"]),
        rules=list(DEFAULT_FAMILY["rules"]),
    )


def _ensure_defaults(st: FamilyState):
    """If the in-memory state was cleared, repopulate defaults."""
    if not st.members:
        st.members = list(DEFAULT_FAMILY["members"])
    if not st.schedule:
        st.schedule = list(DEFAULT_FAMILY["schedule"])
    if not st.rules:
        st.rules = list(DEFAULT_FAMILY["rules"])


def update_family_state(payload: dict | None) -> FamilyState:
    """Merge user-supplied family data into the cached state."""
    st = _family_state()
    _ensure_defaults(st)
    if not payload:
        return st

    if members := payload.get("members"):
        st.members = members
    if schedule := payload.get("schedule"):
        st.schedule = schedule
    if rules := payload.get("rules"):
        st.rules = rules
    return st


def delete_family_event(criteria: dict | None) -> dict:
    """Remove events from the in-memory schedule matching provided fields."""
    st = _family_state()
    if not criteria:
        return {"deleted": 0, "schedule": st.schedule}

    allowed = {"person", "event", "start_time", "location", "needs_driver"}
    filtered = {k: v for k, v in (criteria or {}).items() if k in allowed}

    def _matches(ev: dict) -> bool:
        return all(ev.get(k) == v for k, v in filtered.items())

    before = len(st.schedule)
    st.schedule = [ev for ev in st.schedule if not _matches(ev)]
    deleted = before - len(st.schedule)
    return {"deleted": deleted, "schedule": st.schedule}


def get_family_state() -> dict:
    """Expose current family state snapshot."""
    st = _family_state()
    _ensure_defaults(st)
    return {"members": st.members, "schedule": st.schedule, "rules": st.rules}


def _build_llm(api_key: str | None):
    if ChatOpenAI is None or not api_key:
        return None
    try:
        return ChatOpenAI(
            api_key=api_key,  # langchain-openai
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=220,
        )
    except TypeError:
        # Older versions use different parameter names
        try:
            return ChatOpenAI(
                openai_api_key=api_key,
                model_name="gpt-4o-mini",
                temperature=0.2,
                max_tokens=220,
            )
        except Exception:
            return None
    except Exception:
        return None


def _call_llm(llm: Any, messages: Iterable[Any], fallback: str) -> str:
    if llm is None or HumanMessage is None or SystemMessage is None:
        return fallback
    try:
        res = llm.invoke(list(messages))
        content = getattr(res, "content", None)
        if content:
            return str(content).strip()
    except Exception:
        return fallback
    return fallback


def _memory_agent(llm: Any, family: FamilyState) -> str:
    """Summarize the roster/schedule/rules so downstream agents can rely on it."""
    fallback = (
        "Family roster: "
        + ", ".join([m.get("name", "?") for m in family.members])
        + f". Rules: {family.rules}."
    )
    return _call_llm(
        llm,
        [
            SystemMessage(
                content=(
                    "You memorize household logistics. Keep names, rules, and schedule "
                    "crisply summarized so a second agent can make transport decision."
                )
            ),
            HumanMessage(content=family.to_prompt_block()),
        ],
        fallback=fallback,
    )


def _packing_list_from_weather(weather: dict | None) -> list[str]:
    """Suggest simple packing items based on weather conditions."""
    if not weather:
        return []
    current = weather.get("current")
    if not current:
        return []
    items: list[str] = []
    condition = str(current.get("condition", "")).lower()
    temp_c = current.get("tempC")
    if "rain" in condition or "drizzle" in condition or "showers" in condition:
        items.append("umbrella")
    if isinstance(temp_c, (int, float)) and temp_c < 5:
        items.append("gloves")
    if isinstance(temp_c, (int, float)) and temp_c > 25:
        items.append("water bottle")
    return sorted(list({*items}))


def _fallback_assignment(event: dict, family: FamilyState, weather: dict | None) -> dict:
    """Rule-based suggestion if LLM is unavailable."""
    person = event.get("person") or "Unknown"
    role = next((m.get("role", "") for m in family.members if m.get("name") == person), "")
    pref_mode = event.get("travel_method")
    driver = person if role == "adult" or role == "parent" else None
    if driver is None:
        driver = next((m.get("name") for m in family.members if m.get("can_drive")), "an adult")
    if pref_mode:
        mode = pref_mode
    else:
        # Prefer walk/transit by default, use car only if driver is available AND event needs driver
        if event.get("needs_driver"):
            mode = "car" if driver not in (None, "bus") else "bus"
        else:
            mode = "walk/transit"

    if not event.get("needs_driver") and (role == "adult" or role == "parent"):
        driver = person
        mode = pref_mode or "car"
    packing = _packing_list_from_weather(weather)
    notes = "Kids never travel alone; an adult drives if required." if role != "adult" else "Adult handles their own trip."
    return {
        "person": person,
        "driver": driver,
        "mode": mode,
        "preferred_mode": pref_mode,
        "items": packing,
        "notes": notes,
        "event": event.get("event") or event.get("title") or "",
        "location": event.get("location"),
        "start_time": event.get("start_time"),
    }


def _fallback_modes_by_member(family: FamilyState) -> dict:
    """Very small rule-based heuristic for who uses which mode when sharing one car."""
    modes: dict[str, str] = {}
    car_available = True
    for ev in family.schedule:
        person = ev.get("person") or "Unknown"
        member = next((m for m in family.members if m.get("name") == person), {})
        can_drive = bool(member.get("can_drive"))
        needs_driver = ev.get("needs_driver", False)
        pref = ev.get("travel_method")
        if pref:
            modes[person] = pref
            if pref == "car" and car_available:
                car_available = False
            continue

        # Prefer walk/transit unless we really need the car
        if needs_driver and car_available and (can_drive or any(m.get("can_drive") for m in family.members)):
            modes[person] = "car"
            car_available = False
        elif needs_driver:
            modes[person] = "carpool/public"
        else:
            modes[person] = "walk/transit"

    return modes


def _mode_assignment_agent(
    llm: Any,
    family: FamilyState,
    memory_summary: str,
) -> dict:
    fallback = _fallback_modes_by_member(family)
    if llm is None or HumanMessage is None or SystemMessage is None:
        return {"assignments": fallback, "llm_summary": "LLM disabled; rule-based assignment applied."}

    prompt = (
        "You are the travel mode assignment agent. You should NOT always choose the car by default.\n"
        "Goal: balance between car, bus, and walking, and reduce car usage when safe and practical.\n\n"
        "Rules:\n"
        "- Only one car is available at a time.\n"
        "- Kids never travel alone.\n"
        "- Adults can drive themselves.\n"
        "- If one adult already uses the car at a given time, others should use bus or walk.\n"
        "- If an event has a preferred travel_method, try to respect it, but override it if it breaks safety rules "
        "  or the single-car constraint.\n"
        "- Prefer walking or bus when the weather is acceptable and timing is not extremely tight.\n"
        "- Choose 'car' ONLY when there is a clear reason: long distance, very bad weather, heavy items to carry, "
        "  or a time-critical event.\n\n"
        "For each family member, assign ONE main travel mode (car, bus, or walk) and give a SHORT reason. "
        "Never say that the car is always the best mode of transport.\n"
    )
    msg = _call_llm(
        llm,
        [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Family memory: {memory_summary}\nReturn modes per member in one short paragraph."),
        ],
        fallback="; ".join([f"{k}: {v}" for k, v in fallback.items()]),
    )
    return {"assignments": fallback, "llm_summary": msg}


def _logistics_agent(
    llm: Any,
    event: dict,
    family: FamilyState,
    memory_summary: str,
    weather: dict | None,
    eta: dict | None,
    prefs: dict | None,
) -> dict:
    fallback = _fallback_assignment(event, family, weather)
    if llm is None or HumanMessage is None or SystemMessage is None:
        return fallback

    packing_list = _packing_list_from_weather(weather)
    cond = ((weather or {}).get("current") or {}).get("condition", "")
    temp_c = ((weather or {}).get("current") or {}).get("tempC")
    eta_min = (eta or {}).get("etaMinutes")
    preferred_mode = event.get("travel_method")

    msg = _call_llm(
        llm,
        [
            SystemMessage(
                content=(
        "Tu es l'agent logistique pour une famille.\n"
        "Pour chaque événement, tu dois décider qui accompagne qui, quel mode de transport utiliser "
        "(voiture, bus, marche), quand partir, et quoi emporter.\n\n"
        "Objectifs :\n"
        "Indique toujours quel adulte accompagne quel enfant.\n"
        "- Réduire l'utilisation de la voiture quand c'est raisonnable.\n"
        "- Toujours considérer AU MOINS deux modes de transport (par ex. voiture vs bus, ou bus vs marche).\n"
        "- La voiture ne doit PAS être le choix par défaut.\n\n"
        "Contraintes :\n"
        "- Il n'y a qu'une seule voiture disponible à la fois.\n"
        "- Les enfants ne voyagent jamais seuls : un adulte doit les accompagner.\n"
        "l'adulte peut emmener l'enfant en dans le bus ou marcher avec lui\n"
        "- Les adultes gèrent leur propre trajet.\n"
        "- Si un adulte utilise déjà la voiture à cette heure, propose bus ou marche pour les autres.\n"
        "- Le champ 'travel_method' d'un événement est une PRÉFÉRENCE FAIBLE :\n"
        "  tu peux le modifier si le bus ou la marche sont raisonnables, même si les règles ne sont "
        "  pas strictement violées.\n"
        "pour les distances courtes prefere marcher ou bus\n"
        "- Si tu recommandes la voiture, tu dois EXPLIQUER pourquoi bus/marche ne conviennent pas (météo, "
        "  distance, heure limite, objets à transporter, sécurité...).\n\n"
        "Style de réponse :\n"
        "- Réponds en français.\n"
        "- Ne pas utiliser des phrases génériques comme 'je vous recommande de conduire' ou "
        "  'le meilleur mode de transport est la voiture'.\n"
        "- Au lieu de ça, compare explicitement au moins deux options, par exemple :\n"
        "  'Option 1 – marcher 15 minutes jusqu'à l'école. Option 2 – prendre le bus 5 minutes. "
        "Choix recommandé : le bus, car il pleut et il fait froid.'\n"
        "- Donne une réponse courte, pratique et concrète (1 paragraphe).\n"
        "un seul adulte peut avoir la voiture dans une journée\n"
        )            
    ),
            HumanMessage(
                content=(
                    f"Family memory: {memory_summary}\n"
                    f"Event: {event}\n"
                    f"Weather: condition={cond}, tempC={temp_c}\n"
                    f"ETA baseline (if any): {eta_min}\n"
                    f"Preferred modes: {prefs}\n"
                    f"Event preferred travel_method: {preferred_mode}\n"
                    f"Suggested packing so far: {packing_list}\n"
                    "Reply with driver/mode/leave_time/packing in one short paragraph."
                )
            ),
        ],
        fallback=f"Driver: {fallback['driver']}, mode: {fallback['mode']}, items: {fallback['items']}",
    )

    return {**fallback, "llm_advice": msg}


def orchestrate_family_plan(
    family_payload: dict | None,
    weather: dict | None,
    eta: dict | None,
    prefs: dict | None,
    api_key: str | None,
) -> dict:
    """Public entry point to run the multi-agent planner."""
    family = update_family_state(family_payload)
    llm = _build_llm(api_key)

    memory_summary = _memory_agent(llm, family)
    mode_assignment = _mode_assignment_agent(llm, family, memory_summary)
    advice = [
        _logistics_agent(llm, ev, family, memory_summary, weather, eta, prefs)
        for ev in family.schedule
    ]

    return {
        "memory": memory_summary,
        "schedule": family.schedule,
        "advice": advice,
        "modes_by_member": mode_assignment,
        "llm_enabled": llm is not None,
    }