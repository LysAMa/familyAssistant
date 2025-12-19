from dataclasses import dataclass, asdict
from typing import Optional
from functools import lru_cache


@dataclass
class UserPreferences:
    preferred_modes: list[str]
    weather_sensitivity: int  # 0-10
    time_flex_minutes: int    # flexibility in minutes

    def to_dict(self):
        return asdict(self)


@dataclass
class PreferenceState:
    prefs: UserPreferences
    history: list[dict]

    def to_dict(self):
        return {"preferences": self.prefs.to_dict(), "history": self.history}


@lru_cache(maxsize=1)
def _state() -> PreferenceState:
    # defaults: prefers driving/walking, medium weather sensitivity, 10m flexibility
    return PreferenceState(
        prefs=UserPreferences(
            preferred_modes=["driving", "walking"],
            weather_sensitivity=5,
            time_flex_minutes=10,
        ),
        history=[],
    )


def get_preferences() -> dict:
    return _state().prefs.to_dict()


def update_preferences(preferred_modes: Optional[list[str]] = None, weather_sensitivity: Optional[int] = None, time_flex_minutes: Optional[int] = None) -> dict:
    st = _state()
    if preferred_modes is not None:
        st.prefs.preferred_modes = preferred_modes
    if weather_sensitivity is not None:
        st.prefs.weather_sensitivity = max(0, min(10, weather_sensitivity))
    if time_flex_minutes is not None:
        st.prefs.time_flex_minutes = max(0, time_flex_minutes)
    return st.prefs.to_dict()


def record_query(query: str, chosen_mode: str | None, eta_minutes: int | None):
    st = _state()
    st.history.append({
        "query": query,
        "chosen_mode": chosen_mode,
        "eta_minutes": eta_minutes,
    })
    # keep last 20
    st.history = st.history[-20:]


def get_context() -> dict:
    st = _state()
    return st.to_dict()
