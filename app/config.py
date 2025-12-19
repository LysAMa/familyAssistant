
from pydantic import BaseModel
from functools import lru_cache
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    openweather_key: str | None = os.getenv("OPENWEATHER_KEY")
    ors_key: str | None = os.getenv("ORS_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    google_maps_key: str | None = os.getenv("GOOGLE_MAPS_KEY")
    stm_api_key: str | None = os.getenv("STM_API_KEY")
    stm_trip_updates_url: str | None = os.getenv("STM_TRIP_UPDATES_URL", "https://api.stm.info/pub/od/gtfs-rt/v1/tripUpdates")

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
