import httpx, datetime as dt
from ..config import get_settings


def _require_bindings():
    try:
        from google.transit import gtfs_realtime_pb2  # type: ignore
        return gtfs_realtime_pb2
    except ImportError:
        return None


def _parse_trip_updates(content: bytes, stop_id: str | None, route_id: str | None, limit: int):
    pb = _require_bindings()
    if pb is None:
        return {"error": "Missing dependency gtfs-realtime-bindings. pip install gtfs-realtime-bindings"}

    feed = pb.FeedMessage()
    feed.ParseFromString(content)

    updates = []
    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue
        tu = entity.trip_update
        if route_id and tu.trip.route_id and tu.trip.route_id != route_id:
            continue
        for stu in tu.stop_time_update:
            if stop_id and stu.stop_id and stu.stop_id != stop_id:
                continue
            arrival_ts = stu.arrival.time or stu.departure.time or None
            arrival_iso = dt.datetime.utcfromtimestamp(arrival_ts).isoformat() + "Z" if arrival_ts else None
            updates.append({
                "tripId": tu.trip.trip_id,
                "routeId": tu.trip.route_id,
                "stopId": stu.stop_id,
                "arrivalEpoch": arrival_ts,
                "arrivalIso": arrival_iso,
                "delaySeconds": stu.arrival.delay if stu.HasField("arrival") else None,
            })

    updates.sort(key=lambda x: x.get("arrivalEpoch") or 0)
    if limit and limit > 0:
        updates = updates[:limit]
    return {"provider": "stm", "updates": updates, "count": len(updates)}


async def fetch_stm_trip_updates(stop_id: str | None = None, route_id: str | None = None, limit: int = 5) -> dict:
    """Trip updates from STM GTFS-RT feed."""
    s = get_settings()
    if not s.stm_api_key:
        return {"error": "Missing STM_API_KEY"}

    url = s.stm_trip_updates_url or "https://api.stm.info/pub/od/gtfs-rt/v1/tripUpdates"
    headers = {
        "apikey": s.stm_api_key,
        "x-api-key": s.stm_api_key,  # STM docs vary; include both.
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            content = r.content
    except httpx.HTTPStatusError as exc:
        return {"error": f"STM API error ({exc.response.status_code})"}
    except httpx.RequestError as exc:
        return {"error": f"STM request failed: {exc.__class__.__name__}"}

    try:
        return _parse_trip_updates(content, stop_id, route_id, limit)
    except Exception:
        return {"error": "STM feed parsing failed"}
