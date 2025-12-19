
import httpx, datetime as dt, re

ICS_DT_RE = re.compile(r'DTSTART(?:;[^:]+)?:([0-9TZ]+)')

async def fetch_ics(url: str) -> str:
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.text

def next_event_from_ics(ics_text: str):
    events = []
    current = None
    for line in ics_text.splitlines():
        line = line.strip()
        if line == "BEGIN:VEVENT":
            current = {}
        elif line == "END:VEVENT":
            if current: events.append(current)
            current = None
        elif current is not None:
            if line.startswith("SUMMARY:"):
                current["summary"] = line[len("SUMMARY:"):]
            elif line.startswith("LOCATION:"):
                current["location"] = line[len("LOCATION:"):]
            elif line.startswith("DTSTART"):
                m = ICS_DT_RE.match(line)
                if m:
                    v = m.group(1)
                    if "T" in v:
                        current["start"] = v
                    else:
                        current["start"] = v + "T000000Z"
    now = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    def key(e): return e.get("start","99999999T000000Z")
    future = [e for e in events if e.get("start","") >= now]
    future.sort(key=key)
    return future[0] if future else None
