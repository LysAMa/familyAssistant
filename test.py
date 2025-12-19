import asyncio

origin = {"lat": 45.508888, "lon": -73.561668}
dest   = {"lat": 45.501689, "lon": -73.567256}

from app.services.eta import fetch_eta   # adapte l'import à ta structure

async def test():
    res = await fetch_eta(origin, dest, "driving")
    print("JSON result:", res)

asyncio.run(test())
