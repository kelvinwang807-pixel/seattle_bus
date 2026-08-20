import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

API_URL = "https://api.pugetsound.onebusaway.org/api/where"
AGENCY_ID = "1"  # King County Metro; change if collecting another agency
INTERVAL_SECONDS = 30
OUTPUT_DIR = Path("data") / "oba_snapshots"


def fetch_vehicles(api_key: str) -> dict:
    query = urlencode({"key": api_key})
    url = f"{API_URL}/vehicles-for-agency/{AGENCY_ID}.json?{query}"

    with urlopen(url, timeout=30) as response:
        return json.load(response)


def save_snapshot(payload: dict) -> Path:
    collected_at = datetime.now(UTC)
    day_directory = OUTPUT_DIR / collected_at.strftime("%Y-%m-%d")
    day_directory.mkdir(parents=True, exist_ok=True)

    filename = f"vehicles-{collected_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    path = day_directory / filename
    for vehicle in payload.get("data", {}).get("list", []):
        status = vehicle.get("tripStatus")
        trip_id = vehicle.get("tripId") or status.get("activeTripId")
        vehicle_id = vehicle.get("vehicleId") or status.get("vehicleId")
        if not trip_id or not vehicle_id:
            payload.get("data", {}).get("list", []).remove(vehicle)
    snapshot = {
        "collected_at": collected_at.isoformat(),
        "source": "onebusaway vehicles-for-agency",
        "agency_id": AGENCY_ID,
        "payload": payload,
    }

    path.write_text(json.dumps(snapshot, separators=(",", ":")), encoding="utf-8")
    return path


def main() -> None:
    api_key = os.environ["ONEBUSAWAY_API_KEY"]

    while True:
        try:
            payload = fetch_vehicles(api_key)

            if payload.get("code") != 200:
                print(f"OneBusAway returned: {payload}")
            else:
                path = save_snapshot(payload)
                vehicle_count = len(payload["data"]["list"])
                print(f"Saved {vehicle_count} vehicles to {path}")

        except Exception as exc:
            print(f"Collection failed: {exc}")

        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()