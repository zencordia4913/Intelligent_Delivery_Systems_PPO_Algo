from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
import googlemaps
import requests
from dotenv import load_dotenv
from datetime import datetime
import datetime as dt    
import re
import math

# Load .env variables (including API key)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
INFERENCE_URL = os.getenv("FASTAPI_INFER_URL", "http://localhost:8001/infer")
MAX_ELEMENTS = 625

gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

ROUTES_ENDPOINT = (
    "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
)
HEADERS_TEMPLATE = {
    "Content-Type": "application/json",
    "X-Goog-Api-Key": GOOGLE_API_KEY,
    # only return the fields we use → cheaper & faster
    "X-Goog-FieldMask": (
        "originIndex,destinationIndex,distanceMeters,duration"
    ),
}
_duration_re = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")

def _rfc3339_utc_plus(seconds_ahead: int = 60) -> str:
    """
    RFC-3339 timestamp, e.g. '2025-05-15T09:41:03Z'
    """
    return (
        dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=seconds_ahead)
    ).isoformat(timespec="seconds").replace("+00:00", "Z")

def _parse_duration(iso: str) -> float:
    """Convert ISO-8601 duration (e.g. 'PT1H2M3S', 'PT123.4S') → seconds."""
    if iso.endswith("s"):                 # legacy `"123.4s"` form (rare)
        return float(iso[:-1])
    m = _duration_re.fullmatch(iso)
    if not m:
        raise ValueError(f"Unrecognised duration format: {iso}")
    h = float(m.group(1) or 0)
    m_ = float(m.group(2) or 0)
    s = float(m.group(3) or 0)
    return h * 3600 + m_ * 60 + s


def _distmat(latlons: list[tuple[float, float]]):
    """
    Query **Routes API  v2  computeRouteMatrix** and return
    (distance_km_matrix, duration_sec_matrix) as nested lists.
    """
    n = len(latlons)
    km   = [[0.0] * n for _ in range(n)]
    secs = [[0.0] * n for _ in range(n)]

    # ---------- build waypoint json blobs once ----------
    wp = [
        {
            "waypoint": {
                "location": {
                    "latLng": {"latitude": lat, "longitude": lon}
                }
            }
        }
        for lat, lon in latlons
    ]

    # ---------- chunk so that |origins|×|dest| ≤ 625 ----------
    step = max(1, math.floor(MAX_ELEMENTS / n))  # dest chunk size
    for o0 in range(0, n, step):
        origins = wp[o0 : o0 + step]

        # chunk destinations for each origin chunk
        d_step = max(1, math.floor(MAX_ELEMENTS / len(origins)))
        for d0 in range(0, n, d_step):
            destinations = wp[d0 : d0 + d_step]

            body = {
                "origins": origins,
                "destinations": destinations,
                "travelMode": "DRIVE",
                "routingPreference": "TRAFFIC_AWARE",
                "departureTime": _rfc3339_utc_plus(60),   # ⬅ 60-sec buffer
                "units": "METRIC"
            }

            resp = requests.post(
                ROUTES_ENDPOINT,
                headers=HEADERS_TEMPLATE,
                json=body,                     # <-- correct way
                stream=True,
                timeout=(10, 90),
            )
            if not resp.ok:
                raise RuntimeError(
                    f"Routes API Error {resp.status_code}: {resp.text}"
                )

            entries = resp.json()  # the whole thing is a JSON array
            for obj in entries:
                r = o0 + obj["originIndex"]
                c = d0 + obj["destinationIndex"]
                if "distanceMeters" not in obj:
                    print(f"Missing distanceMeters for origin {r}, dest {c}")
                dist_m = obj.get("distanceMeters", 0)
                dur_s  = obj.get("duration", "PT0S")
                km[r][c]   = dist_m / 1000.0
                secs[r][c] = _parse_duration(dur_s)

    return km, secs


def parse_time_str_to_sec(time_str):
    """Convert HH:MM string to seconds since midnight."""
    try:
        dt = datetime.strptime(time_str.strip(), "%H:%M")
        return dt.hour * 3600 + dt.minute * 60
    except ValueError:
        return None

def to_sec(hhmm: str):
    return datetime.strptime(hhmm.strip(), "%H:%M").hour * 3600 + \
           datetime.strptime(hhmm.strip(), "%H:%M").minute * 60


@csrf_exempt
def convert_addresses(request):
    if request.method != 'POST':
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body)
        

        depot_address = data["depot"]
        truck_capacity = data.get("capacity_limit", 1000)
        departure_time_str = data.get("departure_time", "08:00")  # default: 8:00 AM
        depot_end_time_str = data.get("depot_end_time")  # may be None

        # Parse customer inputs
        customer_info = data["customers"]
        customer_addresses = [c["address"] for c in customer_info]
        demands = [c["demand"] for c in customer_info]

        # Geocode all addresses
        location_dict = {}
        addresses = [depot_address] + customer_addresses
        for address in addresses:
            try:
                result = gmaps.geocode(address)
                if result:
                    loc = result[0]["geometry"]["location"]
                    location_dict[address] = (loc["lat"], loc["lng"])
                else:
                    location_dict[address] = (None, None)
            except Exception:
                location_dict[address] = (None, None)

        depot_coords = location_dict[depot_address]
        customer_coords = [location_dict[addr] for addr in customer_addresses]

        if None in depot_coords or any(None in c for c in customer_coords):
            return JsonResponse({"error": "One or more addresses could not be geocoded."}, status=400)
        
        # Parse times
        departure_sec = parse_time_str_to_sec(departure_time_str)
        if departure_sec is None:
            return JsonResponse({"error": "Invalid departure_time format. Use HH:MM."}, status=400)

        depot_end_time_str = data.get("depot_end_time")
        if depot_end_time_str:
            depot_end_sec = parse_time_str_to_sec(depot_end_time_str)
            if depot_end_sec is None:
                return JsonResponse({"error": "Invalid depot_end_time format. Use HH:MM."}, status=400)
        else:
            depot_end_sec = max(parse_time_str_to_sec(c["end_time"]) for c in customer_info) + 3600  # +1 h buffer

        if depot_end_sec <= departure_sec:
            return JsonResponse({"error": "depot_end_time must be after departure_time"}, status=400)

        depot_delta_sec = depot_end_sec - departure_sec          # <-- define this
        SCALE_MAX = 1000
        scale_factor = SCALE_MAX / depot_delta_sec
        
        start_secs = []
        end_secs   = []     
        for c in customer_info:
            s, e = to_sec(c["start_time"]), to_sec(c["end_time"])
            if s is None or e is None or e <= s:
                return JsonResponse({"error": "Bad start/end times."}, status=400)
            if e > depot_end_sec:
                return JsonResponse({"error": f"Customer end_time {c['end_time']} exceeds depot_end_time"}, status=400)
            start_secs.append(s); end_secs.append(e)

        departure_sec = to_sec(departure_time_str)
        depot_end_sec = to_sec(depot_end_time_str) if depot_end_time_str \
                    else max(end_secs) + 3600
        # scale = 1000 / (depot_end_sec - departure_sec)
        # start_scaled = [int(round((s - departure_sec) * scale)) for s in start_secs]
        # end_scaled   = [int(round((e - departure_sec) * scale)) for e in end_secs]
        start_secs = [(s - departure_sec) for s in start_secs]   # seconds after truck departure
        end_secs   = [(e - departure_sec) for e in end_secs]

        service_times_min = [c.get("service_time", 10) for c in customer_info]
        service_times_sec = [int(s * 60) for s in service_times_min]

        latlons = [depot_coords] + customer_coords          # [(lat,lon), …]
        dist_km , dur_sec = _distmat(latlons)
        print(f"dist_km: {dist_km}")
        print(f"dur_sec: {dur_sec}")

        # Assemble JSON for FastAPI
        payload = {
            "dist_matrix":  dist_km,              # nested lists = JSON serialisable
            "time_matrix":  dur_sec,
            "depot": { "x": depot_coords[1], "y": depot_coords[0],
                    "start_time": departure_sec,               # depot always open
                    "end_time":   depot_end_sec },
            "customers": [
                {
                    "id": i+1,
                    "x": coord[1], "y": coord[0],
                    "demand": demands[i],
                    "start_time": start_secs[i],   # ← real seconds (0 = dispatch time)
                    "end_time":   end_secs[i],     # ← real seconds
                    "service_time": service_times_sec[i]
                } for i, coord in enumerate(customer_coords)
            ],
            "capacity_limit": truck_capacity,
            "num_traj": 50
        }

        # Send to FastAPI
        response = requests.post(INFERENCE_URL, json=payload, timeout=180)
        if response.status_code != 200:
            return JsonResponse({"error": f"Inference server error: {response.text}"}, status=500)

        inference_result = response.json()
        return JsonResponse(inference_result)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)