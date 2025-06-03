import json, numpy as np
from pathlib import Path
from geopy.distance import geodesic 

def build_env_cfg_from_json(path, capacity_limit=40, num_traj=None, mode="deploy"):
    from math import isnan
    data = json.loads(Path(path).read_text())
    print(f"Loaded JSON data: {data}")

    depot_xy   = np.array([data["depot"]["x"], data["depot"]["y"]], dtype=np.float32)
    depot_time = float(data["depot"].get("end_time", 10_000))
    depot_start = float(data["depot"].get("start_time", 0))

    cust = data["customers"]
    assert len(cust) > 0, "JSON has no customers!"

    cust_xy  = np.array([[c["x"], c["y"]] for c in cust], dtype=np.float32)
    cust_dmd  = np.array([c["demand"] for c in cust], dtype=np.float32)
    cust_tw  = np.array([c["end_time"] for c in cust], dtype=np.float32)
    cust_st  = np.array([c["start_time"] for c in cust], dtype=np.float32)
    service  = np.array([c["service_time"] for c in cust], dtype=np.float32)

    # ---------- normalise coords to [0,1] ----------
    all_xy = np.vstack([depot_xy, cust_xy])
    min_xy, max_xy = all_xy.min(0), all_xy.max(0)
    span = np.maximum(max_xy - min_xy, 1e-9)          # avoid /0
    depot_xy_norm = (depot_xy - min_xy) / span
    cust_xy_norm  = (cust_xy  - min_xy) / span

    # ---------- scale demand to [0,1] ----------
    cust_dmd_norm = cust_dmd / capacity_limit

    cfg = dict(
        max_nodes          = len(cust_xy),
        n_traj             = num_traj or len(cust),
        customer_demands   = cust_dmd_norm,
        customer_locs      = cust_xy_norm,
        customer_end_times = cust_tw,
        customer_start_times = cust_st,
        customer_service_times = service,
        depot_loc          = depot_xy_norm,
        depot_start_time   = depot_start,
        depot_end_time     = depot_time,
        capacity_limit     = capacity_limit,
        mode               = mode,
    )

    print("Env cfg:")
    print(cfg)
    return cfg

