from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import List, Optional

import gym
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import matplotlib


def compute_total_distance(coords, route):
    return sum(np.linalg.norm(coords[route[i]] - coords[route[i+1]]) for i in range(len(route)-1))

def compute_violations(demand, start_times, end_times, route, coords, capacity_limit=1.0):
    total_demand = 0
    capacity_violations = 0
    time_violations = 0
    current_time = 0

    for i in range(1, len(route)):
        prev_node = route[i - 1]
        curr_node = route[i]

        # Distance = travel time (assuming 1 unit = 1 time unit)
        travel_time = np.linalg.norm(coords[prev_node] - coords[curr_node])
        current_time += travel_time

        if curr_node == 0:
            total_demand = 0
            current_time = 0  # reset time at depot
            continue

        cust_idx = curr_node - 1  # customer index (since depot is 0)
        total_demand += demand[cust_idx]

        if total_demand > capacity_limit + 1e-6:
            capacity_violations += 1

        if current_time < start_times[cust_idx]:
            current_time = start_times[cust_idx]  # wait until window opens

        if current_time > end_times[cust_idx]:
            time_violations += 1

    return capacity_violations, time_violations



def plot_multi_truck_route(coords,
                           demand,
                           start_times,
                           end_times,
                           route_with_depot,
                           depot_idx=0,
                           time_matrix=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import to_rgba
    from matplotlib.cm import get_cmap

    fig, ax = plt.subplots(figsize=(10, 10))

    # ---------------------- Route segmentation --------------------------
    segments = []
    current_segment = []

    for node in route_with_depot:
        current_segment.append(node)
        if node == depot_idx and len(current_segment) > 1:
            segments.append(current_segment)
            current_segment = [depot_idx]
    if len(current_segment) > 1:
        segments.append(current_segment)

    # ---------------------- Color mapping -------------------------------
    cmap = get_cmap('autumn')
    n_routes = len(segments)
    if n_routes == 0:
        print("No route segments to plot."); return
    elif n_routes == 1:
        route_colors = ['orange']
    else:
        route_colors = [cmap(i / (n_routes - 1)) for i in range(n_routes)]

    for i, (segment, color) in enumerate(zip(segments, route_colors)):
        xy = coords[segment]
        ax.plot(xy[:, 0], xy[:, 1], '-o', color=color, label=f'Truck {i+1}', linewidth=2, markersize=4)

    # Plot depot
    depot_coord = coords[depot_idx]
    ax.plot(depot_coord[0], depot_coord[1], marker='*', color='black', markersize=20, label='Depot')

    # ---------------------- Demand & Time Windows -----------------------
    demand = demand.flatten()
    end_times = end_times.flatten()
    full_demand = np.insert(demand, 0, 0)
    full_end_times = np.insert(end_times, 0, 0)

    x_d, y_d = coords.T

    demand_bar = np.vstack([np.zeros_like(full_demand), full_demand / 4])
    ax.errorbar(x_d, y_d, demand_bar, fmt='None', elinewidth=2, color='grey', alpha=0.6, label='Demand (d)')

    endtime_bar = np.vstack([np.zeros_like(full_end_times), full_end_times / full_end_times.max() / 4])
    ax.errorbar(x_d + 0.005, y_d, endtime_bar, fmt='None', elinewidth=2, color='black', alpha=0.6, label='End Time (tw)')

    for i, (x_pos, y_pos, d, t) in enumerate(zip(x_d, y_d, full_demand, full_end_times)):
        if i == depot_idx:
            continue
        label = f'd={d:.2f}\ntw={int(t)}'
        ax.text(
            x_pos + 0.01, y_pos + 0.01, label,
            fontsize=8, color='black', ha='left', va='bottom',
            bbox=dict(facecolor='lightgrey', edgecolor='none', boxstyle='round,pad=0.3', alpha=0.6)
        )

    # ---------------------- Stats Calculation ---------------------------
    def _total_dist():
        if time_matrix is not None:
            # Just sum the durations for a better sense of realism
            return sum(time_matrix[route_with_depot[i]][route_with_depot[i+1]]
                       for i in range(len(route_with_depot)-1))
        else:
            return compute_total_distance(coords, route_with_depot)

    def _violations():
        if time_matrix is not None:
            # Use time_matrix to simulate timestamps
            time = 0
            total_demand = 0
            cap_viol, time_viol = 0, 0
            for i in range(1, len(route_with_depot)):
                prev, curr = route_with_depot[i-1], route_with_depot[i]
                time += time_matrix[prev][curr]
                if curr == depot_idx:
                    time = 0
                    total_demand = 0
                    continue
                cust = curr - 1
                total_demand += demand[cust]
                if total_demand > 1.0 + 1e-6:
                    cap_viol += 1
                if time < start_times[cust]:
                    time = start_times[cust]
                if time > end_times[cust]:
                    time_viol += 1
            return cap_viol, time_viol
        else:
            return compute_violations(
                demand, start_times, end_times,
                route_with_depot, coords
            )

    total_distance = _total_dist()
    cap_viol, time_viol = _violations()

    stats_text = f"Total Travel: {total_distance:.2f} sec\nCapacity Violations: {cap_viol}\nTime Violations: {time_viol}"
    ax.text(
        0.03, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.8)
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('square')
    ax.set_title('Optimized CVRP-TW Solution using RLOR-ASAP')
    ax.legend(loc='upper right')
    ax.grid(True)
    matplotlib.use("Agg")
    plt.savefig("inference_plot.png")
    print("figure saved as inference_plot.png")



# --------------------------------------------------------------------------
# Project Paths and Imports
# --------------------------------------------------------------------------
PROJECT_ROOT = Path("/data/students/jeryl/RLOR/RLOR-ASAP")
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.append(PROJECT_ROOT.as_posix())

from wrappers.syncVectorEnvPomo import SyncVectorEnv
from wrappers.recordWrapper import RecordEpisodeStatistics
from models.attention_model_wrapper import Agent
from Helper_Function import build_env_cfg_from_json

# --------------------------------------------------------------------------
# Global Init
# --------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = PROJECT_ROOT / "runs/cvrp-v0__ppo_or__42__1744547570/ckpt/2441.pt"

agent = Agent(device=DEVICE, name="cvrp").to(DEVICE)
agent.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True))
agent.eval()

ENV_ID = "cvrp-flex-v0"
try:
    gym.spec(ENV_ID)
except gym.error.Error:
    gym.envs.register(id=ENV_ID, entry_point="envs.cvrp_vector_env:CVRPVectorEnv")

def _make_env(seed: int, cfg: dict):
    def thunk():
        env = gym.make(ENV_ID, **cfg)
        env = RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

# --------------------------------------------------------------------------
# FastAPI Schema
# --------------------------------------------------------------------------
class Customer(BaseModel):
    id: int
    x: float; y: float
    demand: float
    start_time: float   
    end_time:   float
    service_time: int = 600  # 10 minutes

class Depot(BaseModel):
    x: float
    y: float
    start_time: float = Field(0)      # depot always open at “0”
    end_time:   float = Field(10000)

class OrderRequest(BaseModel):
    depot: Depot
    customers: List[Customer]
    capacity_limit: float = 40
    num_traj: int | None = None
    dist_matrix : Optional[list[list[float]]] = None   # km
    time_matrix : Optional[list[list[float]]] = None   # sec

class RouteResponse(BaseModel):
    route: List[int]
    objective: float
    timetable: list[dict]
    coords: List[List[float]]

app = FastAPI(title="RLOR-ASAP CVRP-TW Inference Server")

# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------

def build_timetable(route,
                    coords,
                    start_ts,
                    end_ts,
                    depot_depart_sec,
                    service_times,
                    depot_wait,
                    speed_km_h=50,
                    time_matrix=None):
    """
    Return a list[dict] – one dict per leg – with realistic timestamps.
    If the previous node is the depot (0) and the next node is a
    customer k, we delay the departure by depot_wait[k-1] seconds so the
    truck arrives just when that customer window opens.
    """
    km_per_sec = speed_km_h / 3600
    cur_time   = depot_depart_sec
    print("Depot departure time:", cur_time)

    table = []

    for i in range(1, len(route)):
        frm, to = route[i - 1], route[i]
        print(f"from {frm} to {to}")
        print(f"wait at depot: {depot_wait[to]}")

        if frm == 0 and to == 0:
            continue

        # New truck → apply wait_at_depot
        if frm == 0:
            wait_at_depot = float(depot_wait[to]) if to != 0 else 0.0
            cur_time = depot_depart_sec + wait_at_depot

        # Compute travel time (prefer duration matrix)
        if time_matrix is not None:
            travel_sec = float(time_matrix[frm][to])
        else:
            dist_km = np.linalg.norm(coords[frm] - coords[to])
            travel_sec = dist_km / km_per_sec

        arrive_sec = cur_time + travel_sec

        wait_sec = max(0, start_ts[to - 1] - arrive_sec) if to != 0 else 0
        arrive_sec += wait_sec

        service_sec = service_times[to]
        depart_sec  = arrive_sec + service_sec

        table.append(dict(
            leg         = i,
            from_idx    = int(frm),
            to_idx      = int(to),
            depart      = cur_time,
            travel      = int(round(travel_sec)),
            arrive      = arrive_sec,
            wait        = int(round(wait_sec)),
            service     = int(round(service_sec)),
            depart_next = depart_sec
        ))

        cur_time = depart_sec

    return table




def _solve_instance(req_dict: dict, plot: bool = True) -> dict:
    # Save to tmp JSON file
    json_path = PROJECT_ROOT / "_tmp_request.json"
    json_path.write_text(json.dumps(req_dict))


    print("attempt building")
    env_cfg = build_env_cfg_from_json(
        json_path,
        capacity_limit=req_dict.get("capacity_limit", 40),
        num_traj=req_dict.get("num_traj") or len(req_dict["customers"]),
        mode="deploy",
    )
    print("Done building")

    env_cfg["customer_start_times"] = np.array(
        [c["start_time"] for c in req_dict["customers"]], dtype=np.float32
    )

    env_cfg["customer_service_times"] = np.array(
        [c["service_time"] for c in req_dict["customers"]], dtype=np.float32
    )

    env_cfg["dist_matrix"]  = np.array(req_dict["dist_matrix"], dtype=np.float32)
    env_cfg["time_matrix"]  = np.array(req_dict["time_matrix"], dtype=np.float32)
    print(env_cfg)


    envs = SyncVectorEnv([_make_env(i, env_cfg) for i in range(env_cfg["n_traj"])])

    obs = envs.reset()
    done = np.zeros(env_cfg["n_traj"], dtype=bool)
    trajectories: list[np.ndarray] = []

    with torch.no_grad():
        while not done.all():
            action, _ = agent(obs)
            obs, _, done, info = envs.step(action.cpu().numpy())
            trajectories.append(action.cpu().numpy())
            depot_wait_time = obs["depot_wait_time"][0]

    final_return = info[0]["episode"]["r"]
    print("FINAL RETURN:", final_return)
    best_idx = int(np.argmax(final_return))
    best_route = np.concatenate([[0], np.array(trajectories)[:, 0, best_idx]])
    print("BEST ROUTE:", best_route)
    objective = float(final_return[best_idx])
    depot_wait_time = depot_wait_time[best_idx]
    print("DEPOT WAIT TIME:")
    print(depot_wait_time)

    if plot:
        coords = np.vstack([env_cfg["depot_loc"], env_cfg["customer_locs"]])
        real_coords = np.array([[req_dict["depot"]["y"], req_dict["depot"]["x"]]] + [[c["y"], c["x"]] for c in req_dict["customers"]])
        demand = env_cfg["customer_demands"]
        start_times = env_cfg["customer_start_times"]
        end_times = env_cfg["customer_end_times"]
        plot_multi_truck_route(
            coords=coords,
            demand=demand,
            start_times=start_times,
            end_times=end_times,
            route_with_depot=best_route,
            depot_idx=0,
            time_matrix=env_cfg["time_matrix"]  # ← ✅ pass here
        )
    service_times = [c["service_time"] for c in req_dict["customers"]]

    timetable = build_timetable(
        best_route,
        coords=coords,
        start_ts=start_times,
        end_ts=end_times,
        depot_depart_sec=int(req_dict["depot"]["start_time"]),
        service_times=[0] + service_times,      
        depot_wait=depot_wait_time,
        time_matrix=env_cfg["time_matrix"]  
    )




    return {"route": best_route.tolist(), "objective": objective, "timetable" : timetable, "coords": real_coords.tolist()}


# --------------------------------------------------------------------------
# Endpoint
# --------------------------------------------------------------------------
@app.post("/infer", response_model=RouteResponse)
async def infer(request: OrderRequest):
    try:
        req_dict = request.dict(by_alias=True)
        print("TEST")
        print(f"[INFO] Received request with {len(req_dict['customers'])} customers")
        return _solve_instance(req_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------------------------------
# Debug Entry
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_server:app", host="0.0.0.0", port=8001, reload=True)
