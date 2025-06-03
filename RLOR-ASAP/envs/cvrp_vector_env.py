import gym
import numpy as np
from gym import spaces
import copy

from .gen_vrp_data import VRPDataset


def assign_env_config(self, kwargs):
    """
    Set self.key = value, for each key in kwargs
    """
    for key, value in kwargs.items():
        setattr(self, key, value)


def dist(loc1, loc2):
    return ((loc1[:, 0] - loc2[:, 0]) ** 2 + (loc1[:, 1] - loc2[:, 1]) ** 2) ** 0.5


class CVRPVectorEnv(gym.Env):
    def __init__(self, *args,
                 mode="train",               
                 capacity_limit=None,
                 customer_locs=None,
                 customer_demands=None,
                 customer_end_times=None,
                 customer_start_times=None,
                 customer_service_times=None,
                 depot_loc=None,
                 depot_start_time=0,
                 depot_end_time=10_000,
                 max_nodes=50,
                 n_traj=None,
                 dist_matrix=None,
                 time_matrix=None,
                 **kwargs):
        
        self.mode = mode
        
        if self.mode == "train":
            self.max_nodes = 50
            self.capacity_limit = 40
            self.n_traj = 50
            self.num_envs = 50
            self.eval_data = False
            self.eval_partition = "test"
            self.eval_data_idx = 0
            self.demand_limit = 10

        elif self.mode == "deploy":
            self.max_nodes = max_nodes
            self.capacity_limit = capacity_limit
            self.customer_service_times = customer_service_times
            self.n_traj = n_traj
            self.num_envs = 1
            self.customer_locs = customer_locs
            self.customer_demands = customer_demands
            self.customer_start_times = customer_start_times
            self.customer_end_times = customer_end_times
            self.depot_locs = depot_loc
            self.depot_end_time = depot_end_time
            self._deploy_data = dict(
                locs = customer_locs,
                demands = customer_demands,
                start_times = customer_start_times,
                end_times = customer_end_times,
                service_times = customer_service_times,
                depot = depot_loc,
                depot_st = depot_start_time,
                depot_et = depot_end_time,
            )
            self.horizon_sec = None
            self.depot_wait_time = np.zeros((self.n_traj, self.max_nodes + 1), dtype=np.float32)
            self.dist_matrix = dist_matrix
            self.time_matrix = time_matrix
        
        
        self.average_speed = 50
        self.average_norm_speed = 0.014
        self.mileage = np.zeros(self.n_traj, dtype=np.float32)
        self.time_conversion_factor = 1
        self.i = 0
        self.P = 1
        self.D = 9
        assign_env_config(self, kwargs)

        obs_dict = {"observations": spaces.Box(low=0, high=1, shape=(self.max_nodes, 2))}
        obs_dict["depot"] = spaces.Box(low=0, high=1, shape=(2,))
        obs_dict["demand"] = spaces.Box(low=0, high=1, shape=(self.max_nodes,))
        obs_dict["action_mask"] = spaces.MultiBinary(
            [self.n_traj, self.max_nodes + 1]
        )  # 1: OK, 0: cannot go
        obs_dict["last_node_idx"] = spaces.MultiDiscrete([self.max_nodes + 1] * self.n_traj)
        obs_dict["current_load"] = spaces.Box(low=0, high=1, shape=(self.n_traj,))
        obs_dict["num_vehicles"] = spaces.Box(low=0, high=self.max_nodes, shape=(self.n_traj,), dtype=np.int32)
        obs_dict["start_time"] = spaces.Box(low=0.0, high=np.inf, shape=(self.n_traj, self.max_nodes + 1), dtype=np.float32)
        obs_dict["end_time"]   = spaces.Box(low=0.0, high=np.inf, shape=(self.n_traj, self.max_nodes + 1), dtype=np.float32)
        obs_dict["depot_time"] = spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
        obs_dict["depot_wait_time"]   = spaces.Box(low=0.0, high=np.inf, shape=(self.n_traj, self.max_nodes + 1), dtype=np.float32)
        obs_dict["mileage"] = spaces.Box(low=0, high=np.inf, shape=(self.n_traj,), dtype=np.float32)  # ✅ Add mileage
        obs_dict["actual_node"] = spaces.MultiDiscrete([self.max_nodes + 1] * self.n_traj)  # ✅ Stores last visited node index
        obs_dict["urgency_factor"] = spaces.Box(low=0, high=1, shape=(self.n_traj, self.max_nodes + 1), dtype=np.float32)

        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.MultiDiscrete([self.max_nodes + 1] * self.n_traj)
        self.reward_space = None

        #Temporary
        self.done = np.array([False] * self.n_traj)
        self.actual_node = np.zeros(self.n_traj, dtype=np.int32) 
        self.urgency_factor = np.zeros((self.n_traj, self.max_nodes + 1))
        self.reset()

    def seed(self, seed):
        np.random.seed(seed)

    def _STEP(self, action):
        print("ACTION")
        print(action)
        for traj_idx in range(self.n_traj):
            self.routes[traj_idx].append(action[traj_idx]) 

        self._go_to(action)  # Go to node 'action', modify the reward
        self.num_steps += 1
        self.state = self._update_state()

        # need to revisit the first node after visited all other nodes
        self.done = (action == 0) & self.is_all_visited()

        return self.state, self.reward, self.done, self.info

    # Euclidean cost function
    def cost(self, idx_from: np.ndarray, idx_to: np.ndarray):
        if self.mode == "deploy" and self.dist_matrix is not None:
            return self.dist_matrix[idx_from, idx_to]
        return dist(self.nodes[idx_from], self.nodes[idx_to])

    def is_all_visited(self):
        # assumes no repetition in the first `max_nodes` steps
        return self.visited[:, 1:].all(axis=1)


    def _update_state(self):
        obs = {"observations": self.nodes[1:]}  # n x 2 array
        obs["depot"] = self.nodes[0]
        obs["demand"] = self.demands
        obs["last_node_idx"] = self.last
        obs["current_load"] = self.load
        obs["start_time"] = self.upd_start_time
        obs["end_time"] = self.upd_end_time 
        obs["depot_time"] = np.array([self.depot_time])  # Depot service time
        obs["mileage"] = self.mileage 
        obs["actual_node"] = self.actual_node
        obs["depot_wait_time"] = self.depot_wait_time

        # ✅ Count the number of vehicles by counting non-trailing zeros
        def count_vehicles(route):
            route = np.array(route)
            # Find the last nonzero index to ignore trailing zeros
            last_nonzero_idx = np.max(np.nonzero(route)) if np.any(route) else -1
            # Count the number of zeros **before the trailing zeros**
            return np.sum(route[: last_nonzero_idx + 1] == 0)

        obs["num_vehicles"] = np.array([count_vehicles(traj) for traj in self.routes])  # Shape: (n_traj,)
        obs["urgency_factor"] = self.urgency_factor
        obs["action_mask"] = self._update_mask()

        #print(obs)
        #if self.done.all():
            #print(obs)
            #print("🚀 Debugging Observation Shapes:")
            #for key, value in obs.items():
                #print(f"Key: {key}, Shape: {np.shape(value)}")
        

        return obs


    def _update_mask(self):
        # Only allow to visit unvisited nodes
        action_mask = ~self.visited

        # can only visit depot when last node is not depot or all visited
        action_mask[:, 0] |= self.last != 0
        action_mask[:, 0] |= self.is_all_visited()

        # not allow visit nodes with higher demand than capacity
        action_mask[:, 1:] &= self.demands <= (
            self.load.reshape(-1, 1) + 1e-5
        )  # to handle the floating point subtraction precision

        action_mask[:, 1:] &= self.upd_end_time[:, 1:] >= 0  
        action_mask[:, 1:] &= self.urgency_factor[:, 1:] <= 1  # Remove missed opportunities by removing f_u scores >1
        action_mask[:, 1:] &= self.urgency_factor[:, 1:] >= 0

        #Emergency in numerical stability
        action_mask[:, 0] |= (np.sum(action_mask[:, 1:], axis=1) == 0)
        print("FINAL ACTION MASK")
        print(action_mask[0])

        # no_valid_actions = np.sum(action_mask[:, 1:], axis=1) == 0
        # action_mask[:, 0] |= no_valid_actions & (self.last != 0)
        #print("NUM UNMASKED:", np.sum(action_mask, axis=1).tolist())
        return action_mask
    
    def _DEPLOY_RESET(self):
        self.visited = np.zeros((self.n_traj, self.max_nodes + 1), dtype=bool)
        self.visited[:, 0] = True
        self.num_steps = 0
        self.last = np.zeros(self.n_traj, dtype=int)  # idx of the last elem
        self.load = np.ones(self.n_traj, dtype=float)  # current load
        self.routes = [[0] for _ in range(self.n_traj)]

        d = self._deploy_data
        assert d["locs"] is not None, "Deploy mode but customer_locs not provided"
        assert d["demands"] is not None, "Deploy mode but customer_demands not provided"
        assert d["start_times"] is not None, "Deploy mode but customer_start_times not provided"
        assert d["end_times"] is not None, "Deploy mode but customer_end_times not provided"
        assert d["depot"] is not None, "Deploy mode but depot_loc not provided"
        for key in ("locs", "demands", "end_times", "depot"):
            assert d[key] is not None, f"Deploy mode but {key} not provided"
        self.nodes = np.vstack([d["depot"], d["locs"]]).astype(np.float32)
        self.max_nodes = self.nodes.shape[0] - 1  
        raw_demands = np.array(d["demands"], dtype=np.float32)  # shape (N,)
        if raw_demands.max() > 1.0:                            # weights in kg?
            raw_demands = raw_demands / self.capacity_limit
        self.demands = raw_demands
        self.service_times = np.insert(self._deploy_data["service_times"], 0, 0.0).astype(np.float32)
        self.demands_with_depot = self.demands.copy()

        times_end   = np.insert(d["end_times"],   0, d["depot_et"] - d["depot_st"]).astype(np.float32)
        times_start =  np.insert(d["start_times"], 0, 0).astype(np.float32)
        self.start_time    = np.tile(times_start, (self.n_traj, 1))
        self.end_time     = np.tile(times_end, (self.n_traj, 1))
        # self.start_time = np.clip(self.start_time, 0, 36_000)
        # self.end_time   = np.clip(self.end_time,   0, 36_000)
        print("START TIME CLIPPED")
        print(self.start_time)
        self.depot_time   = float(d["depot_et"])          # seconds
        self.horizon_sec  = self.depot_time               # store for logging
        self.upd_end_time   = self.end_time.copy()
        self.upd_start_time = self.start_time.copy()

        self.state = self._update_state()
        self.info = {}
        self.done = np.array([False] * self.n_traj)
        self.actual_node = np.zeros(self.n_traj, dtype=np.int32)
        self.elapsed_time = np.zeros(self.n_traj, dtype=np.float32)
        self.depot_wait_time = np.zeros((self.n_traj, self.max_nodes + 1), dtype=np.float32)
        self.urgency_factor = self.compute_urgency()

        return self.state       


    def _RESET(self):
        self.visited = np.zeros((self.n_traj, self.max_nodes + 1), dtype=bool)
        self.visited[:, 0] = True
        self.num_steps = 0
        self.last = np.zeros(self.n_traj, dtype=int)  # idx of the last elem
        self.load = np.ones(self.n_traj, dtype=float)  # current load
        self.routes = [[0] for _ in range(self.n_traj)]

        if self.eval_data:
            self._load_orders()
        else:
            self._generate_orders()
        self.state = self._update_state()
        self.info = {}
        self.done = np.array([False] * self.n_traj)
        self.actual_node = np.zeros(self.n_traj, dtype=np.int32)

        return self.state

    def _load_orders(self):
        """
        Load orders from dataset
        """

        data = VRPDataset[self.eval_partition, self.max_nodes, self.eval_data_idx, self.num_envs, 1, self.capacity_limit]
        #self.nodes = np.concatenate((data["depot"][None, ...], data["loc"]))
        self.nodes = data["loc"]
        self.demands = data["demand"]
        self.demands_with_depot = self.demands.copy()
        single_end_time = data["end_time"]  # Load service time
        self.end_time = np.tile(single_end_time[None, :], (self.n_traj, 1))
        self.depot_time = data["depot_time"]  # Load depot service time
        self.end_time[:, 0] = self.depot_time
        self.upd_end_time = self.end_time.copy()
        #print("Load Reset:", self.nodes[1:].shape)

    def _generate_orders(self):
        self.nodes = np.random.rand(self.max_nodes + 1, 2)
        self.demands = (
            np.random.randint(low=1, high=self.demand_limit, size=self.max_nodes)
            / self.capacity_limit
        )
        self.demands_with_depot = self.demands.copy()
        base_end_time = np.clip(
            np.random.normal(
                loc=5000,
                scale=2000,
                size=(self.max_nodes + 1)  # +1 for depot
            ),
            200, 10000
        )

        cust_windows = np.random.uniform(2*3600, 6*3600, size=self.max_nodes)
        depot_close  = 8 * 3600
        times        = np.insert(cust_windows, 0, depot_close).astype(np.float32)

        self.end_time     = np.tile(times, (self.n_traj, 1))
        self.depot_time   = float(depot_close)
        self.horizon_sec  = self.depot_time
        self.upd_end_time = self.end_time.copy()

    def compute_urgency(self):
            print("ACTUAL NODE")
            print(self.actual_node)
            last_node_coords = self.nodes[self.actual_node]
            # print("COORS")
            # print(last_node_coords)
            # print("NODES")
            # print(self.nodes[None, :, :])
            if self.mode == "deploy" and self.time_matrix is not None:
                anticipated_t = self.time_matrix[self.actual_node, :]
            else:
                last_node_coords = self.nodes[self.actual_node]
                dist_nodes = ((last_node_coords[:, None, :] - self.nodes[None, :, :]) ** 2).sum(-1) ** 0.5
                anticipated_t = dist_nodes / self.average_norm_speed
            from_depot   = (self.actual_node == 0)
            print("FROM DEPOT")
            print(from_depot)
            # print("ANTICIPATED")
            # print(anticipated_t)
            print("UPD START TIME")
            print(self.upd_start_time)
            print("UPD END TIME")
            print(self.upd_end_time)
            wait_full = np.maximum(self.upd_start_time - anticipated_t, 0.0)
            self.depot_wait_time[from_depot] = wait_full[from_depot]
            # print("WAIT TIME BEFORE")
            # print(wait_time)
            wait_time = wait_full.copy()
            wait_time[from_depot, :] = 0.0
            late_time = np.maximum(anticipated_t - self.upd_end_time,  0.0)
            window_len = np.clip(self.upd_end_time - self.upd_start_time, 1e-6, None)
            urgency = (wait_time + late_time) / window_len
            print("WINDOW LEN")
            print(window_len)
            print("WAIT TIME")
            print(wait_time)
            print("LATE TIME")
            print(late_time)
            print("URGENCY")
            print(urgency)
            # if self.i == 5:
            #     raise ValueError("Debugging urgency")

            return np.clip(urgency, 0.0, 10.0)  

    def _go_to(self, destination):
        destination = destination.astype(np.int32)
        print("UPD START TIME")
        print(self.upd_start_time)
        print("UPD END TIME")
        print(self.upd_end_time)
        print("Elapsed Time:", self.elapsed_time)
        traj_idx = np.arange(self.n_traj).astype(np.int32)
        print("Start Time:", self.start_time[traj_idx, destination])
        dest_node = self.nodes[destination]

        if self.mode == "deploy" and self.time_matrix is not None and self.dist_matrix is not None:
            print("test")
            last = self.last.astype(np.int32)
            dest = destination.astype(np.int32)
            dist = self.cost(dest, last)
            travel_time = self.time_matrix[last, dest]
            print("done")
        else:
            last = self.last.astype(np.int32)
            dest = destination.astype(np.int32)
            dist = self.cost(dest, last) 
            travel_time = dist / self.average_norm_speed


        self.elapsed_time += travel_time 
        wait_time = np.maximum(
            0,
            self.start_time[np.arange(self.n_traj), destination] - self.elapsed_time
        )
        self.elapsed_time += wait_time
        print("Elapsed Time:", self.elapsed_time)
        self.elapsed_time += self.service_times[destination]
        print("Service Time:", self.service_times[destination])
        print("Travel Time:", travel_time)
        print("Wait Time:", wait_time)
        print("Elapsed Time:", self.elapsed_time)
        self.mileage += dist
        self.actual_node = destination
        self.last = destination
        self.load[destination == 0] = 1
        self.load[destination > 0] -= self.demands[destination[destination > 0] - 1]
        self.demands_with_depot[destination[destination > 0] - 1] = 0
        self.visited[np.arange(self.n_traj), destination] = True
        traj_idx = np.arange(self.n_traj)  # [0, 1, 2, ..., n_traj-1]

        #print("LAST NODE")
        #print(self.last)
        # print("DIST")
        # print(dist)
        #print("BEFORE")
        #print(self.upd_end_time)
        #print("CONST")
        #print(self.end_time)
        # print("mileage")

        # Compute the time penalty per trajectory
        service_time = self.service_times[destination]
        time_penalty = self.time_conversion_factor * (
            travel_time + wait_time + service_time                  # ← travel  + wait + service
        )
        # print("SUBTRACT")
        # print(time_penalty)

        # Ensure correct row-wise subtraction
        print("TIME PENALTY")
        print(-time_penalty)
        self.upd_start_time[traj_idx, :] -= time_penalty[:, None]
        self.upd_end_time[traj_idx, :] -= time_penalty[:, np.newaxis]  # Expands time_penalty for row-wise subtraction
        returning_to_depot = np.where(destination == 0)[0]  # Get indices of returning vehicles

        #print("RETURNING")
        #print(returning_to_depot)
        if returning_to_depot.size > 0:  # Only update if there are returning vehicles
            self.upd_start_time[returning_to_depot, :] = self.start_time[returning_to_depot, :].copy()
            self.upd_end_time[returning_to_depot, :] = self.end_time[returning_to_depot, :].copy()

        return_vector = np.where(destination == 0, 1, 0)

        self.reward = - (self.D * dist) - ( self.P * (1 - (self.load / self.capacity_limit)) * return_vector)
   
        
        self.urgency_factor = self.compute_urgency()
        # plt.figure(figsize=(8,5))
        # plt.hist(self.end_time.flatten(), bins=20, edgecolor='black', alpha=0.7)
        # plt.xlabel("End Time Distribution")
        # plt.ylabel("Frequency")
        # plt.title("Distribution of Time Window Constraints")
        # plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.show()

        # print("URGENCY")
        # print(self.urgency_factor)

    def step(self, action):
        # return last state after done,
        # for the sake of PPO's abuse of ff on done observation
        # see https://github.com/opendilab/DI-engine/issues/497
        # Not needed for CleanRL
        # print("\n✅ Final Routes for All Trajectories:")
        self.i += 1
        # if self.i == 1:  # Initialize only once if needed
        #     self.trajectories = []
        #     self.loads = []
        #     self.times = []
        # self.trajectories.append(copy.deepcopy(self.routes))
        # self.loads.append(copy.deepcopy(self.load))
        # self.times.append(copy.deepcopy(self.upd_end_time))

        # if self.i == 149:
        #     non_zero_counts = [sum(1 for x in route if x != 0) for route in self.routes]
        #     first_index = next((idx for idx, count in enumerate(non_zero_counts) if count == 49), None)

        #     if first_index is not None:
        #         print(f"First index with non_zero_count of 49: {first_index}")
        #         for idx, (routes, loads, times) in enumerate(zip(self.trajectories, self.loads, self.times)):
        #             print(routes[first_index])
        #             print(loads[first_index])
        #             print(times[first_index])
        #     else:
        #         print("No route found with non_zero_count of 49.")


        #non_zero_counts = [sum(1 for x in route if x != 0) for route in self.routes]
        #print(non_zero_counts)  # Print as an array


        return self._STEP(action)

    def reset(self):
        if self.mode == "train":
            return self._RESET()
        elif self.mode == "deploy":
            return self._DEPLOY_RESET()
