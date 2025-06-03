import logging
import numpy as np

def generate_vrp_data_formatted(dataset_size, num_customers=50, num_vehicles=5, vehicle_capacity=40):
    """
    Generate VRP instances following the required format.

    Parameters:
        dataset_size (int): Number of instances to generate.
        num_customers (int): Number of customers in each instance.
        num_vehicles (int): Number of vehicles available in each instance.

    Returns:
        List[dict]: List of dictionaries containing 'loc', 'demand', 'depot', 'capacity', 'service_time', and 'depot_time'.
    """

    # Generate depot locations (one per instance)
    depot_location = np.random.uniform(size=(dataset_size, 2))

    # Generate customer locations
    customer_locations = np.random.uniform(size=(dataset_size, num_customers, 2))

    # Generate demand values uniformly from U(1,10) and normalize by vehicle capacity
    demands = np.random.randint(1, 10, size=(dataset_size, num_customers)) / vehicle_capacity

    # Create an array of vehicle capacities (Each instance has an array of `num_vehicles` filled with 40)
    capacities = np.full((dataset_size, num_vehicles), vehicle_capacity)

    # Generate service end-time constraints
    # service_end_times = np.clip(
    #         np.random.normal(
    #             loc=5000,   # Mean (center of distribution)
    #             scale=2000, # Standard deviation (spread of values)
    #             size=(dataset_size, num_customers + 1)  # +1 if depot is included
    #         ),
    #         200, 10000  # Clip values to ensure they stay within range
    #     )
    base_service_end_times = np.clip(
        np.random.normal(loc=5000, scale=2000, size=(num_customers + 1)),
        200, 10000
    )

    service_end_times = np.tile(base_service_end_times, (dataset_size, 1))

    depot_service_end_time = np.full(dataset_size, 10000)  # Fixed for depot

    # Format the dataset
    dataset = []
    for i in range(dataset_size):
        instance = {
            "loc": np.vstack([depot_location[i], customer_locations[i]]),  # Stack depot + customers
            "demand": demands[i],  # Depot has zero demand
            "depot": depot_location[i],  # Store depot separately
            "capacity": capacities[i],  # Store vehicle capacities as an array of size (num_vehicles,)
            "end_time": service_end_times[i],  # Per-customer service end times
            "depot_time": depot_service_end_time[i]  # Depot's service time
        }
        dataset.append(instance)

    return dataset


class lazyClass:
    data = {
        "test": {},
        "eval": {},
    }

    def __getitem__(self, index):
        """
        Access a VRP instance.

        Args:
            index: Tuple in format (partition, nodes, idx, dataset_size)
                   - partition (str): "test" or "eval"
                   - nodes (int): Number of customers (20, 50, 100)
                   - idx (int): Index of the instance
                   - dataset_size (int, optional): Number of instances to generate

        Returns:
            dict: A single VRP instance.
        """
        if len(index) == 5:
            partition, nodes, idx, num_vehicles, vehicle_capacity = index
            dataset_size = 1  # Default value
        elif len(index) == 6:
            partition, nodes, idx, dataset_size, num_vehicles, vehicle_capacity = index
        else:
            raise ValueError("Index must be (partition, nodes, idx, num_vehicles, vehicle_capacity) or (partition, nodes, idx, dataset_size, num_vehicles, vehicle_capacity)")

        if partition not in self.data or nodes not in self.data[partition]:
            logging.warning(
                f"Data ({partition}, {nodes}) not initialized. Generating {dataset_size} instances."
            )
            generated_data = generate_vrp_data_formatted(dataset_size, num_customers=nodes, num_vehicles=num_vehicles, vehicle_capacity=vehicle_capacity)
            self.data[partition][nodes] = [instance for instance in generated_data]

        return self.data[partition][nodes][idx]


VRPDataset = lazyClass()
