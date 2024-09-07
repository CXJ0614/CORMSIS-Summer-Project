import numpy as np
import random
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import time

# Start time for performance measurement
start_time = time.time()

# Set seed
np.random.seed(50)
random.seed(50)

# Define the parameters
n = 20  # The number of candidate locations of UAM station
m = 40  # The number of demand points
alpha = 0.9  # Capacity probability threshold
beta = 0.9  # Time probability threshold
sigma_C = 2  # Standard deviation of capacity
sigma_T = 5  # Standard deviation of travel time
L = 150  # The lower bound of total capacity
T = 60  # Maximum acceptable travel time
V = 1  # Speed (assuming 60 km/h)
mu_C = 35  # Mean of capacity
sigma_C = 5  # Standard deviation of capacity
mu_T = 2.65  # Mean of travel time
sigma_T = 0.3  # Standard deviation of travel time
alpha_weight = 0.35 # Distance weight
beta_weight = 0.4 # Capacity weight
lambda_ = 0.25 # Time weight

# Generate random coordinates for UAM stations and demand points
uam_locations = np.random.randint(0, 100, size=(n, 2))
demand_locations = np.random.randint(0, 100, size=(m, 2))

# Calculate distance matrix
def calculate_distance_matrix(locations):
    n = len(locations)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = np.linalg.norm(locations[i] - locations[j])
    return distance_matrix

# Calculate distance matrix between demand points and UAM stations
def calculate_demand_uam_distance(demand_locations, uam_locations):
    m = len(demand_locations)
    n = len(uam_locations)
    distance_matrix = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            distance_matrix[i, j] = np.linalg.norm(demand_locations[i] - uam_locations[j])
    return distance_matrix

# Generate random capacity and delay
def generate_random_capacity(mu_C, sigma_C, size, samples=1000):
    return [np.abs(norm.rvs(loc=mu_C, scale=sigma_C, size=samples)) for _ in range(size)]

def generate_random_delay(mu_T, sigma_T, size, samples=1000):
    return [np.random.lognormal(mean=mu_T, sigma=sigma_T, size=samples) for _ in range(size)]

# Monte Carlo simulation
def monte_carlo_simulation(selected_uam, capacities, demand_uam_distances, delays, L, T, iterations=100):
    feasible_capacity_count = 0
    feasible_time_count = 0
    for _ in range(iterations):
        total_capacity = sum([random.choice(capacities[i]) for i in selected_uam])
        if total_capacity >= L:
            feasible_capacity_count += 1

        max_time = 0
        for j in range(m):
            min_time = np.inf
            for i in selected_uam:
                travel_time = (demand_uam_distances[j, i] / V) + random.choice(delays[i])
                if travel_time < min_time:
                    min_time = travel_time
            if min_time > max_time:
                max_time = min_time
        if max_time <= T:
            feasible_time_count += 1

    capacity_feasibility = feasible_capacity_count / iterations
    time_feasibility = feasible_time_count / iterations
    return capacity_feasibility, time_feasibility

# Evaluate initial solution
def evaluate_initial_solution(solution, capacities, demand_uam_distances, delays, L, T):
    capacity_feasibility, time_feasibility = monte_carlo_simulation(solution, capacities, demand_uam_distances, delays, L, T)
    return capacity_feasibility >= alpha and time_feasibility >= beta

# Define objective function
def objective_function(selected_uam, distance_matrix):
    min_distance = np.inf  # Initialize the min_distance(m)
    for i in range(len(selected_uam)):
        for j in range(i + 1, len(selected_uam)):
            dist = distance_matrix[selected_uam[i], selected_uam[j]]
            if dist < min_distance:
                min_distance = dist
    return min_distance

# BR-FP Algorithm to generate initial solution
def forward_construction(uam_distance_matrix, demand_uam_distance_matrix, capacities, delays, L, T, alpha, beta):
    solution = []

    # Step 1: Create a sorted list of edges based on distance (in descending order)
    edges = [(i, j, uam_distance_matrix[i, j]) for i in range(n) for j in range(i + 1, n)]
    edges_sorted = sorted(edges, key=lambda x: -x[2])

    # Step 2: Randomly select an edge and add its endpoints to the solution
    initial_edge = random.choice(edges_sorted[:int(len(edges_sorted) * 0.1)])  # Bias towards longer edges
    solution.extend([initial_edge[0], initial_edge[1]])

    # Initialize remaining capacity and cost
    remaining_capacity = L - capacities[initial_edge[0]][0] - capacities[initial_edge[1]][0]

    # Step 3: Iteratively add nodes until capacity and time constraints are met
    while remaining_capacity > 0 or not evaluate_initial_solution(solution, capacities, demand_uam_distance_matrix, delays, L, T):
        # Create a candidate list of nodes not in the solution
        candidate_list = [i for i in range(n) if i not in solution]

        # Evaluate each candidate based on the modified evaluation function
        def evaluate_candidate(node):
            min_distance = min([uam_distance_matrix[node, j] for j in solution])
            min_travel_time = np.min([demand_uam_distance_matrix[:, node] / V])
            travel_time_penalty = lambda_ * max(0, (min_travel_time - T))
            return alpha_weight * min_distance + beta_weight * capacities[node][0] - travel_time_penalty

        candidate_list = sorted(candidate_list, key=evaluate_candidate, reverse=True)

        # Select a candidate using biased randomization (geometric distribution)
        selected_candidate = random.choices(candidate_list, weights=[0.9 ** i for i in range(len(candidate_list))])[0]

        # Add the selected candidate to the solution
        solution.append(selected_candidate)
        remaining_capacity -= capacities[selected_candidate][0]

    return solution

# Tabu search
def tabu_search(distance_matrix, demand_uam_distances, capacities, delays, L, T, alpha, beta, initial_solution, max_iteration=20, tabu_tenure=5):
    n = len(distance_matrix)
    best_solution = initial_solution.copy()
    best_objective = objective_function(initial_solution, distance_matrix)
    tabu_list = []
    history = []

    current_solution = initial_solution.copy()
    current_objective = best_objective

    for iteration in range(max_iteration):
        neighborhood = []
        for i in range(n):
            if i not in current_solution:
                neighbor = current_solution + [i]
                if neighbor not in tabu_list:
                    neighborhood.append(neighbor)

        for i in current_solution:
            for j in range(n):
                if j not in current_solution:
                    neighbor = current_solution.copy()
                    neighbor.remove(i)
                    neighbor.append(j)
                    if neighbor not in tabu_list:
                        neighborhood.append(neighbor)

        best_neighbor = None
        best_neighbor_objective = -np.inf
        for neighbor in neighborhood:
            capacity_feasibility, time_feasibility = monte_carlo_simulation(neighbor, capacities, demand_uam_distances, delays, L, T)
            if capacity_feasibility >= alpha and time_feasibility >= beta:
                objective_value = objective_function(neighbor, distance_matrix)
                if objective_value > best_neighbor_objective:
                    best_neighbor = neighbor
                    best_neighbor_objective = objective_value

        if best_neighbor is not None:
            current_solution = best_neighbor
            current_objective = best_neighbor_objective
            tabu_list.append(current_solution)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

        if current_objective > best_objective:
            best_solution = current_solution
            best_objective = current_objective
            print(f"Iteration {iteration + 1}: New Best Solution Found: {best_solution}, Best Objective Value: {best_objective}")
            history.append((iteration + 1, best_objective))

    df_history = pd.DataFrame(history, columns=['Iteration', 'Objective'])

    return best_solution, best_objective

# Main program
if __name__ == "__main__":
    capacities = generate_random_capacity(mu_C, sigma_C, n)
    delays = generate_random_delay(mu_T, sigma_T, n)

    distance_matrix = calculate_distance_matrix(uam_locations)
    demand_uam_distances = calculate_demand_uam_distance(demand_locations, uam_locations)

    # Using BR-FP Algorithm to get initial solution
    initial_solution = forward_construction(distance_matrix, demand_uam_distances, capacities, delays, L, T, alpha, beta)

    # Using Tabu Search to optimize the initial solution
    best_solution, best_objective = tabu_search(distance_matrix, demand_uam_distances, capacities, delays, L, T, alpha, beta, initial_solution, max_iteration=20, tabu_tenure=5)

    print(f"Initial Solution: {initial_solution}")
    print(f"Best Solution: {best_solution}, Best Objective Value: {best_objective}")

    # End time for performance measurement
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

