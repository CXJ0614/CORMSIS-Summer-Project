import numpy as np
import random
import pulp
import pandas as pd
import time

# Start time for performance measurement
start_time = time.time()

# Set seed
np.random.seed(50)
random.seed(50)

# Define the parameters
n = 20  # The number of candidate locations of UAM stations
m = 30  # The number of demand points
L = 120  # The lower bound of total capacity
U = 100  # The upper bound of total cost of maintaining
P = 1000  # A very large constant
T = 60  # Maximum acceptable travel time from UAM to demand points
V = 1  # Speed, use V to change the distance into time
alpha = 0.1  # Distance weight
beta = 0.1  # Capacity weight
gamma = 0.2  # Cost weight
lambda_ = 0.6  # Time weight

# Generate random coordinates for UAM stations and demand points
uam_locations = np.random.randint(0, 100, size=(n, 2))
demand_locations = np.random.randint(0, 100, size=(m, 2))

# Generate other parameters
capacities = np.random.randint(20, 30, size=n)
costs = np.random.randint(15, 20, size=n)

# Calculate distance matrix between UAM stations
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

# Generate distance matrices
uam_distance_matrix = calculate_distance_matrix(uam_locations)
demand_uam_distance_matrix = calculate_demand_uam_distance(demand_locations, uam_locations)

def objective_function(selected_uam, distance_matrix):
    min_distance = np.inf
    for i in range(len(selected_uam)):
        for j in range(i + 1, len(selected_uam)):
            dist = distance_matrix[selected_uam[i], selected_uam[j]]
            if dist < min_distance:
                min_distance = dist
    return min_distance

def evaluate_solution(solution, capacities, costs, L, U, demand_uam_matrix, T, V):
    total_capacities = sum([capacities[i] for i in solution])
    total_costs = sum([costs[i] for i in solution])
    min_time = np.inf
    for i in range(len(demand_uam_matrix)):
        travel_time = min([demand_uam_matrix[i][j] / V for j in solution])
        if travel_time < min_time:
            min_time = travel_time
    return total_capacities >= L and total_costs <= U and min_time <= T

# Check the third constraint
def check_third_constraint(selected_uam, distance_matrix, P):
    min_distance = objective_function(selected_uam, distance_matrix)
    for i in selected_uam:
        for j in selected_uam:
            if i != j and min_distance > distance_matrix[i, j] + P * (1 - 1) + P * (1 - 1):
                return False
    return True

# BR-FP Algorithm to generate initial solution
def forward_construction(uam_distance_matrix, demand_uam_distance_matrix, capacities, costs, L, U, T, V):
    solution = []

    # Step 1: Create a sorted list of edges based on distance (in descending order)
    edges = [(i, j, uam_distance_matrix[i, j]) for i in range(n) for j in range(i + 1, n)]
    edges_sorted = sorted(edges, key=lambda x: -x[2])

    # Step 2: Randomly select an edge and add its endpoints to the solution
    initial_edge = random.choice(edges_sorted[:int(len(edges_sorted) * 0.1)])  # Bias towards longer edges
    solution.extend([initial_edge[0], initial_edge[1]])

    # Initialize remaining capacity and cost
    remaining_capacity = L - capacities[initial_edge[0]] - capacities[initial_edge[1]]
    total_cost = costs[initial_edge[0]] + costs[initial_edge[1]]

    # Step 3: Iteratively add nodes until capacity, time, and cost constraints are met
    while remaining_capacity > 0 or not np.all(np.min(demand_uam_distance_matrix[:, solution] / V, axis=1) <= T) or total_cost > U:
        # Create a candidate list of nodes not in the solution
        candidate_list = [i for i in range(n) if i not in solution]

        if not candidate_list:  # If there are no candidates left, return an empty solution
            return []

        # Evaluate each candidate based on the modified evaluation function
        def evaluate_candidate(node):
            min_distance = min([uam_distance_matrix[node, j] for j in solution])
            min_travel_time = np.min([demand_uam_distance_matrix[:, node] / V])
            travel_time_penalty = lambda_ * max(0, (min_travel_time - T))
            return alpha * min_distance + beta * capacities[node] + gamma * (U - costs[node]) - travel_time_penalty

        candidate_list = sorted(candidate_list, key=evaluate_candidate, reverse=True)

        # Select a candidate using biased randomization (geometric distribution)
        selected_candidate = random.choices(candidate_list, weights=[0.9 ** i for i in range(len(candidate_list))])[0]

        # Add the selected candidate to the solution
        solution.append(selected_candidate)
        remaining_capacity -= capacities[selected_candidate]
        total_cost += costs[selected_candidate]

    return solution

# Tabu search
def tabu_search(distance_matrix, demand_uam_matrix, capacities, costs, L, U, T, V, P, initial_solution=None, max_iteration=10, tabu_tenure=5):
    n = len(distance_matrix)
    best_solution = initial_solution.copy() if initial_solution else []
    best_objective = objective_function(initial_solution, distance_matrix) if initial_solution else -np.inf
    tabu_list = []
    history = []

    # Generate the initial solution using BR-F if not provided
    if initial_solution is None or not initial_solution:
        initial_solution = forward_construction(distance_matrix, demand_uam_matrix, capacities, costs, L, U, T, V)
        if not initial_solution:  # If no feasible solution is found, generate a random initial solution
            initial_solution = random.sample(range(n), random.randint(n // 5, n // 2))
            while not evaluate_solution(initial_solution, capacities, costs, L, U, demand_uam_matrix, T, V):
                initial_solution = random.sample(range(n), random.randint(n // 5, n // 2))

    current_solution = initial_solution.copy()
    current_objective = objective_function(initial_solution, distance_matrix)

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
            if evaluate_solution(neighbor, capacities, costs, L, U, demand_uam_matrix, T, V) and check_third_constraint(neighbor, distance_matrix, P):
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

        history.append((iteration + 1, current_solution, current_objective))

    df_history = pd.DataFrame(history, columns=['Iteration', 'Solution', 'Objective'])

    return best_solution, best_objective

# Linear programming part
def linear_programming(uam_distance_matrix, demand_uam_distance_matrix, capacities, costs, L, U, P, T, V):
    model = pulp.LpProblem("UAM_Location", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", [(i, j) for i in range(n) for j in range(m)], cat=pulp.LpBinary)
    min_distance = pulp.LpVariable("min_distance", lowBound=0)

    # Objective function
    model += min_distance, "Objective"

    # Constraints
    model += pulp.lpSum(capacities[i] * x[i] for i in range(n)) >= L, "Capacity"
    model += pulp.lpSum(costs[i] * x[i] for i in range(n)) <= U, "Cost"

    for i in range(n):
        for j in range(i + 1, n):
            model += min_distance <= uam_distance_matrix[i][j] + P * (1 - x[i]) + P * (1 - x[j]), f"Dist_{i}_{j}"

    for j in range(m):
        model += pulp.lpSum(y[(i, j)] for i in range(n)) == 1, f"Assign_{j}"
        for i in range(n):
            model += y[(i, j)] <= x[i], f"Open_{i}_{j}"
            model += (demand_uam_distance_matrix[j][i] / V) * y[(i, j)] <= T, f"Time_{i}_{j}"

    model.solve()

    selected_uam_stations = [i for i in range(n) if pulp.value(x[i]) == 1]
    return selected_uam_stations, pulp.value(min_distance)

# Main program
if __name__ == "__main__":
    # Using BR-FP Algorithm to get initial solution
    initial_solution = forward_construction(uam_distance_matrix, demand_uam_distance_matrix, capacities, costs, L, U, T, V)

    # Using Tabu Search to optimize the initial solution
    tabu_solution, tabu_objective = tabu_search(uam_distance_matrix, demand_uam_distance_matrix, capacities, costs, L, U, T, V, P, initial_solution, max_iteration=20, tabu_tenure=5)

    # Using LP to get solution
    lp_solution, lp_objective = linear_programming(uam_distance_matrix, demand_uam_distance_matrix, capacities, costs, L, U, P, T, V)

    print(f"Initial Solution: {initial_solution}")
    print(f"Tabu Search Solution: {tabu_solution}, Objective Value: {tabu_objective}")
    print(f"Linear Programming Solution: {lp_solution}, Objective Value: {lp_objective}")

    # End time for performance measurement
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")