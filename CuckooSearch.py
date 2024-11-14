import numpy as np
import random
import math

# Cuckoo Search Parameters
nests = 20  # Number of nests (solutions)
max_iter = 100  # Maximum number of iterations
pa = 0.25  # Probability of abandoning a nest (replacing the worst solution)
alpha = 0.01  # Step size for the Levy flight
beta = 1.5  # Levy flight exponent
dim = 6  # For example, 3 warehouse locations and 3 inventory levels

def fitness_function(solution):
    warehouse_locations = solution[:dim // 2]  # Warehouse locations (binary)
    inventory_levels = solution[dim // 2:]  # Inventory levels (continuous)

    # Apply a penalty if there are no active warehouses or inventory is too low
    if np.sum(warehouse_locations) == 0:
        penalty = 1000  # Apply a large penalty if no warehouse is selected
    else:
        penalty = 0

    # Calculate costs
    fixed_warehouse_cost = np.sum(warehouse_locations) * 5000  # $5000 per warehouse
    transportation_cost = np.sum(warehouse_locations) * 100  # $100 per active warehouse
    inventory_cost = np.sum(inventory_levels) * 2  # $2 per unit of inventory

    total_cost = fixed_warehouse_cost + transportation_cost + inventory_cost + penalty
    return total_cost

def levy_flight(alpha, beta):
    """Levy flight function that returns a random step size based on a Levy distribution."""
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
    u = np.random.normal(0, sigma, 1)
    v = np.random.normal(0, 1, 1)
    step = u / abs(v)**(1 / beta)
    return alpha * step

def cuckoo_search(nests, max_iter, dim, fitness_function):
    # Step 1: Initialize nests (solutions)
    nests_position = np.zeros((nests, dim))

    # Initialize: First half binary (for warehouse locations), rest continuous (for inventory)
    nests_position[:, :dim//2] = np.random.randint(1, 2, size=(nests, dim//2))  # Warehouse locations
    nests_position[:, dim//2:] = np.random.rand(nests, dim - dim//2)  # Inventory levels

    fitness_values = np.array([fitness_function(nest) for nest in nests_position])

    # Step 2: Keep track of the best solution found
    best_nest = nests_position[np.argmin(fitness_values)]
    best_fitness = np.min(fitness_values)

    # Step 3: Iterate and improve solutions
    for iteration in range(max_iter):
        new_nests_position = np.copy(nests_position)

        for i in range(nests):
            # Generate a new solution by Levy flight
            step = levy_flight(alpha, beta)
            new_position = nests_position[i] + step  # Apply Levy flight to move

            # Make sure the new position is within bounds
            new_position = np.clip(new_position, 0, 1)

            # Evaluate the new solution
            new_fitness = fitness_function(new_position)

            # If the new solution is better, update the nest
            if new_fitness < fitness_values[i]:
                nests_position[i] = new_position
                fitness_values[i] = new_fitness

        # Abandon worse nests (replace with new random solutions)
        if np.random.rand() < pa:
            worst_index = np.argmax(fitness_values)
            nests_position[worst_index] = np.random.rand(dim)
            fitness_values[worst_index] = fitness_function(nests_position[worst_index])

        # Update the best nest found so far
        current_best_index = np.argmin(fitness_values)
        current_best_nest = nests_position[current_best_index]
        current_best_fitness = fitness_values[current_best_index]

        if current_best_fitness < best_fitness:
            best_nest = current_best_nest
            best_fitness = current_best_fitness

        print(f"Iteration {iteration + 1}/{max_iter}, Best Fitness: {best_fitness}")

    return best_nest, best_fitness

# Run the Cuckoo Search
best_solution, best_cost = cuckoo_search(nests, max_iter, dim, fitness_function)

# Output the results
print("Best Solution (Warehouse Locations and Inventory Levels):", best_solution)
print("Best Total Cost:", best_cost)
