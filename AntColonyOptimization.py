import numpy as np
import random
import matplotlib.pyplot as plt
#Ant Colony Optimization (ACO) algorithm code for solving the Traveling Salesman Problem (TSP) using a predefined set of cities with coordinates. The algorithm iteratively constructs tours for each ant, calculates their lengths, updates pheromones, and seeks the shortest tour across multiple iterations.
# Define city coordinates
cities = np.array([
    [0, 0], [1, 3], [4, 3], [6, 1], [6, 6], [3, 7], [2, 2], [5, 5]
])
num_cities = len(cities)

# ACO Parameters
num_ants = 50
num_iterations = 100
alpha = 1.0
beta = 2.0
rho = 0.1
Q = 100

# Distance Matrix
def euclidean_distance(city1, city2):
    return np.linalg.norm(city1 - city2)

dist_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(i + 1, num_cities):
        dist = euclidean_distance(cities[i], cities[j])
        dist_matrix[i][j] = dist_matrix[j][i] = dist

# Pheromone Matrix Initialization
pheromone_matrix = np.ones((num_cities, num_cities)) / num_cities

# Construct Solution
def construct_solution():
    start_city = random.randint(0, num_cities - 1)
    visited = [False] * num_cities
    visited[start_city] = True
    tour = [start_city]
    current_city = start_city
    for _ in range(num_cities - 1):
        probabilities = []
        for next_city in range(num_cities):
            if not visited[next_city]:
                pheromone = pheromone_matrix[current_city][next_city] ** alpha
                heuristic = (1.0 / dist_matrix[current_city][next_city]) ** beta
                probabilities.append(pheromone * heuristic)
            else:
                probabilities.append(0)
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        next_city = random.choices(range(num_cities), probabilities)[0]
        visited[next_city] = True
        tour.append(next_city)
        current_city = next_city
    tour.append(tour[0])
    return tour

# Pheromone Update
def update_pheromones(ant_solutions, ant_lengths):
    global pheromone_matrix
    pheromone_matrix *= (1 - rho)
    for solution, length in zip(ant_solutions, ant_lengths):
        pheromone_deposit = Q / length
        for i in range(len(solution) - 1):
            pheromone_matrix[solution[i]][solution[i + 1]] += pheromone_deposit

# ACO Main Loop
def aco_tsp():
    best_tour = None
    best_length = float('inf')
    for iteration in range(num_iterations):
        ant_solutions = []
        ant_lengths = []
        for _ in range(num_ants):
            tour = construct_solution()
            tour_length = sum(dist_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
            ant_solutions.append(tour)
            ant_lengths.append(tour_length)
            if tour_length < best_length:
                best_tour = tour
                best_length = tour_length
        update_pheromones(ant_solutions, ant_lengths)
        print(f"Iteration {iteration + 1}/{num_iterations} | Best Length: {best_length:.2f}")
    return best_tour, best_length

# Run ACO
best_tour, best_length = aco_tsp()
print("\nBest tour found:", best_tour)
print("Length of best tour:", best_length)

# Plot Best Tour
tour_coordinates = cities[best_tour]
plt.figure(figsize=(8, 8))
plt.plot(tour_coordinates[:, 0], tour_coordinates[:, 1], marker='o', linestyle='-', color='b')
plt.scatter(cities[:, 0], cities[:, 1], color='r')
for i, city in enumerate(cities):
    plt.text(city[0] + 0.1, city[1] + 0.1, f'City {i}', fontsize=12)
plt.title('Best Tour Found by ACO')
plt.show()
