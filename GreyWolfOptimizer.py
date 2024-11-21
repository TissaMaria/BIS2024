import numpy as np

# Define the Rastrigin function (objective function to minimize)
def rastrigin_function(x):
    A = 10
    return A * len(x) + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)

# Grey Wolf Optimizer (GWO) Algorithm
class GreyWolfOptimizer:
    def __init__(self, function, num_wolves, num_iterations, dim, lb, ub):
        self.function = function  # The objective function to optimize
        self.num_wolves = num_wolves  # Number of wolves
        self.num_iterations = num_iterations  # Number of iterations
        self.dim = dim  # Number of dimensions (problem variables)
        self.lb = lb  # Lower bound for the search space
        self.ub = ub  # Upper bound for the search space

        # Initialize positions and velocities of the wolves
        self.positions = np.random.uniform(self.lb, self.ub, (self.num_wolves, self.dim))
        self.velocities = np.zeros_like(self.positions)

        # Initialize the best positions of alpha, beta, and delta wolves
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = float('inf')
        self.beta_pos = np.zeros(self.dim)
        self.beta_score = float('inf')
        self.delta_pos = np.zeros(self.dim)
        self.delta_score = float('inf')

    def update_positions(self, A, C, wolf, best_pos):
        # Update positions based on the equations
        new_position = wolf - A * (best_pos - wolf) + C * (self.alpha_pos - wolf)
        return np.clip(new_position, self.lb, self.ub)  # Ensure the position is within bounds

    def optimize(self):
        for t in range(self.num_iterations):
            for i in range(self.num_wolves):
                # Evaluate fitness of the current wolf
                fitness = self.function(self.positions[i])

                # Update alpha, beta, and delta wolves based on fitness
                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i].copy()

                elif fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = self.positions[i].copy()

                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i].copy()

            # Update the positions of all wolves based on the hierarchy (alpha, beta, delta)
            for i in range(self.num_wolves):
                # Calculate A and C
                r1, r2 = np.random.random(), np.random.random()
                A = 2 * r1 - 1
                C = 2 * r2

                # Update wolf positions based on alpha, beta, and delta wolves
                if np.random.random() > 0.5:
                    self.positions[i] = self.update_positions(A, C, self.positions[i], self.alpha_pos)
                else:
                    self.positions[i] = self.update_positions(A, C, self.positions[i], self.beta_pos)

            # Print the best solution for every 100 iterations
            if (t+1) % 100 == 0:
                print(f"Iteration {t+1}/{self.num_iterations} | Best Fitness: {self.alpha_score}")

        return self.alpha_pos, self.alpha_score  # Return the best position and its fitness

# Hyperparameters
num_wolves = 30  # Number of wolves (particles)
num_iterations = 1000  # Number of iterations
dim = 10  # Number of dimensions (problem variables)
lb = -5.12  # Lower bound of the search space
ub = 5.12  # Upper bound of the search space

# Initialize and run GWO
gwo = GreyWolfOptimizer(function=rastrigin_function, num_wolves=num_wolves, num_iterations=num_iterations, dim=dim, lb=lb, ub=ub)
best_position, best_fitness = gwo.optimize()

# Output the best result
print("\nBest Position Found: ", best_position)
print("Best Fitness (Rastrigin Value): ", best_fitness)
