# Hyperparameter Optimization of SVM using Particle Swarm Optimization (PSO) on Iris Dataset

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Objective function to minimize (negative accuracy)
def objective_function(hyperparameters):
    C, gamma = hyperparameters
    model = SVC(C=C, gamma=gamma)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return -accuracy

# Load Iris dataset and split it
data = load_iris()
X = data.data
y = data.target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize PSO parameters
num_particles = 30
num_iterations = 100
inertia_weight = 0.5
cognitive_coeff = 1.5
social_coeff = 1.5

# Initialize particles' positions and velocities
positions = np.random.uniform(0.1, 100, (num_particles, 2))
velocities = np.random.uniform(-1, 1, (num_particles, 2))
personal_best_positions = positions.copy()
personal_best_scores = np.array([objective_function(p) for p in positions])
global_best_position = personal_best_positions[np.argmin(personal_best_scores)]

# Ensure valid hyperparameter ranges
def enforce_valid_parameters(position):
    C, gamma = position
    C = max(0.1, C)
    gamma = max(0.001, min(gamma, 10))
    return np.array([C, gamma])

# Update personal bests and global best
def evaluate_fitness():
    global global_best_position
    current_scores = np.array([objective_function(p) for p in positions])
    for i in range(num_particles):
        if current_scores[i] < personal_best_scores[i]:
            personal_best_scores[i] = current_scores[i]
            personal_best_positions[i] = positions[i]
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]

# Update velocities and positions of particles
def update_particles():
    global positions, velocities
    for i in range(num_particles):
        r1, r2 = np.random.random(), np.random.random()
        cognitive_velocity = cognitive_coeff * r1 * (personal_best_positions[i] - positions[i])
        social_velocity = social_coeff * r2 * (global_best_position - positions[i])
        velocities[i] = inertia_weight * velocities[i] + cognitive_velocity + social_velocity
        positions[i] += velocities[i]
        positions[i] = enforce_valid_parameters(positions[i])

# Run PSO optimization loop
for _ in range(num_iterations):
    evaluate_fitness()
    update_particles()

# Output the best hyperparameters and train the SVM model
best_C, best_gamma = global_best_position
print(f"Best C: {best_C}")
print(f"Best Gamma: {best_gamma}")

best_model = SVC(C=best_C, gamma=best_gamma)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"Validation Accuracy: {accuracy}")
