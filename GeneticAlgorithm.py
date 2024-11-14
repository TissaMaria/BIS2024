import random

class Item:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight

class Individual:
    def __init__(self, num_items):
        # Random gene initialization: 1 or 0, where 1 means the item is included
        self.genes = [random.randint(0, 1) for _ in range(num_items)]
        self.fitness = 0
        self.total_weight = 0

    def calculate_fitness(self, items, capacity):
        # Calculate total weight and value based on genes
        self.total_weight = sum(self.genes[i] * items[i].weight for i in range(len(items)))
        self.fitness = sum(self.genes[i] * items[i].value for i in range(len(items)))

        # Penalize individuals that exceed the capacity
        if self.total_weight > capacity:
            self.fitness = 0  # Assign 0 fitness if capacity is exceeded

def create_initial_population(population_size, num_items):
    return [Individual(num_items) for _ in range(population_size)]

def select_parent(population):
    # Tournament selection
    tournament_size = 5
    best = max(random.sample(population, tournament_size), key=lambda ind: ind.fitness)
    return best

def crossover(parent1, parent2):
    # Single-point crossover
    crossover_point = random.randint(1, len(parent1.genes) - 1)
    child_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
    child = Individual(len(child_genes))
    child.genes = child_genes
    return child

def mutate(individual, mutation_rate):
    for i in range(len(individual.genes)):
        if random.random() < mutation_rate:
            individual.genes[i] = 1 - individual.genes[i]  # Flip gene

def evolve(population, items, capacity, mutation_rate, num_generations, log=True):
    for generation in range(num_generations):
        # Evaluate fitness for each individual
        for individual in population:
            individual.calculate_fitness(items, capacity)

        # Logging the progress
        if log and generation % 100 == 0:
            best_individual = max(population, key=lambda ind: ind.fitness)
            print(f"Generation {generation}: Best Fitness = {best_individual.fitness}, Best Total Weight = {best_individual.total_weight}")

        # Create a new population
        new_population = []
        for _ in range(len(population)):
            parent1 = select_parent(population)
            parent2 = select_parent(population)
            offspring = crossover(parent1, parent2)
            mutate(offspring, mutation_rate)
            new_population.append(offspring)

        population = new_population

    return population

def get_best_individual(population):
    return max(population, key=lambda ind: ind.fitness)

def display_solution(best_individual, items):
    print("\nBest Solution Found:")
    print(f"Fitness (Value): {best_individual.fitness}")
    print(f"Total Weight: {best_individual.total_weight}")
    print("Items Included:")
    for i in range(len(best_individual.genes)):
        if best_individual.genes[i] == 1:
            print(f"Item {i+1} (Value: {items[i].value}, Weight: {items[i].weight})")

def main():
    # User input for items and knapsack capacity
    num_items = int(input("Enter the number of items: "))
    items = []
    for i in range(num_items):
        value = int(input(f"Enter value for item {i + 1}: "))
        weight = int(input(f"Enter weight for item {i + 1}: "))
        items.append(Item(value, weight))

    capacity = int(input("Enter the knapsack capacity: "))

    # Genetic Algorithm Parameters
    population_size = int(input("Enter population size: "))
    mutation_rate = float(input("Enter mutation rate (0.01 - 0.1): "))
    num_generations = int(input("Enter number of generations: "))

    # Create initial population
    population = create_initial_population(population_size, num_items)

    # Run the evolution process
    print("\nStarting genetic algorithm...\n")
    final_population = evolve(population, items, capacity, mutation_rate, num_generations)

    # Get and display the best solution
    best_individual = get_best_individual(final_population)
    display_solution(best_individual, items)

if __name__ == "__main__":
    main()
