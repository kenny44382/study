# # file2.py
# import numpy as np
# import matplotlib.pyplot as plt
# import PSO_trial_2 as PS  # Import the PSO_trial_2 module

# # Using the function from file1
# # name = "Alice"
# # greeting = file1.greet(name)
# # print(greeting)

# # Define the problem parameters
# dim = 50  # Number of dimensions (e.g., 50-dimensional problem)
# bounds = (-5, 5)  # Boundaries of the search space (e.g., [-5, 5] for all dimensions)
# num_particles = 50  # Number of particles
# max_iter = 100  # Maximum number of iterations

# # Initialize and run PSO
# pso = PS.PSO(PS.objective_function, dim, bounds, num_particles, max_iter)
# best_position, best_value = pso.optimize()

# print(f"Best Position: {best_position}")
# print(f"Best Value: {best_value}")

# # --- Plotting the optimization process ---
# # Plot fitness values over iterations
# plt.figure(figsize=(10, 6))
# plt.plot(pso.fitness_history)
# plt.title("Optimization Progress - PSO")
# plt.xlabel("Iteration")
# plt.ylabel("Best Fitness (Objective Value)")
# plt.grid(True)
# plt.show()

# ppso = pso.particles
# # Optionally plot particle positions in 2D or 3D at selected iterations

# PS.plot_particles(ppso, 1, dim=2)  # 2D projection at iteration 1

# PS.plot_particles(ppso, max_iter, dim=2)  # 2D projection at the last iteration

import numpy as np
import matplotlib.pyplot as plt

# Objective Function: Rastrigin Function
def rastrigin(x, A=10):
    # A is usually set to 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# Particle class for PSO
class Particle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')

    def evaluate(self, objective_function):
        value = objective_function(self.position)
        if value < self.best_value:
            self.best_value = value
            self.best_position = np.copy(self.position)
        return value

# PSO Class
class PSO:
    def __init__(self, objective_function, dim, bounds, num_particles=30, max_iter=100, w=0.5, c1=1.5, c2=1.5):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive weight
        self.c2 = c2  # Social weight

        # Initialize particles
        self.particles = [Particle(dim, bounds) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_value = float('inf')

        # Store fitness values for plotting
        self.fitness_history = []

    def optimize(self):
        for iter_num in range(self.max_iter):
            iter_best_value = float('inf')
            for particle in self.particles:
                # Evaluate particle fitness
                fitness = particle.evaluate(self.objective_function)
                iter_best_value = min(iter_best_value, fitness)

                # Update global best position if necessary
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = np.copy(particle.position)

            # Store fitness for plotting
            self.fitness_history.append(self.global_best_value)

            # Update particle velocities and positions
            for particle in self.particles:
                r1, r2 = np.random.rand(2)
                # Update velocity
                particle.velocity = (self.w * particle.velocity + 
                                     self.c1 * r1 * (particle.best_position - particle.position) + 
                                     self.c2 * r2 * (self.global_best_position - particle.position))
                # Update position
                particle.position = particle.position + particle.velocity
                
                # Keep particle within bounds
                particle.position = np.clip(particle.position, self.bounds[0], self.bounds[1])

            # Optionally, print progress
            if (iter_num + 1) % 10 == 0 or iter_num == self.max_iter - 1:
                print(f"Iteration {iter_num + 1}/{self.max_iter}, Best Value: {self.global_best_value}")

        return self.global_best_position, self.global_best_value

# Define problem parameters
dim = 100  # Number of dimensions (100-dimensional problem)
bounds = (-5.12, 5.12)  # Rastrigin function is typically evaluated in the range [-5.12, 5.12]
num_particles = 50  # Number of particles
max_iter = 100  # Maximum number of iterations

# Initialize and run PSO
pso = PSO(rastrigin, dim, bounds, num_particles, max_iter)
best_position, best_value = pso.optimize()

print(f"Best Position: {best_position}")
print(f"Best Value: {best_value}")

# Plot fitness values over iterations (this will be a 1D plot showing progress)
plt.figure(figsize=(10, 6))
plt.plot(pso.fitness_history)
plt.title("Optimization Progress - PSO (Rastrigin Function)")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness (Objective Value)")
plt.grid(True)
plt.show()