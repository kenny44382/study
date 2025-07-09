import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Particle:
    def __init__(self, dim, bounds):
        # Random initialization of particle position and velocity
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

class PSO:
    def __init__(self, objective_function, dim, bounds, num_particles=30, max_iter=100, w=0.5, c1=1.5, c2=1.5):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive weight
        self.c2 = c2  # social weight
        
        # Initialize particles
        self.particles = [Particle(dim, bounds) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_value = float('inf')

        # To store fitness values for plotting
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


# Define the objective function to be optimized
def objective_function(x):
    # Example: Sphere function (sum of squares), works well for high dimensions
    return np.sum(x**2)


# # --- Plotting the optimization process ---
# # Plot fitness values over iterations
# plt.figure(figsize=(10, 6))
# plt.plot(pso.fitness_history)
# plt.title("Optimization Progress - PSO")
# plt.xlabel("Iteration")
# plt.ylabel("Best Fitness (Objective Value)")
# plt.grid(True)
# plt.show()

# If the problem is high-dimensional, we can visualize a lower-dimensional projection (2D or 3D).
# We will visualize only the first 2 or 3 dimensions of the particles for visualization purposes.

def plot_particles(particles, iteration, dim=2):
    positions = np.array([particle.position[:dim] for particle in particles])
    if dim == 2:
        plt.scatter(positions[:, 0], positions[:, 1], label=f"Iteration {iteration}")
        plt.title(f"Particles Position at Iteration {iteration}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True)
    elif dim == 3:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], label=f"Iteration {iteration}")
        ax.set_title(f"Particles Position at Iteration {iteration}")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        ax.grid(True)
    plt.show()

# # Optionally plot particle positions in 2D or 3D at selected iterations
# plot_particles(pso.particles, 1, dim=2)  # 2D projection at iteration 1
# plot_particles(pso.particles, max_iter, dim=2)  # 2D projection at the last iteration