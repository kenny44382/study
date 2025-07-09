import numpy as np

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

    def optimize(self):
        for iter_num in range(self.max_iter):
            for particle in self.particles:
                # Evaluate particle fitness
                fitness = particle.evaluate(self.objective_function)
                
                # Update global best position if necessary
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = np.copy(particle.position)
            
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
    # Example: Sphere function (sum of squares), which works well for high dimensions
    return np.sum(x**2)

# Define the problem parameters
dim = 50  # Number of dimensions (e.g., 50-dimensional problem)
bounds = (-5, 5)  # Boundaries of the search space (e.g., [-5, 5] for all dimensions)
num_particles = 50  # Number of particles
max_iter = 100  # Maximum number of iterations

# Initialize and run PSO
pso = PSO(objective_function, dim, bounds, num_particles, max_iter)
best_position, best_value = pso.optimize()

print(f"Best Position: {best_position}")
print(f"Best Value: {best_value}")