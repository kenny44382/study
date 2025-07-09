import optuna
import numpy as np
import matplotlib.pyplot as plt

# Objective Function: Rastrigin Function (for high dimensions)
def rastrigin(x, A=10):
    x = np.array(x)  # Ensure x is a numpy array
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=0)

# Objective function for Optuna
def objective(trial):
    # Define the bounds for the Rastrigin function [-5.12, 5.12] for each dimension
    n = 30  # Number of dimensions (can be increased for higher-dimensional problems)
    bounds = [-5.12, 5.12]  # Rastrigin function search space
    x = [trial.suggest_uniform(f"dim_{i}", bounds[0], bounds[1]) for i in range(n)]
    
    # Evaluate the Rastrigin function at the suggested point
    return rastrigin(x)

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100)  # Perform 100 trials of optimization

# Get the best value and corresponding parameters
best_value = study.best_value
best_params = study.best_params

print(f"Best Value: {best_value}")
print(f"Best Parameters: {best_params}")

# Visualization of optimization progress
opt_values = [trial.value for trial in study.trials]
plt.figure(figsize=(10, 6))
plt.plot(opt_values, marker='o')
plt.title("Optimization Progress - Optuna Bayesian Optimization")
plt.xlabel("Trial Number")
plt.ylabel("Objective Function Value (Rastrigin)")
plt.grid(True)
plt.show()

# Visualize the best parameters (showing only the first few dimensions for clarity)
best_params_subset = {k: best_params[k] for k in list(best_params)[:2]}  # Show first 5 dimensions
print("Best Parameters (subset for visualization):", best_params_subset)