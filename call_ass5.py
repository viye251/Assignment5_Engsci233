from module_ass5 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Code and comments for Task 1.2: numerical error in difference_squares goes here
x = 1 + 2**-29
y = 1 + 2**-30
difference_squares(x, y)

# a, b, c) Comments:

# -------------------------------------------------------------------
# Code and comments for Task 1.4: numerical error in relative_error_subtraction goes here

x = 1 + 1e-15
y = 1 + 2e-15
z_exact = -1e-15

relative_error_subtraction(x, y, z_exact)

# a) What is the relative error in z?

# b) What causes the relative error?

# -------------------------------------------------------------------
# Code for Task 2.4: numerically solving the ODE goes here

# a) Numerical solution using solve_ivp (t(0) to t(2))
t_span = [0, 2]
y_initial = [4]

solution = solve_ivp(derivative_ode1, t_span, y_initial)

# Time values from solver
t_values = solution.t
# Approximate solution
y_numerical = solution.y[0]

#b) Exact solution at same time points
y_exact = exact_solution_ode1(t_values)

#c) Plot exact and numerical solutions
plt.figure()
    #blue lines for Exact solution
plt.plot(t_values, y_exact, 'b-', label='Exact solution')
    # Red circles for numerical solution
plt.plot(t_values, y_numerical, 'ro', label='Numerical solution')
    # Graph Title and Labels
plt.title('Comparison of Exact and Numerical Solutions')
plt.xlabel('Time/(t)')
plt.ylabel('Solution/(y)')
plt.legend()

# d) Save the figure
plt.savefig("ode1ComparisonPlot.jpg")
plt.close()

# --------------------------------------------------------------------------------------------------------------
# Code and comments for Task 2.6: investigating the MAE goes here

mae = mean_absolute_error(y_exact, y_numerical)
print(f"Mean Absolute Error = {mae:.4f}")

# a) What is the value of MAE when solving the given ODE?

# b) Based on this and a visual comparison of the exact and approximate solutions,
# comment on how accurate this solution is.

# c) Note that the calculation of MAE may itself be subject to numerical error.
# Briefly comment on two ways that numerical error may be incurred.

# set Butcher tableau for Euler method
alpha_euler = np.array([1.])
beta_euler = np.array([0.])
gamma_euler = np.array([[0.]])

# --------------------------------------------------------------------------------------------------------------
# Code and comments for Task 3.3 Investigating RK numerical error goes here

# --------------------------------------------------------------------------------------------------------------
# Task 3.3: Solving an ODE using different RK methods and varying step sizes

# Step sizes to investigate
step_sizes = [0.01, 0.025, 0.05, 0.1, 0.2, 0.25]

# Define Butcher Tableaux for each method
rk_tableaux = {
    "Euler": (
        np.array([1.]),
        np.array([0.]),
        np.array([[0.]]) ),
    "Improved": (
        np.array([0.5, 0.5]),
        np.array([0., 1.]),
        np.array([
            [0., 0.],
            [1., 0.]])),
    "RK4": (
        np.array([1/6, 1/3, 1/3, 1/6]),
        np.array([0., 0.5, 0.5, 1.]),
        np.array([
            [0., 0., 0., 0.],
            [0.5, 0., 0., 0.],
            [0., 0.5, 0., 0.],
            [0., 0., 1., 0.]]))}

# Initialise empty lists for storing MAE results
mae_results = {method: [] for method in rk_tableaux.keys()}

# Loop over each step size and each method
for h in step_sizes:
    for method_name, (alpha, beta, gamma) in rk_tableaux.items():
        t_values, y_approx = explicit_rk_solver(derivative_ode1, [0, 2], 4, h, alpha, beta, gamma)
        y_exact = exact_solution_ode1(np.array(t_values))
        mae = mean_absolute_error(y_exact, np.array(y_approx))
        mae_results[method_name].append(mae)

# Display the results in a formatted table
print("\nStep Size   Euler MAE     Improved MAE   RK4 MAE")
for i, h in enumerate(step_sizes):
    euler_mae = mae_results["Euler"][i]
    improved_mae = mae_results["Improved"][i]
    rk4_mae = mae_results["RK4"][i]
    print(f"{h:<10} {euler_mae:<13.6e} {improved_mae:<13.6e} {rk4_mae:.6e}")

# --------------------------------------------------------------------------------------------------------------
# Code and comments for Task 3.4 Investigating RK stability goes here

# Plot MAE vs step size for all three methods
plt.figure()

# Plotting each method
plt.plot(step_sizes, mae_results["Euler"], 'o-', label="Euler")
plt.plot(step_sizes, mae_results["Improved"], 's-', label="Improved Euler")
plt.plot(step_sizes, mae_results["RK4"], '^-', label="Classic RK4")

# Axis labels and title
plt.xlabel("Step size (h)")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("MAE vs Step Size for RK Methods")
plt.legend()
# plt.grid(True)

# Save plot to file
plt.savefig("numericalErrorRKmethods.jpg")
plt.close()


