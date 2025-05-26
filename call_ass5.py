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

# Code and comments for Task 3.3 Investigating RK numerical error goes here



# Code and comments for Task 3.4 Investigating RK stability goes here


pass
