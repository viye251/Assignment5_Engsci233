from module_ass5 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Code and comments for Task 1.2: numerical error in difference_squares goes here
x = 1 + 2**-29
y = 1 + 2**-30
difference_squares(x, y)

"""
a) Comment on how much the two values calculated for the difference of two squares differ

The values differ very slightly:
z1 = 0.00000000186264514923095703125000
z2 = 0.00000000186264515183304224521521. The difference is approximately 2.6020852 x 10^-18

b) Identify two reasons why they differ, providing a brief explanation of each reason and how it applies in this context

Firstly, Floating-Point Precision Limits - The numbers x = 1 + 2^-29 and y = 1 + 2^-30 are very close together.
Squaring these values involves operations beyond the precision limit of the float representation, mainly when it
comes towards the less significant bits. 
Secondly, Numerical Stability of the Formula - The first form, x^2 - y^2, is less numerically stable when x and y 
are close in value. This is because of potential cancellation error. The second form, (x - y)(x + y), is 
mathematically the same, but can reduce numerical error in these situations.
    
c) Explain which result you think is more accurate

The second one where z2 = (x - y)(x + y) is more accurate in the scenario due to its numerical stability
"""

# -------------------------------------------------------------------
# Code and comments for Task 1.4: numerical error in relative_error_subtraction goes here

x = 1 + 1e-15
y = 1 + 2e-15
z_exact = -1e-15

relative_error_subtraction(x, y, z_exact)

"""
a) What is the relative error in z?

The relative error is: 0.1118215802998748

b) What causes the relative error?

Firstly, Catastrophic Cancellation - subtracting two very similar floating-point numbers (x and y) causes most 
significant digits to cancel out. Leading to a large relative error because the small difference is computed from
imprecise representations.

Secondly, Floating-Point Representation Limits - Python uses IEEE 754 double precision, which has a limit to how 
precisely it can represent numbers Because x and y differ by only 1e-15 (which is very close to epsilon), 
their individual representations already include round-off error, reducing overall accuracy
"""

# -------------------------------------------------------------------
# Code for Task 2.4: numerically solving the ODE goes here

# a) Numerical solution using solve_ivp (t(0) to t(2))
t_span = [0, 2]
y_initial = [4]

solution = solve_ivp(derivative_ode1, t_span, y_initial)

# Time values from solver
time_values = solution.t
# Approximate solution
y_numerical = solution.y[0]

# Exact solution at same time points
y_exact = exact_solution_ode1(time_values)

#Plot exact and numerical solutions
plt.figure()
    #blue lines for Exact solution
plt.plot(time_values, y_exact, 'b-', label='Exact solution')
    # Red circles for numerical solution
plt.plot(time_values, y_numerical, 'ro', label='Numerical solution')
    # Graph Title and Labels
plt.title('Exact Solutions vs Numerical Solutions')
plt.xlabel('Time/(t)')
plt.ylabel('Solution/(y)')
plt.legend()
plt.savefig("ode1ComparisonPlot.jpg")
plt.close()

# --------------------------------------------------------------------------------------------------------------
# Code and comments for Task 2.6: investigating the MAE goes here

mae = mean_absolute_error(y_exact, y_numerical)

print(f"Mean Absolute Error = {mae:.4f}")

"""
a) What is the value of MAE when solving the given ODE?

MAE = 0.0002 (4 d.p.)

b) Based on this and a visual comparison of the exact and approximate solutions, comment on on the solution accuracy

The numerical solution - red -  closely follows the exact solution - blue - across the interval [0, 2].
This suggests that the numerical method (`solve_ivp`) produces a very accurate result with minimal deviation.

cc) Note that the calculation of MAE may itself be subject to numerical error.
Briefly comment on two ways that numerical error may be incurred.

Firstly, Catastrophic Cancellation – when the exact and numerical solutions are very close in value, 
subtracting them (as in |y_exact - y_approx|) can lead to loss of significant digits. This causes a larger numerical 
error in the computed difference, which adds up when calculating the MAE.

Secondly, Floating-Point Representation Limits – both y_exact and y_approx are stored in double-precision floating-point
format (IEEE 754), which cannot exactly represent many decimal numbers. These roundoff errors can affect into the MAE 
calculation, even if both inputs are accurate to complete precision.
"""

# set Butcher tableau for Euler method
alpha_euler = np.array([1.])
beta_euler = np.array([0.])
gamma_euler = np.array([[0.]])

# --------------------------------------------------------------------------------------------------------------
# Code and comments for Task 3.3 Investigating RK numerical error goes here

# Step sizes to investigate
step_sizes = [0.01, 0.025, 0.05, 0.1, 0.2, 0.25]

# Define Butcher Tableau for each method
rk_tableau = {
    "Euler": (
        np.array([1.]),
        np.array([0.]),
        np.array([[0.]]) ),
    "Improved": (
        np.array([0.5, 0.5]),
        np.array([0., 1.]),
        np.array([[0., 0.], [1., 0.]])),
    "RK4": (
        np.array([1/6, 1/3, 1/3, 1/6]),
        np.array([0., 0.5, 0.5, 1.]),
        np.array([ [0., 0., 0., 0.], [0.5, 0., 0., 0.], [0., 0.5, 0., 0.], [0., 0., 1., 0.]]))}

# Initialise empty lists for storing MAE results
mae_results = {method: [] for method in rk_tableau.keys()}

# Loop over each step size and each method
for h in step_sizes:
    for method_name, (alpha, beta, gamma) in rk_tableau.items():
        time_values, y_approx = explicit_rk_solver(derivative_ode1, [0, 2], 4, h, alpha, beta, gamma)
        y_exact = exact_solution_ode1(np.array(time_values))
        mae = mean_absolute_error(y_exact, np.array(y_approx))
        mae_results[method_name].append(mae)

# Display all the results
print("\nMAE results each of the different step sizes:")
for i, h in enumerate(step_sizes):
    print("h =", h)
    print("  Euler MAE:    ", mae_results["Euler"][i])
    print("  Improved MAE: ", mae_results["Improved"][i])
    print("  RK4 MAE:      ", mae_results["RK4"][i])
    print()

# --------------------------------------------------------------------------------------------------------------
# Code and comments for Task 3.4 Investigating RK stability goes here

# Plot MAE vs step size for all three methods
plt.figure()

# Plotting each of the 3 methods
plt.plot(step_sizes, mae_results["Euler"], 'b' 'o-', label="Euler")
plt.plot(step_sizes, mae_results["Improved"], 'r' 'o-', label="Improved Euler")
plt.plot(step_sizes, mae_results["RK4"], 'g' 'o-', label="Classic RK4")

# labels and title
plt.grid(True)
plt.xlabel("Step size / (h)")
plt.ylabel("Mean Absolute Error / (MAE)")
plt.title("MAE vs Step Size (RK Methods)")
plt.legend()

# Save plot to file
plt.savefig("numericalErrorRKmethods.jpg")
plt.close()

"""
b) Estimate from your plot the proportionality relationship between MAE and h for the Euler method. Comment on how 
this compares to the expected truncation error of the method

From the plot, the MAE for the Euler method increases almost linearly with step size h. Suggesting a proportional 
relationship between MAE and h, matching the known global truncation error for Euler: O(h). 
This is expected since Euler is a first-order method.

c) Comment on an advantage and a disadvantage of using the Classic RK4 method over the Euler or Improved Euler methods

    Advantage - high accuracy: the MAE is significantly lower than the Euler and Improved Euler for all step sizes there

    Disadvantage - computationally, it is expensive as it requires 4 derivative evaluations per step vs the 1
    for Euler and 2 for Improved Euler.

"""

# --------------------------------------------------------------------------------------------------------------
# Task 3.5: Investigating Numerical Stability in RK Methods

# Step size to test for stability
h_stability = 0.5
tspan = [0, 2]
y0 = 4

# Solve ODE with each RK method using h = 0.5
solutions = {}

for method_name, (alpha, beta, gamma) in rk_tableau.items():
    t_vals, y_vals = explicit_rk_solver(derivative_ode1, tspan, y0, h_stability, alpha, beta, gamma)
    solutions[method_name] = (np.array(t_vals), np.array(y_vals))

# Compute exact solution at same time points as one of the solvers
t_exact = solutions["Euler"][0]
y_exact = exact_solution_ode1(t_exact)

# Plot all solutions
plt.figure()

# Exact solution
plt.plot(t_exact, y_exact, 'k-', label='Exact Solution')

#Numerical methods
plt.plot(*solutions["Euler"], 'r' 'o-', label="Euler")
plt.plot(*solutions["Improved"], 'g' 'o-', label="Improved Euler")
plt.plot(*solutions["RK4"], 'b' 'o-', label="Classic RK4")
# Add labels
plt.grid(True)
plt.title("Stability Comparison with h = 0.5")
plt.xlabel("Time (t)")
plt.ylabel("y(t)")
plt.legend()
plt.savefig("stabilityRKmethods.jpg")
plt.close()

"""
b) Comment on the numerical stability of each method. How does this impact
on the behaviour of the numerical error in each case?

The plot shows that both the Euler method and the Improved Euler method become numerically unstable when using a
larger step size like h = 0.5. The Euler method oscillates crazily and diverges far from the exact solution, producing
large positive and negative outputs, which is inconsistent with the expected decaying behaviour of the solution.
The Improved Euler method also shows instability, with its values growing exponentially instead of decaying. In
contrast, the Classic RK4 method remains most stable and follows the exact solution right through the entire interval.
This highlights that low-order explicit methods and are far more sensitive in comparison to the larger step sizes, 
while higher-order methods like RK4 maintain that stability under the same conditions.

c)  In the context that we are unable to mathematically determine the stability
condition for a given explicit RK method and ODE, briefly comment on how
might we determine numerically whether or not the solution is numerically
stable Hint: you may want to look at the MAE for h = 0.5 in addition to the
previously evaluated steps.

We can alternatively identify instability numerically. One way is by visually comparing the numerical solution to the 
predicted behaviour of the exact solution. If the numerical solution changes in any way, such as diverrging, oscillations
or just even growing when the exact should be steadily decaying, it is an indication oof instability. 
Another method is to examine the Mean Absolute Error (MAE) for a largre step size. A sudden increase in MAE suggests 
that numerical instability has occurred, even if the divergence is not as easily from a visual plot.
"""
