from module_ass5 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Code and comments for Task 1.2: numerical error in difference_squares goes here
x = 1 + 2**-29
y = 1 + 2**-30
difference_squares(x, y)

# Code and comments for Task 1.4: numerical error in relative_error_subtraction goes here


# Code for Task 2.4: numerically solving the ODE goes here


# Code and comments for Task 2.6: investigating the MAE goes here

# set Butcher tableau for Euler method
alpha_euler = np.array([1.])
beta_euler = np.array([0.])
gamma_euler = np.array([[0.]])

# Code and comments for Task 3.3 Investigating RK numerical error goes here


# Code and comments for Task 3.4 Investigating RK stability goes here


pass
