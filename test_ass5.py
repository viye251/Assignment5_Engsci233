from module_ass5 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Code and comments for Task 1.2: numerical error in difference_squares goes here
x = 1 + 2**-29
y = 1 + 2**-30
difference_squares(x, y)

# a) Comment:
# The values of z1 and z2 are very close but may differ slightly in the least significant digits.

# b) Two reasons they differ:
# 1. Floating-Point Precision: In Python, float uses 64-bit precision (IEEE 754). Squaring large, close numbers
# can amplify rounding errors, especially for small differences.
# 2. Subtractive Cancellation: The method z1 = x^2 - y^2 involves subtracting two similar-sized numbers.
# This can lead to loss of significant digits, making the result less accurate.

# c) More Accurate Result:
# z2 = (x - y)(x + y) is generally more accurate in floating-point arithmetic because it avoids squaring and
# then subtracting nearly equal numbers — reducing the effects of cancellation.

# -------------------------------------------------------------------

# Code and comments for Task 1.4: numerical error in relative_error_subtraction goes here
x = 1 + 1e-15
y = 1 + 2e-15
z_exact = -1e-15

relative_error_subtraction(x, y, z_exact)

# a) What is the relative error in z?
# The relative error is: 0.0000000000000000

# b) What causes the relative error?
# Two main causes:
# 1. Floating-point precision limits: At this scale (1e-15), Python may round off or truncate digits.
# 2. Cancellation error: Subtracting two nearly equal numbers can lose significant digits,
#    leading to higher relative error if not computed with enough precision.

# -------------------------------------------------------------------

# Code for Task 2.4: numerically solving the ODE goes here

# Code and comments for Task 2.6: investigating the MAE goes here

# set Butcher tableau for Euler method
alpha_euler = np.array([1.])
beta_euler = np.array([0.])
gamma_euler = np.array([[0.]])

# Code and comments for Task 3.3 Investigating RK numerical error goes here


# Code and comments for Task 3.4 Investigating RK stability goes here


pass
