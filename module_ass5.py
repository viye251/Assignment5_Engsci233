# import statements
import numpy as np
import matplotlib.pyplot as plt


def difference_squares(x, y):
    """
    Calculates the difference of squares using two algebraically equivalent methods
    and prints both to 32 decimal places.

    Parameters:
    x (float): first value
    y (float): second value

    Returns:
    None
    """

    z1 = x ** 2 - y ** 2
    z2 = (x - y) * (x + y)
    print(f"z1 = {z1:.32f}")
    print(f"z2 = {z2:.32f}")


def relative_error_subtraction(x, y, z_exact):
    """
    Calculates and displays the subtraction of two floats,
    and computes the relative error between the approximate and exact result.

    Parameters:
    x (float): first number
    y (float): second number
    z_exact (float): exact value of x - y

    Returns:
    None
    """
    z_approx = x - y

    print("The value of x to 64dp is:")
    print(f"{x:.64f}")

    print("The value of y to 64dp is:")
    print(f"{y:.64f}")

    print("The approximate value of z to 64dp is:")
    print(f"{z_approx:.64f}")

    print("The exact value of z to 64dp is:")
    print(f"{z_exact:.64f}")

    if z_exact != 0:
        relative_error = abs(z_approx - z_exact) / abs(z_exact)

    else:
        relative_error = float('inf')

    print("The relative error to 16dp is:", end=" ")
    print(f"{relative_error:.16f}")



def exact_solution_ode1(t):
    """
    Calculates the exact solution to the ODE:
    dy/dt + 5y = 2e^(-5t), y(0) = 4

    The solution is: y(t) = e^(-5t) * (2t + 4)

    Parameters
    ----------
    t : float or 1D numpy array
        Time value(s) at which to evaluate the solution

    Returns
    -------
    y_exact : float or numpy array
        Exact solution evaluated at t
    """
    return np.exp(-5 * t) * (2 * t + 4)


def mean_absolute_error(y_exact, y_approx):
    """ TODO
    """
    pass


def derivative_ode1(t, y):
    """
    Computes dy/dt for the ODE:
    dy/dt + 5y = 2e^(-5t)

    Rearranged as:
    dy/dt = 2e^(-5t) - 5y

    Parameters
    ----------
    t : float
        Independent variable (time)
    y : float
        Dependent variable

    Returns
    -------
    dydt : float
        Derivative dy/dt at given t and y
    """
    dydt = 2 * np.exp(-5 * t) - 5 * y

    return dydt



def euler_step(f, t, y, h):
    """
    Calculate one step of the Euler method.

    Parameters
    ----------
    f : function
        Derivative function (callable).
    t : float
        Independent variable at start of step.
    y : float
        Dependent variable at start of step.
    h : float
        Step size along independent variable.

    Returns
    -------
    y_new : float
        Dependent variable at end of step.
    """
    f0 = f(t, y)
    y_new = y + h * f0
    return y_new


def improved_euler_step(f, t, y, h):
    """
    Calculate one step of the Improved Euler method.

    Parameters
    ----------
    f : function
        Derivative function (callable).
    t : float
        Independent variable at start of step.
    y : float
        Dependent variable at start of step.
    h : float
        Step size along independent variable.

    Returns
    -------
    y_new : float
        Dependent variable at end of step.
    """
    f0 = f(t, y)
    f1 = f(t + h, y + h * f0)
    y_new = y + h * 0.5 * (f0 + f1)
    return y_new


def classic_rk4_step(f, t, y, h):
    """
    Calculate one step of the Classic RK4 method.

    Parameters
    ----------
    f : function
        Derivative function (callable).
    t : float
        Independent variable at start of step.
    y : float
        Dependent variable at start of step.
    h : float
        Step size along independent variable.

    Returns
    -------
    y_new : float
        Dependent variable at end of step.
    """
    f0 = f(t, y)
    f1 = f(t + h * 0.5, y + h * 0.5 * f0)
    f2 = f(t + h * 0.5, y + h * 0.5 * f1)
    f3 = f(t + h, y + h * f2)
    y_new = y + h * (f0 + 2. * f1 + 2. * f2 + f3) / 6.
    return y_new


def explicit_rk_step(f, t, y, h, alpha, beta, gamma):
    """ TODO
    """
    pass


def explicit_rk_solver(f, tspan, y0, h, alpha, beta, gamma):
    """ TODO
    """
    pass


