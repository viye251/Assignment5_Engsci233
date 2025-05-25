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
    """ TODO
    """
    pass


def exact_solution_ode1(t):
    """ TODO
    """
    pass


def mean_absolute_error(y_exact, y_approx):
    """ TODO
    """
    pass


def derivative_ode1(t, y):
    """ TODO
    """
    pass


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


