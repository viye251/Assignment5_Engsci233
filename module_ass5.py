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
    """
    Calculates the Mean Absolute Error (MAE) between an exact solution and
    a numerical approximation, excluding the initial condition.

    Parameters
    ----------
    y_exact : 1D numpy array
        Exact solution including initial condition.
    y_approx : 1D numpy array
        Approximate (numerical) solution including initial condition.

    Returns
    -------
    mae : float
        Mean absolute error, excluding the initial condition.
    """
    total = 0
    for i in range(1, len(y_exact)):
        total += abs(y_exact[i] - y_approx[i])
    mae = total / (len(y_exact) - 1)
    return mae


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
    """
        Perform a single step of an explicit Runge-Kutta method.

    Parameters
    ----------
    f : function
        Derivative function f(t, y).
    t : float
        Current value of independent variable.
    y : float
        Current value of dependent variable.
    h : float
        Step size.
    alpha : list of float
        Weights from Butcher tableau.
    beta : list of float
        Nodes from Butcher tableau.
    gamma : list of list of float
        RK matrix from Butcher tableau.

    Returns
    -------
    y_new : float
        New value of dependent variable after one RK step.
    """
    n = len(alpha)
    # Store derivative values at each stage
    f_values = [0.0] * n

    for i in range(n):
        # Calc. the time point
        t_i = t + beta[i] * h
        # Cal. immediate y value at this stage
        y_i = y
        for j in range(i):
            y_i += h * gamma[i][j] * f_values[j]
            # Now start calculating the derivative
        f_values[i] = f(t_i, y_i)

    y_new = y
    for i in range(n):
        y_new += h * alpha[i] * f_values[i]
    # new value of y after one RK step
    return y_new


def explicit_rk_solver(f, tspan, y0, h, alpha, beta, gamma):
    """
        Solve a first-order ODE using an explicit Runge-Kutta method.

    Parameters
    ----------
    f : function
        Derivative function f(t, y).
    tspan : list of float
        [t_start, t_end] time interval.
    y0 : float
        Initial y value.
    h : float
        Step size.
    alpha : list of float
        Weights from the Butcher tableau.
    beta : list of float
        Nodes from the Butcher tableau.
    gamma : list of list of float
        RK matrix from the Butcher tableau.

    Returns
    -------
    t : list of float
        Time values from t_start to t_end (inclusive).
    y : list of float
        Corresponding y values at each step.
    """
    t_start = tspan[0]
    t_end = tspan[1]
    # calc. total number of steps
    n_steps = int((t_end - t_start) / h)

    t = [0.0] * (n_steps + 1)
    y = [0.0] * (n_steps + 1)

    # set the initial values
    t[0] = t_start
    y[0] = y0

    for i in range(n_steps):
        t_current = t[i]
        y_current = y[i]
        y_next = explicit_rk_step(f, t_current, y_current, h, alpha, beta, gamma)
        # store the following values for t and y
        t[i + 1] = t_current + h
        y[i + 1] = y_next

    return t, y
