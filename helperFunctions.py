import numpy as np


def derivative(f, x, eps=1e-5, order=1):
    """Calculate the derivative of a function.
    Using Ridders algorithm.
    Calculate up to order x**4.
    Only compute 1st or 2nd order derivative."""
    assert (
        order == 1 or order == 2
    ), "Calculate derivative up to an order, must be 1 or 2."
    δx = x * eps
    if order == 1:
        return (f(x - 2 * δx) - 8 * f(x - δx) + 8 * f(x + δx) - f(x + 2 * δx)) / (
            12 * δx
        )
    if order == 2:
        return (f(x - 2 * δx) - f(x - δx) - f(x + δx) + f(x + 2 * δx)) / (3 * δx**2)


def cs_sq(V, T, vev):
    """Sound speed square."""

    def VT(T):
        return V(vev, T)

    return derivative(VT, T, order=1) / (T * derivative(VT, T, order=2))


def epsilon(V, T, vev):
    """Epsilon."""

    def VT(T):
        return V(vev, T)

    return -0.25 * T * derivative(VT, T) + VT(T)


def a(V, T, vev):
    """The parameter a, means the effective dofs."""

    def VT(T):
        return V(vev, T)

    return -0.75 * derivative(VT, T) / T**3


def alpha_p(V, Tp, Tm, high_vev, low_vev):
    return (epsilon(V, Tp, high_vev) - epsilon(V, Tm, low_vev)) / (
        a(V, Tp, high_vev) * Tp**4
    )


def r_func(V, Tp, Tm, high_vev, low_vev):
    return a(V, Tp, high_vev) * Tp**4 / (a(V, Tm, low_vev) * Tm**4)


def vJ(alphap):
    v = (alphap * (2 + 3 * alphap)) ** 0.5 + 1
    v = v / ((1 + alphap) * 3**0.5)
    return v


def dYdtau(tau, y, *args):
    """Y: (v, xi, T)"""
    v = y[0]
    T = y[1]
    xi = y[2]
    V = args[0]
    vev = args[1]
    dvdtau = 2 * v * cs_sq(V, T, vev) * (1 - v**2) * (1 - xi * v)
    dxidtau = xi * ((xi - v) ** 2 - cs_sq(V, T, vev) * (1 - xi * v) ** 2)
    dvdxi = 2 * v / xi * (1 - v**2)
    dvdxi = dvdxi / (1 - v * xi) / (μ(xi, v) ** 2 / cs_sq(V, T, vev) - 1)
    dTdxi = T * μ(xi, v) * dvdxi / (1 - v**2)
    dTdtau = dTdxi * dxidtau
    return np.array([dvdtau, dTdtau, dxidtau])


def μ(x, v):
    return (x - v) / (1 - x * v)
