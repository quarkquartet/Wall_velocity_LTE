"""
This file is an implementation of the "modified cosmoTransitions".
Modifications focus on the integral function in the 1-loop
finite temperature correction:

    Jb(x) = int[0->inf] dy +y^2 log( 1 - exp(-sqrt(x + y^2)) )
    Jf(x) = int[0->inf] dy -y^2 log( 1 - exp(-sqrt(x + y^2)) )

Called by:

    Jb(x), Jf(x)

Notice that x refers to (m/T)^2. Square is already taken!

The motivations is that the original form of CosmoTransitions does not treat
negative mass square properly. See the appendix of the paper.

This code uses interpolation function to make a splined function. The input
data points for the spline is from
https://gitlab.com/claudius-krause/ew_nr

Most of the code are borrowed from them.

Author: Isaac R. Wang
"""

import os

import numpy
from scipy import integrate, interpolate
from scipy import special

try:
    from scipy.misc import factorial as fac
except ImportError:
    from scipy.special import factorial as fac

pi = numpy.pi
euler_gamma = 0.577215661901532
log, exp, sqrt = numpy.log, numpy.exp, numpy.sqrt
array = numpy.array

spline_data_path = os.path.dirname(__file__)

# Spline fitting Jf
_xfmin = -1000.0
_xfmax = 1.35e3
_Jf_dat_path = spline_data_path + "/finiteT_f.dat.txt"
_xf, _yf = numpy.loadtxt(_Jf_dat_path).T
_tckf = interpolate.splrep(_xf, _yf)


def Jf_spline(X, n=0):
    """Jf interpolated from a saved spline. Input is (m/T)^2."""
    X = numpy.array(X, copy=False)
    x = X.ravel()
    y = interpolate.splev(x, _tckf, der=n).ravel()
    y[x < _xfmin] = interpolate.splev(_xfmin, _tckf, der=n)
    y[x > _xfmax] = 0
    return y.reshape(X.shape)


# Spline fitting Jb
_xbmin = -25000.0
# We're setting the lower acceptable bound as the point where it's a minimum
# This guarantees that it's a monatonically increasing function, and the first
# deriv is continuous.
_xbmax = 1.41e3
_Jb_dat_path = spline_data_path + "/finiteT_b.dat.txt"
_xb, _yb = numpy.loadtxt(_Jb_dat_path).T

_tckb = interpolate.splrep(_xb, _yb)


def Jb_spline(X, n=0):
    """Jb interpolated from a saved spline. Input is (m/T)^2."""
    X = numpy.array(X, copy=False)
    x = X.ravel()
    y = interpolate.splev(x, _tckb, der=n).ravel()
    y[x < _xbmin] = interpolate.splev(_xbmin, _tckb, der=n)
    y[x > _xbmax] = 0
    return y.reshape(X.shape)
