"""
SM_model.py
Author: Isaac Wang
This code implements the SM Higgs potential at the finite temperature,
as well as the computation for quantities related to the electroweak
phase transition.
The SM parameters are varied.
Input parameters should be the g2 gauge coupling, Higgs quartic coupling,
and the Higgs mass term parameter.
The top Yukawa is modified to a smaller value so that a strong first-order
phase transition can be achieved.
"""

import numpy as np
from cosmoTransitions import generic_potential as gp
from cosmoTransitions import tunneling1D as td
from finiteT import Jb_spline as Jb
from finiteT import Jf_spline as Jf
from matplotlib import pyplot as plt
from scipy import interpolate, optimize

v = 246.22
mt = 120.0


def df(f, x, eps):
    """
    Calculate the derivative of a function to the second order,
    using the Ridders algorithm.
    f: callable. The function to be calculated.
    x: the point to calculate.
    eps: the step.
    """
    return (f(x - 2 * eps) - 8 * f(x - eps) + 8 * f(x + eps) - f(x + 2 * eps)) / (
        12.0 * eps
    )


class SM(gp.generic_potential):
    """
    Definition of the SM Higgs potential, as well as computations for
    related quantities in EWPT.
    Should be called for any computations.
    """

    def init(self, g2, λ, μhsq, Teps=0.02):
        """
        Initialization of the potential.
        - g2: gauge coupling for the SU(2) gauge group.
        - λ: Higgs quartic coupling.
        - μhsq: the μ^2 parameter in the Higgs potential. Notice that
        this is the squared one.
        - Teps: the steps in temperature used to search the nucleation temperature.
        """
        self.Ndim = (
            1  # Dimension in field configuration space, i.e. number of scalar fields
        )
        self.g2 = g2  # SU(2) gauge coupling
        self.g1 = 0.358  # U(1) gauge coupling
        self.λ = λ  # Higgs quartic coupling
        self.μh = μhsq**0.5  # The μ parameter in Higgs potential
        self.yt = 2**0.5 * mt / v  # Top Yukawa.
        self.Tmax = 200  # Maximal temperature scanning
        self.Tmin = 20  # Minimal temperature scanning
        self.Tc = None  # Critical temperature
        self.Tn = None  # Nucleation temperature
        self.Tcvev = None  # The vev at the critical temperature
        self.Tnvev = None  # The vev at the nucleation temperature
        self.strengthTc = None  # The v/T at the critical temperature
        self.strengthTn = None  # The v/T at the nucleation temperature
        self.βH = None  # The β/H parameter
        self.α = None  # The α parameter
        self.renormScaleSq = (
            1e4  # The renormalization scale squared in the Coleman-Weinberg potential
        )
        self.Teps = Teps  # T steps in searching for nucleation temperature
        print("Model inialized.")
        print("g2 = " + str(self.g2))
        print("λ = " + str(self.λ))
        print("μh = " + str(self.μh))

    def V0(self, X):
        """
        The tree-level Higgs potential.
        """
        X = np.array(X)
        h = X[..., 0]
        return -0.5 * self.μh**2 * h**2 + 0.25 * self.λ * h**4

    def boson_massSq(self, X, T):
        """
        The mass square of bosons, depending on the scalar field value and temperature.
        Thermal mass is included for the sake of resummation.
        """
        X = np.array(X)
        h = X[..., 0]
        T2 = T * T
        mgs = (
            self.λ * h**2
            - self.λ * v**2
            + (3 * self.g2**2 / 16 + self.g1**2 / 16 + 0.5 * self.λ + 0.25 * self.yt**2)
            * T2
        )  # Goldstone modes
        mhsq = (
            3 * self.λ * h**2
            - self.λ * v**2
            + (3 * self.g2**2 / 16 + self.g1**2 / 16 + 0.5 * self.λ + 0.25 * self.yt**2)
            * T2
        )  # Higgs

        mW = 0.25 * self.g2**2 * h**2  # W boson
        mWL = mW + 11 * self.g2**2 * T2 / 6  # Longitudinal mode for W boson
        mZ = 0.25 * (self.g2**2 + self.g1**2) * h**2  # Z boson

        AZsq = np.sqrt(
            (self.g2**2 + self.g1**2) ** 2 * (3 * h**2 + 22 * T2) ** 2
            - 176 * self.g2**2 * self.g1**2 * T2 * (3 * h**2 + 11 * T2)
        )  # Square-root part of the A-Z mixing matrix eigenvalues

        mZL = (
            (self.g2**2 + self.g1**2) * (3 * h**2 + 22 * T2) + AZsq
        ) / 24  # Longitudinal mode of Z boson
        mAL = (
            (self.g2**2 + self.g1**2) * (3 * h**2 + 22 * T2) - AZsq
        ) / 24  # Longitudinal mode of photon, should be 0 at T=0

        M = np.array([mgs, mhsq, mW, mWL, mZ, mZL, mAL])
        M = np.rollaxis(M, 0, len(M.shape))

        dof = np.array([3, 1, 4, 2, 2, 1, 1])  # Degrees of freedom
        c = np.array(
            [1.5, 1.5, 0.5, 1.5, 0.5, 1.5, 1.5]
        )  # Constants in the CW potential

        return M.real + 1e-16, dof, c

    def fermion_massSq(self, X):
        """
        Fermion mass squared. Only top quark is included.
        """
        X = np.array(X)
        h = X[..., 0]

        mtt = 0.5 * self.yt**2 * h**2
        Mf = np.array([mtt])
        Mf = np.rollaxis(Mf, 0, len(Mf.shape))

        doff = np.array([12.0])
        return Mf, doff

    def V1(self, bosons, fermions):
        """
        Method of CosmoTransitions. Overwritten.

        The 1-loop CW correction at the zero-temperature in the
        MS-bar renormalization scheme.
        """
        Q2 = self.renormScaleSq
        m2, n, c = bosons
        y = np.sum(n * m2 * m2 * (np.log(m2 / Q2 + 1e-100 + 0j) - c), axis=-1)
        m2, n = fermions
        c = 1.5
        y -= np.sum(n * m2 * m2 * (np.log(m2 / Q2 + 1e-100 + 0j) - c), axis=-1)
        return y.real / (64 * np.pi * np.pi)

    def V0T(self, X):
        """
        1-loop corrected effective potential at T=0.
        i.e. the CW potential plus the tree-level one.
        Not an intrinsic method of CosmoTransitions.
        """
        X = np.asanyarray(X, dtype=float)

        bosons = self.boson_massSq(X, 0)
        fermions = self.fermion_massSq(X)

        y = self.V0(X)
        y += self.V1(bosons, fermions)

        return y

    def V1T(self, bosons, fermions, T, include_radiation=True):
        """
        Method of CosmoTransitions. Should be overwritten.
        The 1-loop finite-temperature correction term.

        `Jf` and `Jb` are modified functions.

        TODO: understand this again, write note, and implement it.
        """

        T2 = (T * T) + 1e-100
        T4 = T2 * T2

        m2, nb, _ = bosons
        y = np.sum(nb * Jb(m2 / T2), axis=-1)
        m2, nf = fermions
        y += np.sum(nf * Jf(m2 / T2), axis=-1)

        return y * T4 / (2 * np.pi * np.pi)

    def Vtot(self, X, T, include_radiation=True):
        """
        Method of CosmoTransitions.
        The total finite temperature effective potential.
        """

        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)

        bosons = self.boson_massSq(X, T)
        fermions = self.fermion_massSq(X)
        Vtot = self.V0(X)
        Vtot += self.V1(bosons, fermions)
        Vtot += self.V1T(bosons, fermions, T, include_radiation)
        Vtot -= np.pi**2 * 83.25 * T**4 / 90

        return Vtot

    def Plot(self, xminmax, npoints=500, T=0):
        """
        Function to plot the potential. Just for convenience.
        - xminmax: plot range.
        - npoints: number of points.
        - T: temperature.
        """
        xmin = xminmax[0]
        xmax = xminmax[1]
        hspace = np.linspace(xmin, xmax, npoints)
        H = hspace[:, np.newaxis]
        if T == 0:
            Vspace = self.V0T(H)
        else:
            Vspace = self.Vtot(H, T)
        plt.plot(hspace, Vspace)
        plt.show()

    def findTc(self):
        """
        Find the critical temperature using the bisection method.
        """
        num_i = 30  # Number of steps
        Tmax = self.Tmax
        Tmin = self.Tmin
        T_test = (Tmax + Tmin) * 0.5
        print("Finding Tc...")
        for i in range(num_i + 10):
            # We allow for 10 more steps.
            # This is because sometimes the result Tc is slightly above the true one.
            # (Numerically you never get the true one, you just get a number close to that.)
            # In this case, the non-zero local mimimum is not the global one,
            # and numerical issue may happen in some algorithms.
            # So just allow more steps to get a result lower than the true one.
            minv = optimize.fmin(
                self.Vtot, 100, args=(T_test,), disp=False, full_output=1
            )
            xmin = minv[0]
            ymin = minv[1]
            if ymin < self.Vtot([1e-16], T_test) and xmin > 1:
                # V at the non-zero local minimum is lower than at 0. Temperature lower than Tc.
                # x_min > 1: sometimes fmin function returns the local minimum at zero.
                # In this case the returned number is very small.
                if i > num_i:
                    # Once the number is reached, we can stop computing and use this temperature as Tc.
                    # Notice that this break condition is under requirement of V(v\neq 0) < V(0),
                    # as stated above.
                    self.Tc = T_test
                    self.Tcvev = xmin[0]
                    self.strengthTc = xmin[0] / T_test
                    break
                else:
                    # Iteration number not enough. Continue. T should be higher than this value.
                    Tmin = T_test
                    Tnext = (Tmax + T_test) * 0.5
                    T_test = Tnext
            else:
                # V(v \neq 0) > V(0). Temperature higher than Tc.
                # In this case we should continue, rather than stop iterating and return the temperature,
                # even if the iteration number is reached.
                Tmax = T_test
                Tnext = (Tmin + T_test) * 0.5
                T_test = Tnext

        print("Critical temperature found! Tc = " + str(self.Tc))
        print("v_c/Tc = " + str(self.strengthTc))

    def tunneling_at_T(self, T):
        """
        Computing the tunneling from the false vev to the truevev at a given temperature.
        Call the SinglefieldInstanton class of cosmoTransitions.tunneling1D.
        Notice that by default that class assumes 3d space.
        This is true for high-temperature, but not for low temperature.
        So this method is not applicable for very low T quantum tunneling, e.g. T=0.
        You want me to add a parameter for this? No.
        Return:
        - Profile. This contains the space distance r and the field value phi.
        - 3d Euclidean action S_3.
        """
        if self.Tc is None:
            self.findTc()
        assert T < self.Tc
        h_range = np.linspace(-2.0 * self.Tcvev, 2.0 * self.Tcvev, 1000)
        V_range = np.array([self.Vtot([h], T) for h in h_range])
        Vinter = interpolate.UnivariateSpline(h_range, V_range, s=0)
        tv = optimize.fmin(Vinter, self.Tcvev, disp=False)[0]
        gradV = Vinter.derivative()
        tv = optimize.fmin(self.Vtot, 100, args=(T,), disp=False)[0]

        tobj = td.SingleFieldInstanton(tv, 1e-16, Vinter, gradV)
        profile = tobj.findProfile()
        action = tobj.findAction(profile)

        return {"profile": profile, "action": action}

    def findTn(self, verbose=True):
        """
        Find the nucleation temperature.
        Remember the criteria is S_3/T = 140.0.
        Scan over temperatures, wait for the S_3/T to drops below 140.0 and stop there.
        Use the brentq function to solve the interpolated S_3/T as a function of T.
        """
        if self.Tc is None:
            self.findTc()

        print("Finding Tn...")

        Tmax = self.Tc - self.Teps
        data = []
        for i in range(0, 1000):
            Ttest = Tmax - i * self.Teps
            if verbose:
                print("Tunneling at T = " + str(Ttest))
            try:
                ST = self.tunneling_at_T(Ttest)["action"] / Ttest
                if verbose:
                    print("S3/T = " + str(ST))
                data.append([Ttest, ST])
            except:
                print("One exeption happens.")
            if ST < 140.0:
                break
        Tmin = Ttest
        if verbose:
            print(
                "Tnuc should be within " + str(Tmin) + " and " + str(Tmin + self.Teps)
            )
        data = np.array(data)
        Tdata = data[:, 0]
        S3Tdata = data[:, 1]
        S3Tfunc = interpolate.interp1d(
            Tdata, np.log10(S3Tdata), kind="cubic"
        )  # Use log data of S_3/T to interpolate.
        Tn = optimize.brentq(
            lambda T: S3Tfunc(T) - np.log10(140.0), Tmin, Tmin + self.Teps, disp=False
        )
        print("Tn = " + str(Tn))
        self.Tn = Tn
        Tnvev = optimize.fmin(self.Vtot, self.Tcvev, args=(Tn,), disp=False)[0]
        self.Tnvev = Tnvev
        self.strengthTn = Tnvev / Tn
        print("v_n/Tn = " + str(self.strengthTn))

    def findβH(self):
        """
        Calculate the β/H parameter.
        The derivative is computed using the Ridders algorithm.
        """
        if self.Tn is None:
            self.findTn()
        if self.βH is None:
            eps = self.Teps / 10
            T_list = [
                self.Tn - 2 * eps,
                self.Tn - eps,
                self.Tn + eps,
                self.Tn + 2 * eps,
            ]
            S3Tlist = []
            for T in T_list:
                S3Tlist.append(self.tunneling_at_T(T)["action"] / T)
            dS3T_dT = (S3Tlist[0] - 8 * S3Tlist[1] + 8 * S3Tlist[2] - S3Tlist[3]) / (
                12 * eps
            )
            self.βH = dS3T_dT * self.Tn
            print("β/H = " + str(self.βH))
        else:
            print("You already computed it before!")
            print("β/H = " + str(self.βH))

    def findα(self):
        """
        Calculate the α parameter.
        The derivative is computed using the Ridders algorithm.
        Notice that a factor 1/4 appears in the definition for the latent heat.
        """
        if self.Tn is None:
            self.findTn()
        if self.α is None:
            fv = 1e-16

            def ΔV(T):
                tv = optimize.fmin(self.Vtot, self.Tcvev, args=(T,), disp=False)
                return self.Vtot([fv], T) - self.Vtot(tv, T)

            dΔV_dT = df(ΔV, self.Tn, self.Teps / 10)
            ρrad = np.pi**2 * self.Tn**4 * 106.75 / 30
            latent = ΔV(self.Tn) - 0.25 * self.Tn * dΔV_dT
            self.α = latent / ρrad
            print("α = " + str(self.α))
        else:
            print("You already computed it before!")
            print("α = " + str(self.α))
