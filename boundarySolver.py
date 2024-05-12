"""
This boundarySolver module is used to calculate the
boundary conditions of the fluid equations around
the bubble wall.
This includes:
- For detonation case, T+ = Tn, v+ = vw. Solve for T- and v-.
- For deflagration case, v- = vw, Tsh+ = Tn. Solve for T-, T+ and v+.
- For hybrid case, v- = cs-, Tsh+ = Tn. Solve for T-, T+ and v+.
"""

import numpy as np
from scipy import integrate, interpolate, optimize

from helperFunctions import a, alpha_p, cs_sq, dYdtau, dvTdxi, epsilon, r_func, μ


class boundary:
    def __init__(self, V, Tn, high_vev, low_vev, vw):
        self.V = V
        self.Tn = Tn
        self.hv = high_vev
        self.lv = low_vev
        self.vw = vw
        self.vp = None
        self.Tp = None
        self.vm = None
        self.Tm = None
        self.type = None

    def match(self, vp, vm, Tp, Tm):
        r = r_func(self.V, Tp, Tm, self.hv, self.lv)
        αp = alpha_p(self.V, Tp, Tm, self.hv, self.lv)
        vpvm = 1 - (1 - 3 * αp) * r
        vpvm = vpvm / (3 - 3 * (1 + αp) * r)
        ratio = 3 + (1 - 3 * αp) * r
        ratio = ratio / (1 + 3 * (1 + αp) * r)
        return [vp * vm - vpvm, vp / vm - ratio]

    def guess_det(self, vp, Tp, ap, am, εp, εm):
        αp = (εp - εm) / (ap * Tp**4)
        r = 1 / (1 + 3 * αp)
        Tm = (ap * Tp**4 / (am * r)) ** 0.25
        vpvm = 1 - (1 - 3 * αp) * r
        vpvm = vpvm / (3 - 3 * (1 + αp) * r)
        vm = min(vpvm / vp, 0.67)
        return [vm, Tm]

    def solve_detonation(self):
        """Solve the boundary condition for detonation case.
        Solve T-, v- from T+, v+.
        Return: v-, T-.
        """
        guess = self.guess_det(
            self.vw,
            self.Tn,
            a(self.V, self.Tn, self.hv),
            a(self.V, self.Tn * 1.1, self.lv),
            epsilon(self.V, self.Tn, self.lv),
        )
        vm, Tm = optimize.fsolve(
            lambda x: self.match(self.vw, x[0], self.Tn, x[1]), guess, xtol=1e-10
        )
        return np.array([vm, Tm])

    def find_Tsh_deflagration(self, Tm):
        """Find the Tsh for deflagration case for any given Tm.
        Will be used for shooting method.
        """
        guess_sol = optimize.fsolve(
            lambda x: self.match(x[0], self.vw, x[1], Tm), [0.1, self.Tn]
        )  # Guess a T-, solve for v+ and T+
        Tp = guess_sol[1]
        vp = guess_sol[0]

        try:
            profile_sol = integrate.solve_ivp(
                dYdtau,
                (10, 0.01),
                np.array([μ(self.vw, vp), Tp, self.vw]),
                t_eval=np.linspace(10, 0.01, 1000),
                method="DOP853",
                args=(self.V, self.hv),
            )  # Solve the differential equations of v, T and xi.
            vsol = profile_sol.y[0]
            Tsol = profile_sol.y[1]
            xisol = profile_sol.y[2]
            xi_max = xisol.max()
            xi_max_index = xisol.argmax()
            v_prof = interpolate.interp1d(
                xisol[0 : xi_max_index + 1], vsol[0 : xi_max_index + 1]
            )
            T_prof = interpolate.interp1d(
                xisol[0 : xi_max_index + 1], Tsol[0 : xi_max_index + 1]
            )
            xsh = optimize.brentq(
                lambda x: μ(x, v_prof(x)) * x - cs_sq(self.V, T_prof(x), self.hv),
                xisol[0],
                xi_max * 0.9999,
            )
        except:
            profile_sol = integrate.solve_ivp(
                dvTdxi,
                (self.vw, 1),
                np.array([μ(self.vw, vp), Tp]),
                t_eval = np.linspace(self.vw, 1, 500),
                method = 'DOP853',
                args=(self.V, self.hv),
            )
            xisol = profile_sol.t
            vsol = profile_sol.y[0]
            Tsol = profile_sol.y[1]
            v_prof = interpolate.interp1d(xisol, vsol, kind='cubic')
            T_prof = interpolate.interp1d(xisol, Tsol, kind='cubic')
            xsh = optimize.brentq(lambda x: μ(x, v_prof(x))*x - cs_sq(self.V, T_prof(x), self.hv), self.vw, 1)
        return T_prof(xsh)

    def solve_deflagration(self):
        """Solve the boundary condition for deflagration case."""
        Tmax = self.Tn * 0.999
        for i in range(1000):
            # Let's find the rough range of the result at first.
            # Start from Tmax, the T at xi_sh should be too high.
            Ti = Tmax - i * 0.2
            if self.find_Tsh_deflagration(Ti) < self.Tn - 0.1:
                break
        Tmax = Ti + 0.2
        Tmin = Ti
        for i in range(50):
            # Binormial solution
            Tcal = (Tmax + Tmin) / 2
            Tsh = self.find_Tsh_deflagration(Tcal)
            if Tsh < self.Tn:
                Tmin = Tcal
            else:
                Tmax = Tcal

        # Now we have the correct T-. Solve for T+ and v+ now.
        Tm = Tcal
        bound_sol = optimize.fsolve(
            lambda x: self.match(x[0], self.vw, x[1], Tm), [0.1, self.Tn]
        )

        vp, Tp = bound_sol

        return np.array([vp, Tp])

    def __run_solution(self):
        # TODO: logic structure to check consistency and return result.