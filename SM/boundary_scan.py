#!/usr/bin/env python
# coding: utf-8

# # Bubble wall velocity under LTE
import numpy as np
from numpy import linalg as la
from scipy import integrate
from scipy import optimize
from scipy import interpolate
import scipy as sp
from matplotlib import pyplot as plt
import plotter as pl
#import helperFunctions as hf
from helperFunctions import derivative, alpha_p, cs_sq, dYdtau, dvTdxi, r_func, μ, ω, p, e, s, find_vw_ds
import csv
import sys
import os
import multiprocessing as mp
from functools import partial

# =============================================================
# Preparation
Abs = np.abs
Log = np.log
Log10 = np.log10
Pi = np.pi
Sqrt = np.sqrt
Exp = np.exp
Cos = np.cos
Sin = np.sin
Sech = lambda x: 1/np.cosh(x)
Tanh = np.tanh
ArcSin = np.arcsin
ArcTanh = np.arctanh
Arg = np.angle
BesselK = sp.special.kv
Zeta = sp.special.zeta
HeavisideTheta = lambda x: np.heaviside(x, 0)

def Plot(fun, xminmax, n=100,xfun=np.linspace, xlog=False, ylog=False):
    xlist = xfun(xminmax[0], xminmax[1], n)
    ylist = [fun(x) for x in xlist]
    plt.plot(xlist, ylist)
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')

from helperFunctions import vJ
def test_vJ(vw, mod):
    Tnuc, hv, lv = mod.Tn, mod.hvev, mod.Tnvev
    Vtot = mod.Vtot
    gsol=optimize.fsolve(lambda x:match(vw, x[0], Tnuc, x[1], hv, lv, Vtot),[vw*0.9,Tnuc+2])
    return vJ(alpha_p(Vtot, Tnuc, gsol[1], hv, lv))

def match(vp,vm,Tp,Tm, high_vev, low_vev, Vtot):
    r = r_func(Vtot, Tp, Tm, high_vev, low_vev)
    αp = alpha_p(Vtot, Tp, Tm, high_vev, low_vev)
    vpvm = 1-(1-3*αp)*r
    vpvm = vpvm/(3-3*(1+αp)*r)
    ratio = 3 + (1-3*αp)*r
    ratio = ratio/(1+3*(1+αp)*r)
    return [vp*vm - vpvm, vp/vm - ratio]

def find_Tsh(Tm, vw, mod, Type='def'):
    Tnuc, hv, lv = mod.Tn, mod.hvev, mod.Tnvev
    Vtot = mod.Vtot
    if Type=='def':
        guess_sol = optimize.fsolve(lambda x:match(x[0], vw, x[1], Tm,hv, lv, Vtot),[vw*0.8,Tnuc])
    elif Type=='hyb':
        guess_sol = optimize.fsolve(lambda x:match(x[0], cs_sq(Vtot, Tm, lv)**0.5, x[1], Tm,hv, lv, Vtot),[0.5,Tnuc])

    # Integrate outside the wall to the shock-wave front
    try:
        vsol=integrate.solve_ivp(dYdtau, (10,0.01), np.array([μ(vw, guess_sol[0]), guess_sol[1], vw]),t_eval=np.linspace(10,0.01,1000),method='DOP853',args=(Vtot, hv))
        xi_max = vsol.y[2].max()
        xi_max_index = vsol.y[2].argmax()
        v_prof = interpolate.interp1d(vsol.y[2][0:xi_max_index+1],vsol.y[0][0:xi_max_index+1])
        T_prof = interpolate.interp1d(vsol.y[2][0:xi_max_index+1],vsol.y[1][0:xi_max_index+1])
        try:
            xsh=optimize.brentq(lambda x: μ(x, v_prof(x))*x - cs_sq(Vtot, T_prof(x), hv), vw, xi_max)
        except:
            xsh = xi_max
    except:
        vTsol = integrate.solve_ivp(dvTdxi, (vw, 1), np.array([μ(vw, guess_sol[0]), guess_sol[1]]), t_eval=np.linspace(vw, 1, 500), method='DOP853', args=(Vtot, hv))
        v_prof = interpolate.interp1d(vTsol.t, vTsol.y[0], kind='cubic')
        T_prof = interpolate.interp1d(vTsol.t, vTsol.y[1], kind='cubic')
        xsh = optimize.brentq(lambda x: μ(x, v_prof(x))*x - cs_sq(Vtot, T_prof(x), hv), vw, 1)

    def sh_match_2(Tshp):
        ap = alpha_p(Vtot, Tshp, T_prof(xsh), hv, hv)
        r = r_func(Vtot, Tshp, T_prof(xsh), hv, hv)
        vp = xsh
        vm = μ(xsh, v_prof(xsh))
        ratio = 3 + (1-3*ap)*r
        ratio = ratio/(1+3*(1+ap)*r)
        return vp/vm - ratio
    Tshp = optimize.newton(sh_match_2, T_prof(xsh))
    return Tshp

def det_bc(vp, Tp, vmguess, Tmguess, mod):
    hv, lv, Vtot = mod.hvev, mod.Tnvev, mod.Vtot
    vm,Tm =optimize.fsolve(lambda x:match(vp, x[0], Tp, x[1], hv, lv, Vtot), [vmguess, Tmguess])
    ds = s(vp, Tp) - s(vm, Tm)
    return vm, Tm, ds

def hyb_bc(vw, mod, Tm0):
    Vtot, Tnuc, hv, lv = mod.Vtot, mod.Tn, mod.hvev, mod.Tnvev
    Tm = optimize.newton(lambda T: find_Tsh(T, vw, mod, 'hyb') - Tnuc, Tm0)
    print("Tm: %s" % str(Tm))
    lv_new = optimize.fmin(Vtot, lv, args=(Tm,),disp=0)
    vm = cs_sq(Vtot, Tm, lv)**0.5
    vp, Tp = optimize.fsolve(lambda x:match(x[0], vm, x[1], Tm,hv, lv_new, Vtot),[0.5,Tm + 2])
    return vp, Tp, vm, Tm

# End of preparation
# ===================================================================
# Define scan function

def scan(i, pt_paras):
    mod = m.SM(pt_paras['g2'][i], pt_paras['lambda'][i], pt_paras['muh2'][i])
    Vtot = mod.Vtot
    mod.hvev = np.array([0.0])
    mod.Tnvev = np.array([pt_paras['Tnvev'][i]])
    mod.Tn = pt_paras['Tn'][i]
    alpha_n = pt_paras['alpha'][i]
    Tnuc, hv, lv = mod.Tn, mod.hvev, mod.Tnvev
    print("α: " + str(alpha_n))

    # First find vJ value (assuming detonation?)
    vwmax = 1.0
    eps = 0.005
    for j in range(1000):
        vw = vwmax - j * eps
        if test_vJ(vw, mod) > vw:
            vwmin = vw
            break

    vJvalue = optimize.brentq(lambda vw:test_vJ(vw, mod) - vw, vwmin, vwmin+eps, xtol=1e-6)
    print("vJ: " + str(vJvalue))
        
    # Solve for ds at vJ
    vm, Tm, dsJ = det_bc(vJvalue, Tnuc, vJvalue*0.7, Tnuc+2, mod)
    lv_new = optimize.fmin(Vtot, lv, args=(Tm,),disp=0)
    h0 = lv_new
    
    if dsJ < 0:
        # Solve for ds at wall velocity right below vJ
        vw = round(vJvalue, 2)
        if vw > vJvalue:
            vw -= 0.01

        Tm0list  = [Tnuc, Tnuc + 1, Tnuc - 1, Tnuc + 3, Tnuc - 3]
        vp, Tp, vm, Tm = None, None, None, None
        for Tm0 in Tm0list:
            try:
                vp, Tp, vm, Tm = hyb_bc(vw, mod, Tm0)
                break
            except:
                continue
        if vp is not None:
            if s(vp, Tp) - s(vm, Tm) < 0:
                print("Detonation")
                return True, alpha_n
            else:
                print("Deflagration/Hybrid")
                return False, alpha_n
        else:
            print("Undetermined")
            return None
    else:
        print("Deflagration/Hybrid")
        return False, alpha_n

# End of define scan function
# ==================================================================
import SM_model as m

if __name__ == '__main__':
    outpath = 'output'
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Read phase transition inputs from file
    keylist = ['g2', 'mH', 'lambda', 'muh2', 'Tc', 'Tcvev', 'strengthTc', 'Tn', 'Tnvev', 'strengthTn', 'alpha', 'betaH']
    pt_paras = {}
    for key in keylist:
        pt_paras[key] = []

    csv_file = 'data/SMout.csv'  # Replace 'data.csv' with your actual file path
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)

        for row in reader:
            for ind, key in enumerate(keylist):
                pt_paras[key].append(float(row[ind]))

    inds = []
    udlist = [0.0165006805737692]
    for udalpha in udlist:
        for ind, x in enumerate(pt_paras['alpha']):
            if abs(x - udalpha) < 1e-7:
                inds.append(ind)
    print(inds)

    ncpu = 6
    #inds = range(len(pt_paras['g2']))
    alpha_det, alpha_def = [], []
    scan_p = partial(scan, pt_paras=pt_paras)
    with mp.Pool(processes=ncpu) as pool:
        results = pool.map(scan_p, inds)

    for result in results:
        if result is not None:
            if result[0]:
                alpha_det.append(result[1])
            else:
                alpha_def.append(result[1])

    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(10, 4), dpi=200)
    axs[0].hist(alpha_def, 20, histtype='stepfilled', facecolor='b', alpha=0.75)
    axs[1].hist(alpha_det, 20, histtype='stepfilled', facecolor='orange', alpha=0.75)
    axs[0].set_xlabel(r'$\alpha_n$')
    axs[1].set_xlabel(r'$\alpha_n$')
    axs[0].set_title('Deflagration/Hybrid')
    axs[1].set_title('Detonation')
    plt.savefig(outpath + '/alpha_hist.png')

