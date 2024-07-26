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

def scan(i, logfile, pt_paras):
    log = open(logfile, 'w')
    sys.stdout = log

    mod = m.model(pt_paras['m12'][i], pt_paras['m22'][i], pt_paras['l1'][i], pt_paras['l2'][i], pt_paras['lm'][i], pt_paras['v2re'][i])
    Vtot = mod.Vtot
    mod.hvev = np.array([pt_paras['hvtn1'][i], pt_paras['hvtn2'][i]])
    mod.Tnvev = np.array([pt_paras['lvtn1'][i], pt_paras['lvtn2'][i]])
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
        vw = round(vJvalue, 3)
        if vw > vJvalue:
            vw -= 0.001

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
                log.close()
                return True, alpha_n, i
            else:
                print("Deflagration/Hybrid")
                log.close()
                return False, alpha_n, i
        else:
            print("Undetermined")
            log.close()
            return None
    else:
        print("Deflagration/Hybrid")
        log.close()
        return False, alpha_n, i

def scatter(ax, pt_paras, p1, p2, ind_det, ind_def):
    p1det, p2det, p1def, p2def = [], [], [], []
    for i in ind_det:
        p1det.append(pt_paras[p1][i])
        p2det.append(pt_paras[p2][i])

    for j in ind_def:
        p1def.append(pt_paras[p1][j])
        p2def.append(pt_paras[p2][j])
    
    ax.scatter(p1def, p2def, c='b', alpha=0.75, label='Deflagration/Hybrid')
    ax.scatter(p1det, p2det, c='orange', alpha=0.75, label='Detonation')
    ax.legend(loc = 'lower right') 

# End of define scan function
# ==================================================================
import Z2_model as m

if __name__ == '__main__':
    outpath = 'output/vw'
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Read phase transition inputs from file
    keylist = ['m12', 'm22', 'l1', 'l2', 'lm', 'v2re', 'Tc', 'lvtc1', 'lvtc2', 'hvtc1', 'hvtc2', 'strengthTc', 'Tn', 'lvtn1', 'lvtn2', 'hvtn1', 'hvtn2', 'strengthTn', 'alpha', 'betaH']
    pt_paras = {}
    for key in keylist:
        pt_paras[key] = []

    dirs = ['output/scan_0', 'output/scan_1', 'output/scan_2']
    for d in dirs:
        csv_file = d + '/Z2out.csv'  # Replace 'data.csv' with your actual file path
        with open(csv_file, mode='r', newline='') as file:
            reader = csv.reader(file)

            for row in reader:
                for ind, key in enumerate(keylist):
                    pt_paras[key].append(float(row[ind]))

    inds = range(len(pt_paras['m12'])) 
    pt_paras['ms'] = []
    scan_paras = []
    for i in inds:
        m22 = pt_paras['m22'][i]
        lm = pt_paras['lm'][i]
        v0 = 246.
        pt_paras['ms'].append(Sqrt(m22 + 0.5 * lm * v0**2))
        scan_paras.append((i, f'{outpath}/log_{i}'))

    ncpu = 6
    alpha_det, alpha_def = [], []
    ind_det, ind_def = [], []
    scan_p = partial(scan, pt_paras=pt_paras)
    with mp.Pool(processes=ncpu) as pool:
        results = pool.starmap(scan_p, scan_paras)

    for result in results:
        if result is not None:
            if result[0]:
                alpha_det.append(result[1])
                ind_det.append(result[2])
            else:
                alpha_def.append(result[1])
                ind_def.append(result[2])

    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(10, 4), dpi=200)
    axs[0].hist(alpha_def, 20, histtype='stepfilled', facecolor='b', alpha=0.75)
    axs[1].hist(alpha_det, 20, histtype='stepfilled', facecolor='orange', alpha=0.75)
    axs[0].set_xlabel(r'$\alpha_n$')
    axs[1].set_xlabel(r'$\alpha_n$')
    axs[0].set_title('Deflagration/Hybrid')
    axs[1].set_title('Detonation')
    plt.savefig(outpath + '/alpha_hist.png')

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=200)
    scatter(ax, pt_paras, 'ms', 'lm', ind_det, ind_def)
    ax.set_xlabel(r'$m_s(GeV)$')
    ax.set_ylabel(r'$\lambda_{sh}$')
    plt.savefig(outpath + '/ms_lsh.png')

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=200)
    ax.scatter(pt_paras['strengthTn'], pt_paras['alpha'])
    ax.set_xlabel(r'$v_n/T_n$')
    ax.set_ylabel(r'$\alpha_n$')
    plt.savefig(outpath + '/vnTn_alpha.png')
