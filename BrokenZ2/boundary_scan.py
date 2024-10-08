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
def test_vJ(vw, mod, include_rad):
    Tnuc, hv, lv = mod.Tn, mod.hvev, mod.Tnvev
    Vtot = partial(mod.Vtot, include_radiation=include_rad)
    T0list = [Tnuc+2, Tnuc+1, Tnuc]
    for T0 in T0list:
        gsol=optimize.fsolve(lambda x:match(vw, x[0], Tnuc, x[1], hv, lv, Vtot),[vw*0.9,T0])
        alphap = alpha_p(Vtot, Tnuc, gsol[1], hv, lv)
        if alphap >= 0:
            return vJ(alphap)
        else:
            continue
    print('Negative α+!')
    return None

def match(vp,vm,Tp,Tm, high_vev, low_vev, Vtot):
    r = r_func(Vtot, Tp, Tm, high_vev, low_vev)
    αp = alpha_p(Vtot, Tp, Tm, high_vev, low_vev)
    vpvm = 1-(1-3*αp)*r
    vpvm = vpvm/(3-3*(1+αp)*r)
    ratio = 3 + (1-3*αp)*r
    ratio = ratio/(1+3*(1+αp)*r)
    return [vp*vm - vpvm, vp/vm - ratio]

def find_Tsh(Tm, vw, mod, include_rad, Type='def'):
    Tnuc, hv, lv = mod.Tn, mod.hvev, mod.Tnvev
    Vtot = partial(mod.Vtot, include_radiation=include_rad)
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

def det_bc(vp, Tp, vmguess, Tmguess, mod, include_rad):
    hv, lv, Vtot = mod.hvev, mod.Tnvev, partial(mod.Vtot, include_radiation=include_rad)
    vm,Tm =optimize.fsolve(lambda x:match(vp, x[0], Tp, x[1], hv, lv, Vtot), [vmguess, Tmguess])
    try:
        ds = s(vp, Tp) - s(vm, Tm)
    except:
        print('Invalid ds')
        return vm, Tm, None
    return vm, Tm, ds

def hyb_bc(vw, mod, Tm0, include_rad):
    Vtot, Tnuc, hv, lv = partial(mod.Vtot, include_radiation=include_rad), mod.Tn, mod.hvev, mod.Tnvev
    Tm = optimize.newton(lambda T: find_Tsh(T, vw, mod, include_rad, 'hyb') - Tnuc, Tm0)
    print("Tm: %s" % str(Tm))
    lv_new = optimize.fmin(Vtot, lv, args=(Tm,),disp=0)
    vm = cs_sq(Vtot, Tm, lv)**0.5
    vp, Tp = optimize.fsolve(lambda x:match(x[0], vm, x[1], Tm,hv, lv_new, Vtot),[0.5,Tm + 2])
    return vp, Tp, vm, Tm

def sdiff(vw, mod, include_rad):
    Vtot, Tnuc, hv, lv = partial(mod.Vtot, include_radiation=include_rad), mod.Tn, mod.hvev, mod.Tnvev
    Tm0list = [Tnuc, Tnuc+1, Tnuc-1, Tnuc+3, Tnuc-3]
    for Tm0 in Tm0list:
        try:
            Tm = optimize.newton(lambda T: find_Tsh(T, vw, mod, include_rad, 'hyb') - Tnuc, Tm0)
            break
        except:
            continue
    lv_new = mod.findMinimum(lv, Tm)
    vm = cs_sq(Vtot, Tm, lv_new)**.5
    if vm <= vw:
        print('Hybrid')
        vp, Tp = optimize.fsolve(lambda x:match(x[0], vm, x[1], Tm,hv, lv_new, Vtot),[0.5, Tm+2])
    else:
        print('Deflagration')
        for Tm0 in Tm0list:
            try:
                Tm = optimize.newton(lambda T: find_Tsh(T, vw, mod, include_rad, 'def') - Tnuc, Tm0)
                break
            except:
                continue
        lv_new = mod.findMinimum(lv, Tm)
        vm = vw
        vp, Tp = optimize.fsolve(lambda x:match(x[0], vm, x[1], Tm,hv, lv_new, Vtot),[0.3, Tnuc])
    return s(vp, Tp) - s(vm, Tm)

# End of preparation
# ===================================================================
# Define scan function

def scan(i, logfile, pt_paras, include_rad=True, solve_vw=False):
    log = open(logfile, 'w')
    sys.stdout = log
    outdir = logfile.split('/log')[0]

    mod = m.model(pt_paras['m12'][i], pt_paras['m22'][i], pt_paras['l1'][i], pt_paras['l2'][i], pt_paras['lm'][i], 1000.**2)
    Vtot = partial(mod.Vtot, include_radiation=include_rad)
    mod.hvev = np.array([pt_paras['hvtn1'][i], pt_paras['hvtn2'][i]])
    mod.Tnvev = np.array([pt_paras['lvtn1'][i], pt_paras['lvtn2'][i]])
    mod.Tn = pt_paras['Tn'][i]
    Tnuc, hv, lv = mod.Tn, mod.hvev, mod.Tnvev
    alpha_n = pt_paras['alpha'][i]
    print("α: " + str(alpha_n))

    # Fist find vJ value (assuming detonation?)
    vwmax = 1.0
    eps = 0.001
    for j in range(1000):
        vw = vwmax - j * eps
        testvj = test_vJ(vw, mod, include_rad)
        if testvj is not None:
            if testvj > vw:
                vwmin = vw
                break
        else:
            print('Undetermined')
            log.close()
            return None

    if vwmin < 0:
        print('vJ not found')
        print('Undetermined')
        log.close()
        return None

    print('vwmin: ' + str(vwmin))
    vJvalue = optimize.brentq(lambda vw:test_vJ(vw, mod, include_rad) - vw, vwmin, vwmin+eps, xtol=1e-6)
    print("vJ: " + str(vJvalue))
        
    # Solve for ds at vJ
    '''
    vm, Tm, dsJ = det_bc(vJvalue, Tnuc, vJvalue*0.7, Tnuc+2, mod, include_rad)
    lv_new = optimize.fmin(Vtot, lv, args=(Tm,),disp=0)
    h0 = lv_new
    '''

    # Solve for ds at right below vJ
    vw = 0.9995 * vJvalue
    Tm0list  = [Tnuc, Tnuc + 1, Tnuc - 1, Tnuc + 3, Tnuc - 3]
    vp, Tp, vm, Tm = None, None, None, None
    
    for Tm0 in Tm0list:
        try:
            vp, Tp, vm, Tm = hyb_bc(vw, mod, Tm0, include_rad)
            break
        except:
            continue

    if vp is not None:
        if s(vp, Tp) - s(vm, Tm) < 0:
            print("Detonation")
            log.close()
            return True, i
        else:
            print("Deflagration/Hybrid")
            if solve_vw:
                try:
                    vwlist = np.linspace(0.1, vw, 10)
                    dslist = np.zeros(len(vwlist))
                    for ind, vwi in enumerate(vwlist):
                        ds = sdiff(vwi, mod, include_rad)
                        print(f'ds at vw={vwi} is {ds}')
                        dslist[ind] = ds

                    dsfunc = interpolate.interp1d(vwlist, dslist, kind='cubic')
                    fig, ax = plt.subplots(1, 1, figsize=(6,4), dpi=200)
                    ax.plot(vwlist, dsfunc(vwlist))
                    ax.set_xlabel(r'$v_w$')
                    ax.set_ylabel(r'$\Delta(s\gamma v)$')
                    fig.savefig(f'{outdir}/ds_vw_{i}.png')

                    vwreal = optimize.newton(dsfunc, 0.5)
                    print(f'Solution for vw: {vwreal}')
                    log.close()
                    return False, i, vwreal
                except:
                    print('Failed to find vw.')
                    log.close()
                    return False, i, None
            else:
                log.close()
                return False, i
    else:
        print("Undetermined")
        log.close()
        return None

def scatter(ax, pt_paras, p1, p2, ind_det, ind_def):
    p1det, p2det, p1def, p2def = [], [], [], []
    for i in ind_det:
        p1det.append(pt_paras[p1][i])
        p2det.append(pt_paras[p2][i])

    for j in ind_def:
        p1def.append(pt_paras[p1][j])
        p2def.append(pt_paras[p2][j])
    
    ax.scatter(p1def, p2def, c='b', alpha=0.75, s=10, label='Deflagration/Hybrid')
    ax.scatter(p1det, p2det, c='orange', alpha=0.75, s=10, label='Detonation')
    ax.legend(loc = 'lower right') 

# End of define scan function
# ==================================================================
import Z2sb_model as m
import json

if __name__ == '__main__':
    pttag = '0wvwr'
    outpath = f'output/vw_{pttag}'
    include_rad = True
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Read phase transition inputs from PT input file
    
    keylist = ['l1', 'l2', 'lm', 'm12', 'm22', 'Tn', 'hvtn1', 'hvtn2', 'lvtn1', 'lvtn2', 'dsdt']
    keyinds = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, -1]
    pt_paras = {}
    for key in keylist:
        pt_paras[key] = []

    files = [f'input/Z2_model_data_file/0wvw_tn_all_action_e3_ext_t_all_new_V/{pttag}_tn_all_action_e3_ext_tcwd_new_V']
    for filename in files:
        with open(filename, mode='r', newline='') as file:
            lines = file.readlines()

            for line in lines:
                row = line.split()
                for i, key in enumerate(keylist):
                    ind = keyinds[i]
                    pt_paras[key].append(float(row[ind]))

    # Calculate alpha and beta/H
    pt_paras['alpha'] = []
    pt_paras['betaH'] = []
    for i in range(len(pt_paras['l1'])):
        mod = m.model(pt_paras['m12'][i], pt_paras['m22'][i], pt_paras['l1'][i], pt_paras['l2'][i], pt_paras['lm'][i], 1000.**2, silent=True)
        Tn = pt_paras['Tn'][i]
        Vtot = mod.Vtot
        hv = np.array([pt_paras['hvtn1'][i], pt_paras['hvtn2'][i]])
        lv = np.array([pt_paras['lvtn1'][i], pt_paras['lvtn2'][i]])
        arad = np.pi**2 * 106.75 / 30
        delta_epmod = mod.epsilon(hv, Tn) - mod.epsilon(lv, Tn)
        alpha = delta_epmod / (arad*Tn**4)
        pt_paras['alpha'].append(alpha)
        pt_paras['betaH'].append(Tn*pt_paras['dsdt'][i])

    # Read phase transition inputs from PT output file
    '''
    keylist = ['m12', 'm22', 'l1', 'l2', 'lm', 'v2re', 'strengthTc', 'Tn', 'lvtn1', 'lvtn2', 'hvtn1', 'hvtn2', 'strengthTn', 'alpha', 'betaH']
    keyinds = [0, 1, 2, 3, 4, 5, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    pt_paras = {}
    for key in keylist:
        pt_paras[key] = []

    dirs = ['output/scan_0wvwnr']
    for d in dirs:
        csv_file = d + '/Z2out.csv'  # Replace 'data.csv' with your actual file path
        with open(csv_file, mode='r', newline='') as file:
            reader = csv.reader(file)

            for row in reader:
                for i, key in enumerate(keylist):
                    ind = keyinds[i]
                    pt_paras[key].append(float(row[ind]))
    '''
    inds = range(10) #range(len(pt_paras['m12'])) 
    scan_paras = []
    for i in inds:
        scan_paras.append((i, f'{outpath}/log_{i}'))

    ncpu = 6
    ind_det, ind_def = [], []
    vwsol = []
    solve_vw = True
    scan_p = partial(scan, pt_paras=pt_paras, include_rad = include_rad, solve_vw=solve_vw)
    with mp.Pool(processes=ncpu) as pool:
        results = pool.starmap(scan_p, scan_paras)

    
    for result in results:
        if result is not None:
            if result[0]:
                ind_det.append(result[1])
            else:
                if solve_vw:
                    if result[2] is not None:
                        ind_def.append(result[1])
                        vwsol.append(result[2])
                else:
                    ind_def.append(result[1])

    vwdata={}
    vwdata['ind_def'] = ind_def
    vwdata['ind_det'] = ind_det
    vwdata['alpha_def'] = [pt_paras['alpha'][n] for n in ind_def]
    vwdata['alpha_det'] = [pt_paras['alpha'][n] for n in ind_det]
    vwdata['betaH_def'] = [pt_paras['betaH'][n] for n in ind_def]
    vwdata['betaH_det'] = [pt_paras['betaH'][n] for n in ind_det]
    if solve_vw:
        vwdata['vw_def'] = vwsol
    outfile = outpath + f'/GWinput_{pttag}.json'
    with open(outfile, 'w') as f:
        json.dump(vwdata, f)
