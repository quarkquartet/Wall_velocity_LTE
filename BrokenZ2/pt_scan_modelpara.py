#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 17:10:18 2018

@author: Tong-Ou
"""

"""
In this module, we create a scaning over the parameter space (m12, m22, l1, l2, lm) which enters the model 
defined and used in the module Z2_model.py

We write the parameter info and the 1st order phase transition info, including 
critical temperature, high low T vevs, and the phase transition strength to a csv file. 

Usage functions are created to print/plot the scaning. 
"""

import sys
from matplotlib import pyplot as plt
import Z2sb_model as m
from scipy import linalg, interpolate, optimize
import random as ran
import numpy as np
import time
import os
import multiprocessing as mp
import csv

def physpara(m, tanb=0.5):
    zeroTmin = [np.array([246, tanb*246]), np.array([0, 0]), np.array([0, tanb*246]), np.array([246, 0])]
    vp = m.findMinimum(zeroTmin[0], T=0.)
    vphy = vp[0]
    tanbphy = vp[1] / vp[0]
    if tanbphy < 0.001:
        print('Not Z2 broken')
        return None
    m2phys = m.d2V(vp, T=0.)
    m2physd, eigv = np.linalg.eig(m2phys)
    if all((m2physd[0] >= 0., m2physd[1] >= 0.)):
        m1phy, m2phy = m2physd**.5
        sintphy = eigv[0][1]
    else:
        print('Z2-broken minimum is not a minimum')
        return None

    if abs(vphy - 246) < 2 and any((abs(m1phy - 125) < 2, abs(m2phy - 125) < 2)):
        for i in range(1, 4):
            vtest = m.findMinimum(X=zeroTmin[i], T=0.)
            if ((abs(vtest[0])-abs(vp[0]))**2 + (abs(vtest[1])-abs(vp[1]))**2)**.5 > 2:
                if m.Vtot(vtest, T=0.0) < m.Vtot(vp, T=0.0):
                    print(f'Minimum at h={vtest[0]}, s={vtest[1]} is deeper than physical minimum.')
                    return None
    else:
        print('Z2-broken minimum is not Higgs minimum.')
        return None
       
    return vp, m1phy, m2phy, sintphy, tanbphy
        
def scan(m12, m22, l1, l2, lm, v2re, logfile):
    log = open(logfile, 'w')
    sys.stdout = log

    mod = m.model(m12, m22, l1, l2, lm, v2re)
    phy = physpara(mod)
    try:
        vp, m1phy, m2phy, sintphy, tanbphy = phy[0], phy[1], phy[2], phy[3], phy[4]
    except:
        return None
    print(f'physical vacuum: v0 = {vp}, m1 = {m1phy}, m2 = {m2phy}, sint = {sintphy}, tanb = {tanbphy}')

    output = [m12, m22, l1, l2, lm, v2re, m1phy, m2phy, sintphy, tanbphy]
    fopttc = False
    sfopttn = False

    """
    First calculate Tc.
    """
    try:
        mod.calcTcTrans(vphy=vp)
        trans = mod.TcTrans
        mod.prettyPrintTcTrans()
        for k in range(len(trans)):
            tc = trans[k]['Tcrit']
            lv = trans[k]['low_vev']
            hv = trans[k]['high_vev']
            sh = abs(lv[0]-hv[0])/tc
            if trans[k]['trantype'] == 1 and sh > 0.5: 
                output.extend([tc, lv[0], lv[1], hv[0], hv[1], sh])
                fopttc = True
                break
    except:
        print("Failed to find Tc.")
        return output, fopttc, sfopttn
    
    """
    Then calculate nucleation.
    """
    if fopttc:
        try:
            mod.findAllTransitions(vphy=vp)
        except KeyError as err:
            print('Skipping due to error: %s' % err)
            return output, fopttc, sfopttn
        except ValueError as err:
            print('Skipping due to error: %s' % err)
            return output, fopttc, sfopttn
        except:
            print('Skipping due to other errors')
            return output, fopttc, sfopttn

        if mod.TnTrans is not None:
            mod.dSdT(eps=0.01)
            mod.prettyPrintTnTrans()
            for tran in mod.TnTrans:
                tn = tran['Tnuc']
                lv = tran['low_vev']
                hv = tran['high_vev']
                sh = abs(lv[0] - hv[0])/tn
                if tran['trantype'] == 1 and sh > 0.8:
                    rho_rad = np.pi**2 * tn**4 * 106.75 / 30
                    alpha_n = (tran['ep+'] - tran['ep-']) / rho_rad
                    betaH = tn * tran['dsdt']
                    output.extend([tn, lv[0], lv[1], hv[0], hv[1], sh, alpha_n, betaH])
                    sfopttn = True

    print(f"FOPT at Tc: {fopttc}")
    print(f"FOPT at Tn: {sfopttn}")
    print(output)
    log.close()

    return output, fopttc, sfopttn

#==============================================================
          
if __name__ == '__main__':
    outpath = 'output/scan_00vw'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    scan_paras = []
    '''
    for i in range(100000):
        m12log = ran.uniform(0., 4.)
        m22log = ran.uniform(0., 4.)
        l1 = ran.uniform(0.05, 0.13)
        l2 = ran.uniform(0., 2.)
        lm = ran.uniform(-2., 2.)
        m12 = 10.**m12log
        m22 = -10.**m22log
        v2re = 1000.**2
        logfile = f'{outpath}/log_{i}'
        scan_paras.append((m12, m22, l1, l2, lm, v2re, logfile))
    '''
    
    data_file = 'input/Z2_model_data_file/0wvw_tn_all_action_e3_ext_t_all_new_V/00vw_tn_all_action_e3_ext_tcwd_new_V'
    with open(data_file, mode='r', newline='') as file:
        lines = file.readlines()
        v2re = 1000.**2
        for i, line in enumerate(lines):
            logfile = f'{outpath}/log_{i}'
            row = line.split()
            l1 = float(row[0])
            l2 = float(row[1])
            lm = float(row[2])
            m12 = float(row[3])
            m22 = float(row[4])
            scan_paras.append((m12, m22, l1, l2, lm, v2re, logfile))
    
    with mp.Pool(processes = 6) as pool:
        results = pool.starmap(scan, scan_paras)

    outfile = outpath + '/Z2out.csv'
    with open(outfile, mode='w', newline='') as file:
        writer = csv.writer(file)
        for re in results:
            if re is not None:
                data, fopttc, sfopttn = re
                if fopttc and sfopttn:
                    writer.writerow(data)

    tcfile = outpath + '/Z2TC.csv'
    with open(tcfile, mode='w', newline='') as file:
        writer = csv.writer(file)
        for re in results:
            if re is not None:
                data, fopttc, sfopttn = re
                if fopttc and not sfopttn:
                    writer.writerow(data)

