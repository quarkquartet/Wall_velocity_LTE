#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 17:10:18 2018

@author: Tong-Ou
"""

"""
In this module, we create a scaning over the parameter space (ms, ls, lsh) which enters the model 
defined and used in the module Z2_model.py

We write the parameter info and the 1st order phase transition info, including 
critical temperature, high low T vevs, and the phase transition strength to a csv file. 

Usage functions are created to print/plot the scaning. 
"""

import sys
import Z2_model as m
from scipy import linalg, interpolate, optimize
import random as ran
import numpy as np
import time
import os
import multiprocessing as mp
import csv

def physpara(m):
    vp = m.findMinimum(X=np.array([246, 0]), T=0.)
    v2phy = vp[0]**2.
    m2phys = m.d2V(vp, T=0.)
    m2physd, eigv = np.linalg.eig(m2phys)
    
    if all((m2physd[0] >= 0., m2physd[1] >= 0.)):
        m1phy, m2phy = m2physd**.5
    else:
        m1phy = 0.
        m2phy = 0.                    
        
    return [v2phy, m1phy, m2phy]
        
def scan(ms2, l2, lm, v2re, logfile):
    log = open(logfile, 'w')
    sys.stdout = log

    l1 = 0.129
    m12 = -l1 * 246**2
    m22 = ms2 - 0.5 * lm * 246**2
    print('parameters: ms2 = %s, ls = %s, lsh = %s' % (str(ms2), str(l2), str(lm)))
    mod = m.model(m12, m22, l1, l2, lm, v2re)
    phy = physpara(mod)
    vphy, m1phy, m2phy = phy[0]**.5, phy[1], phy[2]
    print('physical vacuum: v0 = %s, m1 = %s, m2 = %s' % (str(vphy), str(m1phy), str(m2phy)))

    output = [m12, m22, l1, l2, lm, v2re]
    fopttc = False
    sfopttn = False
    if abs(vphy - 246) < 2 and any((abs(m1phy - 125) < 2, abs(m2phy - 125) < 2)):
        """
        First calculate Tc.
        """
        mod.calcTcTrans()
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
        
        """
        Then calculate nucleation.
        """
        if fopttc:
            try:
                mod.findAllTransitions()
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
                mod.dSdT()
                mod.prettyPrintTnTrans()
                for tran in mod.TnTrans:
                    tn = tran['Tnuc']
                    lv = tran['low_vev']
                    hv = tran['high_vev']
                    sh = abs(lv[0] - hv[0])/tn
                    if tran['trantype'] == 1 and sh > 1:
                        rho_rad = np.pi**2 * tn**4 * 106.75 / 30
                        alpha_n = (tran['ep+'] - tran['ep-']) / rho_rad
                        betaH = tn * tran['dsdt']
                        output.extend([tn, lv[0], lv[1], hv[0], hv[1], sh, alpha_n, betaH])
                        sfopttn = True

    print(f"FOPT at Tc: {fopttc}")
    print(f"SFOPT at Tn: {sfopttn}")
    print(output)
    log.close()

    return output, fopttc, sfopttn

#==============================================================
          
if __name__ == '__main__':
    outpath = 'output/scan_3'
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    scan_paras = []
    for i in range(2000):
        ms2 = ran.uniform(62.5, 120) ** 2
        l2 = 1
        lmmin = 2 * ms2 / (246**2)
        lm = ran.uniform(lmmin, 1.2)
        v2re = 1000.**2
        logfile = f'{outpath}/log_{i}'
        scan_paras.append((ms2, l2, lm, v2re, logfile))

    with mp.Pool(processes = 6) as pool:
        results = pool.starmap(scan, scan_paras)

    outfile = outpath + '/Z2out.csv'
    with open(outfile, mode='w', newline='') as file:
        writer = csv.writer(file)
        for re in results:
            data, fopttc, sfopttn = re
            if fopttc and sfopttn:
                writer.writerow(data)

    tcfile = outpath + '/Z2TC.csv'
    with open(tcfile, mode='w', newline='') as file:
        writer = csv.writer(file)
        for re in results:
            data, fopttc, sfopttn = re
            if fopttc and not sfopttn:
                writer.writerow(data)

