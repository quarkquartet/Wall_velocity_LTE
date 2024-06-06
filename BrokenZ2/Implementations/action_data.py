#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:34:20 2019

@author: yik
"""

from __future__ import division


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 16:58:01 2019

@author: yik
"""

import baseMo_s_b_daisy_cw as bmb

import matplotlib.pyplot as plt

import transitionFinder_s as tf

import numpy as np
      
#m = bm.model( 1.08878195e+01  , 4.58813047e+00  , 1.60687673e-01, mh =  125., v2 = 246.**2., v2re = 1000.**2.)

d = np.load('0wvwnr_tn_all.npy')

action = []

def equal(a, b):   # apply to 1-d array

    a = np.array(a)

    #an = np.sum(a**2.)**.5

    b = np.array(b)

    #bn = np.sum(b**2.)**.5

    ab_d = np.sum((a-b)**2.)**.5

    re_d = np.sum(abs((a-b)/(a+b)))

    if any(( re_d <= .1, ab_d <= 1.)):

        return re_d, ab_d, True

    else:

        return re_d, ab_d, False            


def TnCrit(T, start_phase_id, end_phase_id, phitol=1e-8, overlapAngle=45.0, 
           nuclCriterion=lambda S,T: S/(T+1e-100) - 140.0,
           fullTunneling_params = {}, verbose =True, outdict={}):
    
    phases = m.phases.copy()
    
    start_phase = phases[start_phase_id]
    
    for i in range(len(phases)):
        if phases[i].key != end_phase_id:
            del phases[i]
    
    return tf._tunnelFromPhaseAtT(T, phases, start_phase, m.Vtot, m.gradV,
                        phitol, overlapAngle, nuclCriterion,
                        fullTunneling_params, verbose, outdict)



def gradAT(T, eps, start_phase_id, end_phase_id):
    
    dT = np.array([-2., -1., 1., 2.])*eps
    
    coef = np.array([1., -8., 8., -1. ])/(12.*eps)
    
    action = []
    
    for i in dT:
        action.append(TnCrit(T+i, start_phase_id, end_phase_id)+140.)
    
    action = np.array(action)
    
    return np.sum(action*coef)


for i in range(len(d)):
    pt = d[i][0][1][0]
    tran = d[i][0][1][3][0]
    cc = 0
    out = np.append(np.append(pt, d[i][0][0][1], axis = 0), [i], axis = 0)
    
    for k in tran:
        if not equal(k['low_vev'][0], 0.)[-1] and equal(k['high_vev'][0], 0.)[-1]:
            cc += 1
            ew = k
            
    if cc == 1:
        out = np.append(np.append(np.append(out, [ew['Tnuc']], axis = 0), ew['high_vev'], axis = 0), ew['low_vev'], axis = 0)
        #out[-4], out[-3] = ew['Tnuc'], (ew['low_vev'][0] - ew['high_vev'][0])/ew['Tnuc']
        try:
            m = bmb.model(pt[3], pt[4],pt[0], pt[1], pt[2],v2re = 1000.**2.)
            print 'new point'
            print i
            m.getPhases()
            crit = TnCrit(round(ew['Tnuc'], 5), ew['high_phase'], ew['low_phase'])
            if abs(crit) <= 1.:
                rate = gradAT(round(ew['Tnuc'], 5), 1e-3, ew['high_phase'], ew['low_phase'])
                out = np.append(out, [m.Vtot(m.phases[ew['high_phase']].X[0], m.phases[ew['high_phase']].T[0]), m.phases[ew['high_phase']].T[0]], axis=0)
                out = np.append(out, [m.Vtot(m.phases[ew['low_phase']].X[0], m.phases[ew['low_phase']].T[0]), m.phases[ew['low_phase']].T[0]], axis=0)
                out = np.append(out, [crit + 140., rate], axis=0)
                #out[-2] = crit + 140.
                #out[-1] = rate
                print 'success'
                print rate
                action.append(out)
                np.savetxt('0wvwnr_tn_all_action_e3_ext',action)
            else:
                out[-1], out[-2] = 1e+5, 1e+5
                print 'fail'
        except:
            out[-1], out[-2] = 1e+5, 1e+5
            print 'fail'
    
    
    
                
            
            

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
