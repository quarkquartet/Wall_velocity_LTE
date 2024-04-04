#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 17:10:18 2018

@author: yik
"""

"""
In this module, we create a scaning over the parameter space (ms, sint, tanb) which enters the model 
defined and used in the module baseMo.py

We create a dictionary containing the parameter info and the 1st order phase transition info, including 
critical temperature, high low T vevs, and the phase transition strength. 

Usage functions are created to print/plot the scaning. 
"""

import sys
#sys.path.append('/home/tong/Chicago/EWPhT/cosmotransition_z2sb/cosmoTransitions/')

#import baseMo_s_b_cw as bmcw

import baseMo_s_b_d as bm
from scipy import linalg, interpolate, optimize


import random as ran

import numpy as np

import time

import os


import mpi4py.MPI as mpi

import math

comm = mpi.COMM_WORLD

FILE = sys.argv[1]

#BASE_PATH = '/home/yikwang/ewpht_data/'

#DATA_PATH = '/data/yikwang/'

l1r = [0.05, 0.5]

l2r = [0.,0.1]

lmr = [-0.2, 0.2]

m12logr = [0.,4.]

m22logr = [0.,4.]

#m12r = [-100000., 100000.]

#m22r = [-10000., 0.]

npt = int(sys.argv[2])

scan = []

BASE_PATH = os.getcwd() #'/home/tong/Chicago/EWPhT/cosmotransition_z2sb/Implementations/'

def physpara(m):
    
    vp = m.findMinimum(T=0.)
    
    v2phy = vp[0]**2.
    
    tanbphy = vp[1]/vp[0]
    
    m2phys = m.d2V(vp, T=0.)
        
    m2physd, eigv = np.linalg.eig(m2phys)
    
    if all((m2physd[0] >= 0., m2physd[1] >= 0.)):
                      
        sintphy = eigv[0][1]
           
        m1phy, m2phy = m2physd**.5
    
    else:
        
        sintphy = 0.
                
        m1phy = 0.
        
        m2phy = 0.                    
        
    return [v2phy, tanbphy, m1phy, m2phy, sintphy]
        
        

def trans(i, m):

    print "\n check point %s with \n" %[i, m.l1, m.l2, m.lm, m.m12, m.m22] 
    
    m.calcTcTrans() 

    trans = m.TcTrans
    
    check = False
    
    sfoph = []
    
    for k in range(len(trans)):
        tc = trans[k]['Tcrit']
        sh = abs(trans[k]['low_vev'][0]-trans[k]['high_vev'][0])/tc
        if trans[k]['trantype'] == 1 and sh >= 1.: 
            sfoph.append([tc, sh])      
               
    for key in m.phases:
        if m.phases[key].check:               
            check = True
    
    return trans, sfoph, check
    
def t0vev(m):
    
    wvev = m.Vtot(m.findMinimum(T=0.),T=0.) - m.Vtot(m.findMinimum(X=[0.,0.],T=0.),T=0.) > 1.
    
    return wvev


def thvev(m):
    
    htX =  m.findMinimum(T=1000.)
    
    wvev = (abs(htX[...,0]) > 10.**10.) or (abs(htX[...,1]) > 10.**10.)
    
    return wvev

def solve_mu(l1, l2, lm):
    def func(x, l1, l2, lm):
        muh2 = x[0]
        mus2 = x[1]
        vphy2 =  2*(2*l2*muh2+lm*mus2)/(4*l1*l2-lm**2)
        wphy2 = 2*(-2*l1*mus2-lm*muh2)/(4*l1*l2-lm**2)
        veq = vphy2 - 246.**2
        meq = 0.5*((3*l1+lm/2)*vphy2+(3*l2+lm/2)*wphy2-muh2+mus2+(((3*l1-lm/2)*vphy2+(-3*l2+lm/2)*wphy2-muh2-mus2)**2+4*lm**2*vphy2*wphy2)**.5) - 125.**2
        return [veq, meq]
    muroot = optimize.fsolve(func, [5000, -5000], (l1, l2, lm))
    return muroot

def getscani(i, m):
    
    
    if any([m.l1 > 4.*np.pi/3., m.l2 > 4.*np.pi/3., m.lm > 16.*np.pi/3.]): 
           
        print('wrong paras')
        
        scani = None
        
        
    else: 
                    
        scani = []
                         
        scani.append([m.l1, m.l2, m.lm, m.m12, m.m22])
        
        scani.append(physpara(m))
            
        if t0vev(m) or thvev(m):
            
            scani.append([])
            
            scani.append([])
            
            scani.append(True)
                
        else:

            transit, sfoph, check = trans(i, m)
            
            scani.append([sfoph])
            
            scani.append([transit])
            
            scani.append(check)
         
    return scani
                
                                                  

def getscan(l1box, l2box, lmbox, m12box, m22box, npt):
 
    l1min,l1max = l1box
    l2min, l2max = l2box
    lmmin, lmmax = lmbox
    m12min, m12max = m12box
    m22min, m22max = m22box
        
    scan_task = range(npt)
    
    rank = 0 #comm.Get_rank()

    size = comm.Get_size()
        
    scan_rank = []
                
    ran.seed(time.time() + rank)
        
    for n in scan_task:
        
        if n%size != rank: 
           continue
       
        else:
            # Tong: generate random number between the given min and max
            # Why not create a np.linspace and scan one by one?
            l1 = ran.uniform(l1min,l1max)
            l2 = ran.uniform(l2min,l2max)
            lm = ran.uniform(lmmin,lmmax)
            m12, m22 = solve_mu(l1, l2, lm)
            #m12log = ran.uniform(m12min,m12max)
            #m22log = ran.uniform(m22min,m22max)
            
            #m12 = 10.**m12log
            #m22 = -10.**m22log
            #m12 = ((4*l1*l2-lm**2)*246**2-2*lm*m22)/(4*l2)

            #require tree level vev of s to exist
            if (-2*l1*m22-lm*m12)/(4*l1*l2-lm**2) < 0:
                n -= 1
                continue
            
            v2re = 1000.**2.
            
                
            #mcw = bmcw.model(m12, m22, l1, l2, lm, v2re)
            
            mcwd = bm.model(m12, m22, l1, l2, lm, v2re)
            
            phy = physpara(mcwd)
            
            vphy, tanbphy, m1phy, m2phy, sintphy = phy[0]**.5, phy[1], phy[2], phy[3], phy[4]
            
                
            print '%s, %s, %s, %s, %s' % (vphy, tanbphy, m1phy, m2phy, sintphy)
            
            #print '%s, %s, %s' % (vphys, m1phys, m2phys)

            #Tong: If the derived physical parameters are in the given range, save the parameters to .npy file.
                        
            if all((vphy <= 248., vphy >= 244., tanbphy >= 0.001)):
                
                if all((m1phy <= 127., m1phy >= 123., abs(sintphy) <= .4)) or all((m2phy <= 127., m2phy >= 123., (1. - sintphy**2.)**.5 <= .4)):
                    
                    print '.',
            
                    scan_rank.append([m12, m22, l1, l2, lm, v2re, vphy, tanbphy, m1phy, m2phy, sintphy])
              
                    filename = '%s_%s' % (FILE, 1) #rank)
              
                    np.save(os.path.join(BASE_PATH, filename), scan_rank)
                    
                else:
                                        
                    pass
           
            else:
                
                pass
                        
            
                                                  
scan = getscan(l1r, l2r, lmr, m12logr, m22logr, npt)

# np.load("ran_scan_1.npz")

#npfile.files

#npfile['check']

# npfile['scan_1pht'].item().get('sint')






