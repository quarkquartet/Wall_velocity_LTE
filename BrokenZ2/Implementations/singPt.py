from __future__ import division

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 20:33:49 2018

@author: yik
"""

import sys
#sys.path.append('/home/tong/Chicago/EWPhT/cosmotransition_z2sb/cosmoTransitions/')

#import baseMo_s_t as bmt
import baseMo_s_b_cwd as bmt

import matplotlib.pyplot as plt

mt = bmt.model(0.3,0.036,-0.171, 125., 246.**2., 1000.**2.)

print("\n")
print("\n")

print("The T=0 potential of the model reads")

bmt.vsh(mt, (-300., 300., -30., 30.), 0.)

print("\n")
print("\n")

print("Now let's find the phase transitions:")

mt.calcTcTrans()

print("\n \n All the phase transitions of such a model are")

mt.prettyPrintTcTrans()

print("And the T-dependent 'phase norm' reads")

#plt.figure()
#m.plotPhasesPhi()
#plt.show()

#plt.figure()
#mt.plotPhases2D()
#plt.show()

"""
Note: to be completed: 
models may have probolems calculating tunneling (possibly due to the strength of phase transitions).
We need Tc info instead of Tn info. 
So such a step shall be neglected at this point.
"""
print("\n \n")

print("Now let's find the corresponding tunneliings:")

mt.findAllTransitions()

print("\n \n All the tunnelings/phase transitions of such a model are")

mt.prettyPrintTnTrans()


