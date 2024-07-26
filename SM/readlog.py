#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
import plotter as pl
import os

outdir = './output/'
log = open(outdir + 'log', 'r')
log2 = open(outdir + 'log2', 'r')
lines = log.readlines() + log2.readlines()

i = 1
g2, lh, muh = 0., 0., 0.
g2det, lhdet, muhdet, alpha_det = [], [], [], []
g2def, lhdef, muhdef, alpha_def = [], [], [], []
udlist = []
while i < len(lines):
    if 'g2' in lines[i]:
        g2 = float(lines[i].split()[-1])
        lh = float(lines[i+1].split()[-1])
        muh = float(lines[i+2].split()[-1])
        alpha = float(lines[i+3].split()[-1])
        if 'Undetermined' in lines[i+5]:
            udlist.append(alpha)
        elif 'Detonation' in lines[i+5] or 'Detonation' in lines[i+6]:
            g2det.append(g2)
            lhdet.append(lh)
            muhdet.append(muh) 
            alpha_det.append(alpha)
        elif 'Deflagration/Hybrid' in lines[i+5] or 'Deflagration/Hybrid' in lines[i+6]:
            g2def.append(g2)
            lhdef.append(lh)
            muhdef.append(muh)
            alpha_def.append(alpha)
    i += 1


fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
ax[0].scatter(g2det, lhdet, c = 'orange', alpha = 0.75, label = 'Detonation')
ax[0].scatter(g2def, lhdef, c = 'blue', alpha = 0.75, label = 'Deflagration/Hybrid')
ax[0].set_xlabel(r'$g_2$')
ax[0].set_ylabel(r'$\lambda$')
ax[1].scatter(lhdet, muhdet, c = 'orange', alpha = 0.75, label = 'Detonation')
ax[1].scatter(lhdef, muhdef, c = 'blue', alpha = 0.75, label = 'Deflagration/Hybrid')
ax[1].legend(loc='lower right' )
ax[1].set_xlabel(r'$\lambda$')
ax[1].set_ylabel(r'$\mu_h$')
plt.savefig(outdir + 'scatter.png')

fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(10, 4), dpi=200)
axs[0].hist(alpha_def, 20, histtype='stepfilled', facecolor='b', alpha=0.75)
axs[1].hist(alpha_det, 20, histtype='stepfilled', facecolor='orange', alpha=0.75)
axs[0].set_xlabel(r'$\alpha_n$')
axs[1].set_xlabel(r'$\alpha_n$')
axs[0].set_title('Deflagration/Hybrid')
axs[1].set_title('Detonation')
plt.savefig(outdir + 'alpha_hist.png')
print(udlist)
