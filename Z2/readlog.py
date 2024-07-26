#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
import plotter as pl
import os

outdir = './output/vw/'
ofile = outdir + 'combined_log'
with open(ofile, 'w') as outfile:
    for filename in os.listdir(outdir):
        if filename.startswith('log'):
            file_path = os.path.join(outdir, filename)
            with open(file_path, 'r') as infile:
                outfile.write(infile.read())
                outfile.write('\n')
log = open(ofile, 'r')        
lines = log.readlines()

i = 1
alpha_det, alpha_def = [], []
udlist = []
while i < len(lines):
    if 'μh2' in lines[i]:
        m12 = float(lines[i].split()[-1])
        m22 = float(lines[i+1].split()[-1])
        l1 = float(lines[i+2].split()[-1])
        l2 = float(lines[i+3].split()[-1])
        lm = float(lines[i+4].split()[-1])
        alpha = float(lines[i+5].split()[-1])
        if 'Undetermined' in lines[i+7]:
            udlist.append(alpha)
        elif 'Detonation' in lines[i+7] or 'Detonation' in lines[i+8]:
            alpha_det.append(alpha)
        elif 'Deflagration/Hybrid' in lines[i+7] or 'Deflagration/Hybrid' in lines[i+8]:
            alpha_def.append(alpha)
    i += 1

'''
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
'''
fig, axs = plt.subplots(1, 1, figsize=(6, 4), dpi=200)
axs.hist(alpha_def, 20, histtype='stepfilled', facecolor='b', alpha=0.75, label='deflagration')
axs.hist(alpha_det, 20, histtype='stepfilled', facecolor='orange', alpha=0.75, label='detonation')
axs.set_xlabel(r'$\alpha_n$')
axs.legend(loc = 'upper right')
plt.savefig(outdir + 'alpha_hist.png')
print(udlist)
