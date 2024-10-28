#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:46:48 2022

@author: dreardon
"""

from scintools.dynspec import Dynspec
from scintools.scint_utils import make_pickle, load_pickle
import matplotlib.pyplot as plt
import sys
import glob
import numpy as np

datadir = None
plotdir = None

if datadir is None:
    print("Warning, data directory not set up. Point to dynamic spectrum pickle files")
    sys.exit()

# filenames = sorted(glob.glob(datadir + '*.WFinpainted.ds'))
filenames = [datadir + sys.argv[1]]

etamin = 0.01
delmax_array = [2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.01, 0.005]


for file in filenames:

    dyn = load_pickle(file)
    savename = '.'.join(dyn.name.split('.')[0:-1]).replace(':', '-')

    dyn.calc_sspec(lamsteps=True, window='hanning', window_frac=0.2)

    for delmax in delmax_array:
        print('working on delmax {}'.format(delmax))

        dyn.norm_sspec(eta=etamin, delmax=delmax, plot=False,
                       startbin=1, maxnormfac=1, cutmid=0,
                       lamsteps=True, scrunched=True, logsteps=True,
                       plot_fit=False, numsteps=2e5, weighted=False,
                       subtract_artefacts=True)

        # Now make a plot and save as numpy array
        # First, do -ve side
        inds = np.argwhere((dyn.normsspec_fdop < 0) *
                           ~np.isnan(dyn.normsspecavg))
        xdata_neg = -dyn.normsspec_fdop[inds].squeeze()
        ydata_neg = dyn.normsspecavg[inds].squeeze()

        # Now, do +ve side
        inds = np.argwhere((dyn.normsspec_fdop > 0) *
                           ~np.isnan(dyn.normsspecavg))
        xdata_pos = dyn.normsspec_fdop[inds].squeeze()
        ydata_pos = dyn.normsspecavg[inds].squeeze()

        np.savez(datadir + '{0}_normsspec_{1}us_startbin1_1hr_new.npz'.format(savename, delmax),
                 xdata_neg, ydata_neg, xdata_pos, ydata_pos, [etamin], [dyn.mjd], [dyn.tobs])
