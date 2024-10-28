#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 12:32:14 2020

@author: dreardon

Script for reading and processing the dynamic spectra
    - Crop at 1680MHz to avoid noise from less sensitive part of band
    - Refill with biharmonic functions
    - Correct with SVD technique
    - Save as a pickle
    - Plot dynspec and save to file in plotdir
    - Calculate and plot secondary spectrum and save to file in plotdir
    - Add each set of ~4 hour observations into ~12 hour long tracks
    - Save long tracks as pickle

To generate the individual dynspec files, the following was performed:
    - p-scrunch 8s subints
    - psradd subints in each ~4 hour observation
    - b-scrunch by a factor of 16 to reduce file size
    - MeerGuard with threshold of 8-sigma
    - MeerGuard subtract b-scrunched template and analyse off-pulse region
    - psrflux with b-scrunched template
"""

# Import things
import glob
import sys
import os
from scintools.dynspec import Dynspec
from scintools.scint_utils import make_pickle, save_fits
from copy import deepcopy as cp

datadir = None
plotdir = None

if datadir is None:
    print("Warning, data directory not set up. Download dynamic spectra from figshare")
    sys.exit()

dynfiles = sorted(glob.glob(datadir + '*ar.ds'))

for ii in range(0, len(dynfiles)):
    dynfile = dynfiles[ii]

    # Read in dynamic spectrum
    dyn = Dynspec(filename=dynfile, process=False)
    savename = dyn.name.split('.')[0].replace(':', '-')

    # Crop at 1680MHz
    dyn.crop_dyn(fmax=1680)

    # Refill with biharmonic equations
    dyn.refill()

    if ii % 3 == 0:
        dyn_tot = cp(dyn)
    else:
        dyn_tot += dyn

    # Correct with SVD technique
    dyn.correct_dyn()

    # Save as a pickle
    # make_pickle(dyn, datadir+'{0}.pkl'.format(savename))

    # Plot dynspec and save to file in plotdir
    dyn.plot_dyn(filename=plotdir+'{0}_dyn.png'.format(savename))

    # Calculate and plot secondary spectrum and save to file in plotdir
    dyn.plot_sspec(lamsteps=True, colorbar=False, prewhite=False,
                   filename=plotdir+'{0}_sspec.png'.
                   format(savename), subtract_artefacts=True)
    dyn.plot_sspec(lamsteps=True, colorbar=False, maxfdop=35,
                   filename=plotdir+'{0}_sspec_zoom.png'.
                   format(savename), subtract_artefacts=True)

    # Process the long tracks in the same way as above
    if (ii + 1) % 3 == 0:
        savename = 'Long_track_' + dyn_tot.name.split('.')[0].replace(':', '-')
        print(' ')
        print('Analysing long track {0}'.format(savename))

        # SVD-correct
        dyn_tot.correct_dyn()

        # Refill gaps between obs with mean
        dyn_tot.refill(linear=False)

        # Save as a pickle
        make_pickle(dyn_tot, datadir+'{0}.pkl'.format(savename))

        # Plot dynspec and save to file in plotdir
        dyn_tot.plot_dyn(filename=plotdir+'{0}_dyn.png'.format(savename))

        # Calculate and plot secondary spectrum and save to file in plotdir
        dyn_tot.plot_sspec(lamsteps=True, colorbar=False, prewhite=False,
                           filename=plotdir +
                           '{0}_sspec.png'.format(savename),
                           subtract_artefacts=True)
        dyn_tot.plot_sspec(lamsteps=True, colorbar=False, maxfdop=35,
                           filename=plotdir +
                           '{0}_sspec_zoom.png'.format(savename),
                           subtract_artefacts=True)
