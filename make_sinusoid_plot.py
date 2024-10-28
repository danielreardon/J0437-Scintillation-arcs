#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:14:29 2023

@author: dreardon

Interactively select a range in curvature, a delay maximum, and filter options
to identify arcs with a Heuristic model

For each arc identified, save the optimal delmax, phase gradient, search window
and Heuristic model

"""

import matplotlib as mpl
import sys
from copy import deepcopy as cp
from lmfit import Parameters
from scintools.scint_models import fitter
from scintools.scint_utils import get_ssb_delay, read_par, get_true_anomaly
from matplotlib.widgets import Slider, Button
from scipy.stats import pearsonr
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import astropy.units as u
from scipy.ndimage import minimum_filter
from scipy.signal import savgol_filter
import matplotlib.style as mplstyle
mplstyle.use('fast')
mpl.rcParams.update({'axes.facecolor': 'black'})


def centres_to_edges(arr):
    """
    Take an array of pixel-centres, and return an array of pixel-edges
        assumes the pixel-centres are evenly spaced from smallest to largest
    """
    darr = np.abs(arr[1] - arr[0])
    arr_edges = arr - darr/2
    return np.append(arr_edges, arr_edges[-1] + darr)


def smooth_min_filt(ydata, filter_size):

    # if no filter
    if filter_size == 0:
        return ydata

    filt_arr = np.zeros_like(ydata)
    try:
        N = len(ydata[:, 0])
        # smoothed minimum filter to subtract the baseline
        for row in range(N):
            filt = minimum_filter(ydata[row, :], size=filter_size)
            # smooth the minimum filter
            filt_arr[row, :] = savgol_filter(filt, filter_size, 1)
            ydata[row, :] = ydata[row, :] - filt_arr[row, :]
        filt = filt_arr
    except IndexError:  # One row only
        print(ydata)

        filt = minimum_filter(ydata, size=filter_size)
        filt = savgol_filter(filt, filter_size, 1)
        ydata = ydata - filt

    return ydata, filt


def fit_shift_centre(params, xdata, data_pos, data_neg, model, precision):
    """
    Fits for and returns the shift between two arrays of data
    The shift is calculated using the central 50%
    """

    shift = params['shift']

    data_pos_new = np.zeros((3, len(xdata)//2))
    data_neg_new = np.zeros((3, len(xdata)//2))

    for iobs in range(3):

        if model is None:
            minxcp = xdata[int(len(xdata)/4)] + shift/2
            maxxcp = xdata[int(3*len(xdata)/4)] + shift/2
            minxcn = xdata[int(len(xdata)/4)] - shift/2
            maxxcn = xdata[int(3*len(xdata)/4)] - shift/2

            xpos = np.linspace(minxcp, maxxcp, len(xdata)//2)
            xneg = np.linspace(minxcn, maxxcn, len(xdata)//2)
        else:
            minxcp = model[iobs] - model[iobs]*precision + shift/2
            maxxcp = model[iobs] + model[iobs]*precision + shift/2
            minxcn = model[iobs] - model[iobs]*precision - shift/2
            maxxcn = model[iobs] + model[iobs]*precision - shift/2

            xpos = np.linspace(minxcp, maxxcp, len(xdata)//2)
            xneg = np.linspace(minxcn, maxxcn, len(xdata)//2)

        data_pos_new[iobs, :] = np.interp(xpos, xdata, data_pos[iobs, :])
        data_neg_new[iobs, :] = np.interp(xneg, xdata, data_neg[iobs, :])

        data_pos_new[iobs, :] -= np.mean(data_pos_new[iobs, :])
        data_neg_new[iobs, :] -= np.mean(data_neg_new[iobs, :])

    # correlation coefficient
    try:
        corr = pearsonr(data_pos_new.flatten(), data_neg_new.flatten())[0]
    except ValueError:
        corr = -1

    return (corr - 1)**2


def calculate_shifts(ypos_array, yneg_array, xneg, model=None, precision=0.2):

    # Shift of the data
    global shift_array
    shift_array = []

    xneg_new = np.linspace(min(xneg), max(xneg), len(xneg))
    ypos_new_array = []
    yneg_new_array = []

    iskip = 0
    for iday in range(6):  # for each day. 3 observations per day
        shift = 0

        y_pos_uniform = np.zeros((3, len(xneg_new)))
        y_neg_uniform = np.zeros((3, len(xneg_new)))

        # interpolate each observation
        for iobs in range(3):
            y_pos_uniform[iobs, :] = np.interp(
                xneg_new, xneg, ypos_array[3*iday + iobs + iskip, :])
            y_neg_uniform[iobs, :] = np.interp(
                xneg_new, xneg, yneg_array[3*iday + iobs + iskip, :])

        params = Parameters()
        params.add('shift', value=0, vary=True, min=-
                   np.ptp(xneg_new)/2, max=np.ptp(xneg_new)/2)

        if model is not None:
            results = fitter(fit_shift_centre, params,
                             (xneg_new, y_pos_uniform, y_neg_uniform, model[3*iday + iskip: 3*iday + iskip + 3], precision))
        else:
            results = fitter(fit_shift_centre, params,
                             (xneg_new, y_pos_uniform, y_neg_uniform, None, precision))

        shift = results.params['shift'].value

        shift_array.append(shift)

        inds_pos = np.argwhere((xneg + shift/2 > wmin)
                               * (xneg + shift/2 < wmax)).squeeze()
        inds_neg = np.argwhere((xneg - shift/2 > wmin)
                               * (xneg - shift/2 < wmax)).squeeze()

        for iobs in range(3):
            y_pos_uniform = np.interp(
                xneg_new, xneg[inds_pos] - shift/2, ypos_array[3*iday + iobs + iskip, inds_pos]).squeeze()
            y_neg_uniform = np.interp(
                xneg_new, xneg[inds_neg] + shift/2, yneg_array[3*iday + iobs + iskip, inds_neg]).squeeze()

            ypos_new_array.append(
                y_pos_uniform/np.max(y_pos_uniform))
            yneg_new_array.append(
                y_neg_uniform/np.max(y_neg_uniform))

        # Append a nan block after each day
        ypos_new_array.append(np.zeros_like(y_pos_uniform)*np.nan)
        yneg_new_array.append(np.zeros_like(y_neg_uniform)*np.nan)

        iskip += 1

    ypos_array = np.array(ypos_new_array).squeeze()
    yneg_array = np.array(yneg_new_array).squeeze()
    xneg = xneg_new

    return ypos_array, yneg_array, xneg


cwd = os.getcwd()
pardir = cwd
datadir = cwd + '/data/'
plotdir = cwd + '/plot/'
outdir = cwd + '/tuned/'


delmax_arr = [2, 0.5]
wmin_arr = [300, 300]
wmax_arr = [1280, 1280]
filter_size_arr = [800, 400]


figsize = (12, 12)
fig, axC = plt.subplots(figsize=figsize, sharey=True)
axC = plt.subplot(111)

for ijk in range(2):
    delmax = delmax_arr[ijk]
    wmin = wmin_arr[ijk]
    wmax = wmax_arr[ijk]
    filter_size = filter_size_arr[ijk]

    filenames = sorted(
        glob.glob(datadir + 'J*2019-12*_{}us_startbin1_new.npz'.format(delmax)))

    mjd_array = []  # mjds of data edges
    tobs_array = []
    ypos_array = []
    yneg_array = []
    mjds = []  # mjds of data centres

    i_data = 0
    for file in filenames:

        print(file)
        savename = file.replace(':', '-').replace('/data/', '/plot/')

        # Arrays: xdata_neg, ydata_neg, xdata_pos, ydata_pos, [etamin], [mjd], [tobs])
        data = np.load(file)
        xdata_neg = data['arr_0']
        ydata_neg = data['arr_1']
        xdata_pos = data['arr_2']
        ydata_pos = data['arr_3']
        etamin = data['arr_4'][0]
        mjd = data['arr_5'][0]
        tobs = data['arr_6'][0]

        xpos = etamin / xdata_pos**2 / (u.m*u.mHz**2)
        xneg = etamin / xdata_neg**2 / (u.m*u.mHz**2)
        xpos = 1. / np.sqrt(2*xpos)
        xneg = 1. / np.sqrt(2*xneg)
        xpos = xpos.to(u.km/u.s/u.kpc**0.5).value
        xneg = xneg.to(u.km/u.s/u.kpc**0.5).value

        ydata_neg = np.flip(ydata_neg)
        xneg = np.flip(xneg)

        ypos_array.append(ydata_pos)
        yneg_array.append(ydata_neg)
        mjd_array.append(mjd)
        tobs_array.append(tobs/86400)
        mjds.append(mjd + tobs/86400/2)

        i_data += 1  # after every long track, append zeros
        if i_data % 3 == 0 and i_data != 0:
            ypos_array.append(np.zeros(np.shape(ydata_pos))*np.nan)
            yneg_array.append(np.zeros(np.shape(ydata_neg))*np.nan)
            mjd_array.append(mjd+tobs/86400)
            mjds.append(mjd + tobs/86400/2 + tobs/86400)

    ypos_array = np.array(ypos_array).squeeze()
    yneg_array = np.array(yneg_array).squeeze()
    mjd_array = np.array(mjd_array).squeeze()
    xpos = np.array(xpos).squeeze()
    xneg = np.array(xneg).squeeze()
    mjds = np.array(mjds).squeeze()

    # Set median noise level to 0
    nr, nc = np.shape(ypos_array)
    for ir in range(0, nr):
        ind = np.argwhere((xpos > 27500) * (xpos < 40000)).squeeze()
        mnpos = np.mean(ypos_array[ir, ind])
        mnneg = np.mean(yneg_array[ir, ind])
        mn = (mnpos + mnneg) / 2
        mx = np.nanmax([ypos_array[ir, ind], yneg_array[ir, ind]])
        ind = np.argwhere((xpos > 60) * (xpos < 1250)).squeeze()
        ypos_array[ir, :] -= mn
        ypos_array[ir, :] /= mx
        yneg_array[ir, :] -= mn
        yneg_array[ir, :] /= mx

    mean_mjd = round(np.mean(mjd_array))
    mjd_array -= mean_mjd

    # calculate orbital phase
    pars = read_par(pardir + '/J0437-4715.par')
    ssb_delays = get_ssb_delay(mjds, pars['RAJ'], pars['DECJ'])
    mjds += np.divide(ssb_delays, 86400)
    # get true anomaly
    U = get_true_anomaly(mjds, pars)
    phase = U * 180/np.pi + pars['OM']
    phase[phase > 360] = phase[phase > 360] - 360

    # Initially cut the data to 2 times the wmin and wmax range. Allowing room for shifting
    ind = np.argwhere((xneg > wmin - (wmax-wmin)/2)
                      * (xneg < wmax + (wmax-wmin)/2)).squeeze()

    ypos_array = ypos_array[:, ind].squeeze()
    yneg_array = yneg_array[:, ind].squeeze()
    xneg = xneg[ind].squeeze()

    xmid = wmin + (wmax - wmin)/2

    ind = np.argmin(np.abs(xneg - xmid))
    xmn = xneg[ind - filter_size//2]
    xmx = xneg[ind + filter_size//2]

    ypos_array, filt_pos = smooth_min_filt(ypos_array, filter_size)
    yneg_array, filt_neg = smooth_min_filt(yneg_array, filter_size)

    ypos_array_orig = cp(ypos_array)
    yneg_array_orig = cp(yneg_array)
    xneg_orig = cp(xneg)

    ypos_array, yneg_array, xneg = calculate_shifts(
        ypos_array_orig, yneg_array_orig, xneg_orig)

    """
    LOADED DATA, NOW START THE INTERACTIVE PLOT
    """

    global iarc
    iarc = 0

    plt.set_cmap('magma')

    ydata = (yneg_array + ypos_array) / 2

    ind = np.argwhere((xneg > wmin) * (xneg < wmax))
    vmin = np.nanmin(ydata[:, ind])
    vmax = np.nanmax(ydata[:, ind])
    ydata[np.isnan(ydata)] = vmin

    edges = centres_to_edges(xneg)
    lim = [wmin, wmax]

    ind = np.argmin(np.abs(xneg - np.mean(xneg)))
    xmn = xneg[ind]

    T = 5.741045915573534
    om = 2 * np.pi / T

    if ijk == 0:
        ydata_plot = np.zeros_like(ydata)

    division_line = 68 * np.sin(om * mjd_array + 0.6) + 625
    for ix in range(len(mjd_array)):
        if ijk == 0:
            indw = np.argwhere(np.abs(xneg) < division_line[ix])
        else:
            indw = np.argwhere(np.abs(xneg) > division_line[ix])
        ydata_plot[ix, indw] = ydata[ix, indw]

    if ijk == 1:
        vmin = np.nanmin(ydata_plot)
        vmax = np.nanmax(ydata_plot)
        indsort = np.argsort(phase)
        axC.pcolormesh(edges, centres_to_edges(phase[indsort]),
                       ydata_plot[indsort, :], vmin=vmin, vmax=vmax, shading='flat', zorder=0, rasterized=True)
    axC.set_xscale('log')
    lim = (wmin, wmax)
    axC.set_xlabel(r'W (km s$^{-1}\sqrt{\rm kpc}$)')
    axC.set_xlim(lim)

    t = np.linspace(min(centres_to_edges(mjd_array) - pars['PB'] / 2),
                    max(centres_to_edges(mjd_array) + pars['PB'] / 2), 1000)
    tplot = np.linspace(min(centres_to_edges(mjds) - pars['PB'] / 2),
                        max(centres_to_edges(mjds) + pars['PB'] / 2), 1000)
    ssb_delays = get_ssb_delay(tplot, pars['RAJ'], pars['DECJ'])
    tplot += np.divide(ssb_delays, 86400)
    # get true anomaly
    Uplot = get_true_anomaly(tplot, pars)
    phaseplot = Uplot * 180/np.pi + pars['OM']
    phaseplot[phaseplot > 360] = phaseplot[phaseplot > 360] - 360

    division_line = 68 * np.sin(om * t + 0.6) + 625

    A0 = 0.1*lim[1]
    phi0 = np.pi
    C0 = np.mean(lim)

    precision0 = 0.3

    y = A0 * np.sin(om * t + phi0) + C0
    ymin = A0 * np.sin(om * t + phi0) + C0 - A0 * precision0
    ymax = A0 * np.sin(om * t + phi0) + C0 + A0 * precision0

    indsort = np.argsort(phaseplot)
    line3, = axC.plot(division_line[indsort], phaseplot[indsort], 'r')
    # line5, = axC.plot(ymin, t, 'w:', alpha=0.5)
    # line6, = axC.plot(ymax, t, 'w:', alpha=0.5)

    for model_file in sorted(glob.glob(outdir + '*.npz')):
        # do not plot 23 and 25 because they do not appear in full on this figure
        # possible arc 25 is unconvincing is not described in paper
        if ('_23.' in model_file) or ('_25.' in model_file):
            continue

        data = np.load(model_file)
        A = data['A']
        C = data['C']
        phi = data['phi']

        y = A * np.sin(om * t + phi) + C

        indsort = np.argsort(phaseplot)
        axC.plot(y[indsort], phaseplot[indsort], 'w--')

        iarc += 1


plt.ylim([0, 360])
plt.ylabel('Orbital phase (degrees)')
plt.tight_layout()
plt.savefig(os.getcwd()+'/paper_plots/Wphase.pdf')
plt.savefig(os.getcwd()+'/paper_plots/Wphase.png')
plt.show()

mpl.rcParams.update({'axes.facecolor': 'white'})
