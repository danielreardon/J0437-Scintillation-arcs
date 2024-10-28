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

import sys
from copy import deepcopy as cp
from lmfit import Parameters
from scintools.scint_models import fitter
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
datadir = cwd + '/data/'
plotdir = cwd + '/plot/'
outdir = cwd + '/tuned/'

delmax_array = [2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.01, 0.005, 0.0025]

parser = argparse.ArgumentParser(description='Run interactive arc identifier')
parser.add_argument("-delmax", default=None, type=str)
parser.add_argument("-wmin", default=None, type=float)
parser.add_argument("-wmax", default=None, type=float)
parser.add_argument("-filter", default=None, type=int)
parser.add_argument("-arcnum", default=None, type=int)


a = parser.parse_args()
delmax = a.delmax
wmin = a.wmin
wmax = a.wmax
filter_size = a.filter
arcnum = a.arcnum

if arcnum is not None:
    # Load parameters from an initial analysis to save time
    arc = np.load(cwd + '/tuned_old/arcfit_{}.npz'.format(arcnum))

    if delmax is None:
        delmax = arc['delmax']
    if wmin is None:
        wmin = arc['wmin']
    if wmax is None:
        wmax = arc['wmax']
    if filter_size is None:
        filter_size = int(arc['filter_size'])

figsize = (15, 8)

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


# Plot the three observations in the first day, to check the filter size
plt.figure(1, figsize=(7, 8))
for iobs in range(3):
    plt.subplot(311)
    plt.plot(xneg, ypos_array[iobs, :], 'C0', alpha=0.8)
    plt.plot(xneg, yneg_array[iobs, :], 'C1', alpha=0.8)
plt.xlim([wmin, wmax])
plt.title(
    'Day #1. Top: Original unshifted. Middle: Filtered. Bottom: Mean of shifted')
yl = plt.ylim()
yl = (0, yl[1])
plt.ylim(yl)
plt.plot([xmn, xmx], [np.mean(yl), np.mean(yl)], 'r')

ypos_array, filt_pos = smooth_min_filt(ypos_array, filter_size)
yneg_array, filt_neg = smooth_min_filt(yneg_array, filter_size)

# plot the filter for one side
for iobs in range(3):
    plt.subplot(311)
    plt.plot(xneg, filt_pos[iobs, :], 'mediumblue', alpha=0.8, linestyle=':')

ypos_array_orig = cp(ypos_array)
yneg_array_orig = cp(yneg_array)
xneg_orig = cp(xneg)

for iobs in range(3):
    plt.subplot(312)
    plt.plot(xneg_orig, ypos_array_orig[iobs, :], 'C0', alpha=0.8)
    plt.plot(xneg_orig, yneg_array_orig[iobs, :], 'C1', alpha=0.8)

plt.xlim([wmin, wmax])
yl = plt.ylim()
yl = (0, yl[1])
plt.ylim(yl)
# Plot the filter size
plt.plot([xmn, xmx], [np.mean(yl), np.mean(yl)], 'r')

ypos_array, yneg_array, xneg = calculate_shifts(
    ypos_array_orig, yneg_array_orig, xneg_orig)

for iobs in range(3):
    plt.subplot(313)
    plt.plot(xneg, (ypos_array[iobs, :] + yneg_array[iobs, :])/2)
plt.xlabel('W')
plt.xlim([wmin, wmax])
yl = plt.ylim()
yl = (0, yl[1])
plt.ylim(yl)

plt.tight_layout()
plt.savefig(plotdir + '/arc_{0}_filtered_data.png'.format(arcnum))
plt.show()


"""
LOADED DATA, NOW START THE INTERACTIVE PLOT
"""

global iarc
iarc = 0

fig, axC = plt.subplots(figsize=figsize)
plt.set_cmap('magma')


axC = plt.subplot(121)

ydata = (yneg_array + ypos_array) / 2

ind = np.argwhere((xneg > wmin) * (xneg < wmax))
vmin = np.nanmin(ydata[:, ind])
vmax = np.nanmax(ydata[:, ind])
ydata[np.isnan(ydata)] = vmin

edges = centres_to_edges(xneg)
lim = [wmin, wmax]

ind = np.argmin(np.abs(xneg - np.mean(xneg)))
xmn = xneg[ind]

axC.pcolormesh(edges, centres_to_edges(mjd_array),
               ydata, vmin=vmin, vmax=vmax, shading='flat')
axC.set_xscale('log')
# axC.set_xlabel(r'Curvature (m$^{-1}$ mHz$^{-2}$)')
axC.set_xlabel(r'W (km\,s$^{-1}\sqrt{\rm kpc}$)')
axC.set_xlim(lim)

t = np.linspace(min(centres_to_edges(mjd_array)),
                max(centres_to_edges(mjd_array)), 1000)
T = 5.741045915573534
om = 2 * np.pi / T
A0 = 0.1*lim[1]
phi0 = np.pi
C0 = np.mean(lim)
if arcnum is not None:
    A0 = arc['A']
    phi0 = arc['phi']
    C0 = arc['C']

precision0 = 0.3

y = A0 * np.sin(om * t + phi0) + C0
ymin = A0 * np.sin(om * t + phi0) + C0 - A0 * precision0
ymax = A0 * np.sin(om * t + phi0) + C0 + A0 * precision0

# Estimate of a curve that separates bright high-curvature arcs from a
#    set of lower curvature arcs
division_line = 68 * np.sin(om * t + 0.6) + 625

# Right panel: differences
plt.figure(1)
axR = plt.subplot(122)

ydata = yneg_array - ypos_array


ind = np.argwhere((xneg > wmin) * (xneg < wmax))
vmin = np.nanmean(ydata[:, ind]) - 4*np.nanstd(ydata[:, ind])
vmax = np.nanmean(ydata[:, ind]) + 4*np.nanstd(ydata[:, ind])
ydata[np.isnan(ydata)] = 0

plt.set_cmap('seismic')
axR.pcolormesh(edges, centres_to_edges(mjd_array),
               ydata, vmin=vmin, vmax=vmax, shading='flat')
axR.set_xscale('log')
axR.set_xlabel(r'W (km\,s$^{-1}\sqrt{\rm kpc}$)')
axR.set_xlim(lim)

line, = axC.plot(y, t, 'w')
line2, = axR.plot(y, t, 'k')
# line3, = axC.plot(division_line, t, 'g--')
# line4, = axR.plot(division_line, t, 'g--')
line5, = axC.plot(ymin, t, 'w:', alpha=0.5)
line6, = axC.plot(ymax, t, 'w:', alpha=0.5)


for model_file in sorted(glob.glob(outdir + '*.npz')):
    data = np.load(model_file)
    A = data['A']
    C = data['C']
    phi = data['phi']

    y = A * np.sin(om * t + phi) + C

    axC.plot(y, t, 'w--')
    axR.plot(y, t, 'k--')

    iarc += 1


axcolor = 'lightgoldenrodyellow'
# [left, bottom, width, height]
ax_A = plt.axes([0.1, 0.95, 0.20, 0.03], facecolor=axcolor)
ax_p = plt.axes([0.4, 0.95, 0.20, 0.03], facecolor=axcolor)
ax_C = plt.axes([0.1, 0.9, 0.20, 0.03], facecolor=axcolor)
ax_phi = plt.axes([0.4, 0.9, 0.20, 0.03], facecolor=axcolor)
ax_fit = plt.axes([0.65, 0.95, 0.15, 0.03], facecolor=axcolor)
ax_save = plt.axes([0.65, 0.9, 0.15, 0.03], facecolor=axcolor)
ax_arc = plt.axes([0.82, 0.9, 0.08, 0.08], facecolor=axcolor)

button_fit_shift = Button(ax_fit, "fit shift")
button_fit_arc = Button(ax_arc, "fit arc")
button_save = Button(ax_save, "save")

A_sl = Slider(ax_A, 'A', 0, 0.5*lim[1], valinit=A0)
C_sl = Slider(ax_C, 'C', lim[0], lim[1], valinit=C0)
phi_sl = Slider(ax_phi, 'phi', 0, 2*np.pi, valinit=phi0)


precision_sl = Slider(ax_p, 'Prec.', 0, 1, valinit=precision0)


def update(val):

    A = A_sl.val
    C = C_sl.val
    phi = phi_sl.val
    precision = precision_sl.val

    y = A * np.sin(om * t + phi) + C
    ymin = A * np.sin(om * t + phi) + C - A * precision
    ymax = A * np.sin(om * t + phi) + C + A * precision

    line.set_xdata([y])
    line2.set_xdata([y])
    line5.set_xdata([ymin])
    line6.set_xdata([ymax])

    fig.canvas.draw()


def fct_button_save(event):
    global iarc

    plt.savefig(plotdir + '/arcfit_{0}.png'.format(iarc))
    np.savez(outdir + '/arcfit_{0}.npz'.format(iarc), A=A_sl.val, C=C_sl.val, phi=phi_sl.val,
             delmax=delmax, wmin=wmin, wmax=wmax, filter_size=filter_size, shift_array=shift_array)

    y = A_sl.val * np.sin(om * t + phi_sl.val) + C_sl.val

    axC.plot(y, t, 'g--')
    axR.plot(y, t, 'g--')

    iarc += 1


def fct_button_fit_shift(event):
    print('Finding shift...')

    # Get the current model, at the MJD centres
    y = A_sl.val * np.sin(om * mjd_array + phi_sl.val) + C_sl.val

    # Use original data
    ypos_array = cp(ypos_array_orig)
    yneg_array = cp(yneg_array_orig)
    xneg = cp(xneg_orig)

    # Shift of the data
    global shift_array

    ypos_array, yneg_array, xneg = calculate_shifts(
        ypos_array_orig, yneg_array_orig, xneg_orig, model=y, precision=precision_sl.val)

    plt.set_cmap('magma')
    ydata = (yneg_array + ypos_array) / 2

    ind = np.argwhere((xneg > wmin) * (xneg < wmax))
    vmin = np.nanmin(ydata[:, ind])
    vmax = np.nanmax(ydata[:, ind])
    ydata[np.isnan(ydata)] = vmin
    axC.pcolormesh(edges, centres_to_edges(mjd_array),
                   ydata, vmin=vmin, vmax=vmax, shading='flat')

    plt.set_cmap('seismic')
    ydata = yneg_array - ypos_array

    ind = np.argwhere((xneg > wmin) * (xneg < wmax))
    vmin = np.nanmean(ydata[:, ind]) - 4*np.nanstd(ydata[:, ind])
    vmax = np.nanmean(ydata[:, ind]) + 4*np.nanstd(ydata[:, ind])
    ydata[np.isnan(ydata)] = 0
    axR.pcolormesh(edges, centres_to_edges(mjd_array),
                   ydata, vmin=vmin, vmax=vmax, shading='flat')

    fig.canvas.draw()
    print('Done!')


def fct_button_fit_arc(event):
    print('Fitting arc parameters...')
    # Get the current model, at the MJD centres
    y = A_sl.val * np.sin(om * mjd_array + phi_sl.val) + C_sl.val

    mjd_date = mjd_array + mean_mjd

    # get shift prior from shift_array
    shift_std = np.std(shift_array)

    print(shift_std)

    print('Done!')


A_sl.on_changed(update)
C_sl.on_changed(update)
phi_sl.on_changed(update)
precision_sl.on_changed(update)

button_save.on_clicked(fct_button_save)
button_fit_shift.on_clicked(fct_button_fit_shift)
button_fit_arc.on_clicked(fct_button_fit_arc)


plt.show()
