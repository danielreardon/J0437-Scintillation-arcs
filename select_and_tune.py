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

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import astropy.units as u
from scipy.ndimage import minimum_filter
import matplotlib.style as mplstyle
mplstyle.use('fast')
from scipy.stats import pearsonr
from matplotlib.widgets import Slider, Button
from scintools.scint_models import fitter
from lmfit import Parameters
from copy import deepcopy as cp


def centres_to_edges(arr):
    """
    Take an array of pixel-centres, and return an array of pixel-edges
        assumes the pixel-centres are evenly spaced
    """
    darr = np.abs(arr[1] - arr[0])
    arr_edges = arr - darr/2
    return np.append(arr_edges, arr_edges[-1] + darr)


def fit_shift_centre(params, xdata, data_pos, data_neg):
    """
    Fits for and returns the shift between two arrays of data
    The shift is calculated using the central 50%
    """

    shift = params['shift']

    minxcp = xdata[int(len(xdata)/4)] + shift/2
    maxxcp = xdata[int(3*len(xdata)/4)] + shift/2
    minxcn = xdata[int(len(xdata)/4)] - shift/2
    maxxcn = xdata[int(3*len(xdata)/4)] - shift/2

    xpos = np.linspace(minxcp, maxxcp, int(len(data_pos/2)))
    xneg = np.linspace(minxcn, maxxcn, int(len(data_neg/2)))

    data_pos = np.interp(xpos, xdata, data_pos)
    data_neg = np.interp(xneg, xdata, data_neg)

    # correlation coefficient
    corr = pearsonr(data_pos, data_neg)[0]

    return (corr - 1)**2

cwd = os.getcwd()
datadir = cwd + '/data/'
plotdir = cwd + '/plot/'
outdir = cwd + '/tuned/'

delmax_array = [2, 1, 0.5, 0.25 , 0.125, 0.0625, 0.03125, 0.01, 0.005, 0.0025]

parser = argparse.ArgumentParser(description='Run interactive arc identifier')
parser.add_argument("-delmax", default=2, type=str)
parser.add_argument("-wmin", default=300, type=float)
parser.add_argument("-wmax", default=700, type=float)
parser.add_argument("-filter", default=1000, type=int)
parser.add_argument("-do_shift", default=1, type=int)

a = parser.parse_args()
delmax = a.delmax
wmin = a.wmin
wmax = a.wmax
filter_size = a.filter
do_shift = a.do_shift

figsize = (15, 8)

filenames = sorted(glob.glob(datadir + 'J*2019-12*_{}us_startbin1_new.npz'.format(delmax)))

mjd_array = []  # mjds of data edges
tobs_array = []
ypos_array = []
yneg_array = []
mjds = []  # mjds of data centres

i_data = 0
for file in filenames:

    print(file)
    savename = file.replace(':', '-').replace('/data/','/plot/')

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

    i_data +=1  # after every long track, append zeros
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
    mn = (mnpos + mnneg) /2
    mx = np.nanmax([ypos_array[ir, ind], yneg_array[ir, ind]])
    ind = np.argwhere((xpos > 60) * (xpos < 1250)).squeeze()
    ypos_array[ir, :] -= mn
    ypos_array[ir, :] /= mx
    yneg_array[ir, :] -= mn
    yneg_array[ir, :] /= mx


mean_mjd = round(np.mean(mjd_array))
mjd_array -= mean_mjd

# Initially cut the data to 120% the wmin and wmax range. Allowing room for shifting

ind = np.argwhere((xneg > wmin/2) * (xneg < wmax*2))
ypos_array = ypos_array[:, ind].squeeze()
yneg_array = yneg_array[:, ind].squeeze()
xneg = xneg[ind].squeeze()

ypos_array_orig = cp(ypos_array)
yneg_array_orig = cp(yneg_array)
xneg_orig = cp(xneg)

# Shift of the data
shift_array = []
if do_shift == 1:
    shift = 0

    ind = np.argwhere((xneg > wmin) * (xneg < wmax))
    xneg_new = np.linspace(wmin, wmax, len(ind))
    ypos_new_array = []
    yneg_new_array = []

    for iday in range(len(mjd_array)):
        shift = 0

        if np.isnan(ypos_array[iday, :]).all():
            # gap between observations
            nr = len(xneg_new)
            ypos_new_array.append(np.zeros(nr)*np.nan)
            yneg_new_array.append(np.zeros(nr)*np.nan)
            continue

        y_pos_uniform = np.interp(xneg_new, xneg, ypos_array[iday, :])
        y_neg_uniform = np.interp(xneg_new, xneg, yneg_array[iday, :])

        params = Parameters()
        params.add('shift', value=0, vary=True, min=-np.ptp(xneg_new)/2, max=np.ptp(xneg_new)/2)

        results = fitter(fit_shift_centre, params, (xneg_new, y_pos_uniform, y_neg_uniform))

        shift = results.params['shift'].value
        shift_array.append(shift)

        inds_pos = np.argwhere((xneg + shift/2 > wmin) * (xneg + shift/2 < wmax)).squeeze()
        inds_neg = np.argwhere((xneg - shift/2 > wmin) * (xneg - shift/2 < wmax)).squeeze()

        y_pos_uniform = np.interp(xneg_new, xneg[inds_pos] - shift/2, ypos_array[iday, inds_pos]).squeeze()
        y_neg_uniform = np.interp(xneg_new, xneg[inds_neg] + shift/2, yneg_array[iday, inds_neg]).squeeze()

        ypos_new_array.append(y_pos_uniform/np.max(y_pos_uniform))
        yneg_new_array.append(y_neg_uniform/np.max(y_neg_uniform))

    ypos_array = np.array(ypos_new_array).squeeze()
    yneg_array = np.array(yneg_new_array).squeeze()
    xneg = xneg_new

"""
LOADED DATA, NOW START THE INTERACTIVE PLOT
"""

global iarc
iarc = 0

fig, axC = plt.subplots(figsize=figsize)
plt.set_cmap('magma')
# axL = plt.subplot(131)

# ydata = (yneg_array + ypos_array) / 2
# ind = np.argwhere((xneg > emin) * (xneg < emax))
# vmin = np.nanmin(ydata[:, ind])
# vmax = np.nanmax(ydata[:, ind])
# ydata[np.isnan(ydata)] = vmin

# axL.pcolormesh(centres_to_edges(xneg), centres_to_edges(mjd_array),
#                ydata, vmin=vmin, vmax=vmax, shading='flat')
# axL.set_xscale('log')
# axL.set_xlabel(r'Curvature (m$^{-1}$ mHz$^{-2}$)')
# axL.set_ylabel(r'Time (MJD $-\,{0}$)'.format(mean_mjd))
# axL.set_xlim([emin, emax])

# Middle panel: average, min filtered

axC = plt.subplot(121)

ydata = (yneg_array + ypos_array) / 2

# minimum filter
for row in range(0, len(ypos_array[:, 0])):
    ydata[row, :] = ydata[row, :] - minimum_filter(ydata[row, :], size=filter_size)

ind = np.argwhere((xneg > wmin) * (xneg < wmax))
vmin = np.nanmin(ydata[:, ind])
vmax = np.nanmax(ydata[:, ind])
ydata[np.isnan(ydata)] = vmin

edges = centres_to_edges(xneg)
lim = [wmin, wmax]

axC.pcolormesh(edges, centres_to_edges(mjd_array),
               ydata, vmin=vmin, vmax=vmax, shading='flat')
axC.set_xscale('log')
#axC.set_xlabel(r'Curvature (m$^{-1}$ mHz$^{-2}$)')
axC.set_xlabel(r'W (km\,s$^{-1}\sqrt{\rm kpc}$)')
axC.set_xlim(lim)

t = np.linspace(min(centres_to_edges(mjd_array)), max(centres_to_edges(mjd_array)), 1000)
T = 5.741045915573534
om = 2 * np.pi / T
A = 0.1*lim[1]
phi = np.pi
C = np.mean(lim)

y = A * np.sin(om * t + phi ) + C


# Right panel: differences
plt.figure(1)
axR = plt.subplot(122)

ydata = yneg_array - ypos_array
# minimum filter
for row in range(0, len(ypos_array[:, 0])):
    ydata[row, :] = ydata[row, :] - minimum_filter(ydata[row, :], size=filter_size)
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

for model_file in sorted(glob.glob(outdir + '*.npz')):
    data = np.load(model_file)
    A = data['A']
    C = data['C']
    phi = data['phi']

    y = A * np.sin(om * t + phi ) + C

    axC.plot(y, t, 'w--')
    axR.plot(y, t, 'k--')

    iarc += 1


axcolor = 'lightgoldenrodyellow'
ax_A = plt.axes([0.1, 0.95, 0.25, 0.03], facecolor=axcolor)
ax_p = plt.axes([0.45, 0.95, 0.25, 0.03], facecolor=axcolor)
ax_C = plt.axes([0.1, 0.9, 0.25, 0.03], facecolor=axcolor)
ax_phi = plt.axes([0.45, 0.9, 0.25, 0.03], facecolor=axcolor)
ax_fit = plt.axes([0.75, 0.95, 0.15, 0.03], facecolor=axcolor)
ax_save = plt.axes([0.75, 0.9, 0.15, 0.03], facecolor=axcolor)

button_fit = Button(ax_fit, "fit shift")
button_save = Button(ax_save, "save")

A_sl = Slider(ax_A, 'A', 0, 0.5*lim[1], valinit=0.1*lim[1])
C_sl = Slider(ax_C, 'C', lim[0], lim[1], valinit=np.mean(lim))
phi_sl = Slider(ax_phi, 'phi', 0, 2*np.pi, valinit=np.pi)
precision_sl = Slider(ax_p, 'Prec.', 0, 0.5, valinit=0.2)

def update(val):

    A = A_sl.val
    C = C_sl.val
    phi = phi_sl.val

    y = A * np.sin(om * t + phi ) + C

    line.set_xdata([y])
    line2.set_xdata([y])

    fig.canvas.draw()

def fct_button_save(event):
    global iarc

    plt.savefig(plotdir + '/arcfit_{0}.png'.format(iarc))
    np.savez(outdir + '/arcfit_{0}.npz'.format(iarc), A=A_sl.val, C=C_sl.val, phi=phi_sl.val,
             delmax=delmax, wmin=wmin, wmax=wmax, filter_size=filter_size, shift_array=shift_array)

    y = A_sl.val * np.sin(om * t + phi_sl.val ) + C_sl.val

    axC.plot(y, t, 'w--')
    axR.plot(y, t, 'k--')

    iarc += 1

def fct_button_fit(event):
    print('Finding shift...')

    # Get the current model, at the MJD centres
    y = A_sl.val * np.sin(om * mjd_array + phi_sl.val ) + C_sl.val

    # Use original data
    ypos_array = cp(ypos_array_orig)
    yneg_array = cp(yneg_array_orig)
    xneg = cp(xneg_orig)

    # Shift of the data
    global shift_array
    shift_array = []

    ind = np.argwhere((xneg > wmin) * (xneg < wmax))
    xneg_new = np.linspace(wmin, wmax, len(ind))
    ypos_new_array = []
    yneg_new_array = []

    for iday in range(len(mjd_array)):
        shift = 0

        ind = np.argwhere(np.abs(xneg - y[iday]) <= precision_sl.val*y[iday])
        xneg_shift = np.linspace(min(xneg[ind]), max(xneg[ind]), len(ind)).squeeze()

        if np.isnan(ypos_array[iday, :]).all():
            # gap between observations
            nr = len(xneg_new)
            ypos_new_array.append(np.zeros(nr)*np.nan)
            yneg_new_array.append(np.zeros(nr)*np.nan)
            continue

        y_pos_uniform = np.interp(xneg_shift, xneg, ypos_array[iday, :]).squeeze()
        y_neg_uniform = np.interp(xneg_shift, xneg, yneg_array[iday, :]).squeeze()

        y_pos_uniform = y_pos_uniform - minimum_filter(y_pos_uniform, size=100)
        y_neg_uniform = y_neg_uniform - minimum_filter(y_neg_uniform, size=100)

        params = Parameters()
        params.add('shift', value=0, vary=True, min=-np.ptp(xneg_new)/2, max=np.ptp(xneg_new)/2)

        results = fitter(fit_shift_centre, params, (xneg_shift, y_pos_uniform, y_neg_uniform))

        shift = results.params['shift'].value
        shift_array.append(shift)

        inds_pos = np.argwhere((xneg + shift/2 > wmin) * (xneg + shift/2 < wmax)).squeeze()
        inds_neg = np.argwhere((xneg - shift/2 > wmin) * (xneg - shift/2 < wmax)).squeeze()

        y_pos_uniform = np.interp(xneg_new, xneg[inds_pos] - shift/2, ypos_array[iday, inds_pos]).squeeze()
        y_neg_uniform = np.interp(xneg_new, xneg[inds_neg] + shift/2, yneg_array[iday, inds_neg]).squeeze()

        ypos_new_array.append(y_pos_uniform/np.max(y_pos_uniform))
        yneg_new_array.append(y_neg_uniform/np.max(y_neg_uniform))

    ypos_array = np.array(ypos_new_array).squeeze()
    yneg_array = np.array(yneg_new_array).squeeze()
    xneg = xneg_new

    plt.set_cmap('magma')
    ydata = (yneg_array + ypos_array) / 2
    for row in range(0, len(ypos_array[:, 0])):
        ydata[row, :] = ydata[row, :] - minimum_filter(ydata[row, :], size=filter_size)
    ind = np.argwhere((xneg > wmin) * (xneg < wmax))
    vmin = np.nanmin(ydata[:, ind])
    vmax = np.nanmax(ydata[:, ind])
    ydata[np.isnan(ydata)] = vmin
    axC.pcolormesh(edges, centres_to_edges(mjd_array),
                   ydata, vmin=vmin, vmax=vmax, shading='flat')

    plt.set_cmap('seismic')
    ydata = yneg_array - ypos_array
    for row in range(0, len(ypos_array[:, 0])):
        ydata[row, :] = ydata[row, :] - minimum_filter(ydata[row, :], size=filter_size)
    ind = np.argwhere((xneg > wmin) * (xneg < wmax))
    vmin = np.nanmean(ydata[:, ind]) - 4*np.nanstd(ydata[:, ind])
    vmax = np.nanmean(ydata[:, ind]) + 4*np.nanstd(ydata[:, ind])
    ydata[np.isnan(ydata)] = 0
    axR.pcolormesh(edges, centres_to_edges(mjd_array),
                   ydata, vmin=vmin, vmax=vmax, shading='flat')


    fig.canvas.draw()



A_sl.on_changed(update)
C_sl.on_changed(update)
phi_sl.on_changed(update)

button_save.on_clicked(fct_button_save)
button_fit.on_clicked(fct_button_fit)


plt.show()

