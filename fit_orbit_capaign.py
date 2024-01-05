import os
import sys
import glob
import argparse
import signal
import numpy as np
import astropy.units as u
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import minimum_filter
from matplotlib.widgets import Slider, Button
from scintools.scint_utils import get_ssb_delay, read_par

def fit_parabola(x, curv, x0=0, C=0):
    return curv*(x-x0)**2 + C

def fit_arc(x, y, xw=20):
    x0 = x[np.argmax(y)]
    fitrange = np.argwhere( abs(x-x0) < xw).squeeze()
    xfit = x[fitrange]
    yfit = y[fitrange]

    p0 = [-0.1, x0, 1]
    popt, pcov = curve_fit(fit_parabola, xfit, yfit, p0=p0, maxfev=20000,
                             bounds=([-np.inf, min(xfit), -np.inf], [0, max(xfit), np.inf]) )

    fdmax = popt[1]
    fderr = np.sqrt(pcov[1][1])

    return fdmax, fderr, xfit, popt, pcov

def load_veffprofiles(npzfiles):

    mjd_array = []  # mjds of data edges
    tobs_array = []
    ypos_array = []
    yneg_array = []
    mjds = []  # mjds of data centres

    for file in npzfiles:

        print(file)

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

    # Initially cut the data to 120% the wmin and wmax range. Allowing room for shifting

    ind = np.argwhere((xneg > wmin/2) * (xneg < wmax*2))
    ypos_array = ypos_array[:, ind].squeeze()
    yneg_array = yneg_array[:, ind].squeeze()
    xneg = xneg[ind].squeeze()

    profs = [yneg_array, ypos_array]

    return profs, xneg, mjd_array



#### Interactive code functions ####

def signal_handler(sig, frame):
    print('Exiting, writing results in {0}'.format(outfile))
    results.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

cwd = os.getcwd()
datadir = cwd + '/data/'
plotdir = cwd + 'plot/'
outdir = cwd + '/tuned/'

parser = argparse.ArgumentParser(description='Run interactive arc fitter on directory of dynspecs')
parser.add_argument("-imodel", default=0, type=int)

a = parser.parse_args()
imodel = a.imodel
outfile = 'fit_arc_orbit/model_int{0}.txt'.format(imodel)

data = np.load(outdir + '/arcfit_{0}.npz'.format(imodel))
delmax = np.float(data['delmax'])
if delmax == 2.0 or delmax == 1.0:
    delmax = int(delmax)
A = data['A']
C = data['C']
phi = data['phi']
wmin = data['wmin']
wmax = data['wmax']
shift_array = data['shift_array']
filter_size = int(data['filter_size'])

npzfiles = np.sort(glob.glob(datadir + 'J*2019-12*WFinpainted_normsspec_{0}us_startbin1_new.npz'.format(delmax)) )

xrange = [wmin, wmax]
filterpars = [300, 1000]

profs, vax, mjds = load_veffprofiles(npzfiles)

profs = np.array(profs)
Lprofs = profs[0]
Rprofs = profs[1]

parfile = cwd + '/J0437-4715.par'
pars = read_par(parfile)

mjddata = np.copy(mjds)
ssb_delays = get_ssb_delay(mjddata, pars['RAJ'], pars['DECJ'])
mjddata += np.divide(ssb_delays, 86400)

mjds = np.array(mjds) - 58846  # subtract mean mjd from modelling

om = 1.094432164378865
model = A * np.sin(om * mjds + phi ) + C

Ndata = profs.shape[1]
print(Ndata)
print(shift_array)
vwindow = 0.02*np.mean(model)

Lwindowprofs = []
Rwindowprofs = []
vfits = np.zeros( (Ndata, 2) )
vfiterrs = np.zeros( (Ndata, 2) )

#create a ordered grid of axes, not one in top of the others

pr = 0.7
yr = 0.8

results = open(outfile, 'a')

for i in range(Ndata):
    profL = profs[0,i]
    profR = profs[1,i]
    mi = model[i]
    try:
        dphase = shift_array[i] / 2
    except IndexError:
        dphase = np.mean(shift_array) / 2
        if np.isnan(dphase):
            dphase = 0

    Lfitrange = np.argwhere(abs(vax-mi+dphase)<vwindow).squeeze()
    Rfitrange = np.argwhere(abs(vax-mi-dphase)<vwindow).squeeze()

    vfitL, vfitLerr, xfitL, poptL, pcovL = fit_arc(vax[Lfitrange], profL[Lfitrange], xw=vwindow)
    vfitR, vfitRerr, xfitR, poptR, pcovR = fit_arc(vax[Rfitrange], profR[Rfitrange], xw=vwindow)

    if i == 0:
        Lwindowprofs.append(profL[Lfitrange][:-1])
        Rwindowprofs.append(profR[Rfitrange][:-1])
    else:
        Lwindowprofs.append(profL[Lfitrange][:len(Lwindowprofs[0])])
        Rwindowprofs.append(profR[Rfitrange][:len(Rwindowprofs[0])])

    axcolor = 'lightgoldenrodyellow'
    fig = plt.figure(figsize=(15, 8))
    axL = plt.subplot(221)
    axR = plt.subplot(222)

    filtL = minimum_filter(profL, size=filter_size)
    filtR = minimum_filter(profR, size=filter_size)

    axL.plot(vax, profL)
    axR.plot(vax, profR)
    axL.plot(vax, filtL)
    axR.plot(vax, filtR)


    vfits[i, 0] = vfitL
    vfits[i, 1] = vfitR
    vfiterrs[i, 0] = vfitLerr
    vfiterrs[i, 1] = vfitRerr

    lL, = axL.plot([vfitL, vfitL], [-100, 100], color='tab:orange', linestyle='dotted')
    lR, = axR.plot([vfitR, vfitR], [-100, 100], color='tab:orange', linestyle='dotted')
    parL, = axL.plot(xfitL, fit_parabola(xfitL, *poptL), color='tab:red' )
    parR, = axR.plot(xfitR, fit_parabola(xfitR, *poptR), color='tab:red')

    rangeloL, = axL.plot([mi-vwindow-dphase,mi-vwindow-dphase], [-100, 100], color='tab:purple', alpha=0.5 )
    rangehiL, = axL.plot([mi+vwindow-dphase,mi+vwindow-dphase], [-100, 100], color='tab:purple', alpha=0.5 )
    rangeloR, = axR.plot([mi-vwindow+dphase,mi-vwindow+dphase], [-100, 100], color='tab:purple', alpha=0.5 )
    rangehiR, = axR.plot([mi+vwindow+dphase,mi+vwindow+dphase], [-100, 100], color='tab:purple', alpha=0.5 )

    axL.set_xlim(pr*mi, (1+pr)*mi)
    axR.set_xlim(pr*mi, (1+pr)*mi)
    xplotrange = np.argwhere( (vax<(1+pr)*mi) & (vax>pr*mi) ).squeeze()
    axL.set_ylim(min(profL[xplotrange]), max(profL[xplotrange]))
    axR.set_ylim(min(profR[xplotrange]), max(profR[xplotrange]))

    axLm = plt.subplot(223)
    axRm = plt.subplot(224)

    axLm.plot(vax, profL - filtL)
    axRm.plot(vax, profR - filtR)

    vfitLm, vfitLmerr, xfitLm, poptLm, pcovLm = fit_arc(vax[Lfitrange], profL[Lfitrange] - filtL[Lfitrange], xw=vwindow)
    vfitRm, vfitRmerr, xfitRm, poptRm, pcovRm = fit_arc(vax[Rfitrange], profR[Rfitrange] - filtR[Rfitrange], xw=vwindow)

    lLm, = axLm.plot([vfitLm, vfitLm], [-100, 100], color='tab:orange', linestyle='dotted')
    lRm, = axRm.plot([vfitRm, vfitRm], [-100, 100], color='tab:orange', linestyle='dotted')
    parLm, = axLm.plot(xfitL, fit_parabola(xfitL, *poptLm), color='tab:red' )
    parRm, = axRm.plot(xfitR, fit_parabola(xfitR, *poptRm), color='tab:red')

    rangeloLm, = axLm.plot([mi-vwindow-dphase,mi-vwindow-dphase], [-100, 100], color='tab:purple', alpha=0.5 )
    rangehiLm, = axLm.plot([mi+vwindow-dphase,mi+vwindow-dphase], [-100, 100], color='tab:purple', alpha=0.5 )
    rangeloRm, = axRm.plot([mi-vwindow+dphase,mi-vwindow+dphase], [-100, 100], color='tab:purple', alpha=0.5 )
    rangehiRm, = axRm.plot([mi+vwindow+dphase,mi+vwindow+dphase], [-100, 100], color='tab:purple', alpha=0.5 )

    axLm.set_xlim(pr*mi, (1+pr)*mi)
    axRm.set_xlim(pr*mi, (1+pr)*mi)
    axL.set_ylim(min(profL[xplotrange]), max(profL[xplotrange]))
    axR.set_ylim(min(profR[xplotrange]), max(profR[xplotrange]))
    xplotrange = np.argwhere( (vax<(1+pr)*mi) & (vax>pr*mi) ).squeeze()
    axLm.set_ylim(min(profL[xplotrange] - filtL[xplotrange]), max(profL[xplotrange] - filtL[xplotrange]))
    axRm.set_ylim(min(profR[xplotrange] - filtR[xplotrange]), max(profR[xplotrange] - filtR[xplotrange]))


    axRm.set_xlabel('W (km/s/sqrkpc)', fontsize=16)
    axLm.set_xlabel('W (km/s/sqrkpc)', fontsize=16)


    # create your plots in the global space.
    # you are going to reference these lines, so you need to make them visible
    # to the update functione, instead of creating them inside a function
    # (and thus losing them at the end of the function)
    # same as usual
    axcolor = 'lightgoldenrodyellow'
    ax_pos = plt.axes([0.1, 0.95, 0.6, 0.03], facecolor=axcolor)
    ax_phase = plt.axes([0.1, 0.9, 0.25, 0.03], facecolor=axcolor)
    ax_range = plt.axes([0.45, 0.9, 0.25, 0.03], facecolor=axcolor)
    ax_fit = plt.axes([0.75, 0.95, 0.15, 0.03], facecolor=axcolor)
    ax_save = plt.axes([0.75, 0.9, 0.15, 0.03], facecolor=axcolor)

    spos = Slider(ax_pos, 'W', 0, 2*mi, valinit=mi)
    sphase = Slider(ax_phase, 'phasegrad', -mi/10., mi/10., valinit=dphase)
    srange = Slider(ax_range, 'fitrange', 0, mi/5., valinit=vwindow)
    button_fit = Button(ax_fit, "fit")
    button_save = Button(ax_save, "save")

    def update(val):
        # you don't need to declare the variables global, as if you don't
        # assign a value to them python will recognize them as global
        # without problem

        xr = srange.val
        lx = spos.val - sphase.val
        rx = spos.val + sphase.val

        lL.set_xdata([lx, lx])
        lR.set_xdata([rx, rx])

        lLm.set_xdata([lx, lx])
        lRm.set_xdata([rx, rx])

        rangeloL.set_xdata([lx-xr, lx-xr])
        rangehiL.set_xdata([lx+xr, lx+xr])
        rangeloR.set_xdata([rx-xr, rx-xr])
        rangehiR.set_xdata([rx+xr, rx+xr])

        rangeloLm.set_xdata([lx-xr, lx-xr])
        rangehiLm.set_xdata([lx+xr, lx+xr])
        rangeloRm.set_xdata([rx-xr, rx-xr])
        rangehiRm.set_xdata([rx+xr, rx+xr])

        # you need to update only the canvas of the figure
        fig.canvas.draw()

    spos.on_changed(update)
    sphase.on_changed(update)
    srange.on_changed(update)

    # - define functions of widgets
    def fct_button_fit(event):
        global vfitmean
        global vfiterrmean
        global vfitdiff
        global vfitmean_m
        global vfiterrmean_m
        global vfitdiff_m

        Lfitrange_fit = np.argwhere(abs(vax-spos.val+sphase.val) < srange.val).squeeze()
        Rfitrange_fit = np.argwhere(abs(vax-spos.val-sphase.val) < srange.val).squeeze()

        vfitL, vfitLerr, xfitL_fit, poptL, pcovL = fit_arc(vax[Lfitrange_fit], profL[Lfitrange_fit], xw=srange.val )
        vfitR, vfitRerr, xfitR_fit, poptR, pcovR = fit_arc(vax[Rfitrange_fit], profR[Rfitrange_fit], xw=srange.val )
        parL.set_ydata(fit_parabola(xfitL_fit, *poptL))
        parL.set_xdata(xfitL_fit)
        parR.set_ydata(fit_parabola(xfitR_fit, *poptR))
        parR.set_xdata(xfitR_fit)
        # = axL.plot(xfitL, fit_parabola(xfitL, *poptL), color='tab:red' )
        #parR, = axR.plot(xfitR, fit_parabola(xfitR, *poptR), color='tab:red')
        print("Arc measurements:")
        print(vfitL, vfitLerr)
        print(vfitR, vfitRerr)

        vfitmean = (vfitL + vfitR)/2.
        vfiterrmean = (vfitRerr+vfitLerr)/(2*np.sqrt(2))
        vfitdiff = (vfitL - vfitR)

        print(vfitmean, vfiterrmean, vfitdiff)

        vfitLm, vfitLmerr, xfitLm_fit, poptLm, pcovLm = fit_arc(vax[Lfitrange_fit], profL[Lfitrange_fit] - filtL[Lfitrange_fit], xw=srange.val )
        vfitRm, vfitRmerr, xfitRm_fit, poptRm, pcovRm = fit_arc(vax[Rfitrange_fit], profR[Rfitrange_fit] - filtR[Rfitrange_fit], xw=srange.val )
        parLm.set_ydata(fit_parabola(xfitLm_fit, *poptLm))
        parLm.set_xdata(xfitLm_fit)
        parRm.set_ydata(fit_parabola(xfitRm_fit, *poptRm))
        parRm.set_xdata(xfitRm_fit)

        fig.canvas.draw()

        print("\n Min-filtered arc measurements:")
        print(vfitLm, vfitLmerr)
        print(vfitRm, vfitRmerr)

        vfitmean_m = (vfitLm + vfitRm)/2.
        vfiterrmean_m = (vfitRmerr+vfitLmerr)/(2*np.sqrt(2))
        vfitdiff_m = (vfitLm - vfitRm)

        print(vfitmean_m, vfiterrmean_m, vfitdiff_m)
        print(" ")

    def fct_button_save(event):
        path = plotdir + '/fit_{}/'.format(imodel)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + '/fit_{0}_{1}.png'.format(imodel, i))
        results.write("{0} {1} {2} {3} {4} {5} {6} {7} {8} \n".format(imodel, i, mjddata[i], vfitmean, vfiterrmean, vfitdiff, vfitmean_m, vfiterrmean_m, vfitdiff_m) )
        print("Saving to {0}, integration {1} of {2}, model {3}".format(outfile, i, len(mjddata), imodel))

    button_fit.on_clicked(fct_button_fit)
    button_save.on_clicked(fct_button_save)

    plt.show()

results.close()
