#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:07:26 2023

@author: dreardon
"""

import sys
import os
import numpy as np
import bilby
from bilby.core.likelihood import Analytical1DLikelihood
import matplotlib.pyplot as plt
from astropy import units as u
from scintools.scint_utils import read_par, read_results, get_earth_velocity, get_true_anomaly, pars_to_params, get_ssb_delay, make_lsr
from scintools.scint_models import effective_velocity_annual
from matplotlib import rc
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
rc('text', usetex=False)
mpl.rcParams.update({'font.size': 18})


def round_to_sig_figs(x, sig_figs):
    """Rounds a number to a specified number of significant figures."""
    if x == 0:
        return 0.0
    else:
        return round(x, -int(np.floor(np.log10(abs(x)))) + (sig_figs - 1))


def format_latex(value, uncertainty):
    """Formats the number and its uncertainty in LaTeX style with correct significant figures."""
    if uncertainty is None or uncertainty == '--':
        return f"{value:.3f}"  # Format without uncertainty

    if uncertainty == 0:
        return f"{value:.3f}(0)"

    # Determine the number of significant figures for uncertainty
    uncertainty_abs = abs(uncertainty)
    # Convert uncertainty to string in scientific notation
    uncertainty_str_full = "{:.15e}".format(uncertainty_abs)
    # Extract digits without decimal point and leading zeros
    digits_uncertainty = uncertainty_str_full.split(
        'e')[0].replace('.', '').lstrip('0')
    if digits_uncertainty:
        first_sig_digit = digits_uncertainty[0]
    else:
        first_sig_digit = '0'

    # Determine significant figures for uncertainty
    if first_sig_digit == '1':
        sig_figs_uncertainty = 2
    else:
        sig_figs_uncertainty = 1

    # Round uncertainty to significant figures
    uncertainty_rounded = round_to_sig_figs(uncertainty, sig_figs_uncertainty)

    # Find exponent (order of magnitude) of uncertainty
    exponent_uncertainty = int(np.floor(np.log10(abs(uncertainty_rounded))))
    # Calculate decimal places needed
    decimal_places = -exponent_uncertainty + (sig_figs_uncertainty - 1)

    # Round value to match decimal place of uncertainty
    value_rounded = round(value, decimal_places)

    # Prepare uncertainty display
    if decimal_places > 0:
        # For positive decimal places, format normally
        uncertainty_display = int(
            round(uncertainty_rounded * 10**decimal_places))
        value_str = f"{value_rounded:.{decimal_places}f}"
    else:
        # For zero or negative decimal places
        scale = 10 ** (-decimal_places)
        uncertainty_display = int(round(uncertainty_rounded / scale))
        value_str = str(int(round(value_rounded / scale) * scale))

    uncertainty_str = f"({uncertainty_display})"

    return f"{value_str}{uncertainty_str}"


class GaussianLikelihood(Analytical1DLikelihood):
    def __init__(self, x, y, func, sigma=None, freq=None, **kwargs):
        """
        A general Gaussian likelihood for known or unknown noise - the model
        parameters are inferred from the arguments of function

        Parameters
        ==========
        x, y: array_like
            The data to analyse
        func:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments
            will require a prior and will be sampled over (unless a fixed
            value is given).
        sigma: None, float, array_like
            If None, the standard deviation of the noise is unknown and will be
            estimated (note: this requires a prior to be given for sigma). If
            not None, this defines the standard-deviation of the data points.
            This can either be a single float, or an array with length equal
            to that for `x` and `y`.
        """

        super(GaussianLikelihood, self).__init__(x=x, y=y, func=func, **kwargs)
        self.sigma = sigma
        self.freq = freq

        # Check if sigma was provided, if not it is a parameter
        if self.sigma is None:
            self.parameters['sigma'] = None

    def log_likelihood(self):
        Q = self.model_parameters["Q"]
        F = self.model_parameters["F"]
        # Modifying the noise levels within the pdfs
        Sigma = np.sqrt((self.sigma * F)**2 + Q**2)

        log_l = np.sum(- (self.residual / Sigma)**2 / 2 -
                       np.log(2 * np.pi * Sigma**2) / 2)
        return log_l

    def __repr__(self):
        return self.__class__.__name__ + '(x={}, y={}, func={}, sigma={})' \
            .format(self.x, self.y, self.func.__name__, self.sigma)

    @property
    def sigma(self):
        """
        This checks if sigma has been set in parameters. If so, that value
        will be used. Otherwise, the attribute sigma is used. The logic is
        that if sigma is not in parameters the attribute is used which was
        given at init (i.e. the known sigma as either a float or array).
        """
        return self.parameters.get('sigma', self._sigma)

    @sigma.setter
    def sigma(self, sigma):
        if sigma is None:
            self._sigma = sigma
        elif isinstance(sigma, float) or isinstance(sigma, int):
            self._sigma = sigma
        elif len(sigma) == self.n:
            self._sigma = sigma
        else:
            raise ValueError('Sigma must be either float or array-like x.')


def arc_curvature_iso_bilby(U, s, vism_ra, vism_dec, KOM, F, Q,
                            params=None, mjd=None, vearth_ra=None,
                            vearth_dec=None):

    # Other parameters in lower-case
    d = params['d']  # pulsar distance in kpc
    dkm = d * kmpkpc  # kms
    params.add('s', value=s, vary=False)  # psr distance in kpc
    params.add('KOM', value=KOM, vary=False)  # psr distance in kpc

    veff_ra, veff_dec, vp_ra, vp_dec = \
        effective_velocity_annual(params, U,
                                  vearth_ra, vearth_dec, mjd=mjd)

    veff2 = (veff_ra - vism_ra)**2 + (veff_dec - vism_dec)**2

    # Calculate curvature model
    model = dkm * s * (1 - s)/(2 * veff2)  # in 1/(km * Hz**2)
    # Convert to 1/(m * mHz**2) for beta in 1/m and fdop in mHz
    model = model/1e9

    model = model / (u.m*u.mHz**2)
    vmodel = 1. / np.sqrt(2*model)
    vmodel = vmodel.to(u.km/u.s/u.kpc**0.5).value

    return vmodel


def arc_curvature_aniso_bilby(U, s, zeta, vism_zeta, KOM, F, Q,
                              params=None, mjd=None, vearth_ra=None,
                              vearth_dec=None):

    # Other parameters in lower-case
    d = params['d']  # pulsar distance in kpc
    dkm = d * kmpkpc  # kms
    params.add('s', value=s, vary=False)  # psr distance in kpc
    params.add('KOM', value=KOM, vary=False)  # psr distance in kpc

    veff_ra, veff_dec, vp_ra, vp_dec = \
        effective_velocity_annual(params, U,
                                  vearth_ra, vearth_dec, mjd=mjd)

    veff2 = (veff_ra*np.sin(zeta * np.pi/180) + veff_dec*np.cos(zeta * np.pi/180) -
             vism_zeta)**2

    # Calculate curvature model
    model = dkm * s * (1 - s)/(2 * veff2)  # in 1/(km * Hz**2)
    # Convert to 1/(m * mHz**2) for beta in 1/m and fdop in mHz
    model = model/1e9

    model = model / (u.m*u.mHz**2)
    vmodel = 1. / np.sqrt(2*model)
    vmodel = vmodel.to(u.km/u.s/u.kpc**0.5).value

    return vmodel


def arc_curvature_iso_bilby_D(U, s, vism_ra, vism_dec, KOM, F, Q, D,
                              params=None, mjd=None, vearth_ra=None,
                              vearth_dec=None):

    # Other parameters in lower-case
    d = D  # pulsar distance in kpc
    dkm = d * kmpkpc  # kms
    params.add('s', value=s, vary=False)  # psr distance in kpc
    params.add('KOM', value=KOM, vary=False)  # psr distance in kpc

    veff_ra, veff_dec, vp_ra, vp_dec = \
        effective_velocity_annual(params, U,
                                  vearth_ra, vearth_dec, mjd=mjd)

    veff2 = (veff_ra - vism_ra)**2 + (veff_dec - vism_dec)**2

    # Calculate curvature model
    model = dkm * s * (1 - s)/(2 * veff2)  # in 1/(km * Hz**2)
    # Convert to 1/(m * mHz**2) for beta in 1/m and fdop in mHz
    model = model/1e9

    model = model / (u.m*u.mHz**2)
    vmodel = 1. / np.sqrt(2*model)
    vmodel = vmodel.to(u.km/u.s/u.kpc**0.5).value

    return vmodel


def arc_curvature_aniso_bilby_D(U, s, zeta, vism_zeta, KOM, F, Q, D,
                                params=None, mjd=None, vearth_ra=None,
                                vearth_dec=None):

    # Other parameters in lower-case
    d = D  # pulsar distance in kpc
    dkm = d * kmpkpc  # kms
    params.add('s', value=s, vary=False)  # psr distance in kpc
    params.add('KOM', value=KOM, vary=False)  # psr distance in kpc

    veff_ra, veff_dec, vp_ra, vp_dec = \
        effective_velocity_annual(params, U,
                                  vearth_ra, vearth_dec, mjd=mjd)

    veff2 = (veff_ra*np.sin(zeta * np.pi/180) + veff_dec*np.cos(zeta * np.pi/180) -
             vism_zeta)**2

    # Calculate curvature model
    model = dkm * s * (1 - s)/(2 * veff2)  # in 1/(km * Hz**2)
    # Convert to 1/(m * mHz**2) for beta in 1/m and fdop in mHz
    model = model/1e9

    model = model / (u.m*u.mHz**2)
    vmodel = 1. / np.sqrt(2*model)
    vmodel = vmodel.to(u.km/u.s/u.kpc**0.5).value

    return vmodel


def round_sig(x, sig=2):
    return round(float(x), int(sig - np.floor(np.log10(abs(x))) - 1))


cwd = os.getcwd()

datadir = cwd + '/fit_arc_orbit/'
pars = read_par(cwd + '/J0437-4715.par')
kmpkpc = 3.085677581e16

params = pars_to_params(pars)
params.add('d', value=0.157, vary=False)  # psr distance in kpc


def modify_errors(vefferr, F, Q):
    return np.sqrt((vefferr * F)**2 + Q**2)


table_text = ''

table_text_new = ''
# for each arc
Narc = 0
iplot = 0
fig, ax = plt.subplots(5, 5, figsize=(24, 18), sharex=True)
minstds = []
s_sort_array = []
# results of above array are now below after running. Use this for sorting
s_sort_array_ = [0.08083450913421937,
                 0.06864291175141857,
                 0.08635965681983852,
                 0.15182043606532464,
                 0.07584254892084238,
                 0.1452028150368374,
                 0.1783365279612421,
                 0.4686719886816291,
                 0.4451358110809383,
                 0.47070703605686304,
                 0.32278904340039194,
                 0.2509586369154921,
                 0.2887454242637217,
                 0.015222023670964825,
                 0.022619079154248233,
                 0.02179877665324424,
                 0.004411523746437612,
                 0.00014735629723361298,
                 8.448405557075346e-05,
                 0.0032754530621145654,
                 0.0011869340580560556,
                 0.00010757147713708655,
                 0.00010733622186152458,
                 0.27223912836464254,
                 0.32682154177955325]

plotting_index = np.linspace(0, 24, 25)
inds = np.flip(np.argsort(s_sort_array_))

for iarc in inds:
    print(iarc)

    iplot += 1
    ax = plt.subplot(5, 5, iplot)

    label = None
    manual = False

    if iarc == 7:  # Sharpest, most isolated arc. Useful for D and KOM. Use Robert's manual measurements. Index is 0 here because it was the first measured
        data = np.loadtxt('model_int0.txt').squeeze()
        mjd = data[:, 2]
        veff = data[:, 3]
        vefferr = data[:, 4]

        manual = True

    if iarc == 21:  # faint bow shock -- use manual measurements
        label = 'Shock B'
        data = np.loadtxt('low_shock.txt', delimiter=',').squeeze()
        mjd = data[:, 0]
        veff_l = data[:, 1]
        vefferr_l = data[:, 2]
        veff_r = data[:, 3]
        vefferr_r = data[:, 4]
        veff = (veff_l + veff_r)/2
        vefferr = 0.5 * np.sqrt(vefferr_l**2 + vefferr_r**2)
        manual = True

    if iarc == 22:  # retrograde shock -- use manual measurements
        label = 'Shock C'
        data = np.loadtxt('retrograde.txt', delimiter=',').squeeze()
        mjd = data[:, 0]
        veff_l = data[:, 1]
        vefferr_l = data[:, 2]
        veff_r = data[:, 3]
        vefferr_r = data[:, 4]
        veff = (veff_l + veff_r)/2
        vefferr = 0.5 * np.sqrt(vefferr_l**2 + vefferr_r**2)
        manual = True
        # continue

    if iarc == 17:  # bow shock -- use manual measurements
        label = 'Shock A'
        data = read_results('bowshock.txt')

        mjd0 = np.array(data['mjd'], dtype=np.float64).squeeze()
        tobs = np.array(data['tobs'], dtype=np.float64).squeeze()
        bl = np.array(data['betaeta_left'], dtype=np.float64).squeeze()
        bel = np.array(data['betaetaerr_left'], dtype=np.float64).squeeze()
        br = np.array(data['betaeta_right'], dtype=np.float64).squeeze()
        ber = np.array(data['betaetaerr_right'], dtype=np.float64).squeeze()
        b = (bl + br)/2
        be = 0.5 * np.sqrt(bel**2 + ber**2)
        veff = []
        vefferr = []
        for i in range(len(bel)):
            bs = np.random.normal(loc=b[i], scale=be[i], size=1000000)
            bs = bs / (u.m*u.mHz**2)
            vs = 1/np.sqrt(2*bs)
            vs = vs.to(u.km/u.s/u.kpc**0.5).value
            veff.append(np.mean(vs))
            vefferr.append(np.std(vs))

        veff = np.array(veff).squeeze()
        vefferr = np.array(vefferr).squeeze()

        mjd0 += tobs/86400/2
        ssb_delays = get_ssb_delay(mjd0, pars['RAJ'], pars['DECJ'])
        mjd = mjd0 + np.divide(ssb_delays, 86400)

        manual = True

    if iarc == 18:  # secondary bow shock structure
        label = 'Shock D'
        print('Secondary bow shock')

    if label is None:
        Narc += 1
        label = str(Narc)

    data = np.loadtxt(datadir + 'model_int{}.txt'.format(iarc)).squeeze()

    if not manual:
        mjd = data[:, 2]
        veff = data[:, 6]
        vefferr = data[:, 7]

    # use only 5-sigma measurements
    ind = np.argwhere(vefferr <= 0.2*veff).squeeze()
    mjd = mjd[ind]
    veff = veff[ind]
    # Add in quadrature the minimum standard deviation of residuals from an initial analysis
    vefferr = np.sqrt(vefferr[ind]**2 + 2**2)

    """
    Model the curvature
    """
    print('Getting Earth velocity')
    vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'], pars['DECJ'])
    print('Getting true anomaly')
    U = get_true_anomaly(mjd, pars)

    model_mjd = np.linspace(58843.528331966417, 58849.26936321295, 1000)
    model_vearth_ra, model_vearth_dec = get_earth_velocity(
        model_mjd, pars['RAJ'], pars['DECJ'])
    model_U = get_true_anomaly(model_mjd, pars)
    model_p = model_U * 180/np.pi + pars['OM']
    model_p[model_p > 360] = model_p[model_p > 360] - 360

    if manual and iarc in [7]:
        # Also plot the October model
        model_mjd2 = np.linspace(58766.28233712221, 58772.02338303779, 1000)
        model_vearth_ra2, model_vearth_dec2 = get_earth_velocity(
            model_mjd2, pars['RAJ'], pars['DECJ'])
        model_U2 = get_true_anomaly(model_mjd2, pars)

        model_p2 = model_U2 * 180/np.pi + pars['OM']
        model_p2[model_p2 > 360] = model_p2[model_p2 > 360] - 360

    """
    ISOTROPIC
    """

    priors = dict()
    priors['s'] = bilby.core.prior.Uniform(0, 1, 's')
    # km/s
    priors['vism_ra'] = bilby.core.prior.Uniform(-1000, 1000, 'vism_ra')
    # km/s
    priors['vism_dec'] = bilby.core.prior.Uniform(-1000, 1000, 'vism_dec')
    priors['KOM'] = bilby.core.prior.Normal(207, 2.4, 'KOM')  # degrees
    priors['F'] = bilby.core.prior.Uniform(0, 100, 'F')
    priors['Q'] = bilby.core.prior.Uniform(0, np.std(veff), 'Q')  # km/s

    likelihood = GaussianLikelihood(U, veff, arc_curvature_iso_bilby, sigma=vefferr,
                                    params=params, mjd=mjd, vearth_ra=vearth_ra,
                                    vearth_dec=vearth_dec)

    if manual:
        outdir = datadir + '/{}_iso_wideprior_manual'.format(iarc)
    else:
        outdir = datadir + '/{}_iso_wideprior'.format(iarc)

    results = bilby.core.sampler.run_sampler(
        likelihood, priors=priors, sampler='dynesty', label='dynesty',
        nlive=250, verbose=False, resume=True, outdir=outdir,
        check_point_delta_t=600)

    # plt.figure(2)
    # results.plot_corner()
    lnev = results.log_evidence

    imx = np.argmax(results.posterior['log_likelihood'].values)

    F_i = results.posterior['F'][imx]
    Q_i = results.posterior['Q'][imx]

    s_i = results.posterior['s'][imx]
    vism_ra = results.posterior['vism_ra'][imx]
    vism_dec = results.posterior['vism_dec'][imx]
    KOM = results.posterior['KOM'][imx]

    s_i_err = np.std(results.posterior['s'])
    vism_ra_err = np.std(results.posterior['vism_ra'])
    vism_dec_err = np.std(results.posterior['vism_dec'])
    KOM_err = np.std(results.posterior['KOM'])

    model = arc_curvature_iso_bilby(U, s_i, vism_ra, vism_dec, KOM, F_i, Q_i,
                                    params=params, mjd=mjd, vearth_ra=vearth_ra,
                                    vearth_dec=vearth_dec)
    model_plot = arc_curvature_iso_bilby(model_U, s_i, vism_ra, vism_dec, KOM, F_i, Q_i,
                                         params=params, mjd=model_mjd, vearth_ra=model_vearth_ra,
                                         vearth_dec=model_vearth_dec)

    sortind = np.argsort(model_p)

    ax.plot(model_p[sortind],
            model_plot[sortind], color='C0', zorder=0)

    if manual and iarc in [7]:
        model_plot2 = arc_curvature_iso_bilby(model_U2, s_i, vism_ra, vism_dec, KOM, F_i, Q_i,
                                              params=params, mjd=model_mjd2, vearth_ra=model_vearth_ra2,
                                              vearth_dec=model_vearth_dec2)
        sortind2 = np.argsort(model_p2)
        ax.plot(model_p2[sortind2],
                model_plot2[sortind2], color='C0', zorder=0, linestyle=':')

    res1 = veff - model
    err1 = modify_errors(vefferr, F_i, Q_i)

    chisqr1 = np.sum(res1**2 / err1**2)

    """
    ANISOTROPIC
    """

    priors = dict()
    priors['s'] = bilby.core.prior.Uniform(0, 1, 's')
    priors['zeta'] = bilby.core.prior.Uniform(0, 180, 'zeta')  # degrees
    # km/s
    priors['vism_zeta'] = bilby.core.prior.Uniform(-1000, 1000, 'vism_zeta')
    priors['KOM'] = bilby.core.prior.Normal(207, 2.4, 'KOM')  # degrees
    priors['F'] = bilby.core.prior.Uniform(0, 100, 'F')
    priors['Q'] = bilby.core.prior.Uniform(0, np.std(veff), 'Q')  # km/s

    likelihood = GaussianLikelihood(U, veff, arc_curvature_aniso_bilby, sigma=vefferr,
                                    params=params, mjd=mjd, vearth_ra=vearth_ra,
                                    vearth_dec=vearth_dec)

    if manual:
        outdir = datadir + '/{}_aniso_wideprior_manual'.format(iarc)
    else:
        outdir = datadir + '/{}_aniso_wideprior'.format(iarc)

    results = bilby.core.sampler.run_sampler(
        likelihood, priors=priors, sampler='dynesty', label='dynesty',
        nlive=250, verbose=False, resume=True, outdir=outdir,
        check_point_delta_t=600)

    # plt.figure(2)
    # results.plot_corner()
    lnev_aniso = results.log_evidence
    lnbf = np.round(lnev_aniso - lnev, 1)

    imx = np.argmax(results.posterior['log_likelihood'].values)

    F = results.posterior['F'][imx]
    Q = results.posterior['Q'][imx]

    s = results.posterior['s'][imx]
    zeta = results.posterior['zeta'][imx]
    vism_zeta = results.posterior['vism_zeta'][imx]
    KOM = results.posterior['KOM'][imx]

    if lnbf > 0:
        ind = np.argwhere(modify_errors(vefferr, F, Q) < 0.2*veff).squeeze()
        ax.errorbar(U[ind]*180/np.pi, veff[ind],
                    yerr=modify_errors(vefferr, F, Q)[ind], fmt='kx', zorder=10)
        s_sort_array.append(s)
    else:
        ind = np.argwhere(modify_errors(vefferr, F, Q) < 0.2*veff).squeeze()
        ax.errorbar(U[ind]*180/np.pi, veff[ind], yerr=modify_errors(vefferr,
                                                                    F_i, Q_i)[ind], fmt='kx', zorder=10)
        s_sort_array.append(s_i)

    np.savez(datadir + '/arcmodel{}.npz'.format(iarc),
             s=s, zeta=zeta, vism_zeta=vism_zeta)

    s_err = np.std(results.posterior['s'])
    zeta_err = np.std(results.posterior['zeta'])
    vism_zeta_err = np.std(results.posterior['vism_zeta'])
    KOM_err = np.std(results.posterior['KOM'])

    model = arc_curvature_aniso_bilby(U, s, zeta, vism_zeta, KOM, F, Q,
                                      params=params, mjd=mjd, vearth_ra=vearth_ra,
                                      vearth_dec=vearth_dec)

    model_plot = arc_curvature_aniso_bilby(model_U, s, zeta, vism_zeta, KOM, F, Q,
                                           params=params, mjd=model_mjd, vearth_ra=model_vearth_ra,
                                           vearth_dec=model_vearth_dec)

    ax.plot(model_p[sortind],
            model_plot[sortind], color='C1', zorder=1)

    if manual and iarc in [7]:
        model_plot2 = arc_curvature_aniso_bilby(model_U2, s, zeta, vism_zeta, KOM, F, Q,
                                                params=params, mjd=model_mjd2, vearth_ra=model_vearth_ra2,
                                                vearth_dec=model_vearth_dec2)

        ax.plot(model_p2[sortind2],
                model_plot2[sortind2], color='C1', zorder=1, linestyle=':')

    res2 = veff - model
    err2 = modify_errors(vefferr, F, Q)

    chisqr2 = np.sum(res2**2 / err2**2)

    dchisqr = np.round(chisqr2 - chisqr1, 1)

    ax.set_xlim([0, 360])
    yl = ax.get_ylim()

    text = label if label is not None else str(Narc)

    if "Shock" in label:
        ax.text(240, yl[0] + 0.1*(yl[1]-yl[0]), text, color='black',
                bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.4'))
    else:
        ax.text(300, yl[0] + 0.1*(yl[1]-yl[0]), text, color='black',
                bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.4'))

    if iplot in [21, 22, 23, 24, 25]:
        ax.set_xlabel('Orbital phase (degrees)')
    if iplot in [1, 6, 11, 16, 21]:
        ax.set_ylabel(r'$W$ (km s$^{-1}$ $\sqrt{\rm kpc}$)')

    minstds.append(np.min([np.std(res1), np.std(res2)]))

    arcstring1 = "{} & {}$\pm${} & {}$\pm${} & {}$\pm${} & X \\\\ \n".format(
        iarc, s_i, s_i_err, vism_ra, vism_ra_err,
        vism_dec, vism_dec_err)

    arcstring2 = "{} & {}$\pm${} & {}$\pm${} & {}$\pm${} & X & Y \\\\ \n".format(
        iarc, s, s_err, zeta, zeta_err,
        vism_zeta, vism_zeta_err)

    table_text += arcstring1 + arcstring2

    arcstring_new = "{} & {} & {} & {} & {} & {} & {} & {} \\\\ \n".format(label,
                                                                           format_latex(
                                                                               s_i, s_i_err),
                                                                           format_latex(
                                                                               vism_ra, vism_ra_err),
                                                                           format_latex(
                                                                               vism_dec, vism_dec_err),
                                                                           format_latex(
                                                                               s, s_err),
                                                                           format_latex(
                                                                               zeta, zeta_err),
                                                                           format_latex(
                                                                               vism_zeta, vism_zeta_err),
                                                                           round_sig(lnbf, 2))

    table_text_new += arcstring_new

    # print('\n', arcstring_new, '\n')


plt.tight_layout()
plt.savefig(os.getcwd()+'/paper_plots/all_arcs.png')
plt.savefig(os.getcwd()+'/paper_plots/all_arcs.pdf')
plt.show()

print("")
print(table_text_new)
print("")


# Put bow shock velocities in the pulsar frame

vra = [43.2, 104.0, 1.8, 57.0]
vra_err = [2.8, 3.2, 6.3, 1.8]
vdec = [0.8, -71.8, 28.8, -17.3]
vdec_err = [3.0, 4.2, 6.6, 1.9]
ss = [0.000149, 0.00012, 0.000115, 0.000086]
ss_err = [0.000008, 0.00004, 0.000012, 0.000004]

for i in range(4):

    vism_ra = np.random.normal(loc=vra[i], scale=vra_err[i], size=10000)
    vism_dec = np.random.normal(loc=vdec[i], scale=vdec_err[i], size=10000)
    s = np.random.normal(loc=ss[i], scale=ss_err[i], size=10000)

    kmpkpc = 3.085677581e16
    secperyr = 86400*365.2425
    masrad = np.pi/(3600*180*1000)

    D = (1 - s) * 0.15679
    d = D * kmpkpc  # distance in km

    pmra_v = pars['PMRA'] * masrad * d / secperyr
    pmdec_v = pars['PMDEC'] * masrad * d / secperyr

    vism_ra_corr = vism_ra - pmra_v
    vism_dec_corr = vism_dec - pmdec_v

    print(np.mean(vism_ra_corr), np.std(vism_ra_corr))
    print(np.mean(vism_dec_corr), np.std(vism_dec_corr))
    print(" ")


vzeta = [27.4, 124.1, 18.4, 49]
vzeta_err = [2.1, 1.7, 5.9, 2]
zetaarr = [138.5, 140.9, 133.1, 139]
zeta_err = [2.6, 2.6, 3.3, 3]
ss = [0.000146, 0.000108, 0.000114, 0.000084]
ss_err = [0.000006, 0.000004, 0.000011, 0.000004]

# New numbers
vzeta = [26, -15, 125.3, 49]
vzeta_err = [2., 7, 1.6, 2]
zetaarr = [141, 133, 139, 139]
zeta_err = [3, 4, 2, 3]
ss = [0.000147, 0.000108, 0.000107, 0.000084]
ss_err = [0.000006, 0.000014, 0.000005, 0.000004]

for i in range(4):

    vism_zeta = np.random.normal(loc=vzeta[i], scale=vzeta_err[i], size=10000)
    zeta = np.random.normal(loc=zetaarr[i], scale=zeta_err[i], size=10000)
    s = np.random.normal(loc=ss[i], scale=ss_err[i], size=10000)

    kmpkpc = 3.085677581e16
    secperyr = 86400*365.2425
    masrad = np.pi/(3600*180*1000)

    D = (1 - s) * 0.15679
    d = D * kmpkpc  # distance in km

    pmra_v = pars['PMRA'] * masrad * d / secperyr
    pmdec_v = pars['PMDEC'] * masrad * d / secperyr

    print("Proper motion velocity RA, DEC:", pmra_v, pmdec_v)

    vism_ra_corr = vism_zeta * np.sin(zeta * np.pi/180) - pmra_v
    vism_dec_corr = vism_zeta * np.cos(zeta * np.pi/180) - pmdec_v

    vism_zeta_corr = vism_ra_corr * np.sin(zeta * np.pi / 180) + \
        vism_dec_corr * np.cos(zeta * np.pi / 180)

    print(np.mean(vism_zeta_corr), np.std(vism_zeta_corr))
    print(" ")
