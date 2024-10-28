#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:07:26 2023

@author: dreardon
"""

from scipy.optimize import curve_fit
import sys
import numpy as np
import bilby
import os
from bilby.core.likelihood import Analytical1DLikelihood
import matplotlib.pyplot as plt
from astropy import units as u
from scintools.scint_utils import read_par, get_earth_velocity, get_true_anomaly, pars_to_params, make_lsr
from scintools.scint_models import effective_velocity_annual
# from matplotlib import rc
# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)
# rc('text', usetex=False)


def make_lsr_new(d, raj, decj, pmra, pmdec, vr=0):
    from astropy.coordinates import BarycentricTrueEcliptic, LSR, SkyCoord
    from astropy import units as u

    coord = SkyCoord('{0} {1}'.format(raj, decj), unit=(u.hourangle, u.deg))
    ra = coord.ra.value
    dec = coord.dec.value

    # Initialise the barycentric coordinates with the LSR class and v_bary=0
    pm = LSR(ra=ra*u.degree, dec=dec*u.deg,
             pm_ra_cosdec=pmra*u.mas/u.yr,
             pm_dec=pmdec*u.mas/u.yr, distance=d*u.kpc,
             radial_velocity=vr*u.km/u.s,
             v_bary=(0.0*u.km/u.s, 0.0*u.km/u.s, 0.0*u.km/u.s))
    pm_ecliptic = pm.transform_to(BarycentricTrueEcliptic)

    # Get barycentric ecliptic coordinates
    elat = coord.barycentrictrueecliptic.lat.value
    elong = coord.barycentrictrueecliptic.lon.value
    pm_lat = pm_ecliptic.pm_lat.value
    pm_lon_coslat = pm_ecliptic.pm_lon_coslat.value

    bte = BarycentricTrueEcliptic(lon=elong*u.degree, lat=elat*u.degree,
                                  distance=d*u.kpc,
                                  pm_lon_coslat=pm_lon_coslat*u.mas/u.yr,
                                  pm_lat=pm_lat*u.mas/u.yr,
                                  radial_velocity=vr*u.km/u.s)

    # Convert barycentric back to LSR
    lsr_coord = bte.transform_to(LSR(v_bary=(11.1*u.km/u.s,
                                             12.24*u.km/u.s, 7.25*u.km/u.s)))

    return lsr_coord


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
        Sigma = np.sqrt((self.sigma * F)**2 + Q ** 2)

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


"""
Rounds float to number of sig figs
"""
# from math import log10, floor
# def round_sig(x, sig=2, small_value=1.0e-9):
#     #if x < 1:
#     #    sig = abs(int(math.log10(abs(x)))) + 1
#     if x > 10:
#         sig = len(str(int(x))) - 1
#     if sig==1 and str(x*10**10)[0] == '1':
#         sig+=1  # First digit, add a sig fig
#     value = round(x, sig - int(floor(log10(max(abs(x), abs(small_value))))) - 1)

#     while value < x:  # if rounded down,m fix it
#         x += 10**(-sig - 3)
#         value = round(x, sig - int(floor(log10(max(abs(x), abs(small_value))))) - 1)

#     if value >= 2:
#         value = int(value)

#     # #print(x, value)
#     # #print(" ")

#     return value


def get_sig(x, e):

    if abs(x) >= 1:
        sig = np.ceil(np.log10(abs(x))) - np.floor(np.log10(abs(e))) + 1
        if abs(x) < 2:
            sig += 1
    else:
        sig = 0
        i = 0
        c = '{:f}'.format(e)[i]
        while c == '0' or c == '.':
            i += 1
            c = '{:f}'.format(e)[i]
            sig += 1
        i = 0
        c = '{:f}'.format(x).split('.')[1][i]
        while c == '0':
            i += 1
            c = '{:f}'.format(x).split('.')[1][i]
            sig -= 1

    return int(sig)


def round_sig(x, sig=2):
    return round(float(x), int(sig - np.floor(np.log10(abs(x))) - 1))


def format_float(num):
    return np.format_float_positional(float(num), trim='-')


def make_string(x, e):
    sig = get_sig(x, e)
    x = str(round_sig(x, sig=sig))
    if abs(float(x)) <= 1 and len(x.split('.')[1]) < sig:
        x += '0'
    e = str(round_sig(e, sig=2))
    string = format_float(x) + '(' + format_float(e).replace('.', '') + ')'
    # string = '{}({})'.format(x, e).replace('(0.', '(')
    while '(0' in string:
        string = string.replace('(0', '(')
    # while '0)' in string:
    #    string = string.replace('0)', ')')
    #    string = string.replace('0(', '(')
    return string


cwd = os.getcwd()
datadir = cwd + '/fit_arc_orbit/'

pars = read_par(cwd + '/J0437-4715.par')
kmpkpc = 3.085677581e16

params = pars_to_params(pars)
params.add('d', value=0.157, vary=False)  # psr distance in kpc


def modify_errors(vefferr, F, Q):
    return np.sqrt((vefferr * F)**2 + Q ** 2)


def sinusoid(t, A, phi, C, om):
    """Model for the sinusoidal function."""
    return A * np.sin(om * t + phi) + C


def fit_sinusoid(t, y, om):
    """
    Fit a sinusoid of the form y(t) = A * sin(om * t + phi) + C to the data.

    Parameters:
    t : array-like
        Time data array.
    y : array-like
        Data to be fitted.
    om : float
        Known angular frequency.

    Returns:
    popt : array
        Optimal values for the parameters A, phi, C.
    pcov : 2d array
        The estimated covariance of popt.
    """
    # Initial guess for A, phi, and C (can be adjusted if needed)
    initial_guess = [np.max(y) - np.min(y), 0, np.mean(y)]

    # Use curve_fit to fit the sinusoid model to the data
    popt, pcov = curve_fit(lambda t, A, phi, C: sinusoid(
        t, A, phi, C, om), t, y, p0=initial_guess)

    return popt, pcov


# data = np.loadtxt(datadir + 'model_int0.txt').squeeze()
# mjd = data[:, 2]

# mjds = np.linspace(
#     np.mean(mjd) - pars['PB']/2, np.mean(mjd) + pars['PB']/2, 1000)


# print('Getting Earth velocity')
# vearth_ra, vearth_dec = get_earth_velocity(mjds, pars['RAJ'], pars['DECJ'])
# print('Getting true anomaly')
# U = get_true_anomaly(mjds, pars)

# s = 0.291
# vism_ra = 0
# vism_dec = 0
# KOM = 207

# model = arc_curvature_iso_bilby(U, s, vism_ra, vism_dec, KOM, 1, 0,
#                                 params=params, mjd=mjds, vearth_ra=vearth_ra,
#                                 vearth_dec=vearth_dec)

# zeta = 26.4
# vism_zeta = 31

# model_aniso = arc_curvature_aniso_bilby(U, s, zeta, vism_zeta, KOM, 1, 0,
#                                         params=params, mjd=mjds, vearth_ra=vearth_ra,
#                                         vearth_dec=vearth_dec)

# indsort = np.argsort(U)

# plt.plot(U[indsort], model[indsort])

# # Fit the sinusoid
# popt, pcov = fit_sinusoid(U, model, 1)

# # Extract fitted parameters
# A_fit, phi_fit, C_fit = popt
# print(f"Fitted parameters:\n A = {A_fit}\n phi = {phi_fit}\n C = {C_fit}")

# fit = sinusoid(U, A_fit, phi_fit, C_fit, 1)
# plt.plot(U[indsort], fit[indsort])

# print(np.max(np.abs(model - fit)))

# # plt.scatter(U, model_aniso)
# plt.xlabel('True Anomaly (deg)')
# plt.ylabel('W (km/s)')

# plt.show()

# sys.exit()

table_text = ''

table_text_new = ''
# for each arc
Narc = 0
for iarc in range(25):
    # for iarc in range(14):

    label = None

    if iarc == 21:  # faint bow shock -- use manual measurements
        continue

    if iarc == 22:  # retrograde shock -- use manual measurements
        continue

    if iarc == 17:  # bow shock
        continue
        # label = 'Shock A'
    #    print('bow shock')

    if iarc == 18:  # secondary bow shock
        continue
        # label = 'Shock B'

    #    print('Secondary bow shock')

    if label is None:
        Narc += 1
        label = str(Narc)

    data = np.loadtxt(datadir + 'model_int{}.txt'.format(iarc)).squeeze()

    mjd = data[:, 2]
    veff = data[:, 3]
    vefferr = data[:, 4]
    shift = data[:, 5]
    # veff = data[:, 6]
    # vefferr = data[:, 7]
    # shift = data[:, 8]

    """
    Model the curvature
    """
    print('Getting Earth velocity')
    vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'], pars['DECJ'])
    print('Getting true anomaly')
    U = get_true_anomaly(mjd, pars)

    plt.errorbar(U, veff, yerr=vefferr, fmt='kx')

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

    # priors['D'] = bilby.core.prior.Uniform(0.1, 0.2, 'D')

    likelihood = GaussianLikelihood(U, veff, arc_curvature_iso_bilby, sigma=vefferr,
                                    params=params, mjd=mjd, vearth_ra=vearth_ra,
                                    vearth_dec=vearth_dec)

    # outdir = datadir + '/{}_iso_wideprior_robert_D'.format(iarc)
    outdir = datadir + '/{}_iso_wideprior'.format(iarc)

    results = bilby.core.sampler.run_sampler(
        likelihood, priors=priors, sampler='dynesty', label='dynesty',
        nlive=250, verbose=False, resume=True, outdir=outdir,
        check_point_delta_t=600)

    # results.plot_corner()
    lnev = results.log_evidence

    imx = np.argmax(results.posterior['log_likelihood'].values)

    F = results.posterior['F'][imx]
    Q = results.posterior['Q'][imx]
    plt.errorbar(U, veff, yerr=modify_errors(vefferr, F, Q), fmt='.')

    s_i = results.posterior['s'][imx]
    vism_ra = results.posterior['vism_ra'][imx]
    vism_dec = results.posterior['vism_dec'][imx]
    KOM = results.posterior['KOM'][imx]

    s_i_err = np.std(results.posterior['s'])
    vism_ra_err = np.std(results.posterior['vism_ra'])
    vism_dec_err = np.std(results.posterior['vism_dec'])
    KOM_err = np.std(results.posterior['KOM'])

    model = arc_curvature_iso_bilby(U, s_i, vism_ra, vism_dec, KOM, F, Q,
                                    params=params, mjd=mjd, vearth_ra=vearth_ra,
                                    vearth_dec=vearth_dec)

    res1 = veff - model
    err1 = modify_errors(vefferr, F, Q)

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

    # priors['D'] = bilby.core.prior.Uniform(0.1, 0.2, 'D')

    likelihood = GaussianLikelihood(U, veff, arc_curvature_aniso_bilby, sigma=vefferr,
                                    params=params, mjd=mjd, vearth_ra=vearth_ra,
                                    vearth_dec=vearth_dec)

    # outdir = datadir + '/{}_aniso_wideprior_robert_D'.format(iarc)
    outdir = datadir + '/{}_aniso_wideprior'.format(iarc)

    results = bilby.core.sampler.run_sampler(
        likelihood, priors=priors, sampler='dynesty', label='dynesty',
        nlive=250, verbose=False, resume=True, outdir=outdir,
        check_point_delta_t=600)

    # results.plot_corner()
    lnev_aniso = results.log_evidence

    imx = np.argmax(results.posterior['log_likelihood'].values)

    F = results.posterior['F'][imx]
    Q = results.posterior['Q'][imx]
    plt.errorbar(U, veff, yerr=modify_errors(vefferr, F, Q), fmt='.')

    s = results.posterior['s'][imx]
    zeta = results.posterior['zeta'][imx]
    vism_zeta = results.posterior['vism_zeta'][imx]
    KOM = results.posterior['KOM'][imx]

    # np.savez(datadir + '/arcmodel{}_robert_D.npz'.format(iarc), s=s, zeta=zeta, vism_zeta=vism_zeta)
    np.savez(datadir + '/arcmodel{}.npz'.format(iarc),
             s=s, zeta=zeta, vism_zeta=vism_zeta)

    s_err = np.std(results.posterior['s'])
    zeta_err = np.std(results.posterior['zeta'])
    vism_zeta_err = np.std(results.posterior['vism_zeta'])
    KOM_err = np.std(results.posterior['KOM'])

    model = arc_curvature_aniso_bilby(U, s, zeta, vism_zeta, KOM, F, Q,
                                      params=params, mjd=mjd, vearth_ra=vearth_ra,
                                      vearth_dec=vearth_dec)

    res2 = veff - model
    err2 = modify_errors(vefferr, F, Q)

    chisqr2 = np.sum(res2**2 / err2**2)

    lnbf = np.round(lnev_aniso - lnev, 1)

    dchisqr = np.round(chisqr2 - chisqr1, 1)

    plt.title("{}: Log BF = {}, dChisqr = {}".format(iarc, lnbf, dchisqr))

    # plt.savefig(outdir + '/{}_robert.png'.format(iarc))
    # plt.savefig(outdir + '/{}.png'.format(iarc))
    plt.show()

    plt.errorbar(U, res1, yerr=err1, fmt='x')
    plt.errorbar(U, res2, yerr=err2, fmt='x')
    plt.title("{}: Residual".format(iarc))

    # plt.savefig(outdir + '/{}_residual_robert.png'.format(iarc))
    # plt.savefig(outdir + '/{}_residual.png'.format(iarc))
    plt.show()

    # continue

    # Make LSR

    kmpkpc = 3.085677581e16
    secperyr = 86400*365.2425
    masrad = np.pi/(3600*180*1000)

    pmra_v = []
    pmdec_v = []

    pmzeta_v = []
    # zeta_arr = []

    for i in range(100):

        vism_ra_i = np.random.normal(loc=vism_ra, scale=vism_ra_err, size=1)
        vism_dec_i = np.random.normal(loc=vism_dec, scale=vism_dec_err, size=1)

        vism_zeta_i = np.random.normal(
            loc=vism_zeta, scale=vism_zeta_err, size=1)
        zeta_i = np.random.normal(loc=zeta, scale=zeta_err, size=1)

        zeta_i = np.random.uniform(low=0, high=360, size=1)
        # zeta_arr.append(zeta_i)

        KOM_i = np.random.normal(loc=KOM, scale=KOM_err, size=1)

        # make RA and DEC components from vism_zeta
        vism_ra_a_i = vism_zeta_i * np.sin(zeta_i * np.pi/180)
        vism_dec_a_i = vism_zeta_i * np.cos(zeta_i * np.pi/180)

        s_i_i = np.random.normal(loc=s_i, scale=s_i_err, size=1)
        s_a_i = np.random.normal(loc=s, scale=s_err, size=1)
        while s_i_i <= 0 or s_i_i >= 1:
            s_i_i = np.random.normal(loc=s_i, scale=s_i_err, size=1)
        while s_a_i <= 0 or s_a_i >= 1:
            s_a_i = np.random.normal(loc=s, scale=s_err, size=1)

        D = (1 - s_i_i) * 0.15679
        D_a = (1 - s_a_i) * 0.15679

        pm_ra_cosdec = vism_ra_i * secperyr / masrad / D / kmpkpc
        pm_dec = vism_dec_i * secperyr / masrad / D / kmpkpc

        pm_ra_cosdec_a = vism_ra_a_i * secperyr / masrad / D_a / kmpkpc
        pm_dec_a = vism_dec_a_i * secperyr / masrad / D_a / kmpkpc

        pmra, pmdec = make_lsr(
            D, pars["RAJ"], pars["DECJ"], pm_ra_cosdec, pm_dec, vr=0)
        pmra_v.append(pmra * masrad * D * kmpkpc / secperyr)
        pmdec_v.append(pmdec * masrad * D * kmpkpc / secperyr)

        pmra_a, pmdec_a = make_lsr(
            D_a, pars["RAJ"], pars["DECJ"], pm_ra_cosdec_a, pm_dec_a, vr=0)
        # project back to zeta
        pmzeta_v.append(pmra_a * masrad * D_a * kmpkpc / secperyr * np.sin(zeta_i * np.pi / 180) +
                        pmdec_a * masrad * D_a * kmpkpc / secperyr * np.cos(zeta_i * np.pi / 180))

    print("VISM RA ", vism_ra, np.mean(pmra_v), np.mean(pmra_v) - vism_ra)
    print("VISM DEC ", vism_dec, np.mean(pmdec_v), np.mean(pmdec_v) - vism_dec)
    print("VISM ZETA ", vism_zeta, np.mean(
        pmzeta_v), np.mean(pmzeta_v) - vism_zeta)

    print("TOTAL", np.sqrt(vism_ra**2 + vism_dec**2),
          np.sqrt(np.mean(pmra_v)**2 + np.mean(pmdec_v)**2))
    print(" ")
    print(" ")

    # string = make_string(x, e)

    arcstring1 = "{} & {}$\pm${} & {}$\pm${} & {}$\pm${} & X \\\\ \n".format(
        iarc, s_i, s_i_err, np.mean(pmra_v), np.std(pmra_v),
        np.mean(pmdec_v), np.std(pmdec_v))

    arcstring2 = "{} & {}$\pm${} & {}$\pm${} & {}$\pm${} & X & Y \\\\ \n".format(
        iarc, s, s_err, zeta, zeta_err,
        np.mean(pmzeta_v), np.std(pmzeta_v))

    table_text += arcstring1 + arcstring2

    arcstring_new = "{} & {} & {} & {} & {} & {} & {} & {} \\\\ \n".format(label,
                                                                           make_string(
                                                                               s_i, s_i_err),
                                                                           make_string(
                                                                               np.mean(pmra_v), np.std(pmra_v)),
                                                                           make_string(
                                                                               np.mean(pmdec_v), np.std(pmdec_v)),
                                                                           make_string(
                                                                               s, s_err),
                                                                           make_string(
                                                                               zeta, zeta_err),
                                                                           make_string(
                                                                               np.mean(pmzeta_v), np.std(pmzeta_v)),
                                                                           round_sig(lnbf, 2))

    table_text_new += arcstring_new

    print('\n', arcstring_new, '\n')


print(table_text)

print("")

print(table_text_new)


# for lx in np.random.normal(loc=0, scale=3, size=100):
#     x = 10**lx
#     le = np.random.normal(loc=0, scale=1, size=1)[0]
#     e = 10**le
#     print(x, e, make_string(x, e))

sys.exit()

vra = [43.2, 104.0, 1.8]
vra_err = [2.8, 3.2, 6.3]
vdec = [0.8, -71.8, 28.8]
vdec_err = [3.0, 4.2, 6.6]
ss = [0.000149, 0.00012, 0.000115]
ss_err = [0.000008, 0.00004, 0.000012]

for i in range(3):

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


vzeta = [27.4, 124.1, 18.4]
vzeta_err = [2.1, 1.7, 5.9]
zetaarr = [138.5, 140.9, 133.1]
zeta_err = [2.6, 2.6, 3.3]
ss = [0.000146, 0.000108, 0.000114]
ss_err = [0.000006, 0.000004, 0.000011]

for i in range(3):

    vism_zeta = np.random.normal(loc=vzeta[i], scale=vzeta_err[i], size=10000)
    zeta = np.random.normal(loc=zetaarr[i], scale=zeta_err[i], size=10000)
    s = np.random.normal(loc=ss[i], scale=ss_err[i], size=10000)

    np.savez(datadir + '/arcmodel{}.npz'.format(14 + i),
             s=ss[i], zeta=zetaarr[i], vism_zeta=vzeta[i])

    kmpkpc = 3.085677581e16
    secperyr = 86400*365.2425
    masrad = np.pi/(3600*180*1000)

    D = (1 - s) * 0.15679
    d = D * kmpkpc  # distance in km

    pmra_v = pars['PMRA'] * masrad * d / secperyr
    pmdec_v = pars['PMDEC'] * masrad * d / secperyr

    vism_ra_corr = vism_zeta * np.sin(zeta * np.pi/180) - pmra_v
    vism_dec_corr = vism_zeta * np.cos(zeta * np.pi/180) - pmdec_v

    vism_zeta_corr = vism_ra_corr * np.sin(zeta * np.pi / 180) + \
        vism_dec_corr * np.cos(zeta * np.pi / 180)

    print(np.mean(vism_zeta_corr), np.std(vism_zeta_corr))
    print(" ")


lsrcoord = make_lsr_new(
    0.15679, pars["RAJ"], pars["DECJ"], pars['PMRA'], pars['PMDEC'], vr=-60.1)
