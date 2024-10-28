#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:07:26 2023

@author: dreardon
"""

import os
import matplotlib as mpl
from matplotlib import rc
import numpy as np
import bilby
from bilby.core.likelihood import Analytical1DLikelihood
import matplotlib.pyplot as plt
from astropy import units as u
from scintools.scint_utils import read_par, get_earth_velocity, get_true_anomaly, pars_to_params
from scintools.scint_models import effective_velocity_annual


def compute_contour_levels(P, levels):
    """
    Compute contour levels for the given cumulative probability levels.
    """
    P_flat = P.flatten()
    P_sort = np.sort(P_flat)[::-1]  # Sort in descending order
    P_cumsum = np.cumsum(P_sort)
    P_cumsum /= P_cumsum[-1]  # Normalize to 1

    levels_list = []
    for prob in levels:
        idx = np.searchsorted(P_cumsum, prob)
        level = P_sort[idx]
        levels_list.append(level)

    return levels_list[::-1]  # return levels increasing


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


def arc_curvature_iso_bilby_D(U, s, vism_ra, vism_dec, KOM, F, Q, D,
                              params=None, mjd=None, vearth_ra=None,
                              vearth_dec=None):

    # Other parameters in lower-case
    d = D  # pulsar distance in kpc
    dkm = d * kmpkpc  # kms
    params.add('d', value=D, vary=False)  # psr distance in kpc
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
    params.add('d', value=D, vary=False)  # psr distance in kpc
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


datadir = os.getcwd() + '/manual/'
pardir = os.getcwd()
pars = read_par(pardir + '/J0437-4715.par')
kmpkpc = 3.085677581e16

params = pars_to_params(pars)
params.add('d', value=0.157, vary=False)  # psr distance in kpc


def modify_errors(vefferr, F, Q):
    return np.sqrt((vefferr * F)**2 + Q ** 2)


fit = False

if fit:
    for iarc in range(13):
        if iarc == 6 or iarc == 7:
            continue

        data = np.loadtxt(datadir + 'model_int{}.txt'.format(iarc)).squeeze()

        mjd = data[:, 2]
        veff = data[:, 3]
        vefferr = data[:, 4]
        shift = data[:, 5]

        """
        Model the curvature
        """
        print('Getting Earth velocity')
        vearth_ra, vearth_dec = get_earth_velocity(
            mjd, pars['RAJ'], pars['DECJ'])
        print('Getting true anomaly')
        U = get_true_anomaly(mjd, pars)

        plt.errorbar(U, veff, yerr=vefferr, fmt='kx')
        plt.show()

        if iarc == 10:

            """
            ISOTROPIC
            """
            priors = dict()
            priors['s'] = bilby.core.prior.Uniform(0, 1, 's')
            # km/s
            priors['vism_ra'] = bilby.core.prior.Uniform(
                -1000, 1000, 'vism_ra')
            # km/s
            priors['vism_dec'] = bilby.core.prior.Uniform(
                -1000, 1000, 'vism_dec')
            # priors['KOM'] = bilby.core.prior.Normal(207, 2.4, 'KOM')  # degrees
            priors['KOM'] = bilby.core.prior.Uniform(0, 360, 'KOM')  # degrees
            priors['F'] = bilby.core.prior.Uniform(0, 100, 'F')
            priors['Q'] = bilby.core.prior.Uniform(
                0, np.std(veff), 'Q')  # km/s
            priors['D'] = bilby.core.prior.Uniform(0.01, 0.5, 'D')

            likelihood = GaussianLikelihood(U, veff, arc_curvature_iso_bilby_D, sigma=vefferr,
                                            params=params, mjd=mjd, vearth_ra=vearth_ra,
                                            vearth_dec=vearth_dec)

            outdir = datadir + '/{}_iso_KOMwide_D_new'.format(iarc)

            results = bilby.core.sampler.run_sampler(
                likelihood, priors=priors, sampler='ptemcee', label='ptemcee',
                nlive=250, verbose=False, resume=True, outdir=outdir,
                check_point_delta_t=600, nsamples=100000)

            parameters = {}
            parameters['D'] = 0.15679
            results.plot_corner()

            lnev = results.log_evidence

            imx = np.argmax(results.posterior['log_likelihood'].values)

            F = results.posterior['F'][imx]
            Q = results.posterior['Q'][imx]
            # plt.errorbar(U, veff, yerr=modify_errors(vefferr, F, Q), fmt='.')

            s_i = results.posterior['s'][imx]
            D_i = results.posterior['D'][imx]
            vism_ra = results.posterior['vism_ra'][imx]
            vism_dec = results.posterior['vism_dec'][imx]
            KOM = results.posterior['KOM'][imx]

            s_i_err = np.std(results.posterior['s'])
            D_i_err = np.std(results.posterior['D'])
            vism_ra_err = np.std(results.posterior['vism_ra'])
            vism_dec_err = np.std(results.posterior['vism_dec'])
            KOM_err = np.std(results.posterior['KOM'])

            D_i_samps = results.posterior['D']
            KOM_samps = results.posterior['KOM']

            model = arc_curvature_iso_bilby_D(U, s_i, vism_ra, vism_dec, KOM, F, Q, D_i,
                                              params=params, mjd=mjd, vearth_ra=vearth_ra,
                                              vearth_dec=vearth_dec)

            res1 = veff - model
            err1 = modify_errors(vefferr, F, Q)

            chisqr1 = np.sum(res1**2 / err1**2)

            np.savez(datadir + '/arcmodel{}_KOMwide_D.npz'.format(iarc),
                     D_samps=D_i_samps, KOM_samps=KOM_samps)

        else:

            """
            ANISOTROPIC
            """

            priors = dict()
            priors['s'] = bilby.core.prior.Uniform(0, 1, 's')
            priors['zeta'] = bilby.core.prior.Uniform(
                0, 180, 'zeta')  # degrees
            # km/s
            priors['vism_zeta'] = bilby.core.prior.Uniform(
                -1000, 1000, 'vism_zeta')
            priors['KOM'] = bilby.core.prior.Uniform(0, 360, 'KOM')  # degrees
            priors['F'] = bilby.core.prior.Uniform(0, 100, 'F')
            priors['Q'] = bilby.core.prior.Uniform(
                0, np.std(veff), 'Q')  # km/s

            priors['D'] = bilby.core.prior.Uniform(0.01, 0.5, 'D')

            likelihood = GaussianLikelihood(U, veff, arc_curvature_aniso_bilby_D, sigma=vefferr,
                                            params=params, mjd=mjd, vearth_ra=vearth_ra,
                                            vearth_dec=vearth_dec)

            outdir = datadir + '/{}_aniso_KOMwide_D_new'.format(iarc)

            results = bilby.core.sampler.run_sampler(
                likelihood, priors=priors, sampler='ptemcee', label='ptemcee',
                nlive=250, verbose=False, resume=True, outdir=outdir,
                check_point_delta_t=600, nsamples=100000)

            results.plot_corner()
            lnev_aniso = results.log_evidence

            imx = np.argmax(results.posterior['log_likelihood'].values)

            F = results.posterior['F'][imx]
            Q = results.posterior['Q'][imx]
            # plt.errorbar(U, veff, yerr=modify_errors(vefferr, F, Q), fmt='.')

            s = results.posterior['s'][imx]
            D = results.posterior['D'][imx]
            zeta = results.posterior['zeta'][imx]
            vism_zeta = results.posterior['vism_zeta'][imx]
            KOM = results.posterior['KOM'][imx]

            D_samps = results.posterior['D']
            KOM_samps = results.posterior['KOM']

            np.savez(
                datadir + '/arcmodel{}_KOMwide_D.npz'.format(iarc), D_samps=D_samps, KOM_samps=KOM_samps)

            s_err = np.std(results.posterior['s'])
            D_err = np.std(results.posterior['D'])
            zeta_err = np.std(results.posterior['zeta'])
            vism_zeta_err = np.std(results.posterior['vism_zeta'])
            KOM_err = np.std(results.posterior['KOM'])

            model = arc_curvature_aniso_bilby_D(U, s, zeta, vism_zeta, KOM, F, Q, D,
                                                params=params, mjd=mjd, vearth_ra=vearth_ra,
                                                vearth_dec=vearth_dec)

            res2 = veff - model
            err2 = modify_errors(vefferr, F, Q)

            chisqr2 = np.sum(res2**2 / err2**2)


"""
Load distance posteriors for each screen
"""

mpl.rcParams.update(mpl.rcParamsDefault)
rc('text', usetex=False)
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rcParams.update({'font.size': 14})

# plt.subplots(1, 2, figsize=(16, 8))
plt.figure(figsize=(8, 6))
nbin = 50
range_d = [100, 200]

count = 0
for iarc in range(10):
    if iarc == 6 or iarc == 7:
        continue
    # s=s, zeta=zeta, vism_zeta=vism_zeta, D_samps=D_samps, D_i_samps=D_i_samps
    data = np.load(datadir + '/arcmodel{}_D.npz'.format(iarc))

    D_samps = data['D_samps'] * 1000

    # plt.hist(D_samps, range=[0.1, 0.2], bins=50, density=True, alpha=0.5)

    probs, edges = np.histogram(
        D_samps, bins=nbin, range=range_d, density=True)
    plt.stairs(probs, edges, alpha=0.6)

    if iarc == 0:
        tot = np.log(probs)
    else:
        tot += np.log(probs)

    count += 1


p = np.exp(tot)
p /= np.sum(p)

plt.stairs(p / (np.ptp(range_d)/nbin), edges, color='k', linewidth=2)
plt.ylabel(r'Probability density', fontsize=14)
plt.xlabel(r'Pulsar distance, $D_{\rm psr}$ (pc)', fontsize=14)
yl = plt.ylim()
plt.plot([156.96, 156.96], yl, color='k', linestyle=':', linewidth=2)
plt.ylim(yl)
plt.xlim([100, 200])
plt.tight_layout()
plt.savefig(os.getcwd()+'/paper_plots/D_constraint.pdf')
plt.show()

centres = (edges[:-1] + edges[1:]) / 2
arr = np.random.choice(centres, p=p, size=10000)

print(np.mean(arr), np.std(arr))


"""
Make a 2D probability countour of KOM and D
"""

plt.figure(figsize=(8, 8))
# plt.subplot(122)

data = np.load(datadir + '/arcmodel0_KOMwide_D.npz'.format(iarc))

D_samps = data['D_samps'] * 1000
KOM_samps = data['KOM_samps']


# Define range for histograms
KOM_range = [min(KOM_samps)-5, max(KOM_samps)+10]
KOM_range = [208.3-60, 208.3+60]
D_range = [90, max(D_samps)+10]

# KOM_range = [0, 360]
# D_range = [100, max(D_samps)]

# Compute original mass-mass
P, x_edges, y_edges = np.histogram2d(
    D_samps, KOM_samps, range=[D_range, KOM_range], bins=100, density=True)

# Get bin centers from the edges
x = x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2
y = y_edges[:-1] + (y_edges[1] - y_edges[0]) / 2


# Define probability levels for contours (cumulative probabilities)
prob_levels = [0.683, 0.954, 0.997]  # 1, 2, 3-sigma levels

levels = compute_contour_levels(P, prob_levels)


plt.contour(x, y, P.T, levels=levels, colors='C0', zorder=10)

xl = plt.xlim()
yl = plt.ylim()


plt.fill_between(xl, [208.3-0.8, 208.3-0.8],
                 y2=[208.3+0.8, 208.3+0.8], color='C1', alpha=0.5)

plt.fill_betweenx(yl, [156.96-0.11, 156.96-0.11], x2=[156.96+0.11, 156.96+0.11],
                  color='C1', zorder=1, alpha=0.5)

plt.xlim(xl)
plt.ylim(yl)

plt.xlabel(r'Pulsar distance, $D_{\rm psr}$ (pc)', fontsize=14)
plt.ylabel(r'Longitude of asending node, $\Omega$ (degrees)', fontsize=14)
plt.tight_layout()
plt.savefig(os.getcwd()+'/paper_plots/KOM-D.png')
plt.savefig(os.getcwd()+'/paper_plots/KOM-D.pdf')
plt.show()
