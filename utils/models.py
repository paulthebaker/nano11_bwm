from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np

from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise.signals import utils

from enterprise import constants as const

#### Model component building blocks ####

def white_noise_block(vary=False):
    """
    Returns the white noise block of the model:
        1. EFAC per backend/receiver system
        2. EQUAD per backend/receiver system
        3. ECORR per backend/receiver system
    :param vary:
        If set to true we vary these parameters
        with uniform priors. Otherwise they are set to constants
        with values to be set later.
    """

    # define selection by observing backend
    selection = selections.Selection(selections.by_backend)

    # white noise parameters
    if vary:
        efac = parameter.Uniform(0.01, 10.0)
        equad = parameter.Uniform(-8.5, -5)
        ecorr = parameter.Uniform(-8.5, -5)
    else:
        efac = parameter.Constant()
        equad = parameter.Constant()
        ecorr = parameter.Constant()

    # white noise signals
    ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
    eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)
    ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)

    # combine signals
    s = ef + eq + ec

    return s


def red_noise_block(prior='log-uniform', Tspan=None):
    """
    Returns red noise model:
        1. Red noise modeled as a power-law with 30 sampling frequencies
    :param prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for indivicual pulsar.
    """

    # red noise parameters
    if prior == 'uniform':
        log10_A = parameter.LinearExp(-20, -11)
    elif prior == 'log-uniform':
        log10_A = parameter.Uniform(-20, -11)
    else:
        raise ValueError('Unknown prior for red noise amplitude!')

    gamma = parameter.Uniform(0, 7)

    # red noise signal
    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(pl, components=30, Tspan=Tspan)

    return rn


def common_red_noise_block(psd='powerlaw', prior='log-uniform',
                           Tspan=None, gamma_val=None, orf=None,
                           name='gwb'):
    """
    Returns common red noise model:
        1. Red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']
    :param prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for indivicual pulsar.
    :param gamma_val:
        Value of spectral index for power-law and turnover
        models. By default spectral index is varied of range [0,7]
    :param orf:
        String representing which overlap reduction function to use.
        By default we do not use any spatial correlations. Permitted
        values are ['hd', 'dipole', 'monopole'].
    :param name: Name of common red process
    """

    orfs = {'hd': utils.hd_orf(), 'dipole': utils.dipole_orf(),
            'monopole': utils.monopole_orf()}

    # common red noise parameters
    if psd in ['powerlaw', 'turnover']:
        amp_name = 'log10_A_{}'.format(name)
        if prior == 'uniform':
            log10_Agw = parameter.LinearExp(-18, -11)(amp_name)
        elif prior == 'log-uniform' and gamma_val is not None:
            if np.abs(gamma_val - 4.33) < 0.1:
                log10_Agw = parameter.Uniform(-18, -14)(amp_name)
            else:
                log10_Agw = parameter.Uniform(-18, -11)(amp_name)
        else:
            log10_Agw = parameter.Uniform(-18, -11)(amp_name)

        gam_name = 'gamma_{}'.format(name)
        if gamma_val is not None:
            gamma_gw = parameter.Constant(gamma_val)(gam_name)
        else:
            gamma_gw = parameter.Uniform(0, 7)(gam_name)

        # common red noise PSD
        if psd == 'powerlaw':
            cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
        elif psd == 'turnover':
            kappa_name = 'kappa_{}'.format(name)
            lf0_name = 'log10_fbend_{}'.format(name)
            kappa_gw = parameter.Uniform(0, 7)(kappa_name)
            lf0_gw = parameter.Uniform(-9, -7)(lf0_name)
            cpl = utils.turnover(log10_A=log10_Agw, gamma=gamma_gw,
                                 lf0=lf0_gw, kappa=kappa_gw)

    if orf is None:
        crn = gp_signals.FourierBasisGP(cpl, components=30, Tspan=Tspan)
    elif orf in orfs.keys():
        crn = gp_signals.FourierBasisCommonGP(cpl, orfs[orf], components=30, Tspan=Tspan)
    else:
        raise ValueError('ORF {} not recognized'.format(orf))

    return crn


def bwm_block(Tmin, Tmax, amp_prior='log-uniform',
              skyloc=None, logmin=-18, logmax=-11,
              name='bwm'):
    """
    Returns deterministic GW burst with memory model:
        1. Burst event parameterized by time, sky location,
        polarization angle, and amplitude
    :param Tmin:
        Min time to search, probably first TOA (MJD).
    :param Tmax:
        Max time to search, probably last TOA (MJD).
    :param amp_prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param skyloc:
        Fixed sky location of BWM signal search as [cos(theta), phi].
        Search over sky location if ``None`` given.
    :param logmin:
        log of minimum BWM amplitude for prior (log10)
    :param logmax:
        log of maximum BWM amplitude for prior (log10)
    :param name:
        Name of BWM signal.
    """

    # BWM parameters
    amp_name = 'log10_A_{}'.format(name)
    if amp_prior == 'uniform':
        log10_A_bwm = parameter.LinearExp(logmin, logmax)(amp_name)
    elif amp_prior == 'log-uniform':
        log10_A_bwm = parameter.Uniform(logmin, logmax)(amp_name)

    pol_name = 'pol_{}'.format(name)
    pol = parameter.Uniform(0, np.pi)(pol_name)

    t0_name = 't0_{}'.format(name)
    t0 = parameter.Uniform(Tmin, Tmax)(t0_name)

    costh_name = 'costheta_{}'.format(name)
    phi_name = 'phi_{}'.format(name)
    if skyloc is None:
        costh = parameter.Uniform(-1, 1)(costh_name)
        phi = parameter.Uniform(0, 2*np.pi)(phi_name)
    else:
        costh = parameter.Constant(skyloc[0])(costh_name)
        phi = parameter.Constant(skyloc[1])(phi_name)


    # BWM signal
    bwm_wf = utils.bwm_delay(log10_h=log10_A_bwm, t0=t0,
                            cos_gwtheta=costh, gwphi=phi, gwpol=pol)
    bwm = deterministic_signals.Deterministic(bwm_wf, name=name)

    return bwm


#### PTA models ####

def model_noise(psr):
    """
    Reads in enterprise Pulsar instance and returns a PTA
    instantiated with the standard NANOGrav noise model:
        1. EFAC per backend/receiver system
        2. EQUAD per backend/receiver system
        3. ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.
    """

    # white noise
    s = white_noise_block(vary=True)

    # red noise
    s += red_noise_block()

    # timing model
    s += gp_signals.TimingModel()

    # set up PTA
    pta = signal_base.PTA([s(psr)])

    return pta


def model_gwb(psrs, psd='powerlaw', gamma_common=None, orf=None,
             upper_limit=False, bayesephem=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with GWB model:
    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.
    global:
        1.Common red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover']
        2. Optional physical ephemeris modeling.
    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover']. 'powerlaw' is default
        value.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param orf:
        String representing which overlap reduction function to use.
        By default we do not use any spatial correlations. Permitted
        values are ['hd', 'dipole', 'monopole'].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    tmin = np.min([p.toas.min() for p in psrs])
    tmax = np.max([p.toas.max() for p in psrs])
    Tspan = tmax - tmin

    # white noise
    s = white_noise_block(vary=False)

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                gamma_val=gamma_common, name='gwb',
                                orf=orf)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    s += gp_signals.TimingModel()

    # set up PTA
    pta = signal_base.PTA([s(psr) for psr in psrs])

    return pta


def model_bwm(psrs, upper_limit=False, bayesephem=False,
              Tmin_bwm=None, Tmax_bwm=None,
              skyloc=None, logmin=-18, logmax=-11):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with BWM model:
    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.
    global:
        1. Deterministic GW burst with memory signal.
        2. Optional physical ephemeris modeling.
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param Tmin_bwm:
        Min time to search for BWM (MJD). If omitted, uses first TOA.
    :param Tmax_bwm:
        Max time to search for BWM (MJD). If omitted, uses last TOA.
    :param skyloc:
        Fixed sky location of BWM signal search as [cos(theta), phi].
        Search over sky location if ``None`` given.
    :param logmin:
        log of minimum BWM amplitude for prior (log10)
    :param logmax:
        log of maximum BWM amplitude for prior (log10)
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    tmin = np.min([p.toas.min() for p in psrs])
    tmax = np.max([p.toas.max() for p in psrs])
    Tspan = tmax - tmin

    if Tmin_bwm == None:
        Tmin_bwm = tmin/const.day
    if Tmax_bwm == None:
        Tmax_bwm = tmax/const.day

    # white noise
    s = white_noise_block(vary=False)

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan)

    # GW BWM signal block
    s += bwm_block(Tmin_bwm, Tmax_bwm, amp_prior=amp_prior,
                   skyloc=skyloc, logmin=logmin, logmax=logmax,
                   name='bwm')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    s += gp_signals.TimingModel()

    # set up PTA
    pta = signal_base.PTA([s(psr) for psr in psrs])

    return pta


def model_gwb_bwm(psrs, psd='powerlaw', gamma_common=None, orf=None,
              Tmin_bwm=None, Tmax_bwm=None,
              upper_limit=False, bayesephem=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with both a GWB and a GW BWM model:
    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.
    global:
        1. Common red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover']
        2. Deterministic GW burst with memory signal.
        3. Optional physical ephemeris modeling.
    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover']. 'powerlaw' is default
        value.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param orf:
        String representing which overlap reduction function to use.
        By default we do not use any spatial correlations. Permitted
        values are ['hd', 'dipole', 'monopole'].
    :param Tmin_bwm:
        Min time to search for BWM (MJD). If omitted, uses first TOA.
    :param Tmax_bwm:
        Max time to search for BWM (MJD). If omitted, uses last TOA.
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    tmin = np.min([p.toas.min() for p in psrs])
    tmax = np.max([p.toas.max() for p in psrs])
    Tspan = tmax - tmin

    if Tmin_bwm == None:
        Tmin_bwm = tmin/const.day
    if Tmax_bwm == None:
        Tmax_bwm = tmax/const.day

    # white noise
    s = white_noise_block(vary=False)

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan)

    # GW BWM signal block
    s += bwm_block(Tmin_bwm, Tmax_bwm, amp_prior=amp_prior,
                   name='bwm')

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                gamma_val=gamma_common, name='gwb',
                                orf=orf)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    s += gp_signals.TimingModel()

    # set up PTA
    pta = signal_base.PTA([s(psr) for psr in psrs])

    return pta
