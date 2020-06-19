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
@signal_base.function
def free_spectrum(f, log10_rho=None):
    """
    Free spectral model. PSD  amplitude at each frequency
    is a free parameter. Model is parameterized by
    S(f_i) = \rho_i^2 * T,
    where \rho_i is the free parameter and T is the observation
    length.
    """
    return np.repeat(10**(2*log10_rho), 2)

# linear interpolation basis in time with nu^-2 scaling
@signal_base.function
def linear_interp_basis_dm(toas, freqs, dt=30*const.day):
    # get linear interpolation basis in time
    U, avetoas = utils.linear_interp_basis(toas, dt=dt)

    # scale with radio frequency
    Dm = (1400/freqs)**2

    return U * Dm[:, None], avetoas

# linear interpolation in radio frequcny
@signal_base.function
def linear_interp_basis_freq(freqs, df=64):
    return utils.linear_interp_basis(freqs, dt=df)

# DMX-like signal with Gaussian prior
@signal_base.function
def dmx_ridge_prior(avetoas, log10_sigma=-7):
    sigma = 10**log10_sigma
    return sigma**2 * np.ones_like(avetoas)

# quasi-periodic kernel for DM
@signal_base.function
def periodic_kernel(avetoas, log10_sigma=-7, log10_ell=2, gam_p=1, p=1):
    r = np.abs(avetoas[None, :] - avetoas[:, None])

    # convert units to seconds
    sigma = 10**log10_sigma
    l = 10**log10_ell * const.day
    p *= 3.16e7
    d = np.eye(r.shape[0]) * (sigma/500)**2
    K = sigma**2 * np.exp(-r**2/2/l**2 - gam_p*np.sin(np.pi*r/p)**2) + d
    return K

# squared-exponential kernel for FD
@signal_base.function
def se_kernel(avefreqs, log10_sigma=-7, log10_lam=np.log10(1000)):
    tm = np.abs(avefreqs[None, :] - avefreqs[:, None])
    lam = 10**log10_lam
    sigma = 10**log10_sigma
    d = np.eye(tm.shape[0]) * (sigma/500)**2
    return sigma**2 * np.exp(-tm**2/2/lam) + d

# quantization matrix in time and radio frequency to cut down on the kernel size.
@signal_base.function
def get_tf_quantization_matrix(toas, freqs, dt=30*const.day, df=None, dm=False):
    if df is None:
        dfs = [(600, 1000), (1000, 1900), (1900, 3000), (3000, 5000)]
    else:
        fmin = freqs.min()
        fmax = freqs.max()
        fs = np.arange(fmin, fmax+df, df)
        dfs = [(fs[ii], fs[ii+1]) for ii in range(len(fs)-1)]

    Us, avetoas, avefreqs, masks = [], [], [], []
    for rng in dfs:
        mask = np.logical_and(freqs>=rng[0], freqs<rng[1])
        if any(mask):
            masks.append(mask)
            U, _ = utils.create_quantization_matrix(toas[mask],
                                                    dt=dt, nmin=1)
            avetoa = np.array([toas[mask][idx.astype(bool)].mean()
                               for idx in U.T])
            avefreq = np.array([freqs[mask][idx.astype(bool)].mean()
                                for idx in U.T])
            Us.append(U)
            avetoas.append(avetoa)
            avefreqs.append(avefreq)

    nc = np.sum(U.shape[1] for U in Us)
    U = np.zeros((len(toas), nc))
    avetoas = np.concatenate(avetoas)
    idx = np.argsort(avetoas)
    avefreqs = np.concatenate(avefreqs)
    nctot = 0
    for ct, mask in enumerate(masks):
        Umat = Us[ct]
        nn = Umat.shape[1]
        U[mask, nctot:nn+nctot] = Umat
        nctot += nn

    if dm:
         weights = (1400/freqs)**2
    else:
        weights = np.ones_like(freqs)

    return U[:, idx] * weights[:, None], {'avetoas': avetoas[idx],
                                          'avefreqs': avefreqs[idx]}

# kernel is the product of a quasi-periodic time kernel and
# a rational-quadratic frequency kernel.
@signal_base.function
def tf_kernel(labels, log10_sigma=-7, log10_ell=2, gam_p=1,
              p=1, log10_ell2=4, alpha_wgt=2):

    avetoas = labels['avetoas']
    avefreqs = labels['avefreqs']

    r = np.abs(avetoas[None, :] - avetoas[:, None])
    r2 = np.abs(avefreqs[None, :] - avefreqs[:, None])

    # convert units to seconds
    sigma = 10**log10_sigma
    l = 10**log10_ell * const.day
    l2 = 10**log10_ell2
    p *= 3.16e7
    d = np.eye(r.shape[0]) * (sigma/500)**2
    Kt = sigma**2 * np.exp(-r**2/2/l**2 - gam_p*np.sin(np.pi*r/p)**2)
    Kv = (1+r2**2/2/alpha_wgt/l2**2)**(-alpha_wgt)

    return Kt * Kv + d

@signal_base.function
def chrom_exp_decay(toas, freqs, log10_Amp=-7,
                    t0=54000, log10_tau=1.7, idx=2):
    """
    Chromatic exponential-dip delay term in TOAs.

    :param t0: time of exponential minimum [MJD]
    :param tau: 1/e time of exponential [s]
    :param log10_Amp: amplitude of dip
    :param idx: index of chromatic dependence

    :return wf: delay time-series [s]
    """
    t0 *= const.day
    tau = 10**log10_tau * const.day
    wf = -10**log10_Amp * np.heaviside(toas - t0, 1) * \
        np.exp(- (toas - t0) / tau)

    return wf * (1400 / freqs) ** idx

@signal_base.function
def chrom_yearly_sinusoid(toas, freqs, log10_Amp=-7, phase=0, idx=2):
    """
    Chromatic annual sinusoid.

    :param log10_Amp: amplitude of sinusoid
    :param phase: initial phase of sinusoid
    :param idx: index of chromatic dependence

    :return wf: delay time-series [s]
    """

    wf = 10**log10_Amp * np.sin( 2 * np.pi * const.fyr * toas + phase)

def which_psrs(psrs, slice_yr=100, min_yr=3, backward=False):
    """determine pulsars to use for a time slice
    :param psrs:
        list of ``enterprise.Pulsar`` objects
    :param slice_yr:
        length of time slice in years.  If slice is longer than
        dataset then all time is used
    :param min_yr:
        minimum data length to include a pulsar in a slice (years)
    :param backward:
        True for backward slices
    """
    if backward:
        tx = np.max([p.toas.max() for p in psrs])  # last observation
        t0 = tx - slice_yr*const.yr  # start time of slice
    else:
        t0 = np.min([p.toas.min() for p in psrs])  # first observation
        tx = t0 + slice_yr*const.yr  # end time of slice

    which = []
    for p in psrs:
        ms = (p.toas.min()-t0)/const.yr + min_yr
        if slice_yr > ms or backward:
            p.filter_data(start_time=t0/const.day, end_time=tx/const.day)
            which.append(p)

    return which

# sngl psr bwm model
@signal_base.function
def bwm_sngl_delay(toas, pos, log10_h=None, h=None,
                   sign=1.0, t0=55000):
    """
    Function that calculates the pulsar-term gravitational-wave
    burst-with-memory signal, as described in:
    Seto et al, van haasteren and Levin, phsirkov et al, Cordes and Jenet.
    The amplitude h eats up the angular response to simplify the search
    space. i.e. h_this = h_bwm * B(theta, phi).
    The polarization is replaced by a "sign" variable.
    :param toas: Time-of-arrival measurements [s]
    :param pos: Unit vector from Earth to pulsar
    :param log10_h: log10 of GW strain
    :param h: GW strain
    :param sign: parameter to sample sign (glitch/anti-glitch)
    :param t0: Burst central time [day]
    :return: the waveform as induced timing residuals (seconds)
    """

    if h is None and log10_h is None:
        raise TypeError("specify one of 'h' or 'log10_h'")
    # convert
    if h is None:
        h = 10**log10_h
    t0 *= const.day

    # Define the heaviside function
    heaviside = lambda x: 0.5 * (np.sign(x) + 1)

    # Return the time-series for the pulsar
    return np.sign(sign) * h * heaviside(toas-t0) * (toas-t0)


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


def free_noise_block(prior='log-uniform', Tspan=None):
    """Returns free spectrum noise model:
        1. noise PSD with 30 sampling frequencies
    :param prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for indivicual pulsar.
    """

    if prior == 'uniform':
        log10_rho = parameter.LinearExp(-9, -4, size=30)
    elif prior == 'log-uniform':
        log10_rho = parameter.Uniform(-9, -4, size=30)

    spect = free_spectrum(log10_rho=log10_rho)
    fn = gp_signals.FourierBasisGP(spect, components=30, Tspan=Tspan)

    return fn


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

def dm_noise_block(gp_kernel='diag', psd='powerlaw', nondiag_kernel='periodic',
                   prior='log-uniform', Tspan=None, components=30, gamma_val=None):
    """
    Returns DM noise model:

        1. DM noise modeled as a power-law with 30 sampling frequencies

    :param psd:
        PSD function [e.g. powerlaw (default), turnover, free spectrum]
    :param prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for indivicual pulsar.
    :param components:
        Number of frequencies in sampling of DM-variations.
    :param gamma_val:
        If given, this is the fixed slope of the power-law for
        powerlaw or turnover DM-variations
    """
    # dm noise parameters that are common
    if gp_kernel == 'diag':
        if psd in ['powerlaw', 'turnover']:
            # parameters shared by PSD functions
            if prior == 'uniform':
                log10_A_dm = parameter.LinearExp(-20, -11)
            elif prior == 'log-uniform' and gamma_val is not None:
                if np.abs(gamma_val - 4.33) < 0.1:
                    log10_A_dm = parameter.Uniform(-20, -11)
                else:
                    log10_A_dm = parameter.Uniform(-20, -11)
            else:
                log10_A_dm = parameter.Uniform(-20, -11)

            if gamma_val is not None:
                gamma_dm = parameter.Constant(gamma_val)
            else:
                gamma_dm = parameter.Uniform(0, 7)

            # different PSD function parameters
            if psd == 'powerlaw':
                dm_prior = utils.powerlaw(log10_A=log10_A_dm, gamma=gamma_dm)
            elif psd == 'turnover':
                kappa_dm = parameter.Uniform(0, 7)
                lf0_dm = parameter.Uniform(-9, -7)
                dm_prior = utils.turnover(log10_A=log10_A_dm, gamma=gamma_dm,
                                          lf0=lf0_dm, kappa=kappa_dm)

        if psd == 'spectrum':
            if prior == 'uniform':
                log10_rho_dm = parameter.LinearExp(-10, -4, size=components)
            elif prior == 'log-uniform':
                log10_rho_dm = parameter.Uniform(-10, -4, size=components)

            dm_prior = free_spectrum(log10_rho=log10_rho_dm)

        dm_basis = utils.createfourierdesignmatrix_dm(nmodes=components,
                                                      Tspan=Tspan)

    elif gp_kernel == 'nondiag':
        if nondiag_kernel == 'periodic':
            # Periodic GP kernel for DM
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            period = parameter.Uniform(0.2, 5.0)
            gam_p = parameter.Uniform(0.1, 30.0)

            dm_basis = linear_interp_basis_dm(dt=15*const.day)
            dm_prior = periodic_kernel(log10_sigma=log10_sigma,
                                    log10_ell=log10_ell, gam_p=gam_p, p=period)
        elif nondiag_kernel == 'periodic_rfband':
            # Periodic GP kernel for DM with RQ radio-frequency dependence
            log10_sigma = parameter.Uniform(-10, -4)
            log10_ell = parameter.Uniform(1, 4)
            log10_ell2 = parameter.Uniform(2, 7)
            alpha_wgt = parameter.Uniform(0.2, 6)
            period = parameter.Uniform(0.2, 5.0)
            gam_p = parameter.Uniform(0.1, 30.0)

            dm_basis = get_tf_quantization_matrix(df=200, dt=15*const.day, dm=True)
            dm_prior = tf_kernel(log10_sigma=log10_sigma, log10_ell=log10_ell,
                                 gam_p=gam_p, p=period, alpha_wgt=alpha_wgt,
                                 log10_ell2=log10_ell2)
        elif nondiag_kernel == 'dmx_like':
            # DMX-like signal
            log10_sigma = parameter.Uniform(-10, -4)

            dm_basis = linear_interp_basis_dm(dt=30*const.day)
            dm_prior = dmx_ridge_prior(log10_sigma=log10_sigma)

    dmgp = gp_signals.BasisGP(dm_prior, dm_basis, name='dm_gp')

    return dmgp

def dm_annual_signal(idx=2, name='dm_s1yr'):
    """
    Returns chromatic annual signal (i.e. TOA advance):

    :param idx:
        index of radio frequency dependence (i.e. DM is 2). If this is set
        to 'vary' then the index will vary from 1 - 6
    :param name: Name of signal

    :return dm1yr:
        chromatic annual waveform.
    """
    log10_Amp_dm1yr = parameter.Uniform(-10, -2)
    phase_dm1yr = parameter.Uniform(0, 2*np.pi)

    wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_dm1yr,
                               phase=phase_dm1yr, idx=idx)
    dm1yr = deterministic_signals.Deterministic(wf, name=name)

    return dm1yr

def dm_exponential_dip(tmin, tmax, idx=2, name='dmexp'):
    """
    Returns chromatic exponential dip (i.e. TOA advance):

    :param tmin, tmax:
        search window for exponential dip time.
    :param idx:
        index of radio frequency dependence (i.e. DM is 2). If this is set
        to 'vary' then the index will vary from 1 - 6
    :param name: Name of signal

    :return dmexp:
        chromatic exponential dip waveform.
    """
    t0_dmexp = parameter.Uniform(tmin,tmax)
    log10_Amp_dmexp = parameter.Uniform(-10, -2)
    log10_tau_dmexp = parameter.Uniform(np.log10(5), np.log10(100))
    wf = chrom_exp_decay(log10_Amp=log10_Amp_dmexp, t0=t0_dmexp,
                         log10_tau=log10_tau_dmexp, idx=idx)
    dmexp = deterministic_signals.Deterministic(wf, name=name)

    return dmexp


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
        amp_name = '{}_log10_A'.format(name)
        if prior == 'uniform':
            log10_Agw = parameter.LinearExp(-18, -11)(amp_name)
        elif prior == 'log-uniform' and gamma_val is not None:
            if np.abs(gamma_val - 4.33) < 0.1:
                log10_Agw = parameter.Uniform(-18, -14)(amp_name)
            else:
                log10_Agw = parameter.Uniform(-18, -11)(amp_name)
        else:
            log10_Agw = parameter.Uniform(-18, -11)(amp_name)

        gam_name = '{}_gamma'.format(name)
        if gamma_val is not None:
            gamma_gw = parameter.Constant(gamma_val)(gam_name)
        else:
            gamma_gw = parameter.Uniform(0, 7)(gam_name)

        # common red noise PSD
        if psd == 'powerlaw':
            cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
        elif psd == 'turnover':
            kappa_name = '{}_kappa'.format(name)
            lf0_name = '{}_log10_fbend'.format(name)
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
              sngl=False, name='bwm'):
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
    amp_name = '{}_log10_A'.format(name)
    if amp_prior == 'uniform':
        log10_A_bwm = parameter.LinearExp(logmin, logmax)(amp_name)
    elif amp_prior == 'log-uniform':
        log10_A_bwm = parameter.Uniform(logmin, logmax)(amp_name)

    t0_name = '{}_t0'.format(name)
    t0 = parameter.Uniform(Tmin, Tmax)(t0_name)

    if sngl:
        sign_name = '{}_sign'.format(name)
        sign = parameter.Uniform(-1.0, 1.0)(sign_name)

        bwm_wf = bwm_sngl_delay(log10_h=log10_A_bwm, t0=t0, sign=sign)

    else:
        pol_name = '{}_pol'.format(name)
        pol = parameter.Uniform(0, np.pi)(pol_name)

        costh_name = '{}_costheta'.format(name)
        phi_name = '{}_phi'.format(name)
        if skyloc is None:
            costh = parameter.Uniform(-1, 1)(costh_name)
            phi = parameter.Uniform(0, 2*np.pi)(phi_name)
        else:
            costh = parameter.Constant(skyloc[0])(costh_name)
            phi = parameter.Constant(skyloc[1])(phi_name)

        bwm_wf = utils.bwm_delay(log10_h=log10_A_bwm, t0=t0,
                                 cos_gwtheta=costh, gwphi=phi, gwpol=pol)

    # BWM signal
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


def model_bwm(psrs,
              Tmin_bwm=None, Tmax_bwm=None,
              skyloc=None, logmin=-18, logmax=-11,
              upper_limit=False, bayesephem=False, sngl_psr=False,
              dmgp=False, free_rn=False):
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
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param sngl_psr:
        run on a single pulsar only. Uses different BWM parameterization to
        avoid problem with sky nulls. Set to False by default.
    :param free_rn:
        Use free red noise spectrum model. Set to False by default.
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
    if free_rn:
        s += free_noise_block(prior=amp_prior, Tspan=Tspan)
    else:
        s += red_noise_block(prior=amp_prior, Tspan=Tspan)

    # GW BWM signal block
    s += bwm_block(Tmin_bwm, Tmax_bwm, amp_prior=amp_prior,
                   skyloc=skyloc, logmin=logmin, logmax=logmax,
                   sngl=sngl_psr, name='bwm')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    s += gp_signals.TimingModel(use_svd=True)

    # set up PTA
    pta = signal_base.PTA([s(psr) for psr in psrs])

    return pta


def model_gwb_bwm(psrs,
              psd='powerlaw', gamma_common=None, orf=None,
              Tmin_bwm=None, Tmax_bwm=None,
              skyloc=None, logmin=-18, logmax=-11,
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
    :param skyloc:
        Fixed sky location of BWM signal search as [cos(theta), phi].
        Search over sky location if ``None`` given.
    :param logmin:
        log of minimum BWM amplitude for prior (log10)
    :param logmax:
        log of maximum BWM amplitude for prior (log10)
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
                   skyloc=skyloc, logmin=logmin, logmax=logmax,
                   name='bwm')

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                gamma_val=gamma_common, name='gwb',
                                orf=orf)

    # DM variations model
    if dmgp:
        s += dm_noise_block(gp_kernel='diag', psd='powerlaw',
                            prior=amp_prior, Tspan=Tspan)
        s += dm_annual_signal()

        # DM exponential dip for J1713's DM event
        dmexp = dm_exponential_dip(tmin=54500, tmax=54900)
        s2 = s + dmexp

    # set up PTA
    mods = []
    for p in psrs:
        if dmgp and 'J1713+0747' == p.name:
            mods.append(s2(p))
        else:
            mods.append(s(p))

    pta = signal_base.PTA(mods)

    return pta
