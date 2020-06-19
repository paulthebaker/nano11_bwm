from __future__ import division, print_function
import numpy as np
import cPickle as pickle
import os, glob
import argparse 

from utils import models
from utils.sample_helpers import JumpProposal, get_parameter_groups

from enterprise.pulsar import Pulsar
from enterprise import constants as const
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from astropy.time import Time


# post-proc stuff
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd

from utils.UL_uncert import UL_uncert
from acor import acor
from corner import corner

from enterprise.signals import parameter
from enterprise.signals import utils
from enterprise.signals import deterministic_signals
from enterprise.signals import gp_signals
from enterprise.signals import signal_base

parser = argparse.ArgumentParser(
          description='run the BWM analysis to test UL priors')

parser.add_argument('-p', '--amp-prior',
                    dest='amp_prior', default='log-uniform',
                    action='store',
                    help="which amp prior to use: 'uniform', 'true-uniform', 'log-normal', or 'log-uniform'")
parser.add_argument('-a', '--minA', type=float,
                    dest='minA', default=-18,
                    action='store',
                    help="min of log amplitude for *uniform priors")
parser.add_argument('-A', '--maxA', type=float,
                    dest='maxA', default=-9,
                    action='store',
                    help="max of log amplitude for *uniform priors")

args = parser.parse_args()

####################
##  SETUP PARAMS  ##
####################

#psr_name = 'J1909-3744'
psr_name = 'J1713+0747'

amp_prior = args.amp_prior

N = 500000
#ii_t = 29  # 0-40 or None
ii_t = 11  # 0-40 or None

# custom bwm models
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
    elif prior == 'log-normal':
        log10_A = parameter.Normal(-15, 4)
    elif prior == 'true-uniform':
        # use LinearExp for RN... simpler
        log10_A = parameter.LinearExp(-20, -11)
    else:
        raise NotImplementedError('Unknown prior for red noise amplitude!')

    gamma = parameter.Uniform(0, 7)

    # red noise signal
    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(pl, components=30, Tspan=Tspan)

    return rn


def bwm_sngl_block(Tmin, Tmax, amp_prior='log-uniform',
                   logmin=-18, logmax=-9,
                   name='bwm'):
    """
    Returns deterministic single pulsar GW burst with memory model:
        1. Burst event parameterized by time, amplitude, and sign (+/-)
    :param Tmin:
        Min time to search, probably first TOA (MJD).
    :param Tmax:
        Max time to search, probably last TOA (MJD).
    :param amp_prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
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
    elif amp_prior == 'log-normal':
        log10_A_bwm = parameter.Normal(logmin, logmax)(amp_name)
    elif amp_prior == 'true-uniform':
        amp_name = '{}_A'.format(name)
        A_bwm = parameter.Uniform(10**logmin, 10**logmax)(amp_name)
    else:
        raise NotImplementedError('Unknown prior for BWM amplitude!')

    t0_name = '{}_t0'.format(name)
    t0 = parameter.Uniform(Tmin, Tmax)(t0_name)

    sign_name = '{}_sign'.format(name)
    sign = parameter.Uniform(-1.0, 1.0)(sign_name)


    # BWM signal
    if amp_prior == 'true-uniform':
        bwm_wf = bwm_sngl_delay(h=A_bwm, t0=t0, sign=sign)
    else:
        bwm_wf = bwm_sngl_delay(log10_h=log10_A_bwm, t0=t0, sign=sign)
    bwm = deterministic_signals.Deterministic(bwm_wf, name=name)

    return bwm


def model_bwm(psrs,
              Tmin_bwm=None, Tmax_bwm=None,
              skyloc=None, logmin=-18, logmax=-11,
              amp_prior='log-uniform', bayesephem=False, dmgp=False, free_rn=False):
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
    :param free_rn:
        Use free red noise spectrum model. Set to False by default
    """

    # find the maximum time span to set GW frequency sampling
    tmin = np.min([p.toas.min() for p in psrs])
    tmax = np.max([p.toas.max() for p in psrs])
    Tspan = tmax - tmin

    if Tmin_bwm == None:
        Tmin_bwm = tmin/const.day
    if Tmax_bwm == None:
        Tmax_bwm = tmax/const.day

    # white noise
    s = models.white_noise_block(vary=False)

    # red noise
    if free_rn:
        s += models.free_noise_block(prior=amp_prior, Tspan=Tspan)
    else:
        s += red_noise_block(prior=amp_prior, Tspan=Tspan)

    # GW BWM signal block
    s += bwm_sngl_block(Tmin_bwm, Tmax_bwm, amp_prior=amp_prior,
                        logmin=logmin, logmax=logmax,
                        name='bwm')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    s += gp_signals.TimingModel(use_svd=True)

    # DM variations model
    if dmgp:
        s += models.dm_noise_block(gp_kernel='diag', psd='powerlaw',
                                   prior=amp_prior, Tspan=Tspan)
        s += models.dm_annual_signal()

        # DM exponential dip for J1713's DM event
        dmexp = models.dm_exponential_dip(tmin=54500, tmax=54900)
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

## setup params that are the same for all runs
ephem = 'DE436'

TMIN = 53217.0
TMAX = 57387.0
tchunk = np.linspace(TMIN, TMAX, 41)  # break in 2.5% chunks
tlim = []
for ii in range(len(tchunk)-2):
    tlim.append(tchunk[ii:ii+3])

datadir = '/home/pbaker/nanograv/data/nano11'
noisefile = '/home/pbaker/nanograv/data/nano11_setpars.pkl'

if ii_t:
    TMIN, CENTER, TMAX = tlim[ii_t]
    chunk = '{:.2f}'.format(CENTER)
else:
    chunk = 'all'

rundir = '/home/pbaker/nanograv/bwm/sngl/{0:s}_{1:s}_trick/{2:s}/'.format(psr_name, amp_prior, chunk)
try:
    os.makedirs(rundir)
except:
    pass

# read in data from .par / .tim
par = glob.glob(datadir +'/'+ psr_name +'*.par')[0]
tim = glob.glob(datadir +'/'+ psr_name +'*.tim')[0]
psr = Pulsar(par, tim, ephem=ephem, timing_package='tempo2')

with open(noisefile, "rb") as f:
    setpars = pickle.load(f)

#################
##  pta model  ##
#################
if amp_prior == 'log-normal':
    logminA = -13 # mean for log-normal
    logmaxA = 3   # stdev for log-normal
else:
    logminA = args.minA
    logmaxA = args.maxA


tmin = psr.toas.min() / 86400
tmax = psr.toas.max() / 86400

if TMIN is not None and TMAX is not None:
    if TMIN<tmin:
        err = "tmin ({:.1f}) BEFORE first TOA ({:.1f})".format(TMIN, tmin)
        raise RuntimeError(err)
    elif TMAX>tmax:
        err = "tmax ({:.1f}) AFTER last TOA ({:.1f})".format(TMAX, tmax)
        raise RuntimeError(err)
    elif TMIN>TMAX:
        err = "tmin ({:.1f}) BEFORE last tmax ({:.1f})".format(TMIN, TMAX)
        raise RuntimeError(err)
    else:
        t0min = TMIN
        t0max = TMAX
else:
    tclip = (tmax - tmin) * 0.05
    t0min = tmin + tclip*2  # clip first 10%
    t0max = tmax - tclip    # clip last 5%

pta = model_bwm([psr],
                amp_prior=amp_prior, bayesephem=False,
                logmin=logminA, logmax=logmaxA,
                Tmin_bwm=t0min, Tmax_bwm=t0max)
pta.set_default_params(setpars)


outfile = os.path.join(rundir, 'params.txt')
with open(outfile, 'w') as f:
    for pname in pta.param_names:
        f.write(pname+'\n')


###############
##  sampler  ##
###############
# dimension of parameter space
x0 = np.hstack(p.sample() for p in pta.params)
ndim = len(x0)

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.1**2)

# parameter groupings
groups = get_parameter_groups(pta)

sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior,
                 cov, groups=groups, outDir=rundir, resume=False)

# add prior draws to proposal cycle
jp = JumpProposal(pta)
sampler.addProposalToCycle(jp.draw_from_prior, 5)
sampler.addProposalToCycle(jp.draw_from_bwm_prior, 5)

if amp_prior == 'uniform':
    draw_bwm_loguni = jp.build_log_uni_draw('bwm_log10_A', logminA, logmaxA, logparam=True)
    sampler.addProposalToCycle(draw_bwm_loguni, 10)
#elif amp_prior == 'true-uniform':
#    draw_bwm_loguni = jp.build_log_uni_draw('bwm_A', logminA, logmaxA, logparam=False)
#    sampler.addProposalToCycle(draw_bwm_loguni, 10)


# SAMPLE!!
sampler.sample(x0, N, SCAMweight=35, AMweight=10, DEweight=50)


#################
##  post proc  ##
#################
def trace_plot(chain, pars,
               cols=3, wid_per_col=4, aspect=4/3,
               kwargs={}):

    rows = len(pars)//cols
    if rows*cols < len(pars):
        rows += 1

    ax = []
    width = wid_per_col * cols
    height = wid_per_col * rows / aspect
    fig = plt.figure(figsize=(width, height))

    for pp, par in enumerate(pars):
        ax.append(fig.add_subplot(rows, cols, pp+1))
        ax[pp].plot(chain[:,pp], **kwargs)
        ax[pp].set_xlabel(par)
    plt.tight_layout()
    return fig


def hist_plot(chain, pars, bins=30,
              cols=3, wid_per_col=4, aspect=4/3,
              kwargs={}):
    hist_kwargs = {
        'density':True,
        'histtype':'step',
    }
    for key, val in kwargs.items():
        hist_kwargs[key] = val

    rows = len(pars)//cols
    if rows*cols < len(pars):
        rows += 1

    ax = []
    width = wid_per_col * cols
    height = wid_per_col * rows / aspect
    fig = plt.figure(figsize=(width, height))

    for pp, par in enumerate(pars):
        ax.append(fig.add_subplot(rows, cols, pp+1))
        ax[pp].hist(chain[:,pp], bins=bins, **hist_kwargs)
        ax[pp].set_xlabel(par)
    plt.tight_layout()
    return fig

with open(rundir + 'params.txt', 'r') as f:
    params = [line.rstrip('\n') for line in f]

# get just bwm params
par_bwm = []
for par in params:
    if par.startswith('bwm_'):
        par_bwm.append(par)
idx_bwm = [params.index(p) for p in par_bwm]
try:
    idx_A = par_bwm.index('bwm_log10_A')
except:
    idx_A = par_bwm.index('bwm_A')
idx_t0 = par_bwm.index('bwm_t0')

chain_raw = pd.read_csv(rundir + 'chain_1.txt',
                        sep='\t', dtype=float, header=None).values

burnfrac = 0.25
thin = 2

burn = int(burnfrac * len(chain_raw))
chain = chain_raw[burn::thin]

chain_bwm = chain[:,idx_bwm]
chain_L = chain[:,-5]  # likelihood w/ pandas load (-4 for numpy load)

corL = acor(chain_L)[0]
N = len(chain_bwm)
print("N = {}, corL = {}".format(N, corL))

ch_plt = np.hstack([chain_bwm, chain_L.reshape(N,1)])
par_plt = par_bwm + ['logL']
fig = trace_plot(ch_plt, par_plt, cols=3, wid_per_col=4);
fig.savefig(rundir + '/trace.png')

fig = hist_plot(ch_plt, par_plt, cols=3, wid_per_col=4)
for ax in fig.axes:
    ax.set_yscale('log')
fig.savefig(rundir +'/hist.png')

corner_kwargs = {'bins':30,
                 'show_titles':True,
                 'labels':par_bwm,
                 'smooth':0.5,
                 'plot_datapoints':False,
                 'plot_density':True,
                 'plot_contours':True,
                 'fill_contours':False,}

fig = corner(chain_bwm, color='C0', **corner_kwargs);
fig.savefig(rundir +'/corner.png')

if amp_prior == 'true-uniform':
    UL = UL_uncert(chain_bwm[:, idx_A])
else:
    UL = UL_uncert(10**chain_bwm[:, idx_A])
print(UL)
