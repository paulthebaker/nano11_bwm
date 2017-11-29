from __future__ import division, print_function
import numpy as np
import cPickle as pickle
import argparse

import enterprise
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const
from enterprise.signals import utils

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from sample_helpers import JumpProposal, get_parameter_groups


### ARG PARSER
parser = argparse.ArgumentParser(
          description='run the BWM analysis with enterprise')

parser.add_argument('--ephem',
                    dest='ephem', default='DE436',
                    action='store',
                    help="JPL ephemeris to use or 'BayesEphem'")

args = parser.parse_args()

datadir = '/home/pbaker/nanograv/data/'
outdir = '/home/pbaker/nanograv/bwm/{}/'.format(args.ephem)

# read in data pickles
filename = datadir + 'nano11_{}.pkl'.format(args.ephem)
with open(filename, "rb") as f:
    psrs = pickle.load(f)

filename = datadir + 'nano11_setpars.pkl'
with open(filename, "rb") as f:
    setpars = pickle.load(f)


# find max time span to set min GW/RN freq
tmin = [p.toas.min() for p in psrs]
tmax = [p.toas.max() for p in psrs]
Tspan = np.max(tmax) - np.min(tmin)

# selection class, white noise by backend
selection = selections.Selection(selections.by_backend)

###################
##  init params  ##
###################
# white noise params (held fixed to setpars values later)
efac = parameter.Constant()
equad = parameter.Constant()
ecorr = parameter.Constant()

# red noise params (uniform on A, sample in log10_A)
log10_A = parameter.LinearExp(-20, -11)
gamma = parameter.Uniform(0, 7)

# burst w/ memory !
tstart = min(tmin)/86400
tend = min(tmax)/86400
log10_h = parameter.LinearExp(-20, -11)('bwm_log10_h')  # strain amplitude
cos_gwtheta = parameter.Uniform(-1, 1)('bwm_cos_gwtheta')  # sky loc
gwphi = parameter.Uniform(0, 2*np.pi)('bwm_gwphi')  # sky loc
gwpol = parameter.Uniform(0, np.pi)('bwm_gwpol')  # pol angle
t0 = parameter.Uniform(tstart, tend)('bwm_t0')  # time of arrival (earth)


####################
##  init signals  ##
####################
# white noise
ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)
ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)

# red noise
pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
rn = gp_signals.FourierBasisGP(pl, components=30, Tspan=Tspan)

# GW -- BWM
bwm_wf = utils.bwm_delay(log10_h=log10_h, t0=t0,
                         cos_gwtheta=cos_gwtheta, gwphi=gwphi, gwpol=gwpol)
bwm = deterministic_signals.Deterministic(bwm_wf, name='bwm')

# linear timing model
tm = gp_signals.TimingModel()

# physical ephemeris model
if ephem=='BayesEphem':
    eph = deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

###########
##  PTA  ##
###########
if ephem=='BayesEphem':
    s = ef + eq + ec + rn + tm + eph + bwm
else:
    s = ef + eq + ec + rn + tm + bwm

pta = signal_base.PTA([s(p) for p in psrs])

# set Constant() params
pta.set_default_params(setpars)


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
                 cov, groups=groups, outDir=outdir, resume=True)

# add prior draws to proposal cycle
jp = JumpProposal(pta)
sampler.addProposalToCycle(jp.draw_from_prior, 15)
sampler.addProposalToCycle(jp.draw_from_bwm_prior, 15)
if ephem=='BayesEphem':
    sampler.addProposalToCycle(jp.draw_from_ephem_prior, 15)

# SAMPLE!!
N = 5000000
sampler.sample(x0, N, SCAMweight=35, AMweight=10, DEweight=50)
