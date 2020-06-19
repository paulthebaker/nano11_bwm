# run a BWM search for a fixed source orientation (theta, phi, psi, t0)
import numpy as np
import argparse, os
import pickle

from enterprise.pulsar import Pulsar
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from utils import sample_utils as su

### ARG PARSER
parser = argparse.ArgumentParser(
          description='run the BWM analysis with enterprise')

parser.add_argument('-d', '--datafile',
                    dest='datafile', default='/home/pbaker/nanograv/data/nano11.pkl',
                    action='store',
                    help="pickle file containing array of enterprise Pulsar objects")

parser.add_argument('-n', '--noisefile',
                    dest='noisefile', default='/home/pbaker/nanograv/data/nano11_setpars.pkl',
                    action='store',
                    help="pickle file containing noise parameters for all pulsars")

parser.add_argument('-o', '--outdir',
                    dest='outdir', default='~/nanograv/bwm/',
                    action='store',
                    help="location to write output")

parser.add_argument('--costheta', type=float,
                    dest='costh', default=None,
                    action='store',
                    help="sky position: cos(theta)")

parser.add_argument('--phi', type=float,
                    dest='phi', default=None,
                    action='store',
                    help="sky position: phi")

parser.add_argument('--psi', type=float,
                    dest='psi', default=None,
                    action='store',
                    help="polarization angle: psi")

parser.add_argument('--t0', type=float,
                    dest='t0', default=None,
                    action='store',
                    help="fixed t0 burst epoch (MJD)")

parser.add_argument('-u', '--upper-limit',
                    dest='UL', default=False,
                    action='store_true',
                    help=["use uniform priors suitable for upper limit ",
                          "calculation. Omit for log-uniform priors for ",
                          "detection"])

parser.add_argument('-b', '--bayesephem',
                    dest='BE', default=False,
                    action='store_true',
                    help="use 'BayesEphem' ephemeris modeling")

parser.add_argument('-N', '--Nsamp', type=int,
                    dest='N', default=int(1.0e+05),
                    action='store',
                    help="number of samples to collect (after thinning!!)")

parser.add_argument('-t', '--thin', type=int,
                    dest='thin', default=10,
                    action='store',
                    help="thinning factor (keep every [thin]th sample)")

parser.add_argument('-R', '--RN-distr',
                    dest='RNdistr', default=None,
                    action='store',
                    help="empirical distribution pickle file to use for RN moves")

parser.add_argument('-J', '--jup-kde',
                    dest='jupdistr', default=None,
                    action='store',
                    help="gaussian KDE pickle file to use for BE moves")

args = parser.parse_args()

outdir = os.path.abspath(args.outdir)
os.system('mkdir -p {}'.format(outdir))

# adjust Nsamp for existing chain
chfile = os.path.join(outdir, 'chain_1.txt')
if os.path.exists(chfile):
    ct = sum(1 for i in open(chfile, 'rb'))
    if ct >= args.N:
        print("{:s} has {:d} samples... exiting".format(chfile, ct))
        exit(0)
    else:
        args.N -= ct

if args.costh is not None and args.phi is not None and args.psi is not None:
    if args.costh > 1 or args.costh < -1:
        raise ValueError("costheta must be in range [-1, 1]")
    if args.phi > 2*np.pi or args.phi < 0:
        raise ValueError("phi must be in range [0, 2*pi]")
    if args.psi > 2*np.pi or args.psi < 0:
        raise ValueError("psi must be in range [0, 2*pi]")
else:
    err = "for fixed source must provide phi, costheta, and psi"
    raise RuntimeError(err)


# read in data pickles
with open(args.datafile, "rb") as f:
    psrs = pickle.load(f)

with open(args.noisefile, "rb") as f:
    noise_params = pickle.load(f)

print("loaded pickles")

#################
##  PTA model  ##
#################
tmin = np.min([p.toas.min() for p in psrs])
tmax = np.max([p.toas.max() for p in psrs])
Tspan = tmax - tmin

# White Noise
selection = selections.Selection(selections.by_backend)

efac = parameter.Constant()
equad = parameter.Constant()
ecorr = parameter.Constant()

ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)
ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)

wn = ef + eq + ec

# Red Noise
if args.UL:
    rn_log10_A = parameter.LinearExp(-20, -11)
else:
    rn_log10_A = parameter.Uniform(-20, -11)
rn_gamma = parameter.Uniform(0, 7)

rn_pl = utils.powerlaw(log10_A=rn_log10_A, gamma=rn_gamma)
rn = gp_signals.FourierBasisGP(rn_pl, components=30, Tspan=Tspan)

# GW BWM
amp_name = 'bwm_log10_A'
if args.UL:
    bwm_log10_A = parameter.LinearExp(-18, -11)(amp_name)
else:
    bwm_log10_A = parameter.Uniform(-18, -11)(amp_name)

t0 = parameter.Constant(args.t0)('bwm_t0')
pol = parameter.Constant(args.psi)('bwm_pol')
phi = parameter.Constant(args.phi)('bwm_phi')
costh = parameter.Constant(args.costh)('bwm_costheta')

bwm_wf = utils.bwm_delay(log10_h=bwm_log10_A, t0=t0,
                         cos_gwtheta=costh, gwphi=phi, gwpol=pol)
# BWM signal
bwm = deterministic_signals.Deterministic(bwm_wf, name='bwm')

# Timing Model
tm = gp_signals.TimingModel(use_svd=True)

# BayesEphem
be = deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

# construct PTA
mod = tm + wn + rn + bwm
if args.BE:
    mod += be

pta = signal_base.PTA([mod(p) for p in psrs])
pta.set_default_params(noise_params)

sumfile = os.path.join(outdir, 'summary.txt')
with open(sumfile, 'w') as f:
    f.write(pta.summary())

print("generated model")

outfile = os.path.join(outdir, 'params.txt')
with open(outfile, 'w') as f:
    for pname in pta.param_names:
        f.write(pname+'\n')


###############
##  sampler  ##
###############
x0 = np.hstack([noise_params[p.name] if p.name in noise_params.keys()
                else p.sample() for p in pta.params])  # initial point
ndim = len(x0)

# initial jump covariance matrix
# set initial cov stdev to (starting order of magnitude)/10
stdev = np.array([10**np.floor(np.log10(abs(x)))/10 for x in x0])
cov = np.diag(stdev**2)

# generate custom sampling groups
groups = [list(range(ndim))]  # all params

# pulsar noise groups (RN)
for psr in psrs:
    this_group = [pta.param_names.index(par)
                  for par in pta.param_names if psr.name in par]
    groups.append(this_group)

# bwm params
this_group = [pta.param_names.index(par)
              for par in pta.param_names if 'bwm_' in par]
for ii in range(5):
    # multiple copies of BWM group!
    groups.append(this_group)

if args.BE:
    # all BE params
    BE_group = [pta.param_names.index(par)
                  for par in pta.param_names
                  if 'jup_orb' in par or 'mass' in par or 'frame_drift' in par]
    groups.append(BE_group)

    # jup_orb elements + GWs
    this_group = [pta.param_names.index(par)
                  for par in pta.param_names if 'jup_orb' in par]
    this_group += [pta.param_names.index(par)
                   for par in pta.param_names if 'bwm_' in par]
    groups.append(this_group)

sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups,
                 outDir=outdir, resume=True)

# additional proposals
full_prior = su.build_prior_draw(pta, pta.param_names, name='full_prior')
sampler.addProposalToCycle(full_prior, 1)

if args.RNdistr:
    from utils.sample_utils import EmpiricalDistribution2D
    print("using empirical RN proposal")
    with open(args.RNdistr, "rb") as f:
        distr = pickle.load(f)
    Non4 = len(distr) // 4
    RN_emp = su.EmpDistrDraw(distr, pta.param_names, Nmax=Non4, name='RN_empirical')
    sampler.addProposalToCycle(RN_emp, 10)
else:
    # use log-uniform draw for RN
    print("using log-uniform RN proposal")
    RNA_params = [pname for pname in pta.param_names if 'red_noise_log10_A' in pname]
    RN_loguni = su.build_loguni_draw(pta, RNA_params, (-20,-11), name='RN_loguni')
    sampler.addProposalToCycle(RN_loguni, 5)

GWA_loguni = su.build_loguni_draw(pta, 'bwm_log10_A', (-18,-11), name='GWA_loguni')
sampler.addProposalToCycle(GWA_loguni, 5)

if args.BE:
    # start jup params near zero
    for p in pta.param_names:
        if "jup_" in p:
            x0[pta.param_names.index(p)] = np.random.normal(scale=0.01)

    if args.jupdistr:
        from scipy.stats import gaussian_kde
        print("using KDE empirical BE proposal")
        with open(args.jupdistr, "rb") as f:
            jup_kde = pickle.load(f)
        BE_kde = su.JupOrb_KDE_Draw(jup_kde, pta.param_names, 'jup_kde')
        sampler.addProposalToCycle(BE_kde, 5)

    BE_params = [pta.param_names[ii] for ii in BE_group]
    BE_prior = su.build_prior_draw(pta, BE_params, name='BE_prior')
    sampler.addProposalToCycle(BE_prior, 5)

# SAMPLE!!
Nsamp = args.N * args.thin
sampler.sample(x0, Nsamp,
               SCAMweight=30, AMweight=20, DEweight=50,
               burn=int(5e4), thin=args.thin)
