
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import argparse, subprocess 

try:
    import pickle
except:
    # Python 2.7 ... harumph!
    import cPickle as pickle

from utils import models
from utils.sample_helpers import JumpProposal, get_parameter_groups, count_lines

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc


### ARG PARSER
parser = argparse.ArgumentParser(
          description='run the BWM analysis with enterprise')

parser.add_argument('-e', '--ephem',
                    dest='ephem', default='DE436',
                    action='store',
                    help="JPL ephemeris version to use")

parser.add_argument('-d', '--datadir',
                    dest='datadir', default='~/nanograv/data/',
                    action='store',
                    help="location of data and noise pickles")

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

parser.add_argument('--tmin', type=float,
                    dest='tmin', default=None,
                    action='store',
                    help="min search time (MJD)")

parser.add_argument('--tmax', type=float,
                    dest='tmax', default=None,
                    action='store',
                    help="max search time (MJD)")

parser.add_argument('-u', '--upper-limit',
                    dest='UL', default=False,
                    action='store_true',
                    help="use uniform priors suitable for upper limit\
                          calculation. False for log-uniform priors for\
                          detection")

parser.add_argument('-b', '--bayes-ephem',
                    dest='BE', default=False,
                    action='store_true',
                    help="use 'BayesEphem' ephemeris modeling")

parser.add_argument('-N', '--Nsamp', type=int,
                    dest='N', default=int(1.0e+06),
                    action='store',
                    help="number of samples to collect (before thinning)")

args = parser.parse_args()


if args.costh is not None and args.phi is not None:
    if args.costh > 1 or args.costh < -1:
        raise ValueError("costheta must be in range [-1, 1]")
    if args.phi > 2*np.pi or args.phi < 0:
        raise ValueError("phi must be in range [0, 2*pi]")

    skyloc = [args.costh, args.phi]

elif not args.costh and not args.phi:
    skyloc = None

else:
    err = "for fixed sky location must provide BOTH phi and costheta"
    raise RuntimeError(err)

try:
    subprocess.run(['mkdir', '-p', args.outdir])
except:
    # Python 2.7 ... harumph!
    subprocess.call('mkdir -p ' + args.outdir, shell=True)

chainfile = outdir + '/chain_1.txt'
thin = 10  # default PTMCMC thinning
if os.path.isfile(chainfile)
    Ndone = count_lines(chainfile)
    args.N = int((args.N - thin*Ndone) + Ndone)

# read in data pickles
filename = args.datadir + 'nano11_{}.pkl'.format(args.ephem)
with open(filename, "rb") as f:
    psrs = pickle.load(f)

filename = args.datadir + 'nano11_setpars.pkl'
with open(filename, "rb") as f:
    setpars = pickle.load(f)


#################
##  pta model  ##
#################
logminA = -18
logmaxA = -11

tmin = np.min([p.toas.min() for p in psrs]) / 86400
tmax = np.max([p.toas.max() for p in psrs]) / 86400

if args.tmin is not None and args.tmax is not None:
    if args.tmin<tmin:
        err = "tmin ({:.1f}) BEFORE first TOA ({:.1f})".format(args.tmin, tmin)
        raise RuntimeError(err)
    elif args.tmax>tmax:
        err = "tmax ({:.1f}) AFTER last TOA ({:.1f})".format(args.tmax, tmax)
        raise RuntimeError(err)
    elif args.tmin>args.tmax:
        err = "tmin ({:.1f}) BEFORE last tmax ({:.1f})".format(args.tmin, args.tmax)
        raise RuntimeError(err)
    else:
        t0min = args.tmin
        t0max = args.tmax
else:
    tclip = (tmax - tmin) * 0.05
    t0min = tmin + tclip*2  # clip first 10%
    t0max = tmax - tclip    # last 5%


pta = models.model_bwm(psrs,
                       upper_limit=args.UL, bayesephem=args.BE,
                       logmin=logminA, logmax=logmaxA,
                       Tmin_bwm=t0min, Tmax_bwm=t0max,
                       skyloc=skyloc)
pta.set_default_params(setpars)


outfile = args.outdir + '/params.txt'
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
try:
    cov = np.load(outdir+'/cov.npy')
except:
    cov = np.diag(np.ones(ndim) * 0.1**2)

# parameter groupings
groups = get_parameter_groups(pta)

sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior,
                 cov, groups=groups, outDir=args.outdir, resume=True)

# add prior draws to proposal cycle
jp = JumpProposal(pta)
sampler.addProposalToCycle(jp.draw_from_prior, 15)
sampler.addProposalToCycle(jp.draw_from_bwm_prior, 15)

draw_bwm_loguni = jp.build_log_uni_draw('log10_A_bwm', logminA, logmaxA)
sampler.addProposalToCycle(draw_bwm_loguni, 20)


# SAMPLE!!
sampler.sample(x0, args.N, SCAMweight=35, AMweight=10, DEweight=50)
