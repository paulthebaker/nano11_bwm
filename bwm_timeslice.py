
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import argparse, subprocess 

try:
    import pickle
except:
    # Python 2.7 ... harumph!
    import cPickle as pickle

from enterprise import constants as const

from utils import models
from utils.sample_helpers import JumpProposal, get_parameter_groups

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

parser.add_argument('-y', '--slice-yr',
                    dest='yrs', default=100, type=float,
                    action='store',
                    help="length of time slice in years. If slice is\
                          longer than dataset all time is used.")

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

try:
    subprocess.run(['mkdir', '-p', args.outdir])
except:
    # Python 2.7 ... harumph!
    subprocess.call('mkdir -p ' + args.outdir, shell=True)

# read in data pickles
filename = args.datadir + 'nano11_{}.pkl'.format(args.ephem)
with open(filename, "rb") as f:
    psrs = pickle.load(f)

filename = args.datadir + 'nano11_setpars.pkl'
with open(filename, "rb") as f:
    setpars = pickle.load(f)

# clip 2% of FULL data set at each end
# use same clip time for all slices
tmin = np.min([p.toas.min() for p in psrs]) / const.day
tmax = np.max([p.toas.max() for p in psrs]) / const.day
tclip = (tmax - tmin) * 0.02

psrs = models.which_psrs(psrs, args.yrs, 3)  # select pulsars


#################
##  pta model  ##
#################
logminA = -18
logmaxA = -11

# get tmax for this slice, use universal clip
tmax = np.min([p.toas.max() for p in psrs]) / const.day
t0min = tmin + tclip
t0max = tmax - tclip

pta = models.model_bwm(psrs,
                       upper_limit=args.UL, bayesephem=args.BE,
                       logmin=logminA, logmax=logmaxA,
                       Tmin_bwm=t0min, Tmax_bwm=t0max)
pta.set_default_params(setpars)


outfile = args.outdir + 'params.txt'
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
                 cov, groups=groups, outDir=args.outdir, resume=True)

# add prior draws to proposal cycle
jp = JumpProposal(pta)
sampler.addProposalToCycle(jp.draw_from_prior, 15)
sampler.addProposalToCycle(jp.draw_from_bwm_prior, 15)

draw_bwm_loguni = jp.build_log_uni_draw('log10_A_bwm', logminA, logmaxA)
sampler.addProposalToCycle(draw_bwm_loguni, 20)


# SAMPLE!!
sampler.sample(x0, args.N, SCAMweight=35, AMweight=10, DEweight=50)
