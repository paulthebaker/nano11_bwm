from __future__ import (absolute_import, division,
                        print_function) # , unicode_literals)
import numpy as np
import argparse, subprocess, glob

try:
    import pickle
except:
    # Python 2.7 ... harumph!
    import cPickle as pickle

from utils import models
from utils.sample_helpers import JumpProposal, get_parameter_groups

from enterprise.pulsar import Pulsar
from enterprise.signals import utils
from enterprise import constants as const
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc


### ARG PARSER
parser = argparse.ArgumentParser(
          description='run the BWM analysis with enterprise')

parser.add_argument('-p', '--psr',
                    dest='psr_name', default=None,
                    action='store',
                    help="pulsar to analyze")

parser.add_argument('-e', '--ephem',
                    dest='ephem', default='DE436',
                    action='store',
                    help="JPL ephemeris version to use")

parser.add_argument('-d', '--datadir',
                    dest='datadir', default='~/nanograv/data/',
                    action='store',
                    help="location of par/tim files")

parser.add_argument('-n', '--noisefile',
                    dest='noisefile', default=None,
                    action='store',
                    help="location of noise pickle")

parser.add_argument('-o', '--outdir',
                    dest='outdir', default='~/nanograv/bwm/{psr:s}/',
                    action='store',
                    help="location to write output")

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

parser.add_argument('-N', '--Nsamp', type=int,
                    dest='N', default=int(1.0e+06),
                    action='store',
                    help="number of samples to collect (before thinning)")

args = parser.parse_args()


try:
    subprocess.run(['mkdir', '-p', args.outdir])
except:
    # Python 2.7 ... harumph!
    subprocess.call(['mkdir', '-p', args.outdir])

# read in data from .par / .tim
par = glob.glob(args.datadir +'/'+ args.psr_name +'*.par')[0]
tim = glob.glob(args.datadir +'/'+ args.psr_name +'*.tim')[0]
psr = Pulsar(par, tim, ephem=args.ephem, timing_package='tempo2')

with open(args.noisefile, "rb") as f:
    setpars = pickle.load(f)


#################
##  pta model  ##
#################
logminA = -18
logmaxA = -9

tmin = psr.toas.min() / const.day
tmax = psr.toas.max() / const.day

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
    U,_ = utils.create_quantization_matrix(psr.toas)
    eps = 9  # clip first and last N observing epochs
    t0min = np.floor(max(U[:,eps] * psr.toas/const.day))
    t0max = np.ceil(max(U[:,-eps] * psr.toas/const.day))
    #tclip = (tmax - tmin) * 0.05
    #t0min = tmin + tclip*2  # clip first 10%
    #t0max = tmax - tclip    # clip last 5%

pta = models.model_bwm([psr], sngl_psr=True,
                       upper_limit=args.UL, bayesephem=False,
                       logmin=logminA, logmax=logmaxA,
                       Tmin_bwm=t0min, Tmax_bwm=t0max) 
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
cov = np.diag(np.ones(ndim) * 0.1**2)

# parameter groupings
groups = get_parameter_groups(pta)

sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior,
                 cov, groups=groups, outDir=args.outdir, resume=True)

# add prior draws to proposal cycle
jp = JumpProposal(pta)
sampler.addProposalToCycle(jp.draw_from_prior, 5)
sampler.addProposalToCycle(jp.draw_from_bwm_prior, 10)

draw_bwm_loguni = jp.build_log_uni_draw('bwm_log10_A', logminA, logmaxA)
sampler.addProposalToCycle(draw_bwm_loguni, 10)


# SAMPLE!!
sampler.sample(x0, args.N, SCAMweight=35, AMweight=10, DEweight=50)
