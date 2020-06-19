
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import argparse, subprocess, os

try:
    import pickle
except:
    # Python 2.7 ... harumph!
    import cPickle as pickle

from enterprise_extensions import models
from enterprise_extensions import model_utils

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc


### ARG PARSER
parser = argparse.ArgumentParser(
          description='run the BWM analysis with enterprise')

parser.add_argument('-d', '--datadir',
                    dest='datadir', default='~/nanograv/data/nano11_nodmx',
                    action='store',
                    help="location of .par and .tim files")

parser.add_argument('-n', '--noisefile',
                    dest='noisefile', default='~/nanograv/data/nano11_setpars.pkl',
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
                    help=["use uniform priors suitable for upper limit ",
                          "calculation. False for log-uniform priors for ",
                          "detection"])

parser.add_argument('-b', '--bayes-ephem',
                    dest='BE', default=False,
                    action='store_true',
                    help="use 'BayesEphem' ephemeris modeling")

parser.add_argument('-g', '--dmgp',
                    dest='DMGP', default=False,
                    action='store_true',
                    help=["use gaussian process DM variation modeling",
                          "(instead of DMX"])

parser.add_argument('-N', '--Nsamp', type=int,
                    dest='N', default=int(1.0e+06),
                    action='store',
                    help="number of samples to collect (before thinning)")

parser.add_argument('--Nmax', type=int,
                    dest='Nmax', default=int(1.0e+05),
                    action='store',
                    help="Maximum number of thinned samples when resuming")

parser.add_argument('--write-hot',
                    dest='write_hot', default=False,
                    action='store_true',
                    help="write hot PT chains")


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

# get list of pulsars
filename = os.path.join(args.datadir, '11yr_34.txt')
psrlist = []
for line in open(filename, 'r'):
    psrlist.append(line.strip())
psrlist.sort()

# load pulsars
parbase = '{}_NANOGrav_11yv0.gls.par'
timbase = '{}_NANOGrav_11yv0.tim'
psrs = []
for pname in psrlist:
    pfile = os.path.join(args.datadir, parbase.format(pname))
    tfile = os.path.join(args.datadir, timbase.format(pname))
    psrs.append(Pulsar(pfile, tfile, ephem='DE436', timing_package='tempo2'))

# read in noise pickle
with open(args.noisefile, "rb") as f:
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

pta = models.model_bwm(psrs, noisedict=noisepars,
                       Tmin_bwm=t0min, Tmax_bwm=t0max, skyloc=None,
                       red_psd='powerlaw', components=30,
                       dm_var=True, dm_psd='powerlaw', dm_annual=True,
                       upper_limit=args.UL, bayesephem=args.BE, dmgp=args.DMGP,
                       wideband=False)

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
jp = model_utils.JumpProposal(pta)
sampler.addProposalToCycle(jp.draw_from_prior, 5)
sampler.addProposalToCycle(jp.draw_from_bwm_prior, 10)
if args.BE:
    sampler.addProposalToCycle(jp.draw_from_ephem_prior, 10)        
if args.DMGP:
    sampler.addProposalToCycle(jp.draw_from_dmgp_prior, 10)      
    sampler.addProposalToCycle(jp.draw_from_dm1yr_prior, 10)      
if args.UL:
    draw_bwm_loguni = jp.build_log_uni_draw('bwm_log10_A', logminA, logmaxA)
    sampler.addProposalToCycle(draw_bwm_loguni, 10)


# SAMPLE!!
sampler.sample(x0, args.N,
               SCAMweight=30, AMweight=20, DEweight=50,
               writeHotChains=args.write_hot)
