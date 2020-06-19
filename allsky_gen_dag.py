from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import random, os
import numpy as np

TMIN = 53217.0
TMAX = 57387.0

tchunk = np.linspace(TMIN, TMAX, 51)  # break in 2% chunks
tlim = []
for ii in range(len(tchunk)-2):
    tlim.append(tchunk[ii:ii+3])

datadir = '/home/pbaker/nanograv/data'

#EPHs = ['DE421', 'DE430', 'DE436']
EPHs = ['DE436']

subdir = 'condor_sub/'
subfile = 'bwm_allsky.sub'

dag_dir = '/home/pbaker/nanograv/bwm/allsky_rerun/'
if not os.path.exists(dag_dir):
    os.makedirs(dag_dir)
os.system('cp {0:s} {1:s}'.format(subdir+subfile, dag_dir))

dag_name = 'allsky.dag'
dag = dag_dir + dag_name

config = dag_dir + dag_name + '.config'
with open(config, 'w') as f:
    f.write('DAGMAN_DEFAULT_NODE_LOG = '+ dag +'.nodes.log')

with open(dag_dir + dag_name, 'w') as f:
    f.write('CONFIG {:s}\n\n'.format(config))

    # all sky, all time
    N = int(5.0e+06)
    for ephem in EPHs:
        datafile = os.path.join(datadir, 'nano11_{}.pkl'.format(ephem))
        noisefile = os.path.join(datadir, 'nano11_setpars.pkl')

        # no BayesEphem
        outdir = os.path.join(dag_dir,"{0:s}/all/".format(ephem))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        job_ID = random.getrandbits(128)
        f.write('JOB {0:d} {1:s}\n'.format(job_ID, subfile))
        f.write('VARS {0:d} datafile="{1:s}"\n'.format(job_ID, datafile))
        f.write('VARS {0:d} noisefile="{1:s}"\n'.format(job_ID, noisefile))
        f.write('VARS {0:d} outdir="{1:s}"\n'.format(job_ID, outdir))
        f.write('VARS {0:d} tmin=""\n'.format(job_ID))
        f.write('VARS {0:d} tmax=""\n'.format(job_ID))
        f.write('VARS {0:d} BE=""\n'.format(job_ID))
        f.write('VARS {0:d} bayesephem=""\n'.format(job_ID))
        f.write('VARS {0:d} dmgp=""\n'.format(job_ID))
        f.write('VARS {0:d} N="{1:d}"\n'.format(job_ID, N))
        f.write('\n')

        # with BayesEphem
        outdir = os.path.join(dag_dir,"{0:s}_BE/all/".format(ephem))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        job_ID = random.getrandbits(128)
        f.write('JOB {0:d} {1:s}\n'.format(job_ID, subfile))
        f.write('VARS {0:d} datafile="{1:s}"\n'.format(job_ID, datafile))
        f.write('VARS {0:d} noisefile="{1:s}"\n'.format(job_ID, noisefile))
        f.write('VARS {0:d} outdir="{1:s}"\n'.format(job_ID, outdir))
        f.write('VARS {0:d} tmin=""\n'.format(job_ID))
        f.write('VARS {0:d} tmax=""\n'.format(job_ID))
        f.write('VARS {0:d} BE="_BE"\n'.format(job_ID))
        f.write('VARS {0:d} bayesephem="--bayes-ephem"\n'.format(job_ID))
        f.write('VARS {0:d} dmgp=""\n'.format(job_ID))
        f.write('VARS {0:d} N="{1:d}"\n'.format(job_ID, N))
        f.write('\n')

    # all sky, time chunks
    N = int(1.0e+06)
    for ephem in EPHs:
        # no BayesEphem
        for tmin,cent,tmax in tlim:
            outdir = os.path.join(dag_dir,"{0:s}/{1:.2f}/".format(ephem, cent))
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            job_ID = random.getrandbits(128)
            f.write('JOB {0:d} {1:s}\n'.format(job_ID, subfile))
            f.write('VARS {0:d} datafile="{1:s}"\n'.format(job_ID, datafile))
            f.write('VARS {0:d} noisefile="{1:s}"\n'.format(job_ID, noisefile))
            f.write('VARS {0:d} outdir="{1:s}"\n'.format(job_ID, outdir))
            f.write('VARS {0:d} tmin="--tmin {1:.2f}"\n'.format(job_ID, tmin))
            f.write('VARS {0:d} tmax="--tmax {1:.2f}"\n'.format(job_ID, tmax))
            f.write('VARS {0:d} BE=""\n'.format(job_ID))
            f.write('VARS {0:d} bayesephem=""\n'.format(job_ID))
            f.write('VARS {0:d} dmgp=""\n'.format(job_ID))
            f.write('VARS {0:d} N="{1:d}"\n'.format(job_ID, N))
            f.write('\n')

        # with BayesEphem
        for tmin,cent,tmax in tlim:
            outdir = os.path.join(dag_dir,"{0:s}_BE/{1:.2f}/".format(ephem, cent))
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            job_ID = random.getrandbits(128)
            f.write('JOB {0:d} {1:s}\n'.format(job_ID, subfile))
            f.write('VARS {0:d} datafile="{1:s}"\n'.format(job_ID, datafile))
            f.write('VARS {0:d} noisefile="{1:s}"\n'.format(job_ID, noisefile))
            f.write('VARS {0:d} outdir="{1:s}"\n'.format(job_ID, outdir))
            f.write('VARS {0:d} tmin="--tmin {1:.2f}"\n'.format(job_ID, tmin))
            f.write('VARS {0:d} tmax="--tmax {1:.2f}"\n'.format(job_ID, tmax))
            f.write('VARS {0:d} BE="_BE"\n'.format(job_ID))
            f.write('VARS {0:d} bayesephem="--bayes-ephem"\n'.format(job_ID))
            f.write('VARS {0:d} dmgp=""\n'.format(job_ID))
            f.write('VARS {0:d} N="{1:d}"\n'.format(job_ID, N))
            f.write('\n')
