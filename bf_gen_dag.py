from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import random, os
import numpy as np

TMIN = 53217.0
TMAX = 57387.0

EPHs = ['DE436']

subdir = 'condor_sub/'
subfile = 'bwm_BF.sub'

dag_dir = '/home/pbaker/nanograv/bwm/allsky/'
if not os.path.exists(dag_dir):
    os.makedirs(dag_dir)
os.system('cp {0:s} {1:s}'.format(subdir+subfile, dag_dir))

dag_name = 'BF.dag'
dag = dag_dir + dag_name

config = dag_dir + dag_name + '.config'
with open(config, 'w') as f:
    f.write('DAGMAN_DEFAULT_NODE_LOG = '+ dag +'.nodes.log')

with open(dag_dir + dag_name, 'w') as f:
    f.write('CONFIG {:s}\n\n'.format(config))

    # all sky, all time
    N = int(5.0e+06)
    for ephem in EPHs:
        
        # with BayesEphem
        outdir = dag_dir + "{0:s}_BE/detection/".format(ephem)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        job_ID = random.getrandbits(128)
        f.write('JOB {0:d} {1:s}\n'.format(job_ID, subfile))
        f.write('VARS {0:d} ephem="{1:s}"\n'.format(job_ID, ephem))
        f.write('VARS {0:d} BE="_BE"\n'.format(job_ID))
        f.write('VARS {0:d} bayesephem="--bayes-ephem"\n'.format(job_ID))
        f.write('VARS {0:d} N="{1:d}"\n'.format(job_ID, N))
        f.write('\n')
