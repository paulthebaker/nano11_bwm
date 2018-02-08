from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import random, os
import numpy as np


slices = np.arange(5, 12, 0.5)

EPHs = ['DE421', 'DE436']

subfile = 'bwm_slice.sub'

dag_dir = '/home/pbaker/nanograv/bwm/slices/'
if not os.path.exists(dag_dir):
    os.makedirs(dag_dir)
os.system('cp {0:s} {1:s}'.format(subfile, dag_dir))

dag_name = 'slices.dag'
dag = dag_dir + dag_name

config = dag_dir + dag_name + '.config'
with open(config, 'w') as f:
    f.write('DAGMAN_DEFAULT_NODE_LOG = '+ dag +'.nodes.log')

with open(dag_dir + dag_name, 'w') as f:
    f.write('CONFIG {:s}\n\n'.format(config))

    for sl in slices:
        for ephem in EPHs:
            outdir = dag_dir + "{0:.1f}/{1:s}/".format(sl, ephem)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            job_ID = random.getrandbits(128)
            f.write('JOB {0:d} {1:s}\n'.format(job_ID, subfile))
            f.write('VARS {0:d} slice="{1:.1f}"\n'.format(job_ID, sl))
            f.write('VARS {0:d} ephem="{1:s}"\n'.format(job_ID, ephem))
            f.write('\n')
