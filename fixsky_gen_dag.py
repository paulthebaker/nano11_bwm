from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import random, os
import numpy as np
import healpy as hp

nside = 8
npix = hp.nside2npix(nside)

EPHs = ['DE421', 'DE436']

subdr = 'condor_sub/'
subfile = 'bwm_fixsky.sub'

dag_dir = '/home/pbaker/nanograv/bwm/fixsky/'
if not os.path.exists(dag_dir):
    os.makedirs(dag_dir)
os.system('cp {0:s} {1:s}'.format(subdir+subfile, dag_dir))

dag_name = 'fixsky.dag'
dag = dag_dir + dag_name

config = dag_dir + dag_name + '.config'
with open(config, 'w') as f:
    f.write('DAGMAN_DEFAULT_NODE_LOG = '+ dag +'.nodes.log')

with open(dag_dir + dag_name, 'w') as f:
    f.write('CONFIG {:s}\n\n'.format(config))

    for ephem in EPHs:
        for loc in range(npix):
            theta, phi = hp.pix2ang(nside,ii)
            costh = np.cos(theta)

            outdir = dag_dir + "{0:s}/{1:03d}/".format(ephem, loc)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            job_ID = random.getrandbits(128)
            f.write('JOB {0:d} {1:s}\n'.format(job_ID, subfile))
            f.write('VARS {0:d} ephem="{1:s}"\n'.format(job_ID, ephem))
            f.write('VARS {0:d} loc="{1:03d}"\n'.format(job_ID, loc))
            f.write('VARS {0:d} costh="{1:.8f}"\n'.format(job_ID, costh))
            f.write('VARS {0:d} phi="{1:.8f}"\n'.format(job_ID, phi))
            f.write('\n')
