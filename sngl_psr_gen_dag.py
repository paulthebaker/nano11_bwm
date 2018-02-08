
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import random, os

Dustys_PSRs = ['J0030+0451',
               'J0613-0200',
               'J1012+5307',
               'J1909-3744',
               'J1918-0642',
               'J2145-0750']

extra_PSRs = ['J1713+0747',
              'J1643-1224',
              'J1744-1134',
              'B1937+21']

PSRs = Dustys_PSRs + extra_PSRs

EPHs = ['DE421', 'DE436']

subfile = 'bwm_sngl.sub'

dag_dir = '/home/pbaker/nanograv/bwm/sngl/'
if not os.path.exists(dag_dir):
    os.makedirs(dag_dir)
os.system('cp {0:s} {1:s}'.format(subfile, dag_dir))

dag_name = 'sngl_psr.dag'
dag = dag_dir + dag_name

config = dag_dir + dag_name + '.config'
with open(config, 'w') as f:
    f.write('DAGMAN_DEFAULT_NODE_LOG = '+ dag +'.nodes.log')

with open(dag_dir + dag_name, 'w') as f:
    f.write('CONFIG {:s}\n\n'.format(config))

    for psr in PSRs:
        for ephem in EPHs:
            outdir = dag_dir + psr +'/'+ ephem +'/'
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            job_ID = random.getrandbits(128)
            f.write('JOB {0:d} {1:s}\n'.format(job_ID, subfile))
            f.write('VARS {0:d} psr="{1:s}"\n'.format(job_ID, psr))
            f.write('VARS {0:d} ephem="{1:s}"\n'.format(job_ID, ephem))
            f.write('\n')

