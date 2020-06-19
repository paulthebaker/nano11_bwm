
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import random, os
import numpy as np

# for UL v. time (use same sampling as 11yr earth term)
TMIN = 53217.0
TMAX = 57387.0

tchunk = np.linspace(TMIN, TMAX, 41)  # break in 2.5% chunks
tlim = []
for ii in range(len(tchunk)-2):
    tlim.append(tchunk[ii:ii+3])

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

psrlist = '/home/pbaker/nanograv/data/nano11/11yr_34.txt'
with open(psrlist, 'r') as f:
    all_PSRs = [line.strip() for line in f]


PSRs = all_PSRs
EPHs = ['DE436']

subdir = 'condor_sub/'
subfile = 'bwm_sngl.sub'

dag_dir = '/home/pbaker/nanograv/bwm/sngl/'
if not os.path.exists(dag_dir):
    os.makedirs(dag_dir)
os.system('cp {0:s} {1:s}'.format(subdir+subfile, dag_dir))

dag_name = 'sngl_psr.dag'
dag = dag_dir + dag_name

config = dag_dir + dag_name + '.config'
with open(config, 'w') as f:
    f.write('DAGMAN_DEFAULT_NODE_LOG = '+ dag +'.nodes.log')

with open(dag_dir + dag_name, 'w') as f:
    f.write('CONFIG {:s}\n\n'.format(config))

    for psr in PSRs:
        for ephem in EPHs:
            # Detection
            outdir = os.path.join(dag_dir, "detect_"+ephem, psr)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            job_ID = random.getrandbits(128)
            f.write('JOB {0:d} {1:s}\n'.format(job_ID, subfile))
            f.write('VARS {0:d} outdir="{1:s}"\n'.format(job_ID, outdir))
            f.write('VARS {0:d} psr="{1:s}"\n'.format(job_ID, psr))
            f.write('VARS {0:d} ephem="{1:s}"\n'.format(job_ID, ephem))
            f.write('VARS {0:d} UL=""\n'.format(job_ID))
            f.write('VARS {0:d} tmin=""\n'.format(job_ID))
            f.write('VARS {0:d} tmax=""\n'.format(job_ID))
            f.write('\n')

            # ULs
            outdir = os.path.join(dag_dir, "uplim_"+ephem, psr, "all")
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            job_ID = random.getrandbits(128)
            f.write('JOB {0:d} {1:s}\n'.format(job_ID, subfile))
            f.write('VARS {0:d} outdir="{1:s}"\n'.format(job_ID, outdir))
            f.write('VARS {0:d} psr="{1:s}"\n'.format(job_ID, psr))
            f.write('VARS {0:d} ephem="{1:s}"\n'.format(job_ID, ephem))
            f.write('VARS {0:d} UL="--upper-limit"\n'.format(job_ID))
            f.write('VARS {0:d} tmin=""\n'.format(job_ID))
            f.write('VARS {0:d} tmax=""\n'.format(job_ID))
            f.write('\n')
        
            # UL v. T
            for tmin,cent,tmax in tlim:
                outdir = os.path.join(dag_dir, "uplim_"+ephem, psr, "{0:.2f}".format(cent))
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                job_ID = random.getrandbits(128)
                f.write('JOB {0:d} {1:s}\n'.format(job_ID, subfile))
                f.write('VARS {0:d} outdir="{1:s}"\n'.format(job_ID, outdir))
                f.write('VARS {0:d} psr="{1:s}"\n'.format(job_ID, psr))
                f.write('VARS {0:d} ephem="{1:s}"\n'.format(job_ID, ephem))
                f.write('VARS {0:d} UL="--upper-limit"\n'.format(job_ID))
                f.write('VARS {0:d} tmin="--tmin {1:.2f}"\n'.format(job_ID, tmin))
                f.write('VARS {0:d} tmax="--tmax {1:.2f}"\n'.format(job_ID, tmax))
                f.write('\n')

