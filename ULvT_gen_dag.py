
import random, os
import numpy as np
import healpy as hp

nside = 2
npix = hp.nside2npix(nside)  # 48 sky points

Npol = 8  # number of polarization bins
POLs = np.arange(Npol) * np.pi / Npol

Nts = 40  # number of time bins
TMIN = 53217.0
TMAX = 57387.0
Ts = np.linspace(TMIN, TMAX, Nts+2)[1:-1]  # don't use first/last t0

datadir = '/home/pbaker/nanograv/data'
datafile_root = 'nano11_{:s}.pkl'

EPHs = ['DE430', 'DE436', 'DE436_BE']

subdir = 'condor_sub/'
subfile = 'bwm_fixsrc.sub'

dag_dir = '/home/pbaker/nanograv/bwm/ULvT/'
if not os.path.exists(dag_dir):
    os.makedirs(dag_dir)
os.system('cp {0:s} {1:s}'.format(subdir+subfile, dag_dir))

dag_root = 'ULvT_{:s}.dag'

# each ephem gets its own .dag
for ephem in EPHs:
    bayesephem = ""
    if '_BE' in ephem:
        bayesephem = "--bayesephem"
        ee = ephem.split('_')[0]
        datafile = os.path.join(datadir, datafile_root.format(ee))
    else:
        bayesephem = ""
        datafile = os.path.join(datadir, datafile_root.format(ephem))
    
    dag_name = dag_root.format(ephem)
    dag = os.path.join(dag_dir, dag_name)

    config = os.path.join(dag_dir, dag_name+'.config')
    with open(config, 'w') as f:
        f.write('DAGMAN_DEFAULT_NODE_LOG = '+ dag +'.nodes.log')

    with open(dag, 'w') as f:
        f.write('CONFIG {:s}\n\n'.format(config))

        for t0 in Ts:
            for loc in range(npix):
                theta, phi = hp.pix2ang(nside, loc)
                costh = np.cos(theta)
                for pol in POLs:
                    runpath = "{:s}/{:.2f}/{:02d}/{:.2f}".format(ephem, t0, loc, pol)
                    outdir = os.path.join(dag_dir, runpath)
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)

                    job_ID = random.getrandbits(128)

                    f.write('JOB {0:d} {1:s}\n'.format(job_ID, subfile))
                    f.write('VARS {0:d} outdir="{1:s}"\n'.format(job_ID, outdir))
                    f.write('VARS {0:d} datafile="{1:s}"\n'.format(job_ID, datafile))
                    f.write('VARS {0:d} BE="{1:s}"\n'.format(job_ID, bayesephem))

                    f.write('VARS {0:d} costh="{1:.8f}"\n'.format(job_ID, costh))
                    f.write('VARS {0:d} phi="{1:.8f}"\n'.format(job_ID, phi))
                    f.write('VARS {0:d} psi="{1:.8f}"\n'.format(job_ID, pol))
                    f.write('VARS {0:d} t0="{1:.2f}"\n'.format(job_ID, t0))

                    f.write('\n')
