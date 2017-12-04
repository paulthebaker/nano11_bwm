
import numpy as np
import scipy.stats as ss
from acor import acor


def UL_uncert(chain, p=0.95):
    corr = acor(chain)[0]
    N = len(chain)
    Neff = N/corr

    hist = np.histogram(chain, bins=100)
    pdf = ss.rv_histogram(hist).pdf

    UL = np.percentile(chain, 100*p)  # 95 for 95% (not 0.95)
    pUL = pdf(UL)
    dUL = np.sqrt(p*(1-p)/Neff) / pUL

    return UL, dUL
