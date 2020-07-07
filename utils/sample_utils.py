# sampling utilities for IPTA DR2 runs
#  at some point should merge into enterprise extensions

import numpy as np
from scipy.stats import gaussian_kde

class UserDraw(object):
    """object for user specified proposal distributions
    """
    def __init__(self, idxs, samplers, log_qs=None, name=None):
        """
        :param idxs: list of parameter indices to use for this jump
        :param samplers: dict of callable samplers
            keys should include all idxs
        :param lqxys: dict of callable log proposal distributions
            keys should include all idxs
            for symmetric proposals set `log_qs=None`, then `log_qxy=0`
        :param name: name for PTMCMC bookkeeping
        """
        #TODO check all idxs in keys!
        self.idxs = idxs
        self.samplers = samplers
        self.log_qs = log_qs

        if name is None:
            namestr = 'draw'
            for ii in samplers.keys():
                namestr += '_{}'.format(ii)
            self.__name__ = namestr
        else:
            self.__name__ = name

    def __call__(self, x, iter, beta):
        """proposal from parameter prior distribution
        """
        y = x.copy()

        # draw parameter from idxs
        ii = np.random.choice(self.idxs)

        try: # vector parameter
            y[ii] = self.samplers[ii]()[0]
        except (IndexError, TypeError) as e:
            y[ii] = self.samplers[ii]()

        if self.log_qs is None:
            lqxy = 0
        else:
            lqxy = self.log_qs[ii](x[ii]) - self.log_qs[ii](y[ii])

        return y, lqxy


def build_prior_draw(pta, parlist, name=None):
    """create a callable object to perfom a prior draw
    :param pta:
        instantiated PTA object
    :param parlist:
        single string or list of strings of parameter name(s) to
        use for this jump.
    :param name:
        display name for PTMCMCSampler bookkeeping
    """
    if not isinstance(parlist, list):
        parlist = [parlist]
    idxs = [pta.param_names.index(par) for par in parlist]

    # parameter map
    pmap = []
    ct = 0
    for ii, pp in enumerate(pta.params):
        size = pp.size or 1
        for nn in range(size):
            pmap.append(ii)
        ct += size

    sampler = {ii: pta.params[pmap[ii]].sample for ii in idxs}
    log_q = {ii: pta.params[pmap[ii]].get_logpdf for ii in idxs}

    return UserDraw(idxs, sampler, log_q, name=name)

def build_loguni_draw(pta, parlist, bounds, name=None):
    """create a callable object to perfom a log-uniform draw
    :param pta:
        instantiated PTA object
    :param parlist:
        single string or list of strings of parameter name(s) to
        use for this jump.
    :param bounds:
        tuple of (pmin, pmax) for draw
    :param name:
        display name for PTMCMCSampler bookkeeping
    """
    if not isinstance(parlist, list):
        parlist = [parlist]
    idxs = [pta.param_names.index(par) for par in parlist]
    pmin, pmax = bounds
    sampler = {ii: (lambda : np.random.uniform(pmin,pmax)) for ii in idxs}

    return UserDraw(idxs, sampler, None, name=name)


class EmpiricalDistribution2D(object):
    def __init__(self, param_names, samples, bins):
        """
        :param samples: samples for hist
        :param bins: edges to use for hist (left and right)
            make sure bins cover whole prior!
        """
        self.ndim = 2
        self.param_names = param_names
        self._Nbins = [len(b)-1 for b in bins]
        hist, x_bins, y_bins = np.histogram2d(*samples, bins=bins)

        self._edges = np.array([x_bins[:-1], y_bins[:-1]])
        self._wids = np.diff([x_bins, y_bins])

        area = np.outer(*self._wids)
        hist += 1  # add a sample to every bin
        counts = np.sum(hist)
        self._pdf = hist / counts / area
        self._cdf = np.cumsum((self._pdf*area).ravel())

        self._logpdf = np.log(self._pdf)

    def draw(self):
        """draw a sample from the distribution"""
        draw = np.random.rand()
        draw_bin = np.searchsorted(self._cdf, draw)

        idx = np.unravel_index(draw_bin, self._Nbins)
        samp = [self._edges[ii, idx[ii]] + self._wids[ii, idx[ii]]*np.random.rand()
                for ii in range(2)]
        return np.array(samp)

    def prob(self, X):
        """pdf of distribution
        :param X: vector point in parameter space
        """
        ix, iy = (min(np.searchsorted(self._edges[ii], X[ii]),
                      self._Nbins[ii]-1) for ii in range(2))

        return self._pdf[ix, iy]

    def logprob(self, X):
        """log(pdf) of distribution at
        :param X: vector point in parameter space
        """
        ix, iy = (min(np.searchsorted(self._edges[ii], X[ii]),
                      self._Nbins[ii]-1) for ii in range(2))

        return self._logpdf[ix, iy]


class EmpDistrDraw(object):
    """object for empirical proposal distributions
    """
    def __init__(self, distr, parlist, Nmax=None, name=None):
        """
        :param distr: list of EmpiricalDistribution2D objects
        :param parlist: list of all model params (pta.param_names)
            to figure out which indices to use
        :param Nmax: maximum number of distributions to propose
            simultaneously
        :param name: name for PTMCMC bookkeeping
        """
        self._distr = distr
        self.Nmax = Nmax if Nmax else len(distr)        
        self.__name__ = name if name else 'draw_empirical'

        # which model indices go with which distr?
        for dd in self._distr:
            dd._idx = []
            for pp in parlist:
                if pp in dd.param_names:
                    dd._idx.append(parlist.index(pp))

    def __call__(self, x, iter, beta):
        """propose a move from empirical distribution
        """
        y = x.copy()
        lqxy = 0

        # which distrs to propose moves
        N = np.random.randint(1, self.Nmax)
        which = np.random.choice(self._distr, size=N, replace=False)

        for distr in which:
            old = x[distr._idx]
            new = distr.draw()
            y[distr._idx] = new

            lqxy += (distr.logprob(old) -
                     distr.logprob(new))

        return y, lqxy

class JupOrb_KDE_Draw(object):
    """object for bayesephem proposal distribution
    """
    def __init__(self, jup_kde, parlist, name='draw_jup'):
        """
        :param jup_kde: jup_orb_elements gaussian_KDE object
        :param parlist: list of all model params (pta.param_names)
            to figure out which indices to use
        :param name: name for PTMCMC bookkeeping
        """
        self.__name__ = name

        # which model indices?
        jup_kde._idx = [parlist.index(pp)
                        for pp in parlist if pp.startswith('jup_orb')]

        self._distr = jup_kde

    def __call__(self, x, iter, beta):
        """propose a move from KDE distribution
        """
        y = x.copy()
        
        # move all of 'em simultaneously
        dd = self._distr
        old = x[dd._idx]
        new = dd.resample(size=1).T
        y[dd._idx] = new

        lqxy = (dd.logpdf(old) -
                dd.logpdf(new))[0]

        return y, lqxy
