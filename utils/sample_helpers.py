from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import os

def count_lines(fname):
    """count lines in a file

    :param fname: filename including path
    """
    if not os.path.exists(fname):
        return 0
    with open(chain_path) as f:
        rows = sum(1 for _ in f)
    return int(rows)

class JumpProposal(object):
    
    def __init__(self, pta):
        """Set up some custom jump proposals
        
        :param pta: an `enterprise` PTA instance
        """
        self.params = pta.params
        self.pnames = pta.param_names
        self.npar = len(pta.params)
        self.ndim = sum(p.size or 1 for p in pta.params)
        
        # parameter map
        self.pmap = {}
        ct = 0
        for p in pta.params:
            size = p.size or 1
            self.pmap[p] = slice(ct, ct+size)
            ct += size
            
        # parameter indices map
        self.pimap = {}
        for ct, p in enumerate(pta.param_names):
            self.pimap[p] = ct
            
        self.snames = {}
        for sc in pta._signalcollections:
            for signal in sc._signals:
                self.snames[signal.signal_name] = signal.params


    def draw_from_prior(self, x, iter, beta):
        """Prior draw.
        
        The function signature is specific to PTMCMCSampler.
        """
        
        q = x.copy()
        lqxy = 0
        
        # randomly choose parameter
        idx = np.random.randint(0, self.npar)
        
        # if vector parameter jump in random component
        param = self.params[idx]
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[param]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[idx] = param.sample()
        
        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[param]]) - param.get_logpdf(q[self.pmap[param]])
                
        return q, float(lqxy)


    def draw_from_gwb_prior(self, x, iter, beta):
        q = x.copy()
        lqxy = 0
        
        signal_name = 'red noise'
        
        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[param]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[param]] = param.sample()
        
        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[param]]) - param.get_logpdf(q[self.pmap[param]])
                        
        return q, float(lqxy)


    def draw_from_bwm_prior(self, x, iter, beta):
        q = x.copy()
        lqxy = 0
        
        signal_name = 'bwm'
        
        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[param]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[param]] = param.sample()
        
        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[param]]) - param.get_logpdf(q[self.pmap[param]])
                        
        return q, float(lqxy)


    def draw_from_ephem_prior(self, x, iter, beta):        
        q = x.copy()
        lqxy = 0
        
        signal_name = 'phys_ephem'
        
        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[param]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[param]] = param.sample()
        
        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[param]]) - param.get_logpdf(q[self.pmap[param]])
                
        return q, float(lqxy)


    def build_log_uni_draw(self, plist, logmin, logmax):
        """create a callable object to perfom a log-uniform draw

        :param plist:
            single string or list of strings of parameter name(s) to
            use for this jump.
        :param logmin:
            log of min of uniform distr (log10)
        :param logmax:
            log of max of uniform distr (log10)
        """
        if not isinstance(plist, list):
            plist = [plist]
        idxs = []
        for pn in plist:
            idxs.append(self.pnames.index(pn))
        
        lud = LogUniDraw(idxs, logmin, logmax)

        return lud


class LogUniDraw(object):
    """object for custom log-uniform draws
    """
    def __init__(self, idxs, logmin, logmax):
        """
        :param idx: index of parameter to use for jump
        """
        self.idxs = idxs
        self.logmin = logmin
        self.logmax = logmax
        
        namestr = 'logunidraw'
        for ii in idxs:
            namestr += '_{}'.format(ii)
        self.__name__ = namestr

    def __call__(self, x, iter, beta):
        """proposal from log-uniform distribution
        """
        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        for ii in self.idxs:
            q[ii] = np.random.uniform(self.logmin, self.logmax)

        return q, 0


# utility function for finding global parameters
def get_global_parameters(pta):
    pars = []
    for sc in pta._signalcollections:
        pars.extend(sc.param_names)
    
    gpars = np.unique(list(filter(lambda x: pars.count(x)>1, pars)))
    ipars = np.array([p for p in pars if p not in gpars])
        
    return gpars, ipars


# utility function to get parameter groupings for sampling
def get_parameter_groups(pta):
    ndim = len(pta.param_names)
    groups  = [range(0, ndim)]
    params = pta.param_names
    
    # get global and individual parameters
    gpars, ipars = get_global_parameters(pta)
    if any(gpars):
        groups.extend([[params.index(gp) for gp in gpars]])

    for sc in pta._signalcollections:
        for signal in sc._signals:
            ind = [params.index(p) for p in signal.param_names if p not in gpars]
            if ind:
                groups.extend([ind])
    
    return groups
