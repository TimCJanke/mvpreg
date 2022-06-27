"""
@author: Tim Janke, Energy Information Networks & Systems Lab @ TU Darmstadt, Germany

A collection of useful functions for probabilistic forecasting.

classes:     
----------
    qr_interpolator: Create smooth functions for the CDF and PPF from a grid of quantiles


functions:
----------
    sample_gaussiancopula: Sample from multivariate distribution for given marginals and
                            covariance matrix using a Gaussian copula

"""

import numpy as np
import scipy
from scipy.interpolate import PchipInterpolator, interp1d


class qr_interpolator(object):
    """
    Interpolation for non-parametric 1-D distribution p(x) represented by quantiles as obtained e.g. by quantile regression.
    Uses scipy's PchipInterpolator to define CDF and PPF from given quantiles.
    Values in q and taus will be automatically sorted in ascending order.
    The interpolation can also deal with censored data in the sense that if one or several quantile predictions are equal to x_min or x_max,
    they will be modelled as point masses at x_min and x_max and only quantile prediction larger/smaller than x_min/x_max will be 
    modelled by a continuous CDF.

     Parameters
     ----------
     q : array, shape (N,)
         A 1-D array of unique and monotonically increasing real values representing the quantile values.
     taus : array, shape (N,)
         A 1-D array of corresponding unique and monotonically increasing values in (0,1) representing the quantile levels.
     x_min : float
         The smallest possible real value.
     x_max : float
         The largest possible real value.

     Methods
     -------
     ppf(tau):
         Interpolated PPF.
     cdf(x):
        Interpolated CDF. 
    sample(size):
        Sample from distribution.
    """    
    def __init__(self, q, taus, x_min=None, x_max=None, eps=1e-6, tail_extrapolation="linear", interpolation="cubic"):
        self.q = np.sort(q)
        self.taus = np.sort(taus)
        self.eps = eps
        self.tail_extrapolation = tail_extrapolation
        self.interpolation = interpolation
        
        #TODO: handle censoring and x_min and x_max seperately: x_min -> q_min, i.e. the smallest predicted quantile, censored could even smaller
        # atm if x_min is specified, we will always linearly/cubically interpolate till x_min, even if "cutoff" is specified! 
        # desired would be to have x_min=censored_left that handles the point masses and have a q_0/q_min parameter that handles the last quantile
        
        # linear interpolation of minimum and maximum values
        if x_min is None:
            if self.tail_extrapolation == "linear":
                self.x_min = q[1] - ((q[1]-q[0]) / (taus[1]-taus[0])) * (taus[1]-0.0) # q_min = q_1 - m*(tau_1-tau_0)
            elif self.tail_extrapolation == "cutoff":
                self.x_min = q[0] - 2*self.eps
            else:
                raise ValueError("Unknown tail interpolation type.")
        else:
            self.x_min = x_min
        
        if x_max is None:
            if self.tail_extrapolation == "linear":
                self.x_max = q[-1] + ((q[-1]-q[-2]) / (taus[-1]-taus[-2])) * (1.0-taus[-1]) # q_max = q_99 - m*(tau_max-tau_99)
            elif self.tail_extrapolation == "cutoff":
                self.x_max = q[-1] + 2*self.eps
            else:
                raise ValueError("Unknown tail interpolation type.")
        else:
            self.x_max = x_max


        
        mask_low = self.q<=self.x_min # True where q is equal or samller than lower bound
        if any(mask_low):
            self.tau_x_min = self.taus[mask_low][-1]
        else:
            self.tau_x_min = 0.0


        mask_high = self.q>=self.x_max # True where q is equal or larger than upper bound
        if any(mask_high):
            self.tau_x_max = self.taus[mask_high][0]
        else:
            self.tau_x_max = 1.0

        # we only want to interpolate between non-censored values
        self.q_ = self.q[np.logical_and(np.invert(mask_low), np.invert(mask_high))]
        self.q_ = np.concatenate(([self.x_min], self.q_, [self.x_max]))
        
        # for scipy interpolation we have to ensure strictly montone values
        self.q_ = self._ensure_strictly_monotone(self.q_)
        
        # prepare corresponding taus
        self.taus_ = self.taus[np.logical_and(np.invert(mask_low), np.invert(mask_high))]
        self.taus_ = np.concatenate(([self.tau_x_min], self.taus_, [self.tau_x_max]))
        
        
        # create interpolated CDF and inverse CDF using scipy cubic monotne interpolation 
        self.ppf_fun = self._interpolate_ppf()
        self.cdf_fun = self._interpolate_cdf()

    def _ensure_strictly_monotone(self, q):
        # ensure strictly monotone function by recursively adding epsilon until all values in q are increasing
        mask_non_increasing = np.diff(q, prepend=self.x_min-self.eps)<=0
        while any(mask_non_increasing):
            q[mask_non_increasing] = q[mask_non_increasing] + self.eps
            mask_non_increasing = np.diff(q, prepend=self.x_min-self.eps)<=0
        return q


    def _interpolate_ppf(self):
        # interpolate quantile function based on taus_ and q_
        
        if self.interpolation == "cubic":
            return PchipInterpolator(x=self.taus_, y=self.q_, extrapolate=False)
        
        elif self.interpolation == "linear":
            return interp1d(x=self.taus_, y=self.q_, bounds_error=False, fill_value=np.nan)     


    def _interpolate_cdf(self, n_grid_divides=2):
        # interpolate CDF by numerically inverting interpolated cdf
        tau_grid = self.taus_
        for i in range(n_grid_divides):
            tau_grid = np.ravel(np.column_stack((tau_grid, tau_grid+np.diff(tau_grid, append=tau_grid[-1])/2)))[0:-1]

        q_grid = self._ensure_strictly_monotone(self.ppf(tau_grid))
        
        if self.interpolation == "cubic":
            return PchipInterpolator(x=q_grid, y=tau_grid, extrapolate=False)
        
        elif self.interpolation == "linear":
            return interp1d(x=q_grid, y=tau_grid, bounds_error=False, fill_value=np.nan)
    

    def ppf(self, u):
        """
        Interpolated PPF aka inverse CDF aka Quantile Function

        Parameters
        ----------
        u : array of floats bewtween (0,1)
            Query point for PPF.

        Returns
        -------
        array of same shape as u
            Values of the PPF at points in q.

        """
        x = self.ppf_fun(np.atleast_1d(u)) # u values out of interpolation range will be NaNs
        x[np.where(u <= self.tau_x_min)] = self.x_min
        x[np.where(u >= self.tau_x_max)] = self.x_max

        # if we pass a float or 0-D array we get a float back
        if x.size == 1:
            x = x.item()

        return x
        

    def cdf(self, x):
        """
        Interpolated CDF.

        Parameters
        ----------
        x : array of floats
            Query point for CDF.

        Returns
        -------
        array of same shape as x
            Values of the CDF at points in x.

        """
        
        cdf_values = self.cdf_fun(np.atleast_1d(x)) # else this returns a zero dim array if x is scalar
        idx_x_min = np.nonzero(x <= self.x_min)[0]
        idx_x_max = np.nonzero(x >= self.x_max)[0]
        if idx_x_min.size != 0:
            cdf_values[idx_x_min] = np.random.uniform(low=0.0,
                                                      high=self.tau_x_min, 
                                                      size=len(idx_x_min)
                                                      )
        if idx_x_max.size != 0:
            cdf_values[idx_x_max] = np.random.uniform(low=self.tau_x_max, 
                                                      high=1.0, 
                                                      size=len(idx_x_max)
                                                      )
        
        # if we pass a float or 0-D array we get a float back
        if cdf_values.size == 1:
            cdf_values = cdf_values.item()
        
        #return np.clip(cdf_values, 1e-8, 1.0-1e-8)
        return cdf_values
    

    def sample(self, size=1):
        """
        Draw random samples from distribution.

        Parameters
        ----------
        size : int or tuple of ints
            Defines number and shape of returned samples.

        Returns
        -------
        float or array
            Random samples from sitribution.
        
        """
        return self.ppf(np.random.uniform(low=0.0, high=1.0, size=size))




def rank_data_random_tiebreaker(a):
    """ Ranks data in 1d array using ordinal ranking (i.e. all ranks are assigned) and breaks ties at random """
    
    idx = np.random.permutation(len(a)) # indexes for random shuffle
    
    ranks_shuffled = scipy.stats.rankdata(a[idx], method="ordinal") # compute ranks of shuffled index
    
    idx_r = np.zeros_like(idx) # indexes to invert permutation
    idx_r[idx] = np.arange(len(idx))
    
    ranks = ranks_shuffled[idx_r] # restore original order
    
    return ranks
    