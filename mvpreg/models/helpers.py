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
    
    def __init__(self, q, taus, censored_left=-np.inf, censored_right=np.inf, eps=1e-6, tail_extrapolation="cutoff", interpolation="cubic"):
        self.q = np.sort(q)
        self.taus = np.sort(taus)
        self.eps = eps
        self.tail_extrapolation = tail_extrapolation
        self.interpolation = interpolation
        
        self.censored_left = censored_left
        self.censored_right = censored_right

        if any(q < self.censored_left):
            print(f"Supplied quantile values in 'q' contain values that are smaller than 'censored_left'({censored_left}).")

        if any(q > self.censored_right):
            print(f"Supplied quantile values in 'q' contain values that are larger than 'censored_right'({censored_right}).")

        # extrapolation of minimum and maximum quantile values
        if self.tail_extrapolation == "linear":
            self.q_min = q[1] - np.maximum(((q[1]-q[0]) / (taus[1]-taus[0])) * (taus[1]-0.0), 2*self.eps) # q_min = q_1 - m*(tau_1-tau_0)
            self.q_max = q[-1] + np.maximum(((q[-1]-q[-2]) / (taus[-1]-taus[-2])) * (1.0-taus[-1]), 2*self.eps) # q_max = q_99 - m*(tau_max-tau_99)
        elif self.tail_extrapolation == "cutoff":
            self.q_min = q[0] - 2*self.eps
            self.q_max = q[-1] + 2*self.eps
        else:
            raise ValueError("Unknown tail interpolation type.")

        # if censored set to censored values
        if self.q_min <= self.censored_left:
            self.q_min = self.censored_left

        if self.q_max >= self.censored_right:
            self.q_max = self.censored_right


        mask_low = self.q <= self.censored_left # True where q is equal or smaller than lower bound
        if any(mask_low):
            self.tau_q_min = self.taus[mask_low][-1] # get the largest tau that corresponds to a censored quantile value
        else:
            self.tau_q_min = 0.0

        mask_high = self.q>=self.censored_right # True where q is equal or larger than upper bound
        if any(mask_high):
            self.tau_q_max = self.taus[mask_high][0] # get the smallest tau that corresponds to a censored quantile value
        else:
            self.tau_q_max = 1.0

        # we only want to interpolate between non-censored values
        self.q_ = self.q[np.logical_and(np.invert(mask_low), np.invert(mask_high))] # all the q that are not below or above censored values
        self.q_ = np.concatenate(([self.q_min], self.q_, [self.q_max])) # q_min and q_max are either extrapolated or equal to bounds
        
        
        # for scipy interpolation we have to ensure strictly montone values
        self.q_ = self._ensure_strictly_monotone(self.q_)
        
        # prepare corresponding taus
        self.taus_ = self.taus[np.logical_and(np.invert(mask_low), np.invert(mask_high))]
        self.taus_ = np.concatenate(([self.tau_q_min], self.taus_, [self.tau_q_max]))
        
        
        # create interpolated CDF and inverse CDF using scipy cubic monotone or linear interpolation 
        self.ppf_fun = self._interpolate_ppf()
        self.cdf_fun = self._interpolate_cdf()

    def _ensure_strictly_monotone(self, q):
        # ensure strictly monotone function by recursively adding epsilon until all values in q are increasing
        mask_non_increasing = np.diff(q, prepend=self.censored_left-self.eps)<=0
        while any(mask_non_increasing):
            q[mask_non_increasing] = q[mask_non_increasing] + self.eps
            mask_non_increasing = np.diff(q, prepend=self.censored_left-self.eps)<=0
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
            Values of the PPF at points in u.

        """
        if any(0.0 > np.atleast_1d(u)) or any(np.atleast_1d(u) > 1.0):
            raise ValueError("u has to be in (0,1)")
        
        x = self.ppf_fun(np.atleast_1d(u)) # u values out of interpolation range will lead to NaNs
        x[np.where(u <= self.tau_q_min)] = self.q_min
        x[np.where(u >= self.tau_q_max)] = self.q_max

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

        # cdf outside smallest q is 0
        idx_x_min = np.nonzero(x <= self.q_min)[0]
        if idx_x_min.size != 0:
            cdf_values[idx_x_min] = 0.0

        # cdf outside largest q is 1
        idx_x_max = np.nonzero(x >= self.q_max)[0]
        if idx_x_max.size != 0:
            cdf_values[idx_x_max] = 1.0


        # in cases where censoring exists:
        # ensure that the returned taus in the left censored point mass case are random between [0,tau_q_min]
        if self.q_min == self.censored_left and self.tau_q_min > 0.0:
            idx_x_lc = np.nonzero(x == self.censored_left)[0]
            if idx_x_lc.size != 0:
                cdf_values[idx_x_lc] = np.random.uniform(low=1e-8,
                                                        high=self.tau_q_min, 
                                                        size=len(idx_x_lc)
                                                        )

        # ensure that the returned taus in the right censored point mass case are random between [tau_q_max, 1] 
        if self.q_max == self.censored_right and self.tau_q_max < 1.0:
            idx_x_rc = np.nonzero(x == self.censored_right)[0]
            if idx_x_rc.size != 0:
                cdf_values[idx_x_rc] = np.random.uniform(low=self.tau_q_max, 
                                                        high=1.0-1e-8, 
                                                        size=len(idx_x_rc)
                                                        )
        

 
        
        # if we pass a float or 0-D array we get a float back
        if cdf_values.size == 1:
            cdf_values = cdf_values.item()
        
        
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
    