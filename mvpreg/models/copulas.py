import numpy as np
import scipy
import sys
import warnings
import pyvinecopulib as pv


class IndependenceCopula(object):
    
    def __init__(self, dim):
        self.dim = dim

    def fit(self, u):
        return self

    def cdf(self, u):
        # evaluate cdf at u
        return np.prod(u, axis=1)

    def simulate(self, n=1):
        # simulate n samples
        return np.random.uniform(0.0,1.0,size=(n, self.dim))
    
    
    
class SchaakeShuffle(object):

    def __init__(self):
        pass

    def fit(self, u):
        # u is a (N,D) vector of pseudo observations, i.e. u~U(0,1)^D
        self.u = u
        return self

    def simulate(self, n=1):
        # simulate n samples
        return np.random.permutation(self.u)[0:n,:]



class GaussianCopula(object): 
    def __init__(self):
        self.epsilon = 1e-8

    def _get_sigma(self, y):
        # compute nomalized covariance matrix
        sigma = np.corrcoef(y, rowvar=False)
        sigma = np.nan_to_num(sigma, nan=0.0)
        # If singular, add some noise to the diagonal
        if np.linalg.cond(sigma) > 1.0 / sys.float_info.epsilon:
            sigma = sigma + np.identity(sigma.shape[0]) * self.epsilon
            warnings.warn("singular covariance matrix, added noise to diagonal")
        
        #TODO: normalize cov matrix s.t. diagonal is 1.0?
        return sigma

    def fit(self, u):
        if np.amin(u) <=0.0 or np.amax(u)>=1.0:
            warnings.warn("pseudo observations contain values smaller/larger than 0.0/1.0. Values will be clipped.")
        # distribution
        self.sigma = self._get_sigma(scipy.stats.norm.ppf(np.clip(u, a_min=self.epsilon, a_max=1.0-self.epsilon)))
        self.mvn = scipy.stats.multivariate_normal(mean=None, cov=self.sigma)
        return self

    def simulate(self, n=1):
        # simulate n samples
        return np.reshape(scipy.stats.norm.cdf(self.mvn.rvs(n)), (n,-1)) # reshape to ensure 2D array also for n=1 


class VineCopula(object): 
    def __init__(self, pair_copula_families="nonparametric", vine_structure=None, vine_type="r-vine"):
        
        if pair_copula_families=="all":
            family_set = pv.all
        
        elif pair_copula_families=="nonparametric":
            family_set = [pv.BicopFamily.indep, pv.BicopFamily.tll]
        
        elif pair_copula_families=="parametric":
            family_set = [pv.BicopFamily.indep,
                          pv.BicopFamily.gaussian, 
                          pv.BicopFamily.student, 
                          pv.BicopFamily.clayton, 
                          pv.BicopFamily.gumbel, 
                          pv.BicopFamily.frank, 
                          pv.BicopFamily.joe, 
                          pv.BicopFamily.bb1, 
                          pv.BicopFamily.bb6, 
                          pv.BicopFamily.bb7, 
                          pv.BicopFamily.bb8]                
        
        self.controls = pv.FitControlsVinecop(family_set=family_set)
        self.vine_structure = vine_structure
        self.vine_type = vine_type


    def fit(self, u):
        if self.vine_type == "c-vine":
            self.copula = pv.Vinecop(data=u, structure=pv.CVineStructure(order=self.vine_structure), controls=self.controls)
        
        elif self.vine_type == "d-vine":
            self.copula = pv.Vinecop(data=u, structure=pv.DVineStructure(order=self.vine_structure), controls=self.controls)
        
        elif self.vine_type == "r-vine":
            if self.vine_structure is None:
                self.copula = pv.Vinecop(data=u, controls=self.controls)
            else:
                self.copula = pv.Vinecop(data=u, controls=self.controls, structure=pv.RVineStructure(order=self.vine_structure))
        
        print("\nVine copula fit:\n")
        print(self.copula.str())
        
        return self


    def simulate(self, n=1):
        # simulate n samples
        return self.copula.simulate(n)





