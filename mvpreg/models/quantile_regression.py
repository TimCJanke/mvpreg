"""
@author: Tim Janke, Energy Information Networks & Systems @ TU Darmstadt
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .base_model import MarginsAndCopulaModel
from .losses import QuantileLoss
from .layers import ClipValues
from .helpers import qr_interpolator



################ MVPReg Models with Quantile Regression ################
class DeepQuantileRegression(MarginsAndCopulaModel):
    def __init__(self, 
                 taus=[0.1, 0.25, 0.5, 0.75, 0.9], 
                 quantile_interpolation = "cubic",
                 tail_extrapolation="linear",
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.taus = np.array(taus)
        self.quantile_interpolation = quantile_interpolation # liner or cubic
        self.tail_extrapolation = tail_extrapolation # how to deal with the tails of the distributions outside of estimated quantiles

        self.model = self._build_model()
        self.model.summary(expand_nested=True)

        
    def _build_model(self):
        
        x = tf.keras.Input(shape=(self.dim_in,), name="exogeneous_input")
        
        h = x
        for i in range(self.n_layers):
            h = layers.Dense(self.n_neurons, activation=self.activation)(h)  
        
        q = layers.Dense(self.dim_out*len(self.taus), activation=self.output_activation)(h)
        q = layers.Reshape((self.dim_out, len(self.taus)))(q)
        q = ClipValues(low=np.reshape(self.censored_left, (1,self.dim_out,1)), high=np.reshape(self.censored_right, (1,self.dim_out,1)))(q)

        mdl = tf.keras.models.Model(inputs=x, outputs=q)
        mdl.compile(loss=QuantileLoss(taus=self.taus), optimizer=self.optimizer)
        
        return mdl


    def predict_quantiles(self, x):
        q_pred = self.model.predict(self._scale_x(x))
        q_pred = np.sort(q_pred, axis=2) # ensure increasing quantile values
        return self._rescale_y_samples(q_pred)
    
    
    def predict_distributions(self, x):
        q = self.predict_quantiles(x)
        p_pred = self._get_distributions_from_quantiles(q)
        return p_pred
  
  
    def simulate(self, x, n_samples=1):
        if self.copula_type == "independence":
            return self.simulate_marginals(x, n_samples)
        else:
            p_pred = self.predict_distributions(x)
            y_pred = np.empty(shape=(p_pred.shape[0], p_pred.shape[1], n_samples))
            for i in range(p_pred.shape[0]):
                u = self.simulate_copula(n_samples) # simulate from copula (not conditional on x) in unit space
                for d in range(p_pred.shape[1]):
                    y_pred[i,d,:] = p_pred[i,d].ppf(u[:,d]) # use conditional inverse CDFs to obtain samples in data space
            return y_pred


    def simulate_marginals(self, x, n_samples=1):
        p_pred = self.predict_distributions(x)
        y_pred = np.empty(shape=(p_pred.shape[0], p_pred.shape[1], n_samples))
        for i in range(p_pred.shape[0]):
            for d in range(p_pred.shape[1]):
                y_pred[i,d,:] = p_pred[i,d].sample(n_samples)
        return y_pred

    
    def cdf(self, x, y):
        pobs = np.zeros_like(y)
        p_pred = self.predict_distributions(x)
        #TODO: parallel for loop with joblib?
        for i in range(p_pred.shape[0]):
            for d in range(p_pred.shape[1]):
                pobs[i,d] = p_pred[i,d].cdf(y[i,d])
        return pobs


    def ppf(self, x, tau):
        p_pred = self.predict_distributions(x)
        
        dim = p_pred.shape
        taus = np.squeeze(tau)
        if len(taus.shape)==0:
            taus = np.zeros(dim) + taus
        elif len(taus.shape)==1:
            taus = np.zeros(dim) + np.reshape(taus, (1,-1))
        elif len(taus.shape)==2:
            taus = tau
        else:
            raise ValueError(f"taus must be float, 1D, or 2D array but has dimension {len(tau)}.")
        
        y_pred = np.empty(shape=(p_pred.shape[0], p_pred.shape[1]))
        for i in range(p_pred.shape[0]):
            for d in range(p_pred.shape[1]):
                y_pred[i,d] = p_pred[i,d].ppf(taus[i,d]) # use conditional inverse CDFs to obtain samples in data space
        
        return y_pred
        
    
    
    def _get_distributions_from_quantiles(self, q):
        p_pred = np.empty(shape=(q.shape[0], q.shape[1]), dtype=object)
        #TODO: parallel for loop with joblib?
        for i in range(q.shape[0]):
            for d in range(q.shape[1]):
                p_pred[i,d] = qr_interpolator(q=q[i,d,:], 
                                              taus=self.taus, 
                                              censored_left=self.censored_left[0,d], 
                                              censored_right=self.censored_right[0,d], 
                                              tail_extrapolation=self.tail_extrapolation,
                                              interpolation=self.quantile_interpolation) 
        return p_pred
