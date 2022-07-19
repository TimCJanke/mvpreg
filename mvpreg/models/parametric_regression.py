"""
@author: Tim Janke, Energy Information Networks & Systems @ TU Darmstadt
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers

from .base_model import MarginsAndCopulaModel
from .losses import LogScore


################ Class for MVPReg Models with univariate parametric output distributions ################
class DeepParametricRegression(MarginsAndCopulaModel):
    def __init__(self,
                 distribution="Normal",
                 link_function_param_1 = "linear",
                 link_function_param_2 = "softplus",
                 link_function_param_3 = "softplus",
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.link_function_param_1 = link_function_param_1
        self.link_function_param_2 = link_function_param_2
        self.link_function_param_3 = link_function_param_3
        self.distribution = distribution
        
        if self.distribution in ("Normal", "LogitNormal", "TruncatedNormal"):
            self.n_params_distribution = 2
        elif self.distribution in ("StudentT"):
            self.n_params_distribution = 3
        
        self.model = self._build_model()
        if self.show_model_summary:
            self.model.summary(expand_nested=True)

    
    def _build_model(self):
        
            x = tf.keras.Input(shape=(self.dim_in,), name="exogeneous_input")
            h = x

            for i in range(self.n_layers):
                h = layers.Dense(self.n_neurons, activation=self.activation)(h)  
            
            params_1 = layers.Dense(self.dim_out, activation=self.link_function_param_1)(h)
            params_2 = layers.Dense(self.dim_out, activation=self.link_function_param_2)(h)
            params_2 = layers.Lambda(lambda x: x + 0.05)(params_2)
            
            
            if self.n_params_distribution == 2:
                params_predict = [params_1, params_2]
                
            elif self.n_params_distribution == 3:
                params_3 = layers.Dense(self.dim_out, activation=self.link_function_param_3)(h)
                params_predict = [params_1, params_2, params_3]
            
            # workaround for being able to inspect predicted parameters
            params_predict = layers.Concatenate(axis=-1, name="params_predict")(params_predict) # list to tensor
            params_predict = layers.Lambda(lambda arg: tf.unstack(arg, num=self.n_params_distribution, axis=-1))(params_predict) # tensor to list

            if self.distribution == "Normal":
                dist_pred = tfp.layers.DistributionLambda(lambda params: tfp.distributions.Normal(loc=params[0], scale=params[1]))(params_predict)
            
            if self.distribution == "StudentT":
                dist_pred = tfp.layers.DistributionLambda(lambda params: tfp.distributions.StudentT(loc=params[0], scale=params[1], df=params[2]+2.01))(params_predict)

            if self.distribution == "LogitNormal":
                dist_pred = tfp.layers.DistributionLambda(lambda params: tfp.distributions.LogitNormal(loc=params[0], scale=params[1]))(params_predict)
            
            if self.distribution == "TruncatedNormal":
                dist_pred = tfp.layers.DistributionLambda(lambda params: tfp.distributions.TruncatedNormal(loc=params[0], 
                                                                                                           scale=params[1], 
                                                                                                           low=self.censored_left.astype(np.float32), 
                                                                                                           high=self.censored_right.astype(np.float32)))(params_predict)
                
            mdl = tf.keras.models.Model(inputs=x, outputs=dist_pred)
            mdl.compile(loss=LogScore(), optimizer=self.optimizer)
            
            return mdl


    def predict_distributions(self, x):
        return self.model(self._scale_x(x))

    
    def simulate(self, x, n_samples=1):
        if self.copula_type == "independence":
            return self.simulate_marginals(x, n_samples)        
        else:
            p_pred = self.predict_distributions(x)
            y_pred = []
            for i in range(n_samples):
                u = self.simulate_copula(n_samples=p_pred.shape[0]) # possible because copulas is unconditional on x 
                y_pred.append(p_pred.quantile(u))            
            return self._rescale_y_samples(np.stack(y_pred, axis=2))
        

    def simulate_marginals(self, x, n_samples=1):
        dist_pred = self.predict_distributions(x)
        y_pred = np.transpose(dist_pred.sample(n_samples).numpy(), (1,2,0)) 
        return self._rescale_y_samples(y_pred)


    def predict_params(self, x):
        try:
            params = self.params_model.predict(self._scale_x(x))
        except AttributeError:
            self.params_model = self._get_params_model() 
            params = self.params_model.predict(self._scale_x(x))
        return params

    def _get_params_model(self):
        return tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer("params_predict").output)


    def cdf(self, x, y):
        p_pred = self.predict_distributions(x)
        return p_pred.cdf(self._scale_y(y)).numpy()
    
    
    def ppf(self, x, tau):
        #TODO: unstable and not implemented for StudentT --> use empirical quantiles?
        dist_pred = self.predict_distributions(x)
        return self._rescale_y_samples(dist_pred.quantile(tau).numpy())
        # q = []
        # for tau in taus:
        #     q.append(np.expand_dims(dist_pred.quantile(tau).numpy(), axis=2))
        # return self._rescale_y_samples(np.concatenate(q, axis=2))


    def pdf(self, x, y):
        p_pred = self.predict_distributions(x)
        return p_pred.pdf(self._scale_y(y)).numpy()