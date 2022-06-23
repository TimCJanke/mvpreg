"""
@author: Tim Janke, Energy Information Networks & Systems @ TU Darmstadt
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd

from .base_model import MVPRegModel
from .losses import EnergyScore, VariogramScore
from .layers import FiLM, ClipValues, UnconditionalGaussianSampling

### MVPRegModel child class for generative models
class DeepGenerativeRegression(MVPRegModel):
    
    def __init__(self,
                 dim_latent = None,
                 conditioning="concatenate", 
                 FiLM_modulation = "constant",
                 n_layers_encoder=0, 
                 n_neurons_encoder=100, 
                 activation_encoder="relu", 
                 n_samples_train=10,
                 n_samples_val = None, 
                 loss = "ES",
                 p_vs = 0.5,
                 **kwargs):

        super().__init__(**kwargs)

        # additional forward model hyperparameters
        if dim_latent is None:
            self.dim_latent = self.dim_out
        else:
            self.dim_latent = dim_latent
        
        # conditioning of x and noise
        self.conditioning = conditioning
        self.FiLM_modulation = FiLM_modulation
        self.n_layers_encoder = n_layers_encoder
        self.n_neurons_encoder = n_neurons_encoder
        self.activation_encoder = activation_encoder
        
        # training and evaluation
        self.n_samples_train = n_samples_train
        if n_samples_val is None:
            self.n_samples_val = n_samples_train
        else:
            self.n_samples_val = n_samples_val
        
        if loss == "ES":
            self.loss = EnergyScore()
        elif loss == "VS":
            self.p_vs = p_vs
            self.loss = VariogramScore(p=self.p_vs)
        else:
            raise ValueError("Unknown loss function. 'loss' must be ES or VS.")
        
        self._expand_y_dim = True
        
        self.model = self._build_model()
        self.model.summary(expand_nested=True)

    def _build_model(self):
        model = ConditionalImplicitGenerativeModel(dim_latent = self.dim_latent,
                                                   n_layers=self.n_layers, 
                                                   n_neurons=self.n_neurons, 
                                                   activation=self.activation,
                                                   conditioning=self.conditioning,
                                                   output_activation=self.output_activation, 
                                                   censored_left=np.reshape(self.censored_left, (1,self.dim_out,1)),
                                                   censored_right=np.reshape(self.censored_right, (1,self.dim_out,1)), 
                                                   dim_out=self.dim_out, 
                                                   n_samples_train=self.n_samples_train)

        _ = model(np.random.normal(size=(1, self.dim_in, 1))) # need to do one forward pass to build model graph
        #model.summary(expand_nested=True) # now we can call summary
        model.compile(loss=self.loss, optimizer=self.optimizer)
        
        return model


    def simulate(self, x, n_samples=1, max_array_size_mb=256.0):
        """ draw n_samples randomly from conditional distribution p(y|x)"""
        
        size = x.nbytes*n_samples/1e6
        
        if size > max_array_size_mb:
            x_ = np.array_split(x, np.ceil(size/max_array_size_mb).astype(int))
            y = []
            for x_i in x_:
                y.append(self.model(np.repeat(np.expand_dims(self._scale_x(x_i), axis=2), repeats=n_samples, axis=2)))
            y = np.concatenate(y, axis=0)
        
        else:
            y = self.model(np.repeat(np.expand_dims(self._scale_x(x), axis=2), repeats=n_samples, axis=2))

        return self._rescale_y_samples(y)


### defining the custom models ####
class ConditionalImplicitGenerativeModel(tf.keras.Model):
    def __init__(self, 
                dim_out, 
                n_layers,
                n_neurons,
                activation,
                conditioning="concatenate",
                dim_latent=None,
                n_samples_train=10, 
                output_activation="linear", 
                censored_left=-np.inf, 
                censored_right=np.inf, 
                n_samples_val=None, 
                **kwargs):
        super().__init__(**kwargs)

        # hyperparams
        self.n_samples_train = n_samples_train
        if n_samples_val is None:
            self.n_samples_val = n_samples_train
        else:
            self.n_samples_val = n_samples_val

        self.dim_out = dim_out
        self.n_layers= n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.conditioning = conditioning
        if dim_latent is None:
            self.dim_latent = dim_out
        else:
            self.dim_latent = dim_latent
        self.output_activation = output_activation

        self.censored_left = censored_left
        self.censored_right = censored_right
        
        # init layers
        if self.conditioning == "concatenate":
            self.generator = ConcatGenerator(dim_latent = self.dim_latent,
                                              n_layers=self.n_layers, 
                                              n_neurons=self.n_neurons, 
                                              activation=self.activation)
            
        elif self.conditioning == "FiLM":
            self.generator = FiLMGenerator(dim_latent = self.dim_latent,
                                            n_layers=self.n_layers, 
                                            n_neurons=self.n_neurons, 
                                            activation=self.activation)
        
        self.output_layer = layers.Conv1D(filters=self.dim_out, kernel_size=1, strides=1, activation=self.output_activation)
        self.permute = layers.Permute((2,1))
        self.clip = ClipValues(low=self.censored_left, high=self.censored_right)


    def call(self, x):
        # 'x' is a tensor
        h = self.generator(x)
        y = self.output_layer(h)
        y = self.permute(y)
        y = self.clip(y)
        return y


    def train_step(self, data):
        x_train, y_train = data
        
        with tf.GradientTape() as tape:
            y_predict = self(tf.repeat(tf.expand_dims(x_train, axis=2), repeats=self.n_samples_train, axis=2), training=True)
            loss = self.compiled_loss(y_train, y_predict)
        grads = tape.gradient(loss, self.generator.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        self.compiled_metrics.update_state(y_train, y_predict)
        return {m.name: m.result() for m in self.metrics}

    
    def test_step(self, data):
        x_test, y_test = data
        y_predict = self(tf.repeat(tf.expand_dims(x_test, axis=2), repeats=self.n_samples_val, axis=2), training=False)
        self.compiled_loss(y_test, y_predict)
        self.compiled_metrics.update_state(y_test, y_predict)
        return {m.name: m.result() for m in self.metrics}
    



### defining different generator architectures ###
class GeneratorModel(tf.keras.Model):
    def __init__(self, dim_latent, n_layers, n_neurons, activation, **kwargs):
        super().__init__(**kwargs)
        
        self.dim_latent = dim_latent
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation



class ConcatGenerator(GeneratorModel):
    def __init__(self, dim_latent, n_layers, n_neurons, activation, **kwargs):
        super().__init__(dim_latent, n_layers, n_neurons, activation, **kwargs)
        
        # x has input dim (bs, n_features, n_samples)
        self.sample_noise = UnconditionalGaussianSampling(dim_latent=self.dim_latent)
        self.concat_x_and_noise = layers.Concatenate(axis=1)
        self.permute = layers.Permute((2,1))
        
        self.dense_layers = []
        for i in range(self.n_layers):
            self.dense_layers.append(layers.Conv1D(filters=self.n_neurons, kernel_size=1, strides=1, activation=self.activation))
        
        
    def call(self, inputs):
        #unpack and prepare inputs
        x = inputs # (bs, n_features, n_samples_train)        
        z = self.sample_noise(x) # (bs, dim_latent, n_samples_train)

        # forward pass
        h = self.concat_x_and_noise([x, z]) # -> (bs, n_features+dim_latent , n_samples_train)
        h = self.permute(h) # -> (bs, n_samples_train, n_features+dim_latent)
        
        for layer_i in self.dense_layers:
            h = layer_i(h)
        return h



class FiLMGenerator(GeneratorModel):
    def __init__(self, dim_latent, n_layers, n_neurons, activation,  modulation="per_layer", **kwargs):
        super().__init__(dim_latent, n_layers, n_neurons, activation, **kwargs)

        self.modulation = modulation

        # init layers
        self.sample_noise = UnconditionalGaussianSampling(dim_latent=self.dim_latent)
        self.permute_layer = layers.Permute((2,1))
        
        if self.modulation == "per_layer":
            self.gamma_layers = []
            self.beta_layers = []
            for i in range(self.n_layers):
                self.gamma_layers.append(layers.Conv1D(filters=self.n_neurons, kernel_size=1, strides=1,  data_format="channels_last"))
                self.beta_layers.append(layers.Conv1D(filters=self.n_neurons, kernel_size=1, strides=1,  data_format="channels_last"))
        
        elif self.modulation == "constant":
            self.gamma_layer = layers.Conv1D(filters=self.n_neurons, kernel_size=1, strides=1,  data_format="channels_last", name="film_gamma")
            self.beta_layer = layers.Conv1D(filters=self.n_neurons, kernel_size=1, strides=1,  data_format="channels_last", name="film_beta")
        
        else:
            raise ValueError("Unknown modulation.")
        
        self.FiLM_layers = []
        for i in range(self.n_layers):
            self.FiLM_layers.append(FiLM(n_neurons=self.n_neurons, activation=self.activation))


    def call(self, inputs):
        # inputs
        x = inputs # (bs, n_features, n_samples_train)
        z = self.sample_noise(x) # (bs, dim_latent, n_samples_train)
        x = self.permute_layer(x) # (bs, n_features, n_samples) --> (bs, n_samples, n_features)
        z = self.permute_layer(z) # (bs, dim_latent, n_samples) --> (bs, n_samples, dim_latent)
        
        # compute gamma and beta for constant modulation
        if self.modulation == "constant":
            gamma = self.gamma_layer(z) # (bs, n_samples, dim_latent) --> (bs, n_samples, n_neurons)
            beta = self.beta_layer(z) # (bs, n_samples, dim_latent) --> (bs, n_samples, n_neurons)
        
        # forward pass
        h = x
        for i in range(len(self.FiLM_layers)):
            if self.modulation == "per_layer":
                gamma = self.gamma_layers[i](z)
                beta = self.beta_layers[i](z)
            h = self.FiLM_layers[i]([h, gamma, beta]) # (bs, n_samples, n_neurons)x3 --> (bs, n_samples, n_neurons)
        return h

# class EncDecGenerator(layers.Layer):
# class Discriminator(layers.Layer):



