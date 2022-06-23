import tensorflow as tf
from tensorflow.keras import layers

### defining some layers ####
class FiLM(layers.Layer):
    def __init__(self, n_neurons, activation, **kwargs):
        super().__init__(**kwargs)
        self.n_neurons = n_neurons
        self.activation = activation
    
        self.conv = layers.Conv1D(filters=self.n_neurons, kernel_size=1, strides=1, data_format="channels_last")
        self.multiply = layers.Multiply()
        self.add = layers.Add()
        self.activation = layers.Activation(self.activation)
        
    def call(self, inputs):
        # inputs:
        # z:     (bs, n_samples, dim)
        # gamma: (bs, n_samples, n_neurons)
        # beta:  (bs, n_samples, n_neurons)
        
        z, gamma, beta = inputs
        z = self.conv(z) # (bs, n_samples, dim_in) --> (bs, n_samples, n_neurons)
        z = self.multiply([z,gamma]) # (bs, n_samples, n_neurons) --> (bs, n_samples, n_neurons)
        z = self.add([z,beta]) # (bs, n_samples, n_neurons) --> (bs, n_samples, n_neurons)
        
        return self.activation(z)


    # do we need this?
    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0][0:2], self.n_neurons)
    
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "n_neurons": self.n_neurons,
                "activation": self.activation}



class GaussianSampling(layers.Layer):
    """ produces isotropic Gaussian noise conditional on the input"""

    def __init__(self, dim_laten, n_samples, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        mus, sigmas = inputs[0], inputs[1]
        batch_size = tf.shape(mus)[0]
        dim = tf.shape(mus)[1:]
        epsilons = tf.keras.backend.random_normal(shape=(batch_size, *dim))
        return mus + sigmas*epsilons

    # do we need this?
    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0])


class UnconditionalGaussianSampling(layers.Layer):
    """ produces unconditional standard Gaussian noise, still needs an input to determine batchsize"""

    def __init__(self, dim_latent, **kwargs):
        super().__init__(**kwargs)
        self.dim_latent = dim_latent # ensure that this is a list we can unpack in call
    
    def call(self, inputs):
        # inputs is a tensor of shape (bs, n_features, n_samples)
        batch_size = tf.shape(inputs)[0] # inputs is a single tensor
        n_samples = tf.shape(inputs)[2] 
        return tf.keras.backend.random_normal(shape=(batch_size, self.dim_latent, n_samples))

    # do we need this:
    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0], self.dim_latent, input_shape[2])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "dim_latent": self.dim_latent}
    

class ClipValues(layers.Layer):
    def __init__(self, low, high, **kwargs):
        super().__init__(**kwargs)
        self.low = low
        self.high = high

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.low, self.high)
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "low": self.low,
                "high": self.high}