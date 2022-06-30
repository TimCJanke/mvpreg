"""
@author: Tim Janke, Energy Information Networks & Systems @ TU Darmstadt
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .base_model import MVPRegModel
from .losses import EnergyScore, VariogramScore, EnergyScoreMetric, VariogramScoreMetric
from .layers import FiLM, ClipValues, UnconditionalGaussianSampling

### MVPRegModel child class for generative models
class DeepGenerativeRegression(MVPRegModel):
    def __init__(self,
                 dim_latent = None,
                 conditioning = "concatenate", 
                 FiLM_modulation = "constant",
                 n_samples_val = 10, 
                 **kwargs):

        super().__init__(**kwargs)

        # additional forward model hyperparameters
        if dim_latent is None:
            self.dim_latent = self.dim_out
        else:
            self.dim_latent = dim_latent
        
        # generator conditioning on x and noise
        self.conditioning = conditioning
        self.FiLM_modulation = FiLM_modulation
        
        # training and evaluation
        self.n_samples_val = n_samples_val
        self._expand_y_dim = True
        
    
    def simulate(self, x, n_samples=1, max_array_size_mb=256.0):
        """ draw n_samples randomly from conditional distribution p(y|x)"""
        
        size = x.nbytes*n_samples/1e6
        
        if size > max_array_size_mb:
            x_ = np.array_split(x, np.ceil(size/max_array_size_mb).astype(int))
            y = []
            for x_i in x_:
                y.append(self.model(np.repeat(np.expand_dims(self._scale_x(x_i), axis=2), repeats=n_samples, axis=2)).numpy())
            y = np.concatenate(y, axis=0)
        
        else:
            y = self.model(np.repeat(np.expand_dims(self._scale_x(x), axis=2), repeats=n_samples, axis=2)).numpy()

        return self._rescale_y_samples(y)
             


class ScoringRuleDGR(DeepGenerativeRegression):
    def __init__(self,
                 n_samples_train=10,
                 loss = "ES",
                 p_vs = 0.5,
                 **kwargs):

        super().__init__(**kwargs)
        
        # training and evaluation
        self.n_samples_train = n_samples_train
        
        if loss == "ES":
            self.loss = EnergyScore()
        elif loss == "VS":
            self.p_vs = p_vs
            self.loss = VariogramScore(p=self.p_vs)
        else:
            raise ValueError("Unknown loss function. 'loss' must be ES or VS.")
        
        self.model = self._build_model()
        if self.show_model_summary:
            self.model.summary(expand_nested=True)


    def _build_model(self):
        model = ScoringRuleCIGM(dim_out=self.dim_out, 
                                dim_latent = self.dim_latent,
                                n_layers_generator=self.n_layers, 
                                n_neurons_generator=self.n_neurons, 
                                activation_generator=self.activation,
                                output_activation_generator=self.output_activation,
                                conditioning_generator=self.conditioning,
                                FiLM_modulation=self.FiLM_modulation,
                                censored_left=self.censored_left,
                                censored_right=self.censored_right, 
                                n_samples_train=self.n_samples_train,
                                n_samples_val=self.n_samples_val)

        _ = model(np.random.normal(size=(1, self.dim_in, 1))) # need to do one forward pass to build model graph
        #model.summary(expand_nested=True) # now we can call summary
        model.compile(loss=self.loss, optimizer=self.optimizer)
        
        return model


class AdversarialDGR(DeepGenerativeRegression):
    def __init__(self,
                 n_layers_discriminator=None, 
                 n_neurons_discriminator=None, 
                 activation_discriminator=None,
                 output_activation_discriminator="sigmoid",
                 optimizer_discriminator = "Adam",
                 label_smoothing = False,
                 generator_train_steps = 1,
                 discriminator_train_steps = 1,
                 n_samples_val=10,
                 validation_metric="ES",
                 p_vs_eval=0.5,
                 **kwargs):

        super().__init__(**kwargs)

        if n_layers_discriminator is None:
            self.n_layers_discriminator = self.n_layers
        else:
            self.n_layers_discriminator = n_layers_discriminator
        
        if n_neurons_discriminator is None:
            self.n_neurons_discriminator = self.n_neurons
        else:
            self.n_neurons_discriminator = n_neurons_discriminator            

        if activation_discriminator is None:
            self.activation_discriminator = self.activation
        else:
            self.activation_discriminator = activation_discriminator
        
        self.output_activation_discriminator = output_activation_discriminator     
        self.optimizer_discriminator = optimizer_discriminator

        # training loop
        self.generator_train_steps = generator_train_steps
        self.discriminator_train_steps = discriminator_train_steps
        self.label_smoothing = label_smoothing

        # evaluation
        self.n_samples_val = n_samples_val
        if validation_metric == "ES":
            self.validation_metric = EnergyScore()
        elif validation_metric == "VS":
            self.p_vs_eval = p_vs_eval
            self.validation_metric = VariogramScore(p=self.p_vs_eval)
        else:
            raise ValueError("Unknown validation metric function. 'validation_metric' must be 'ES' or 'VS'.")

        self.model = self._build_model()
        if self.show_model_summary:
            self.model.summary(expand_nested=True)

    def _build_model(self):
        model = ConditionalGAN(dim_out=self.dim_out, 
                                n_layers_generator=self.n_layers,
                                n_neurons_generator=self.n_neurons,
                                activation_generator=self.activation,
                                output_activation_generator=self.output_activation,
                                conditioning_generator=self.conditioning,
                                FiLM_modulation=self.FiLM_modulation,
                                dim_latent=self.dim_latent,
                                censored_left=self.censored_left,
                                censored_right=self.censored_right, 
                                n_layers_discriminator=self.n_layers_discriminator, 
                                n_neurons_discriminator=self.n_neurons_discriminator, 
                                activation_discriminator=self.activation_discriminator,
                                output_activation_discriminator=self.output_activation_discriminator,
                                generator_train_steps=self.generator_train_steps,
                                discriminator_train_steps=self.discriminator_train_steps,
                                n_samples_val=self.n_samples_val)

        _ = model(np.random.normal(size=(1, self.dim_in, 1))) # need to do one forward pass to build model graph
        _ = model.discriminator([np.random.normal(size=(1, self.dim_in)), np.random.normal(size=(1, self.dim_out))])
        
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=self.label_smoothing), 
                      g_optimizer=self.optimizer, 
                      d_optimizer=self.optimizer_discriminator,
                      val_loss_metric=EnergyScoreMetric())
        
        
        return model
        
        


### defining the core models ####
class ConditionalImplicitGenerativeModel(tf.keras.Model):
    def __init__(self, 
                dim_out,
                n_layers_generator,
                n_neurons_generator,
                activation_generator,
                output_activation_generator="linear", 
                conditioning_generator="concatenate",
                FiLM_modulation="constant",
                dim_latent=None,
                n_samples_train=10, 
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
        
        if dim_latent is None:
            dim_latent = dim_out

        self.generator_config = {"dim_out": dim_out,
                                 "dim_latent": dim_latent,
                                 "n_layers": n_layers_generator, 
                                 "n_neurons": n_neurons_generator, 
                                 "activation": activation_generator,
                                 "output_activation": output_activation_generator,
                                 "censored_left": censored_left,
                                 "censored_right": censored_right}
        
        # init generator model
        if conditioning_generator == "concatenate":
            self.generator = ConcatGenerator(**self.generator_config)
            
        elif conditioning_generator == "FiLM":
            self.generator_config["modulation"] = FiLM_modulation
            self.generator = FiLMGenerator(**self.generator_config)

    
    def call(self, x):
        # 'x' is a tensor of shape (batch_size, dim_in, n_samples)
        # 'y' is a tensor of shape (batch_size, dim_out, n_samples)
        y = self.generator(x)
        return y


class ScoringRuleCIGM(ConditionalImplicitGenerativeModel):
    def __init__(self,
                 dim_out,
                 n_layers_generator,
                 n_neurons_generator,
                 activation_generator,
                 n_samples_train=10, 
                 **kwargs):
        
        super().__init__(dim_out, 
                         n_layers_generator,
                         n_neurons_generator,
                         activation_generator,
                         **kwargs)

        # hyperparams
        self.n_samples_train = n_samples_train


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
    


### defining the core models ####
class ConditionalGAN(ConditionalImplicitGenerativeModel):
    def __init__(self, 
                dim_out, 
                n_layers_generator,
                n_neurons_generator,
                activation_generator,
                n_layers_discriminator, 
                n_neurons_discriminator, 
                activation_discriminator,
                output_activation_discriminator="sigmoid",
                generator_train_steps = 1,
                discriminator_train_steps = 1,
                n_samples_val=10,
                **kwargs):
        
        super().__init__(dim_out, 
                         n_layers_generator,
                         n_neurons_generator,
                         activation_generator,
                         **kwargs)

        # discriminator hyperparams
        self.discriminator_config = {"n_layers": n_layers_discriminator,
                                    "n_neurons": n_neurons_discriminator,
                                    "activation": activation_discriminator,
                                    "output_activation": output_activation_discriminator}

        # training loop
        self.generator_train_steps = generator_train_steps
        self.discriminator_train_steps = discriminator_train_steps
        self.n_samples_train = 1 # this has to be 1 for adversarial training

        # evaluation
        self.n_samples_val = n_samples_val
        
        # generator model is initiated in parent class
        
        # init discriminator model
        self.discriminator = DiscriminatorModel(**self.discriminator_config)


    def compile(self, loss, d_optimizer, g_optimizer, val_loss_metric):
        super().compile()
        
        if isinstance(d_optimizer, str):
            self.d_optimizer = getattr(tf.keras.optimizers, d_optimizer)()
        else:
            self.d_optimizer = d_optimizer
        
        if isinstance(g_optimizer, str):
            self.g_optimizer = getattr(tf.keras.optimizers, g_optimizer)()
        else:
            self.g_optimizer = g_optimizer
                
        self.loss_fn = loss
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")
        self.val_loss_metric = val_loss_metric


    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric, self.val_loss_metric]


    def train_step(self, data):
        x_train, y_train = data
        #batch_size = y_train.shape[0]
        batch_size = tf.shape(y_train)[0]
        
        ### Train the discriminator ###
        for i in range(self.discriminator_train_steps):
            # get fake and true data
            y_fake = self.generator(tf.expand_dims(x_train, axis=2), training=True)
            y_fake_true = tf.squeeze(tf.concat([y_fake, y_train], axis=0), axis=2)
            labels_fake_true = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
            
        # Update D's weights
            with tf.GradientTape() as tape:
                d_output = self.discriminator((tf.tile(x_train, (2,1)), y_fake_true), training=True)
                d_loss = self.loss_fn(labels_fake_true, d_output)
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        
        ### Train the generator ###
        for i in range(self.generator_train_steps):
            #get misleading labels
            labels_fake_misleading = tf.zeros((batch_size, 1))

            # Update G's weights
            with tf.GradientTape() as tape:
                y_fake = tf.squeeze(self.generator(tf.expand_dims(x_train, axis=2), training=True), axis=2)
                d_out = self.discriminator((x_train, y_fake), training=True)
                g_loss = self.loss_fn(labels_fake_misleading, d_out)
            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))


        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {"d_loss": self.d_loss_metric.result(),
                "g_loss": self.g_loss_metric.result()}
    
    
    def test_step(self, data):
        x_test, y_test = data
        y_predict = self.generator(tf.repeat(tf.expand_dims(x_test, axis=2), repeats=self.n_samples_val, axis=2), training=False)
        self.val_loss_metric.update_state(y_test, y_predict)
        return {"loss": self.val_loss_metric.result()}



### defining different generator architectures ###
class GeneratorModel(tf.keras.Model):
    def __init__(self,
                 dim_out,
                 dim_latent, 
                 n_layers, 
                 n_neurons, 
                 activation, 
                 output_activation="linear",
                 censored_left=-np.inf,
                 censored_right=np.inf,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.dim_out = dim_out
        self.dim_latent = dim_latent
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.output_activation = output_activation
        
        # censored_left and censored_right should be arrays of shape (dim_out, 1)
        if isinstance(censored_left, (list, tuple, np.ndarray)):
            self.censored_left = np.reshape(censored_left, (1, self.dim_out, 1)) # if it's an array with length dim_out
        else:
            self.censored_left = np.zeros((1, self.dim_out, 1)) + censored_left # if it's a float
        
        if isinstance(censored_right, (list, tuple, np.ndarray)):
            self.censored_right = np.reshape(censored_right, (1, self.dim_out, 1))
        else:
            self.censored_right = np.zeros((1, self.dim_out, 1)) + censored_right


class ConcatGenerator(GeneratorModel):
    def __init__(self, dim_out, dim_latent, n_layers, n_neurons, activation, **kwargs):
        super().__init__(dim_out, dim_latent, n_layers, n_neurons, activation, **kwargs)
        
        # x has input dim (bs, n_features, n_samples)
        self.sample_noise = UnconditionalGaussianSampling(dim_latent=self.dim_latent)
        self.concat_x_and_noise = layers.Concatenate(axis=1)
        self.permute_inputs = layers.Permute((2,1))
        
        self.dense_layers = []
        for i in range(self.n_layers):
            self.dense_layers.append(layers.Conv1D(filters=self.n_neurons, kernel_size=1, strides=1, activation=self.activation))

        self.output_layer = layers.Conv1D(filters=self.dim_out, kernel_size=1, strides=1, activation=self.output_activation)
        self.permute_outputs = layers.Permute((2,1))
        self.clip = ClipValues(low=self.censored_left, high=self.censored_right)

        
    def call(self, inputs):
        #unpack and prepare inputs
        x = inputs # (bs, n_features, n_samples_train)        
        z = self.sample_noise(x) # (bs, dim_latent, n_samples_train)

        # forward pass
        h = self.concat_x_and_noise([x, z]) # -> (bs, n_features+dim_latent , n_samples_train)
        h = self.permute_inputs(h) # -> (bs, n_samples_train, n_features+dim_latent)
        
        for layer_i in self.dense_layers:
            h = layer_i(h)

        y = self.output_layer(h)
        y = self.permute_outputs(y)
        y = self.clip(y)
        
        return y


class FiLMGenerator(GeneratorModel):
    def __init__(self, dim_out, dim_latent, n_layers, n_neurons, activation,  modulation="per_layer", **kwargs):
        super().__init__(dim_out, dim_latent, n_layers, n_neurons, activation, **kwargs)

        self.modulation = modulation

        # init layers
        self.sample_noise = UnconditionalGaussianSampling(dim_latent=self.dim_latent)
        self.permute_inputs = layers.Permute((2,1))
        
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
        
        self.output_layer = layers.Conv1D(filters=self.dim_out, kernel_size=1, strides=1, activation=self.output_activation)
        self.permute_outputs = layers.Permute((2,1))
        self.clip = ClipValues(low=self.censored_left, high=self.censored_right)



    def call(self, inputs):
        # inputs
        x = inputs # (bs, n_features, n_samples_train)
        z = self.sample_noise(x) # (bs, dim_latent, n_samples_train)
        x = self.permute_inputs(x) # (bs, n_features, n_samples) --> (bs, n_samples, n_features)
        z = self.permute_inputs(z) # (bs, dim_latent, n_samples) --> (bs, n_samples, dim_latent)
        
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

        y = self.output_layer(h)
        y = self.permute_outputs(y)
        y = self.clip(y)
        
        return y



class DiscriminatorModel(tf.keras.Model):
    def __init__(self, n_layers, n_neurons, activation, output_activation, **kwargs):
        super().__init__(**kwargs)
        
        # hyperparams
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.output_activation = output_activation
        
        # layers
        self.concat_inputs = layers.Concatenate(axis=1)
        self.hidden_layers = []
        for i in range(self.n_layers):
            self.hidden_layers.append(layers.Dense(units=self.n_neurons, activation=self.activation))
        self.output_layer = layers.Dense(1, activation=self.output_activation)
        

    def call(self, inputs):
        x, y = inputs # [(bs, n_features), (bs, dim_out)]
        h = self.concat_inputs([x,y]) # [(bs, n_features), (bs, dim_out)] --> (bs, n_features+dim_out)
        for layer_i in self.hidden_layers:
            h = layer_i(h)
        y = self.output_layer(h)
        return y

