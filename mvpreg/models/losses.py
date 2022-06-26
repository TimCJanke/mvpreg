import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss


############### loss function ################
def compute_quantile_loss(y_true, q_pred, taus):
    """Computes energy score

    Args:
        y_true (tf.tensor, shape BSxD): Samples from true distribution.
        q_pred (tf. tensor, shape BSxDxQ): Predicted quantiles from model.
        taus (tf. tensor, shape 1x1xQ): Quantile levels 

    Returns:
        loss (tf.tensor, shape BS): Average quantile losses for batch
    """
    
    res = tf.expand_dims(y_true,axis=2) - q_pred
    L = tf.nn.relu(res) * taus + tf.nn.relu(-res) * (1.0-taus)
    
    return tf.reduce_mean(L, axis=(1,2))


# subclass Keras loss
class QuantileLoss(Loss):
    def __init__(self, taus ,name="QuantileLoss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.taus = tf.reshape(taus, (1,1,-1))

    def call(self, y_true, q_pred):
        return compute_quantile_loss(y_true, q_pred, tf.cast(self.taus, q_pred.dtype))

    def get_config(self):
        cfg = super().get_config({"taus": self.taus})
        return cfg



# define energy score 
def compute_energy_score(y_true, y_pred):
    """Computes energy score

    Args:
        y_true (tf.tensor, shape BSxDx1): Samples from true distribution.
        y_pred (tf. tensor, shape BSxDxM): Samples from model.

    Returns:
        loss (tf.tensor, shape BS): Energy score for batch
    """
    n_samples_model = tf.cast(tf.shape(y_pred)[2], dtype=tf.float32)
    
    #N = y_pred.shape[-1]
    #es_12 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(y_pred - tf.repeat(y_true, repeats=N, axis=2)), axis=1)+K.epsilon()), axis=1) # should give array of length N
    #es_22 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(tf.tile(y_pred, multiples=(1,1,N)) - tf.repeat(y_pred, repeats=N, axis=2)), axis=1)+K.epsilon()), axis=1) # should give array of length N

    es_12 = tf.reduce_sum(tf.sqrt(tf.clip_by_value(tf.matmul(y_true, y_true, transpose_a=True, transpose_b=False) + tf.square(tf.linalg.norm(y_pred, axis=1, keepdims=True)) - 2*tf.matmul(y_true, y_pred, transpose_a=True, transpose_b=False), K.epsilon(), 1e10)), axis=(1,2))    
    G = tf.linalg.matmul(y_pred, y_pred, transpose_a=True, transpose_b=False)
    d = tf.expand_dims(tf.linalg.diag_part(G, k=0), axis=1)
    es_22 = tf.reduce_sum(tf.sqrt(tf.clip_by_value(d + tf.transpose(d, perm=(0,2,1)) - 2*G, K.epsilon(), 1e10)), axis=(1,2))

    loss = es_12/(n_samples_model) -  es_22/(2*n_samples_model*(n_samples_model-1))
    
    return loss


# subclass Keras loss
class EnergyScore(Loss):
    def __init__(self, name="EnergyScore", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_data, y_model):
        return compute_energy_score(y_data, y_model)
        #return compute_es_and_vs(y_data, y_model)
    
    

# define variogram score 
def compute_variogram_score(y_true, y_pred, p):
    """Computes variogram score

    Args:
        y_true (tf.tensor, shape BSxDx1): Samples from true distribution.
        y_pred (tf. tensor, shape BSxDxM): Samples from model.

    Returns:
        loss (tf.tensor, shape BS): Variogram score for batch
    """
    
    D = y_pred.shape[1]

    vs_1 = tf.squeeze(tf.pow(tf.abs(tf.repeat(y_true, D, axis=1) - tf.tile(y_true, (1,D,1)))+K.epsilon(), p)) # (N,D*D) array
    vs_2 = tf.reduce_mean(tf.pow(tf.abs(tf.repeat(y_pred, D, axis=1) - tf.tile(y_pred, (1,D,1)))+K.epsilon(), p), axis=2) # (N,D*D) array
        
    return tf.reduce_sum(tf.pow(vs_1-vs_2, 2.0), axis=1) # (N,D*D) -> (N,)


# subclass Keras loss
class VariogramScore(Loss):
    def __init__(self, name="VariogramScore", p=0.5, **kwargs):
        self.p = p
        super().__init__(name=name, **kwargs)

    def call(self, y_data, y_model):
        return compute_variogram_score(y_data, y_model, p=self.p)

    def get_config(self):
        cfg = super().get_config()
        cfg["p"] = self.p
        return cfg
    


# LogScore aka negative log likelihood
class LogScore(Loss):
    def __init__(self, name="LogScore", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, dist_pred):
        #return -dist_pred.log_prob(y_true)
        return tf.reduce_sum(-dist_pred.log_prob(y_true), axis=-1) # do we need this?



### Keras metrics ### 
class EnergyScoreMetric(tf.keras.metrics.Metric):
  def __init__(self, name='ES', **kwargs):
    super().__init__(name=name, **kwargs)
    self.es = self.add_weight(name='es', initializer = 'zeros')

  def update_state(self, y_true, y_pred):
      self.es.assign_add(tf.reduce_mean(compute_energy_score(y_true, y_pred)))

  def result(self):
    return self.es

  def reset_state(self):
    self.es.assign(0.0)
    

class VariogramScoreMetric(tf.keras.metrics.Metric):
  def __init__(self, name='VS', p=0.5, **kwargs):
    super().__init__(name=name, **kwargs)
    self.vs = self.add_weight(name='vs', initializer = 'zeros')
    self.p = p
    
  def update_state(self, y_true, y_pred):
      self.vs.assign_add(tf.reduce_mean(compute_variogram_score(y_true, y_pred, p=self.p)))

  def result(self):
    return self.vs

  def reset_state(self):
    self.vs.assign(0.0)