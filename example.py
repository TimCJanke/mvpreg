#%%
import numpy as np
import tensorflow as tf

from mvpreg.data import data_utils
from mvpreg import DeepQuantileRegression as DQR
from mvpreg import DeepParametricRegression as PRM
from mvpreg import ScoringRuleDGR as DGR
from mvpreg import AdversarialDGR as CGAN


#%% Data and hyperparams
ZONES = [1,4,5,6,7,8]
data = data_utils.fetch_wind_spatial(zones=[1,4,5,6,7,8])

features = data["features"]
x = np.reshape(data["X"], (-1,len(features)*len(ZONES)))
y = data["y"]

x = x[0:3000,...]
y = y[0:3000, ...]

N_VAL = 500
N_TEST = 1000

x_test = x[-N_TEST:,...]
y_test = y[-N_TEST:,...]

x_val = x[-N_VAL-N_TEST:-N_TEST,...]
y_val = y[-N_VAL-N_TEST:-N_TEST,...]

x_train = x[0:-N_VAL-N_TEST,...]
y_train = y[0:-N_VAL-N_TEST,...]


nn_base_config = {"dim_in": x_train.shape[1],
                  "dim_out": y_train.shape[1],
                  "n_layers": 3,
                  "n_neurons": 200,
                  "activation": "relu",
                  "output_activation": None,
                  "censored_left": 0.0, 
                  "censored_right": 1.0, 
                  "input_scaler": "Standard",
                  "output_scaler": None}

y_predict = {}
TAUS = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])

#%% QR + Copula
model_dqr_mimo = DQR(**nn_base_config, taus=TAUS, copula_type="gaussian")
model_dqr_mimo.fit(x_train, 
                    y_train, 
                    x_val=x_val,
                    y_val=y_val, 
                    epochs=100,
                    callbacks=tf.keras.callbacks.EarlyStopping(patience=10, verbose=1, mode="min", restore_best_weights=True)
                    )

y_predict["QR"] = model_dqr_mimo.simulate(x_test, n_samples=1000)

#%% Parametric + Copula
model_param = PRM(**nn_base_config, distribution="LogitNormal", copula_type="gaussian")
model_param.fit(x_train, 
                np.clip(y_train, nn_base_config["censored_left"]+1e-2, nn_base_config["censored_right"]-1e-2), 
                x_val=x_val,
                y_val=np.clip(y_val, nn_base_config["censored_left"]+1e-2, nn_base_config["censored_right"]-1e-2), 
                epochs=100,
                callbacks=tf.keras.callbacks.EarlyStopping(patience=10, verbose=1, mode="min", restore_best_weights=True))

y_predict["PARAM"] = model_param.simulate(x_test, n_samples=1000)


#%% Scoring Rule Deep Generative Model
model_dgr = DGR(**nn_base_config, n_samples_train=20, n_samples_val=50, dim_latent=y_train.shape[1], conditioning="FiLM")
model_dgr.fit(x_train, 
              y_train, 
              x_val=x_val,
              y_val=y_val, 
              epochs=100,
              callbacks=tf.keras.callbacks.EarlyStopping(patience=20, verbose=1, mode="min", restore_best_weights=True))

y_predict["DGR"] = model_dgr.simulate(x_test, n_samples=1000)


#%% Conditional GAN
model_gan = CGAN(**nn_base_config, 
                 dim_latent=y_train.shape[1], 
                 conditioning="FiLM", 
                 optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.1),
                 optimizer_discriminator = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.1),
                 n_samples_val=50,
                 label_smoothing=0.1
                 )

model_gan.fit(x_train, 
              y_train, 
              x_val=x_val,
              y_val=y_val, 
              epochs=1000,
              callbacks=tf.keras.callbacks.EarlyStopping(patience=25, verbose=1, mode="min", restore_best_weights=True))

y_predict["GAN"] = model_gan.simulate(x_test, n_samples=1000)
