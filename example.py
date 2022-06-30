#%%
import numpy as np
import tensorflow as tf
import pandas as pd

from mvpreg.data import data_utils
from mvpreg import DeepQuantileRegression as DQR
from mvpreg import DeepParametricRegression as PRM
from mvpreg import ScoringRuleDGR as DGR
from mvpreg import AdversarialDGR as CGAN

from mvpreg.evaluation import scoring_rules
from mvpreg.evaluation import visualization


#%% Data and hyperparams
#get data set
ZONES = [4, 5, 6, 1, 7, 8]
data = data_utils.fetch_wind_spatial(zones=ZONES)
features = data["features"]
x = np.reshape(data["X"], (-1,len(features)*len(ZONES)))
y = data["y"]

# make train, val, test sets
x = x[0:3500,...]
y = y[0:3500, ...]

N_VAL = 500
N_TEST = 1000

x_test = x[-N_TEST:,...]
y_test = y[-N_TEST:,...]

x_val = x[-N_VAL-N_TEST:-N_TEST,...]
y_val = y[-N_VAL-N_TEST:-N_TEST,...]

x_train = x[0:-N_VAL-N_TEST,...]
y_train = y[0:-N_VAL-N_TEST,...]

# misc
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

TAUS = np.arange(0.05,1.0, 0.05) # for qr model

y_predict = {} # to store predictions

#%% QR + Copula
model_dqr = DQR(**nn_base_config, taus=TAUS, copula_type="gaussian")
model_dqr.fit(x_train, 
                    y_train, 
                    x_val=x_val,
                    y_val=y_val, 
                    epochs=100,
                    early_stopping=True)

y_predict["QR"] = model_dqr.simulate(x_test, n_samples=1000)

#%% Parametric + Copula
model_param = PRM(**nn_base_config, distribution="LogitNormal", copula_type="gaussian")
model_param.fit(x_train, 
                np.clip(y_train, nn_base_config["censored_left"]+1e-2, nn_base_config["censored_right"]-1e-2), # because LogitNormal is only define on (0,1)
                x_val=x_val,
                y_val=np.clip(y_val, nn_base_config["censored_left"]+1e-2, nn_base_config["censored_right"]-1e-2), # because LogitNormal is only define on (0,1)
                epochs=100,
                early_stopping=True)

y_predict["PARAM"] = model_param.simulate(x_test, n_samples=1000)


#%% Scoring Rule Deep Generative Model
model_dgr = DGR(**nn_base_config, 
                n_samples_train=10, 
                n_samples_val=100, 
                dim_latent=y_train.shape[1], 
                conditioning="FiLM")

model_dgr.fit(x_train, 
              y_train, 
              x_val=x_val,
              y_val=y_val, 
              epochs=100,
              early_stopping=True)

y_predict["DGR"] = model_dgr.simulate(x_test, n_samples=1000)


#%% Conditional GAN
model_gan = CGAN(**nn_base_config, 
                 dim_latent=y_train.shape[1], 
                 conditioning="FiLM", 
                 optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.1),
                 optimizer_discriminator = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.1),
                 n_samples_val=100,
                 label_smoothing=0.1)

model_gan.fit(x_train, 
              y_train, 
              x_val=x_val,
              y_val=y_val, 
              epochs=1000,
              early_stopping=True)

y_predict["GAN"] = model_gan.simulate(x_test, n_samples=1000)


#%% evalutaion
# first let's have a look at two examplary hours
names=dim_names=["ZONE_"+str(i) for i in ZONES[0:3]]
visualization.scenario_plot_spatial(y_test[0,0:3], y_predict["DGR"][0,0:3,0:500].T, dim_names=dim_names)
visualization.scenario_plot_spatial(y_test[400,0:3], y_predict["DGR"][400,0:3,0:500].T, dim_names=dim_names)

# let's compare models based on several scores
scores={}
for key in y_predict:
    scores[key] = scoring_rules.all_scores_mv_sample(y_test, y_predict[key][:,:,0:500])
scores = pd.DataFrame(scores).T
print(scores)

# let's assess significance of score differences via DM test and plot the results
es_series={}
for key in y_predict:
    es_series[key] = scoring_rules.es_sample(y_test, y_predict[key][:,:,0:500], return_single_scores=True)

dm_results_matrix = scoring_rules.dm_test_matrix(es_series)
visualization.plot_dm_test_matrix(dm_results_matrix)