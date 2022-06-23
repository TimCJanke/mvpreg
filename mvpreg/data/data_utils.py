import numpy as np
import pandas as pd
import os
import copy

from . import get_data_path


def read_wind_data(zones=np.arange(1,11)):
    path_to_dataset = os.path.join(get_data_path(), "datasets/wind")
    
    dataset = {}
    for zone in zones:
        dataset[zone] = pd.read_csv(os.path.join(path_to_dataset,"wind_data_zone_"+str(zone)+".csv"), index_col=0, parse_dates=True, infer_datetime_format=True)
    
    return dataset

def read_solar_data(zones=np.arange(1,4)):
    path_to_dataset = os.path.join(get_data_path(), "datasets/solar")
    
    dataset = {}
    for zone in zones:
        dataset[zone] = pd.read_csv(os.path.join(path_to_dataset,"solar_data_zone_"+str(zone)+".csv"), index_col=0, parse_dates=True, infer_datetime_format=True)
    
    return dataset


def read_load_data():
    path_to_dataset = os.path.join(get_data_path(), "datasets/load")
    
    dataset  = pd.read_csv(os.path.join(path_to_dataset,"load_data.csv"), index_col=0, parse_dates=True, infer_datetime_format=True)
    
    return dataset



def build_wind_features(dataset, features=['WE10', 'WE100', 'WD10', 'WD100', 'WD_difference', 'WS_ratio'], hours=np.arange(0,24)):
    
    """
    inspired from: https://www.sciencedirect.com/science/article/pii/S0169207016000145
    
    possible features:
    
    U10: u wind component at 10 m
    ws: wind speed
    we: wind energy
    wd: wind direction    
    
    ZONE: zone number i.e. {1,...,10}
    ZONE-DUMMY: zone as dummy 
    """
    dataset_out = copy.deepcopy(dataset)
    
    for zone, df in dataset_out.items():
        
        df['WS10'] = np.sqrt(df['U10'].values ** 2 + df['V10'].values ** 2)
        df['WS100'] = np.sqrt(df['U100'].values ** 2 + df['V100'].values ** 2)
        df['WE10'] = 0.5 * 1 * df['WS10'].values ** 3
        df['WE100'] = 0.5 * 1 * df['WS100'].values ** 3
        df['WD10'] = np.arctan2(df['U10'].values, df['V10'].values) * 180 / np.pi
        df['WD100'] = np.arctan2(df['U100'].values, df['V100'].values) * 180 / np.pi
        
        df["WD_difference"] = (df['WD10'] - df['WD100'] + 180.0)%360.0 - 180.0
        df["WS_ratio"] = np.clip(df['WS100']/df['WS10'], 0.0, 5.0)
        df["WE_ratio"] = np.clip(df['WE100']/df['WE10'], 0.0, 150.0)

        df['ZONE'] = zone
        
        df = df[["TARGETVAR", *features]]

        if "ZONE_DUMMY" in features:
            n_zones = len(dataset_out.keys())
            dummy_matrix = np.zeros(len(df), n_zones)
            dummy_matrix[:,zone] = 1.0
            df[["ZONE_"+str(i) for i in range(1,n_zones+1)]] = dummy_matrix

        df = df[[hh in hours for hh in df.index.hour]] # only select examples specified in hours
        
        dataset_out[zone] = df
        
    return dataset_out


def build_solar_features(dataset, features=['T', 'SSRD', 'RH', 'WS10'], hours=np.arange(6,20)):
    """
    possible features:
    
    T: 2m temperature
    SSRD: Solar Irradiance
    RH: rel. hunidity
    WS10:
    
    ZONE: zone number i.e. {1,...,10}
    ZONE-DUMMY: zone as dummy 
    """
    """
    Total column liquid water (tclw) kg/m2 Vertical integral of cloud liquid water content
    Total column ice water (tciw) kg/m2 Vertical integral of cloud ice water content
    Surface pressure (SP) Pa
    Relative humidity at 1000 mbar (RH) % Relative humidity 
    Total cloud cover (TCC)
    10-metre U wind component (10u) m/s
    10-metre V wind component (10v) m/s
    2-metre temperature (T) in C
    Surface solar rad down (SSRD) W/m2 Accumulated field
    Surface thermal rad down (STRD) W/m2 Accumulated field
    Top net solar rad (TSR) W/m2 Net solar radiation at the top of the atmosphere.
    Total precipitation (TP) m Convective precipitation + stratiform precipitation
    
    #TODO:
    HoD: Hour of day
    HoD-DUMMY: One-hot encoded hour of day
    
    MONTH: Month of year
    MONTH-DUMMY: One hot encoded hour of year
        
    
    ZONE: zone number i.e. {1,...,10}
    ZONE-DUMMY: zone as dummy
    """
    dataset_out = copy.deepcopy(dataset)

    for zone, df in dataset_out.items():
        df['WS10'] = np.sqrt(df['U10'].values ** 2 + df['V10'].values ** 2)
        df['ZONE'] = zone
        
        # hour of day
        #df['HoD'] = df.index.hour
        #df['sin_hour_day'] = np.sin(2*np.pi*df['HoD']/24)
        #df['cos_hour_day'] = np.cos(2*np.pi*df['HoD']/24)

        # hour of year
        #df['HoY'] = df.index.dayofyear*24 + df.index.hour
        #df['sin_hour_year'] = np.sin(2*np.pi*df['HoY']/8760)
        #df['cos_hour_year'] = np.cos(2*np.pi*df['HoY']/8760)

        df = df[["TARGETVAR", *features]]

        if "ZONE_DUMMY" in features:
            n_zones = len(dataset_out.keys())
            dummy_matrix = np.zeros(len(df), n_zones)
            dummy_matrix[:,zone] = 1.0
            df[["ZONE_"+str(i) for i in range(1,n_zones+1)]] = dummy_matrix

        df = df[[hh in hours for hh in df.index.hour]] # only select examples specified in hours
        
        dataset_out[zone] = df
    
    return dataset_out


def build_load_features(dataset, features=['TEMP', 'DoW_DUMMY', 'MoY_DUMMY'], load_lags=[1,2,7], temp_lags=[1,2,7], hours=np.arange(0,24)):
    """
    possible features:
    TEMP (avergae of all stations)
    HoD (number/one-hot)
    DoW (number/one-hot)
    MoY (number/one-hot)
    LOAD-24h, LOAD-48h, ...
    TEMP-24h, TEMP-48h, ...
    """
    
    df = copy.deepcopy(dataset)
    features_out = features.copy()
    max_lag = 0
    
    # add features
    df["TEMP"] = df[["w"+str(i) for i in range(1,26)]].mean(axis=1)
    
    df["HoD"] = df.index.hour
    df["DoW"] = df.index.weekday
    df["MoY"] = df.index.month
        
    if "HoD_DUMMY" in features:
        df, features_out = _add_dummies(df, features_out, "HoD")
    
    if "DoW_DUMMY" in features:
        df, features_out = _add_dummies(df, features_out, "DoW")    
    
    if "MoY_DUMMY" in features:
        df, features_out = _add_dummies(df, features_out, "MoY")        
    
    if load_lags is not None:
        for d in load_lags:
            df[f"LOAD_d-{d}"] = df["TARGETVAR"].shift(d*24)
            features_out.append(f"LOAD_d-{d}")
        max_lag = np.max([max_lag, np.max(load_lags)])

    if temp_lags is not None:
        for d in temp_lags:
            df[f"TEMP_d-{d}"] = df["TEMP"].shift(d*24)
            features_out.append(f"TEMP_d-{d}")
        max_lag = np.max([max_lag, np.max(temp_lags)])
    
    df = df.iloc[max_lag*24:,:] #cut off NAs from lag shift
    
    df = df[["TARGETVAR", *features_out]] # only return specified features
    
    df = df[[hh in hours for hh in df.index.hour]]
    
    return df


def _add_dummies(df, features, prefix):
    
    dummies = pd.get_dummies(df[prefix],  prefix=prefix, prefix_sep="_")
    features.remove(prefix+"_DUMMY")
    features = [*features, *list(dummies.columns)]
    
    return pd.concat([df, dummies], axis=1), features


def get_Xy_arrays_spatial(dataset):
    # assumes dataset is a dictionary with dataframes from different zones
    X = []
    y = []
    for _, df in dataset.items():
        X.append(np.expand_dims(df.drop(["TARGETVAR"], axis=1).values, axis=1)) #(N,1,M) array
        y.append(df[["TARGETVAR"]].values) # (N,1) array
        dates = df.index
        features = list(df.columns)
        features.remove("TARGETVAR")
        
    X = np.concatenate(X, axis=1) # [(N,1,M)] --> (N,D,M)
    y = np.concatenate(y, axis=1) # [(N,1)] --> (N,D)
    
    return {"X": X, "y": y, "dates": dates, "features": features}


def get_Xy_arrays_temporal(df, freq=24):
    # assumes df is a single dataframe from one zone  
    X = df.drop(["TARGETVAR"], axis=1).values
    y = df[["TARGETVAR"]].values
    
    X = np.reshape(X, (-1, freq, X.shape[1]))
    y = np.reshape(y, (-1, freq))
    
    dates = pd.date_range(df.index[0].date(), df.index[-1].date(), freq="D")
    features = list(df.columns)
    features.remove("TARGETVAR")
    
    return {"X": X, "y": y, "dates": dates, "features": features}



  
def fetch_wind_spatial(zones=np.arange(1,11), features=['WE10', 'WE100', 'WD10', 'WD100', 'WD_difference', 'WS_ratio'], hours=np.arange(0,24)):
    return get_Xy_arrays_spatial(build_wind_features(read_wind_data(zones=zones), features=features, hours=hours))


def fetch_solar_spatial(zones=np.arange(1,4), features=['T', 'SSRD', 'RH', 'WS10'], hours=np.arange(6,20)):
    return get_Xy_arrays_spatial(build_solar_features(read_solar_data(zones=zones), features=features, hours=hours))


def fetch_wind_temporal(zone, features=['WE10', 'WE100', 'WD10', 'WD100', 'WD_difference', 'WS_ratio'], hours=np.arange(0,24)):
    return  get_Xy_arrays_temporal(build_wind_features(read_wind_data(zones=[zone]), features=features, hours=hours)[zone], freq=len(hours))


def fetch_solar_temporal(zone, features=['T', 'SSRD', 'RH', 'WS10'], hours=np.arange(6,20)):
    return  get_Xy_arrays_temporal(build_solar_features(read_solar_data(zones=[zone]), features=features, hours=hours)[zone], freq=len(hours))


def fetch_load(features=['TEMP', 'DoW_DUMMY', 'MoY_DUMMY'], load_lags=[1,2,7], temp_lags=[1,2,7], hours=np.arange(0,24)):
    return get_Xy_arrays_temporal(build_load_features(read_load_data(), features=features, load_lags=load_lags, temp_lags=temp_lags, hours=hours), freq=len(hours))

#TODO:
# def fetch_price():
#     pass