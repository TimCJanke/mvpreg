# script to be run once at package intallation to create data sets in nice shape

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import os

def make_wind_dataset(zone, path_to_datafolder):
    # load orginal data files 
    data_train = pd.read_csv(os.path.join(path_to_datafolder,"GEFCom2014Data/wind/wind_train/Task15_W_Zone"+str(zone)+".csv"), index_col=1, parse_dates=True)
    data_train = data_train.drop(["ZONEID"], axis=1)    
    
    data_test_input = pd.read_csv(os.path.join(path_to_datafolder, "GEFCom2014Data/wind/wind_test_input/TaskExpVars15_W_Zone"+str(zone)+".csv"), index_col=1, parse_dates=True)
    data_test_input = data_test_input.drop(["ZONEID"], axis=1)

    data_test_target = pd.read_csv(os.path.join(path_to_datafolder,"GEFCom2014Data/wind/wind_test_target.csv"), index_col=1, parse_dates=True)
    data_test_target = data_test_target[data_test_target.ZONEID==zone]
    data_test_target = data_test_target.drop(["ZONEID"], axis=1)

    data_test = pd.concat((data_test_target, data_test_input), axis=1)
    data = pd.concat((data_train, data_test), axis=0)
    data = data.iloc[0:-24,:] # cut off the last day (many missing values)
    
    data.index = pd.Index(pd.date_range(start=data.index[0], end=data.index[-1], freq='H'), name="TIMESTAMP")
    data.index = data.index.shift(-1) # ensures that days start at 0:00 and end at 23:00

    #print("\nNumber of NAs")
    #print(np.sum(data.isna()))

    data = replace_missing_values(data)

    return data


def replace_missing_values(data, max_gap_interpolation=3):
    
    # interpolate small gaps
    if np.sum(data.isna().values)>0:
        data = data.interpolate(limit=max_gap_interpolation, limit_direction="both")

        #print("\nNumber of NAs after interpolation")
        #print(np.sum(data.isna()))


    # impute large gaps by nearest neighbours
    if np.sum(data.isna().values)>0:
        idx = data["TARGETVAR"].isna().values
        
        X_train = data.drop(["TARGETVAR"], axis=1).values[~idx,:]
        y_train = data["TARGETVAR"].values[~idx]
        X_test = data.drop(["TARGETVAR"], axis=1).values[idx,:]
        
        mdl = KNeighborsRegressor(n_neighbors=5).fit(X_train,y_train)
        data.loc[idx,"TARGETVAR"] = mdl.predict(X_test)
    
        #print("\nNumber of NAs after KNN")
        #print(np.sum(data.isna()))
    
    return data



def make_solar_dataset(zone, path_to_datafolder):
    data = pd.read_csv(os.path.join(path_to_datafolder, "GEFCom2014Data/solar/solar.csv"), index_col=1, parse_dates=True)
    data = data[data.ZONEID==zone]
    data = data.drop(["ZONEID"], axis=1)
    
    data = data.rename(columns={"POWER":"TARGETVAR",
                                'VAR78':"TCLW",
                                'VAR79':"TCIW",
                                'VAR134':"SP",
                                'VAR157':"RH",
                                'VAR164':"TCC",
                                'VAR165':"U10",
                                'VAR166':"V10",
                                'VAR167':"T",
                                'VAR169':"SSRD",
                                'VAR175':"STRD",
                                'VAR178':"TSR",
                                'VAR228':"TP"})

    
    data['SSRD'] = data['SSRD']/3600 # from J/m2 to W/m2
    data['SSRD'] = deaccumulate(data['SSRD'].values)

    data['STRD'] = data['STRD']/3600 # from J/m2 to W/m2
    data['STRD'] = deaccumulate(data['STRD'].values)

    data['TSR'] = data['TSR']/3600 # from J/m2 to W/m2
    data['TSR'] = deaccumulate(data['TSR'].values)

    data['TP'] = deaccumulate(data['TP'].values)
    
    data.index = pd.Index(pd.date_range(start=data.index[0], end=data.index[-1], freq='H'), name="TIMESTAMP")
    data.index = data.index.shift(-1) # ensures that days start at 0:00 and end at 23:00
    data.index = data.index.shift(11) # adjust for Australian time zone, timestamps are probably UTC
    
    data = data[pd.to_datetime("2012-04-02 00:00"):pd.to_datetime("2014-06-30 23:00")] # ensure we have full 24 hours for each day
    
    return data


def deaccumulate(x, freq=24):
    
    x = np.reshape(x, (-1, freq))
    y = np.diff(x, axis=1, prepend=0.0)
    y = np.reshape(y, (-1,))
    
    return np.clip(y, a_min=0.0, a_max=None)
    
    
    

def make_load_dataset(path_to_datafolder):
    data_train = []
    for i in range(1,16):
        data_train.append(pd.read_csv(os.path.join(path_to_datafolder,"GEFCom2014Data/load/Task "+str(i)+"/L"+str(i)+"-train.csv"), index_col=1, parse_dates=False))
    data_train = pd.concat(data_train, axis=0)
    data_train = data_train.drop(["ZONEID"], axis=1)
    
    data_test = pd.read_csv(os.path.join(path_to_datafolder, "GEFCom2014Data/load/Solution to Task 15/solution15_L_temperature.csv"), index_col=None, parse_dates=False)
    data_test = data_test.drop(["date", "hour"], axis=1)
    
    data = pd.concat([data_train, data_test], axis=0, ignore_index=True)
    
    data.index = pd.Index(pd.date_range(start="2005-01-01 00:00", end="2011-12-31 23:00", freq='H'), name="TIMESTAMP")
    data = data.rename(columns={"LOAD":"TARGETVAR"})
    
    data.loc[:,data.columns!="TARGETVAR"] = (data.loc[:,data.columns!="TARGETVAR"] - 32.0)*5/9 # Fahrenheit --> Celsius

    return data


if __name__ == '__main__':
    import pathlib
    ### get path of this file and create datasets folder ###
    path_to_datafolder = pathlib.Path(__file__).resolve().parent
    os.makedirs(os.path.join(path_to_datafolder, "datasets"))

    ### create wind data set from raw data ###
    print("\nCreating wind data set...")
    os.makedirs(os.path.join(path_to_datafolder, "datasets/wind"))
    for zone in range(1,11):
        make_wind_dataset(zone=zone, path_to_datafolder=path_to_datafolder).to_csv(os.path.join(path_to_datafolder, "datasets/wind/wind_data_zone_"+str(zone)+".csv"), float_format='%.6f')
    print("Done.\n\n")


    ### create solar data set from raw data ###
    print("Creating solar data set...")
    os.makedirs(os.path.join(path_to_datafolder, "datasets/solar"))
    for zone in range(1,4):
        make_solar_dataset(zone=zone, path_to_datafolder=path_to_datafolder).to_csv(os.path.join(path_to_datafolder, "datasets/solar/solar_data_zone_"+str(zone)+".csv"), float_format='%.6f')
    print("Done.\n\n")


    ### create load data set from raw data ###
    print("Creating load data set...")
    os.makedirs(os.path.join(path_to_datafolder, "datasets/load"))
    make_load_dataset(path_to_datafolder=path_to_datafolder).to_csv(os.path.join(path_to_datafolder, "datasets/load/load_data.csv"), float_format='%.2f')
    print("Done.\n\n")


