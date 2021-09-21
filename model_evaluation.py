# Copyright 2020 Â© Michal Narajewski, Florian Ziel
#%%
import locale
import os
import pandas as pd
import numpy as np
from my_functions import DST_trafo
from calendar import day_abbr
from scipy.stats import t
import matplotlib.pyplot as plt
from forecast_model import *

# %% set language setting
# English US Linux
locale.setlocale(locale.LC_ALL, "en_US.utf8")
# English US Mac
# locale.setlocale(locale.LC_ALL, 'EN_US')
# English US Windows
# locale.setlocale(locale.LC_ALL,  locale="English_United States.1252")
locale.getlocale()

# %% check the working directory
os.getcwd()

# %% read the data
data = pd.read_csv("Final_Dataset.csv")
data.drop('Price_MBA', axis = 1, inplace = True)
data.dropna(inplace = True)
# %% select the price and time
time_utc = pd.to_datetime(data["DateTime"],
                          utc=True, format="%Y-%m-%d %H:%M:%S")
local_time_zone = "CET"
time_lt = time_utc.dt.tz_convert(local_time_zone)

S = 24

# %% Save the start and end-time
start_end_time_S = time_lt.iloc[[0, -1]
                                ].dt.tz_localize(None).dt.tz_localize("UTC")

# %% creating 'fake' local time
start_end_time_S_num = pd.to_numeric(start_end_time_S)
time_S_numeric = np.arange(
    start=start_end_time_S_num.iloc[0],
    stop=start_end_time_S_num.iloc[1] + 24 * 60 * 60 * 10 ** 9 / S,
    step=24 * 60 * 60 * 10 ** 9 / S,
)

# %% 'fake' local time
time_S = pd.Series(pd.to_datetime(time_S_numeric, utc=True))
dates_S = pd.Series(time_S.dt.date.unique())

# %% import DST_trafo function and use it on data
data_array = DST_trafo(X=data.iloc[:, 1:], Xtime=time_utc,
                       tz=local_time_zone)
data_array.shape

# %% save as dataframe
price_S = pd.DataFrame(data_array[..., 0], index=dates_S)

# %% Forecasting and Evaluation #
D = 365 * 3
N = price_S.shape[0] - D

oos_dates = dates_S[D:N+D]


#%% Benchmark models

model_names = ["true", "naive",
               "expert", "expert.REDADV"]
n_models = len(model_names)

# %%
forecasts = np.empty((N, S, n_models))
forecasts[::] = np.nan


# %%
dim_names = [oos_dates.values, np.arange(S), model_names]


# %%
coeffs_list = {}
coeffs_list["expert"] = np.full((7, N, S), np.nan)
coeffs_list["expert.REDADV"] = np.full((15,N,S), np.nan)
# %%
price = data_array[:, :, data.columns[1:] == "Price"][..., 0]

lag1_min = price.min(axis = 1)
lag1_max = price.max(axis = 1)



for n in range(N):
    # 'model' 1 is the true price to forecast (prediction target)
    forecasts[n, :, 0] = price_S.iloc[D+n]

    Y = price_S.iloc[n:D+n]
    days = pd.to_datetime(dates_S[n:D+n], utc=True)
    
    Xlag1_max = pd.Series([np.nan])
    Xlag1_max = Xlag1_max.append(pd.Series(lag1_max[n:D+n]))
    Xlag1_min = pd.Series([np.nan])
    Xlag1_min = Xlag1_min.append(pd.Series(lag1_min[n:D+n]))
    
    dat = data_array[n:D+n, :, :]

    forecasts[n, :, 1] = forecast_naive(Y, days)["forecasts"]

    fc_expert = forecast_expert(Y, days)
    forecasts[n, :, 2] = fc_expert["forecasts"]
    coeffs_list["expert"][:, n] = fc_expert["coefficients"].transpose()
    
    fc_expertREDADV = forecast_expert_REDADV(dat, days, Xlag1_min, Xlag1_max, reg_names = data.columns[1:])
    forecasts[n, :, 3] = fc_expertREDADV["forecasts"]
    coeffs_list["expert.REDADV"][:, n] = fc_expertREDADV["coefficients"].transpose()
    
    
    print("\r-> "+str(np.round((n+1)/N*100, 4))+"% done", end='')
    
#%%  errors of benchmark models
np.save(file="D:/TU_DORTMUND/EEM/out/result/benchmark", arr=forecasts)

errors_benchmark = forecasts - forecasts[:, :, 0:1]


#%% the hourly MAE and RMSE
MAE_benchmark_hourly = pd.DataFrame(np.mean(np.abs(errors_benchmark), 0))
MAE_benchmark_hourly.columns = ["true", "naive",
               "expert", "expert.REDADV"]
fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(MAE_benchmark_hourly.iloc[:, 1:])
plt.title("MAE of benchmark models over hours")
plt.legend(model_names[1:], loc='center left', bbox_to_anchor=(1, .5),
          ncol=1)
plt.xlabel("Hour of the day")
plt.show()

#%% modified expert models

model_names = ["true",
               "expert_addlag1421", "expert_addTueFri", "expert_plusLastHour", 
               "expert_withSeason", "expert_addnoHour6", "expert_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6",
               "mv_expert_mix_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6"]
n_models = len(model_names)

# %%
forecasts = np.empty((N, S, n_models))
forecasts[::] = np.nan


# %%
dim_names = [oos_dates.values, np.arange(S), model_names]


# %%
coeffs_list = {}
coeffs_list["expert_addlag1421"] = np.full((9, N, S), np.nan)
coeffs_list["expert_addTueFri"] = np.full((9,N,S), np.nan)
coeffs_list["expert_plusLastHour"] = np.full((8,N,S), np.nan)
coeffs_list["expert_withSeason"] = np.full((10,N,S), np.nan)
coeffs_list["expert_addnoHour6"] = np.full((13,N,S), np.nan)
coeffs_list["expert_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6"] = np.full((21,N,S), np.nan)
coeffs_list["mv_expert_mix_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6"] = np.full((389, N), np.nan)


for n in range(N):
    # 'model' 1 is the true price to forecast (prediction target)
    forecasts[n, :, 0] = price_S.iloc[D+n]

    Y = price_S.iloc[n:D+n]
    days = pd.to_datetime(dates_S[n:D+n], utc=True)

    fc_expert_addlag1421 = forecast_expert(Y, days, expert_lags = [1,2,7,14,21])
    forecasts[n, :, 1] = fc_expert_addlag1421["forecasts"]
    coeffs_list["expert_addlag1421"][:, n] = fc_expert_addlag1421["coefficients"].transpose()
    
    fc_expert_addTueFri = forecast_expert(Y, days, expert_wd = [1,2,5,6,7])
    forecasts[n, :, 2] = fc_expert_addTueFri["forecasts"]
    coeffs_list["expert_addTueFri"][:, n] = fc_expert_addTueFri["coefficients"].transpose()
    
    fc_expert_plusLastHour = forecast_expert_plusLastHour(Y, days)
    forecasts[n, :, 3] = fc_expert_plusLastHour["forecasts"]
    coeffs_list["expert_plusLastHour"][:, n] = fc_expert_plusLastHour["coefficients"].transpose()
    
    fc_expert_withSeason = forecast_expert_withSeason(Y, days)
    forecasts[n, :, 4] = fc_expert_withSeason["forecasts"]
    coeffs_list["expert_withSeason"][:, n] = fc_expert_withSeason["coefficients"].transpose()
    
    fc_expert_addnoHour6 = forecast_expert_addNoHour6(Y, days)
    forecasts[n, :, 5] = fc_expert_addnoHour6["forecasts"]
    coeffs_list["expert_addnoHour6"][:, n] = fc_expert_addnoHour6["coefficients"].transpose()
    
    fc_expert_addlag1421_addTueFri_plusLastHour_withSS_addnoHour6 = forecast_expert_addlag1421_addTueFri_plusLastHour_withSeason_addNoHour6(Y, days)
    forecasts[n, :, 6] = fc_expert_addlag1421_addTueFri_plusLastHour_withSS_addnoHour6["forecasts"]
    coeffs_list["expert_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6"][:, n] = fc_expert_addlag1421_addTueFri_plusLastHour_withSS_addnoHour6["coefficients"].transpose()
    
    fc_mv_expert_mix_mix_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6 = forecast_mixexpert_sparse_addlag1421_addTueFri_plusLastHour_withSeason_addNoHour6(Y, days)
    forecasts[n, :, 7] = fc_mv_expert_mix_mix_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6["forecasts"]
    coeffs_list["mv_expert_mix_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6"][:, n] = fc_mv_expert_mix_mix_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6["coefficients"]
    
    print("\r-> "+str(np.round((n+1)/N*100, 4))+"% done", end='')

np.save(file="D:/TU_DORTMUND/EEM/out/result/expert_modified", arr=forecasts)

errors_expert = forecasts - forecasts[:, :, 0:1]

MAE_expert_hourly = pd.DataFrame(np.mean(np.abs(errors_expert), 0))

MAE_expert_hourly.columns = ["true",
               "expert_addlag1421", "expert_addTueFri", "expert_plusLastHour", 
               "expert_withSeason", "expert_addnoHour6", "expert_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6",
               "mv_expert_mix_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6"]

MAE_expert_hourly['expert.REDADV'] = MAE_benchmark_hourly['expert.REDADV']

fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(MAE_expert_hourly.iloc[:, 1:])
plt.title("MAE of modified expert models vs the best benchmark model (expert.REDADV) over hours")
plt.legend(["expert_addlag1421", "expert_addTueFri", "expert_plusLastHour", 
               "expert_withSeason", "expert_addnoHour6", "expert_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6",
               "mv_expert_mix_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6", "expert.REDADV"], loc='center left', bbox_to_anchor=(1, .5),
          ncol=1)
plt.xlabel("Hour of the day")
plt.show()


#%% advanced extending models vs advanced extending model with advanced estimation

model_names = ["true",
               "expert.REDADV_ext",
               "expert.REDADV_ext_advEst"]
n_models = len(model_names)

# %%
forecasts = np.empty((N, S, n_models))
forecasts[::] = np.nan


# %%
dim_names = [oos_dates.values, np.arange(S), model_names]


# %%
coeffs_list = {}
coeffs_list["expert.REDADV_ext"] = np.full((28, N, S), np.nan)
coeffs_list["expert.REDADV_ext_advEst"] = np.full((28,N,S), np.nan)

price = data_array[:, :, data.columns[1:] == "Price"][..., 0]

lag1_min = price.min(axis = 1)
lag1_max = price.max(axis = 1)

for n in range(N):
    # 'model' 1 is the true price to forecast (prediction target)
    forecasts[n, :, 0] = price_S.iloc[D+n]

    days = pd.to_datetime(dates_S[n:D+n], utc=True)
    
    Xlag1_max = pd.Series([np.nan])
    Xlag1_max = Xlag1_max.append(pd.Series(lag1_max[n:D+n]))
    Xlag1_min = pd.Series([np.nan])
    Xlag1_min = Xlag1_min.append(pd.Series(lag1_min[n:D+n]))
    
    dat = data_array[n:D+n, :, :]
    
    fc_expert_adv = forecast_expert_REDADV_ext(dat, days, data.columns[1:], Xlag1_min, Xlag1_max)
    forecasts[n, :, 1] = fc_expert_adv["forecasts"]
    coeffs_list["expert.REDADV_ext"][:, n] = fc_expert_adv["coefficients"].transpose()
    
    fc_expert_adv2 = forecast_expert_REDADV_ext_advEstimation(dat, days, data.columns[1:], Xlag1_min, Xlag1_max)
    forecasts[n, :, 2] = fc_expert_adv2["forecasts"]
    coeffs_list["expert.REDADV_ext_advEst"][:, n] = fc_expert_adv2["coefficients"].transpose()
    
    
    print("\r-> "+str(np.round((n+1)/N*100, 4))+"% done", end='')


np.save(file="D:/TU_DORTMUND/EEM/out/result/advancedextent_advancedestimation", arr=forecasts)
np.save(file="D:/TU_DORTMUND/EEM/out/result/expertREADVext_coff", arr=coeffs_list['expert.REDADV_ext'])
np.save(file="D:/TU_DORTMUND/EEM/out/result/expertREADVext_advEst_coff", arr=coeffs_list['expert.REDADV_ext_advEst'])

errors_advanced = forecasts - forecasts[:, :, 0:1]

MAE_advanced_hourly = pd.DataFrame(np.mean(np.abs(errors_advanced), 0))

MAE_advanced_hourly.columns = ["true",
               "expert.REDADV_ext",
               "expert.REDADV_ext_advEst"]

MAE_advanced_hourly['expert.REDADV'] = MAE_benchmark_hourly['expert.REDADV']
MAE_advanced_hourly['expert_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6'] = MAE_expert_hourly['expert_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6']
fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(MAE_advanced_hourly.iloc[:, 1:])
plt.title("MAE of advanced models vs the best benchmark model (expert.REDADV) over hours")
plt.legend(["expert.REDADV_ext",
               "expert.REDADV_ext_advEst", "expert.REDADV", "expert_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6"], loc='center left', bbox_to_anchor=(1, .5),
          ncol=1)
plt.xlabel("Hour of the day")
plt.show()






