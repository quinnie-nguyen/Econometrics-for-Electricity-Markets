# Copyright 2020 Â© Michal Narajewski, Florian Ziel
# %% load packages
import locale
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from forecast_model import *
from scipy.stats import t


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

#%% loading the forecasted prices

forecasts_benchmark = np.load('D:/TU_DORTMUND/EEM/out/result/benchmark.npy')
forecasts_expert = np.load('D:/TU_DORTMUND/EEM/out/result/expert_modified.npy')
forecasts_advanced = np.load('D:/TU_DORTMUND/EEM/out/result/advancedextent_advancedestimation.npy')

#%%
errors_benchmark = forecasts_benchmark - forecasts_benchmark[:, :, 0:1]
MAE_benchmark_hourly = pd.DataFrame(np.mean(np.abs(errors_benchmark), 0))
MAE_benchmark_hourly.columns = ["true", "naive",
               "expert", "expert.REDADV"]

#%%
forecasts = np.dstack((forecasts_expert, forecasts_advanced[:,:,1:]))
errors = forecasts - forecasts[:, :, 0:1]
MAE_hourly = pd.DataFrame(np.mean(np.abs(errors), 0))

navive_combi = np.full((1187, 24, 1), np.nan)
ls_combi = np.full((1187, 24, 1), np.nan)

for s in range(24):
    weights_naive = np.ones(9)/9
    combination_naive = np.expand_dims(
        weights_naive, axis=0) @ forecasts[:, s, 1:].transpose()
    navive_combi[:,s, 0] = combination_naive
    weights_naive_ls = np.hstack([0,0,0.25,0,0,0.25,0,0.25,0.25])
    # same weight on long and short term
    combination_naive_ls = np.expand_dims(
        weights_naive_ls, axis=0) @ forecasts[:, s, 1:].transpose()
    ls_combi[:,s, 0] = combination_naive_ls

error_combi = navive_combi - forecasts[:, :, 0:1]
error_combi_ls = ls_combi - forecasts[:, :, 0:1]

arr_naive = []
    
for i in range(len(np.mean(np.abs(error_combi), 0))):
    
    arr_naive.append(np.mean(np.abs(error_combi), 0)[i][0])
    
arr_ls = []
    
for i in range(len(np.mean(np.abs(error_combi_ls), 0))):
    
    arr_ls.append(np.mean(np.abs(error_combi_ls), 0)[i][0])
    
MAE_hourly['naive_combination'] = arr_naive

MAE_hourly['longshort_combination'] = arr_ls

MAE_hourly.columns = ["true",
               "expert_addlag1421", "expert_addTueFri", "expert_plusLastHour", 
               "expert_withSeason", "expert_addnoHour6", "expert_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6",
               "mv_expert_mix_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6",
               "expert.REDADV_ext",
               "expert.REDADV_ext_advEst",
               "naive_combination", "longshort_combination"]
    
fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(MAE_hourly.iloc[:, 1:])
plt.title("MAE of models over hours")
plt.legend(["expert_addlag1421", "expert_addTueFri", "expert_plusLastHour", 
            "expert_withSeason", "expert_addnoHour6", "expert_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6",
            "mv_expert_mix_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6",
            "expert.REDADV_ext",
            "expert.REDADV_ext_advEst",
            "naive_combination", "longshort_combination"], loc='center left', bbox_to_anchor=(1, .5),
          ncol=1)
plt.xlabel("Hour of the day")
plt.show()

#%% comparison between top best models

fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(MAE_hourly.iloc[:, [3,6,8,9,11]])
plt.title("MAE of the best models over hours")
plt.legend(["expert_plusLastHour",
            "expert_addlag1421_addTueFri_plusLastHour_withSeason_addnoHour6",
            "expert.REDADV_ext",
            "expert.REDADV_ext_advEst",
            "longshort_combination"], loc='center left', bbox_to_anchor=(1, .5),
          ncol=1)
plt.xlabel("Hour of the day")
plt.show()

#%% dm test

def dm_test(error_a, error_b, hmax=1, power=1):
    # as dm_test with alternative == "less"
    loss_a = (np.abs(error_a)**power).sum(1)**(1/power)
    loss_b = (np.abs(error_b)**power).sum(1)**(1/power)
    delta = loss_a - loss_b
    # estimation of the variance
    delta_var = np.var(delta) / delta.shape[0]
    statistic = delta.mean() / np.sqrt(delta_var)
    delta_length = delta.shape[0]
    k = ((delta_length + 1 - 2 * hmax + (hmax / delta_length)
         * (hmax - 1)) / delta_length)**(1 / 2)
    statistic = statistic * k
    p_value = t.cdf(statistic, df=delta_length-1)

    return {"stat": statistic, "p_val": p_value}

#%% 
    
errors = np.dstack((errors, error_combi, error_combi_ls))



for i in range(11):
    
    print(dm_test(errors[...,11], errors[...,i]))
    print('--')

# expert.REDADV_ext_advEst VS expert.REDADV_ext
dm_test(errors[...,9], errors[...,8])



#%% comparison between benchmark models and final model

MAE_benchmark_hourly['longshort_combination'] = arr_ls

fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(MAE_benchmark_hourly.iloc[:, 1:])
#plt.title("MAE of final model vs benchmark models over hours")
plt.legend(["naive","expert", "expert.REDADV", "longshort_combination"], loc='center left', bbox_to_anchor=(1, .5),
          ncol=1)
plt.xlabel("Hour of the day")
plt.show()    
    

errors_benchmark = np.dstack((errors_benchmark, error_combi_ls))
# %% Now we want to test every model vs. every model
errors_wo_true = errors_benchmark[..., 1:]

# long-short combination models VS Naive model
dm_test(errors_wo_true[...,3], errors_wo_true[...,0])

# long-short combination models VS Expert model
dm_test(errors_wo_true[...,3], errors_wo_true[...,1])

# long-short combination models VS Expert.REDADV model
dm_test(errors_wo_true[...,3], errors_wo_true[...,2])

    

