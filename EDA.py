# Copyright 2020 Â© Michal Narajewski, Florian Ziel
#%%
'''Exploratory Data Analysis

1. Regressors selection

'''


import locale
import os
import pandas as pd
import numpy as np
from my_functions import DST_trafo
from calendar import month_abbr
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

#%%

data = pd.read_csv('Final_Dataset.csv')
data.drop(['Price_MBA'], axis = 1, inplace = True)
data.dropna(inplace = True)
data.reset_index(drop = True, inplace = True)
price = data['Price']
#%%

time_utc = pd.to_datetime(data["DateTime"],
                          utc=True, format="%Y-%m-%d %H:%M:%S")
time_utc
# %% time as numeric
time_numeric = pd.to_numeric(time_utc)
time_numeric 

# %% convert to local time
local_time_zone = "CET"  # local time zone abbrv
time_lt = time_utc.dt.tz_convert(local_time_zone)
time_lt

S = 24

# %% Save the start and end-time
start_end_time_S = time_lt.iloc[[0, -1]
                                ].dt.tz_localize(None).dt.tz_localize("UTC")
start_end_time_S

# %% creating 'fake' local time
start_end_time_S_num = pd.to_numeric(start_end_time_S)
time_S_numeric = np.arange(
    start=start_end_time_S_num.iloc[0],
    stop=start_end_time_S_num.iloc[1]+24*60*60*10**9/S,
    step=24*60*60*10**9/S
)

# %% 'fake' local time
time_S = pd.Series(pd.to_datetime(time_S_numeric, utc=True))
dates_S = pd.Series(time_S.dt.date.unique())

data_array = DST_trafo(X=data.iloc[:, 1:], Xtime=time_utc,
                       tz=local_time_zone)

# %% save as dataframe
price_S = pd.DataFrame(data_array[..., 0], index=dates_S)


# %% in-sample and forecasting study sizes
D = 365*3
N = price_S.shape[0] - D
index = np.arange(D)
days = pd.to_datetime(dates_S[:365*3], utc=True)


Y = price_S.iloc[index]
Y.index = days
#%%
for s in range(24):
    fig, axs = plt.subplots(2, sharex=True)
    sm.graphics.tsa.plot_acf(Y.values[:, s], lags=50, ax=axs[0],
                             title=f"Sample ACF: hour {s+1}", marker=None)
    sm.graphics.tsa.plot_pacf(Y.values[:, s], lags=50, ax=axs[1],
                              title=f"Sample PACF: hour {s+1}", marker=None)
    #plt.savefig(f"out/EDA/PACF/PACF_{s}")
    plt.show()
    
# %% Cross-Period Dependencies


def get_cpacf(y, k=1):
    S = y.shape[1]
    n = y.shape[0]
    cpacf = np.full((S, S), np.nan)
    for s in range(S):
        for l in range(S):
            y_s = y[k:n, s]
            y_l_lagged = y[:(n-k), l]
            cpacf[s, l] = np.corrcoef(y_s, y_l_lagged)[0, 1]
    return cpacf

# %% plot the matrix
    
k = 1
cpacf = get_cpacf(price_S.tail(5*D).values, k=k)
pd.DataFrame(cpacf).round(2)


plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(15, 15))
cax = ax.matshow(cpacf, cmap=plt.cm.rainbow, vmin=-1, vmax=1)
cb = fig.colorbar(cax, fraction=0.046, pad=0.04)
for (i, j), z in np.ndenumerate(cpacf):
    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
plt.xticks(ticks=np.arange(S), labels=np.arange(S), rotation=45)
plt.yticks(ticks=np.arange(S), labels=np.arange(S))
plt.tight_layout()
plt.xlabel("l")
plt.ylabel("s")
#plt.savefig("out/EDA/cross1")
plt.show()

# %% More on partial correlation


def pcor(x, y, z):
    XREG = np.column_stack((np.ones(z.shape[0]), z))
    model_y = LinearRegression(fit_intercept=False).fit(X=XREG, y=y)
    model_x = LinearRegression(fit_intercept=False).fit(X=XREG, y=x)
    cor = np.corrcoef(y - model_y.predict(XREG), x -
                      model_x.predict(XREG))[0, 1]
    return cor


maxlag = 3
index = np.arange(maxlag, 3*365)
index_ext = np.arange(3*365)

# %% Diagonal + Last
cp_pacf_diag_last = np.full((S, S), np.nan)
for s in range(S):
    for l in range(S):
        x = price_S.iloc[index, s]
        y = price_S.iloc[index-1, l]
        z = np.column_stack(
            (price_S.iloc[index-1, s], price_S.iloc[index-1, S-1])
        )
        cp_pacf_diag_last[s, l] = pcor(x, y, z)

# %% plot the matrix
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(15, 15))
cax = ax.matshow(cp_pacf_diag_last, cmap=plt.cm.rainbow, vmin=-1, vmax=1)
cb = fig.colorbar(cax, fraction=0.046, pad=0.04)
for (i, j), z in np.ndenumerate(cp_pacf_diag_last):
    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
plt.xticks(ticks=np.arange(S), labels=np.arange(S), rotation=45)
plt.yticks(ticks=np.arange(S), labels=np.arange(S))
plt.tight_layout()
plt.xlabel("l")
plt.ylabel("s")
#plt.savefig("out/EDA/partial_diagonal_last")
plt.show()

#%%

wmean = Y.groupby(by=Y.index.strftime("%A"), sort=False).mean()
days_order = [6,0,1,2,3,4,5]
wmean = wmean.iloc[days_order]

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(10, 5))
markers = ["o", "^", "+", "x", "D", "v", "h"]
for day in range(len(markers)):
    plt.plot(wmean.columns, wmean.iloc[day], marker=markers[day],
             markerfacecolor='none')
plt.ylabel("Weekly mean price in EUR/MWh")
plt.xlabel("s, "+str(Y.index.min().date())+" to "+str(Y.index.max().date()))
plt.tight_layout()
plt.grid()
plt.legend(wmean.index)
os.makedirs("out/04_seasonal_structures", exist_ok=True)
#plt.savefig("out/EDA/weekly_mean_sample")
plt.show()

# %% annual seasonal parameters
index = np.arange(365*4)  # 4 years

Y = price_S.iloc[index]
days = pd.to_datetime(dates_S[:365*4], utc=True)
Y.index = days

month_list = [[3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 1, 2]]

season_names = ["+".join([month_abbr[month_num]
                         for month_num in months]) for months in month_list]
season_names

# %%
amean = np.full((S, len(month_list)), np.nan)

for j in range(len(month_list)):
    sindex = Y.index.month.isin(month_list[j])
    amean[:, j] = Y[sindex].mean(0)

# %% plot it now
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(10, 5))
markers = ["o", "^", "+", "x"]
for j in range(len(month_list)):
    plt.plot(np.arange(S), amean[:, j],
             marker=markers[j], markerfacecolor='none')
plt.ylabel("Annual mean price in EUR/MWh")
plt.xlabel("s, "+str(Y.index.min().date())+" to "+str(Y.index.max().date()))
plt.tight_layout()
plt.grid()
plt.legend(season_names)
#plt.savefig("out/EDA/annual_mean_sample")
plt.show()

# %% Weekly sample correlations for expert_last specification
weekdays_num = pd.to_datetime(dates_S, utc=True).dt.weekday+1
WD = np.transpose([(weekdays_num == x) + 0 for x in range(1, 8)])
maxlag = 7
index = index + 7

wd_pacf = np.full((S, 7), np.nan)
for s in range(S):
    for l in range(7):
        x = price_S.iloc[index, s]
        y = WD[index, l]
        z = np.column_stack(
            (price_S.iloc[index-1, s], price_S.iloc[index-2, s],
             price_S.iloc[index-7, s], price_S.iloc[index-1, S-1],
             WD[index][:, [0, 5, 6]])
        )
        wd_pacf[s, l] = pcor(x, y, z)

# %%
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(8, 7))
cax = ax.matshow(wd_pacf, cmap=plt.cm.rainbow, vmin=-1, vmax=1,
                 aspect='auto')
cb = fig.colorbar(cax)
for (i, j), z in np.ndenumerate(wd_pacf):
    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
plt.xticks(ticks=np.arange(7), labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=45)
plt.yticks(ticks=np.arange(S), labels=np.arange(S))
plt.tight_layout()
plt.xlabel("l")
plt.ylabel("s")
#plt.savefig("out/EDA/pacfs_wd")
plt.show()

#%% external regressors

# day-ahead price
price = data_array[:, :, data.columns[1:] == "Price"][..., 0]
# day-ahead load
load = data_array[:, :, data.columns[1:] == "Load_DA"][..., 0]
# EUA
eua = data_array[:, :, data.columns[1:] == "EUA"][...,0]
# EUA
brentOil = data_array[:, :, data.columns[1:] == "BrentOil"][...,0]
# EUA
ttfGas = data_array[:, :, data.columns[1:] == "TTFGas"][...,0]
# EUA
api2Coal = data_array[:, :, data.columns[1:] == "API2Coal"][...,0]

#%% price vs load

fig, axs = plt.subplots(4, figsize=(15, 15.5))

colors = ["C"+str(i) for i in range(7)]
weekdays = pd.to_datetime(dates_S).dt.weekday.values

s= 0
# day-ahead load
for day in np.unique(weekdays):
    axs[0].scatter(load[weekdays == day, s], price[weekdays == day, s],
                   color=colors[day], facecolors='none')
axs[0].set_ylabel("Price in EUR/MWh")
axs[0].set_xlabel("Day-ahead Load, s = "+str(s))


s = 6
# day-ahead load
for day in np.unique(weekdays):
    axs[1].scatter(load[weekdays == day, s], price[weekdays == day, s],
                   color=colors[day], facecolors='none')
axs[1].set_ylabel("Price in EUR/MWh")
axs[1].set_xlabel("Day-ahead Load, s = "+str(s))

s = 12
# day-ahead load
for day in np.unique(weekdays):
    axs[2].scatter(load[weekdays == day, s], price[weekdays == day, s],
                   color=colors[day], facecolors='none')
axs[2].set_ylabel("Price in EUR/MWh")
axs[2].set_xlabel("Day-ahead Load, s = "+str(s))


s = 18
# day-ahead load
for day in np.unique(weekdays):
    axs[3].scatter(load[weekdays == day, s], price[weekdays == day, s],
                   color=colors[day], facecolors='none')
axs[3].set_ylabel("Price in EUR/MWh")
axs[3].set_xlabel("Day-ahead Load, s = "+str(s))

#%% correlation between day-ahead load, EUA, BrentOil,TFFGas, API2Coal and price
l_p = []
eua_p = []
brentoil_p = []
api2coal_p = []
gas_p = []

for s in range(24):
    
    l_p.append(round(np.corrcoef(price[:, s], load[:, s])[0,1],3))
    eua_p.append(round(np.corrcoef(price[:,s], eua[:,s])[0,1],3))
    brentoil_p.append(round(np.corrcoef(price[:,s], brentOil[:,s])[0,1],3))
    api2coal_p.append(round(np.corrcoef(price[:,s], api2Coal[:,s])[0,1],3))
    gas_p.append(round(np.corrcoef(price[:,s], ttfGas[:,s])[0,1],3))
    
external_price_corr = pd.DataFrame()
external_price_corr['Load_EPrice'] = l_p
external_price_corr['EUA_EPrice'] = eua_p
external_price_corr['BrentOil_EPrice'] = brentoil_p
external_price_corr['Gas_EPrice'] = gas_p
external_price_corr = external_price_corr.T



