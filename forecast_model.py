# Copyright 2020 Â© Michal Narajewski, Florian Ziel

import numpy as np
import pandas as pd
from calendar import day_abbr
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from scipy import sparse

## Naive Models ##
def forecast_naive(Y, days):
    weekday = (days.iloc[-1] + pd.DateOffset(1)
               ).weekday() + 1  # 1 - Mon, ... , 7 - Sun
    if weekday in [1, 6, 7]:  # if Mon, Sat, Sun, use the price from previous week
        forecast = Y.values[-7]
    else:  # otherwise, use the price of yesterday
        forecast = Y.values[-1]
    return {"forecasts": forecast, "coefficients": None}

## Expert Models
    
def forecast_expert(Y, days, expert_wd=[1, 6, 7], expert_lags=[1, 2, 7]):
    S = Y.shape[1]
    forecast = np.repeat(np.nan, S)

    days_ext = days.append(pd.Series(days.iloc[-1] + pd.DateOffset(1),
                                     index=[len(days)]))
    # preparation of weekday dummies including the day to forecast
    weekdays_num = days_ext.dt.weekday+1  # 1 = Mon, 2 = Tue, ..., 7 = Sun
    WD = np.transpose([(weekdays_num == x) + 0 for x in expert_wd])

    # preparation of lags :
    def get_lagged(lag, Z):
        return np.concatenate((np.repeat(np.nan, lag), Z[:(len(Z) + 1-lag)]))

    coefs = np.empty((S, len(expert_wd)+len(expert_lags)+1))

    for s in range(S):
        # prepare the Y vector
        YREG = Y.iloc[:, s].values

        # get lags
        XLAG = np.transpose([get_lagged(lag=lag, Z=YREG)
                            for lag in expert_lags])

        # combine to X matrix
        XREG = np.column_stack((np.ones(Y.shape[0]+1), WD, XLAG))

        act_index = ~ np.isnan(XREG).any(axis=1)
        act_index[-1] = False  # no NAs and no last obs
        model = LinearRegression(fit_intercept=False).fit(
            X=XREG[act_index], y=YREG[act_index[:-1]])

        forecast[s] = model.coef_ @ XREG[-1]
        coefs[s] = model.coef_

    regressor_names = ["intercept"]+[
        day_abbr[i-1]
        for i in expert_wd]+["lag "+str(lag)
                             for lag in expert_lags]

    coefs_df = pd.DataFrame(coefs, columns=regressor_names)

    return {"forecasts": forecast, "coefficients": coefs_df}

def forecast_expert_REDADV(dat, days, Xlag1_min, Xlag1_max, reg_names, wd=[1, 6, 7],
                        price_s_lags=[1, 2, 7], fuel_lags=[2]):
    S = dat.shape[1]
    forecast = np.repeat(np.nan, S)

    days_ext = days.append(pd.Series(days.iloc[-1] + pd.DateOffset(1),
                                     index=[len(days)]))
    
    
    # preparation of weekday dummies including the day to forecast
    weekdays_num = days_ext.dt.weekday+1  # 1 = Mon, 2 = Tue, ..., 7 = Sun
    WD = np.transpose([(weekdays_num == x) + 0 for x in wd])
    

    # Names for subsetting:
    da_forecast_names = ["Load_DA"]

    fuel_names = ["EUA", "API2Coal", "BrentOil", "TTFGas"]

    # preparation of lags :
    def get_lagged(Z, lag):
        return np.concatenate((np.repeat(np.nan, lag), Z[:(len(Z) + 1-lag)]))

    # Lag of 2 as end of day data... at d-1
    mat_fuels = np.concatenate([np.apply_along_axis(
        get_lagged, 0, dat[:, 0, reg_names.isin(fuel_names)], lag=l)
        for l in fuel_lags], axis=-1)
    price_last = get_lagged(
        Z=dat[:, S-1, reg_names == "Price"][..., 0], lag=1)

    coefs = np.empty((S, len(wd)+len(price_s_lags) +
                     len(fuel_names)*len(fuel_lags)+len(da_forecast_names) + 2 + 2))
    
   
    for s in range(S):
        # prepare the Y vector
        acty = dat[:, s, reg_names == "Price"][..., 0]

        # get lags
        mat_price_lags = np.transpose([get_lagged(lag=lag, Z=acty)
                                       for lag in price_s_lags])
        mat_da_forecasts = np.concatenate([np.full(
            (1, len(da_forecast_names)), np.nan), dat[:, s, reg_names.isin(
                da_forecast_names)]])

        # combine all regressors to a matrix
        regmat = np.column_stack(
            (np.append(acty, np.nan), np.ones(acty.shape[0]+1), Xlag1_min, Xlag1_max, WD,
             mat_price_lags, price_last, mat_da_forecasts, mat_fuels))

        act_index = ~ np.isnan(regmat).any(axis=1)

        model = LinearRegression(fit_intercept=False).fit(
            X=regmat[act_index, 1:], y=regmat[act_index, 0])

        # deal with singularities
        model.coef_[np.isnan(model.coef_)] = 0

        forecast[s] = model.coef_ @ regmat[-1, 1:]
        coefs[s] = model.coef_

    regressor_names = ["intercept"]+ ["Lag1_min", "Lag1_max"] + [
        day_abbr[i-1] for i in wd]  +["Price lag "+str(lag)
                                    for lag in price_s_lags]+[
        "Price last lag 1"]+da_forecast_names+[
        fuel+" lag "+str(lag) for lag in fuel_lags for fuel in fuel_names]

    coefs_df = pd.DataFrame(coefs, columns=regressor_names)

    return {"forecasts": forecast, "coefficients": coefs_df}

def forecast_expert_plusLastHour(Y, days, expert_wd=[1, 6, 7], expert_lags=[1, 2, 7], no_lasthour = 1):
    S = Y.shape[1]
    forecast = np.repeat(np.nan, S)

    days_ext = days.append(pd.Series(days.iloc[-1] + pd.DateOffset(1),
                                     index=[len(days)]))
    # preparation of weekday dummies including the day to forecast
    weekdays_num = days_ext.dt.weekday+1  # 1 = Mon, 2 = Tue, ..., 7 = Sun
    WD = np.transpose([(weekdays_num == x) + 0 for x in expert_wd])

    # preparation of lags :
    def get_lagged(lag, Z):
        return np.concatenate((np.repeat(np.nan, lag), Z[:(len(Z) + 1-lag)]))
    

    coefs = np.empty((S, len(expert_wd)+len(expert_lags)+1 +  no_lasthour))
    
    XLastHour = pd.Series([np.nan])
    XLastHour = XLastHour.append(Y.iloc[:, S-1])
    
    for s in range(S):
        # prepare the Y vector
        YREG = Y.iloc[:, s].values

        # get lags
        XLAG = np.transpose([get_lagged(lag=lag, Z=YREG)
                            for lag in expert_lags])
        
        # combine to X matrix
        XREG = np.column_stack((np.ones(Y.shape[0] +1), WD, XLAG, XLastHour))

        act_index = ~ np.isnan(XREG).any(axis=1)
        act_index[-1] = False  # no NAs and no last obs
        model = LinearRegression(fit_intercept=False).fit(
            X=XREG[act_index], y=YREG[act_index[:-1]])

        forecast[s] = model.coef_ @ XREG[-1]
        coefs[s] = model.coef_

    regressor_names = ["intercept"]+[
        day_abbr[i-1]
        for i in expert_wd]+["lag "+str(lag)
                             for lag in expert_lags] + ["LastHour"]

    coefs_df = pd.DataFrame(coefs, columns=regressor_names)

    return {"forecasts": forecast, "coefficients": coefs_df}

def forecast_expert_withSeason(Y, days, expert_wd=[1, 6, 7], expert_lags=[1, 2, 7],
                               season_regr = ['DJF', 'MAM', 'JJA', 'SON']):
    S = Y.shape[1]
    forecast = np.repeat(np.nan, S)

    days_ext = days.append(pd.Series(days.iloc[-1] + pd.DateOffset(1),
                                     index=[len(days)]))
    
    month_ext = days_ext.dt.month
    
    SEASONS = np.array(['DJF', 'MAM', 'JJA', 'SON'])
    season = SEASONS[(month_ext // 3) % 4]
            
    # preparation of weekday dummies including the day to forecast
    weekdays_num = days_ext.dt.weekday+1  # 1 = Mon, 2 = Tue, ..., 7 = Sun
    WD = np.transpose([(weekdays_num == x) + 0 for x in expert_wd])
    SS = np.transpose([(season == x) + 0 for x in season_regr])
    SS = np.delete(SS, 3, axis = 1)
    
    # preparation of lags :
    def get_lagged(lag, Z):
        return np.concatenate((np.repeat(np.nan, lag), Z[:(len(Z) + 1-lag)]))

    coefs = np.empty((S, len(expert_wd)+len(expert_lags)+1 +len(season_regr) - 1))

    for s in range(S):
        # prepare the Y vector
        YREG = Y.iloc[:, s].values

        # get lags
        XLAG = np.transpose([get_lagged(lag=lag, Z=YREG)
                            for lag in expert_lags])

        # combine to X matrix
        XREG = np.column_stack((np.ones(Y.shape[0]+1), WD, SS, XLAG))

        act_index = ~ np.isnan(XREG).any(axis=1)
        act_index[-1] = False  # no NAs and no last obs
        model = LinearRegression(fit_intercept=False).fit(
            X=XREG[act_index], y=YREG[act_index[:-1]])

        forecast[s] = model.coef_ @ XREG[-1]
        coefs[s] = model.coef_

    regressor_names = ["intercept"]+[
        day_abbr[i-1]
        for i in expert_wd] + ["Winter", "Spring", "Summer"] + ["lag "+str(lag)
                             for lag in expert_lags]

    coefs_df = pd.DataFrame(coefs, columns=regressor_names)

    return {"forecasts": forecast, "coefficients": coefs_df}

def forecast_expert_addNoHour6(Y, days, expert_wd=[1, 6, 7], 
                               expert_lags=[1, 2, 7]):
    S = Y.shape[1]
    forecast = np.repeat(np.nan, S)

    days_ext = days.append(pd.Series(days.iloc[-1] + pd.DateOffset(1),
                                     index=[len(days)]))
    
            
    # preparation of weekday dummies including the day to forecast
    weekdays_num = days_ext.dt.weekday+1  # 1 = Mon, 2 = Tue, ..., 7 = Sun
    WD = np.transpose([(weekdays_num == x) + 0 for x in expert_wd])

    
    # preparation of lags :
    def get_lagged(lag, Z):
        return np.concatenate((np.repeat(np.nan, lag), Z[:(len(Z) + 1-lag)]))

    coefs = np.empty((S, len(expert_wd)+len(expert_lags)+1 + 6))
    

    for s in range(S):
        # prepare the Y vector
        YREG = Y.iloc[:, s].values

        # get lags
        XLAG = np.transpose([get_lagged(lag=lag, Z=YREG)
                            for lag in expert_lags])
        
        XHour1 = np.repeat(s, Y.shape[0]+1)
        XHour2 = np.repeat(s, Y.shape[0]+1) ** 2
        XHour3 = np.repeat(s, Y.shape[0]+1) ** 3
        XHour4 = np.repeat(s, Y.shape[0]+1) ** 4
        XHour5 = np.repeat(s, Y.shape[0]+1) ** 5
        XHour6 = np.repeat(s, Y.shape[0]+1) ** 6
        
        # combine to X matrix
        XREG = np.column_stack((np.ones(Y.shape[0]+1), WD, XLAG, XHour1, XHour2, XHour3, XHour4, XHour5, XHour6))

        act_index = ~ np.isnan(XREG).any(axis=1)
        act_index[-1] = False  # no NAs and no last obs
        model = LinearRegression(fit_intercept=False).fit(
            X=XREG[act_index], y=YREG[act_index[:-1]])

        forecast[s] = model.coef_ @ XREG[-1]
        coefs[s] = model.coef_

    regressor_names = ["intercept"]+[
        day_abbr[i-1]
        for i in expert_wd] + ["lag "+str(lag)
                             for lag in expert_lags] + ["NoHour1", "NoHour2", "NoHour3", "NoHour4", "NoHour5", "NoHour6"]

    coefs_df = pd.DataFrame(coefs, columns=regressor_names)

    return {"forecasts": forecast, "coefficients": coefs_df}

def forecast_expert_addlag1421_addTueFri_plusLastHour_withSeason_addNoHour6(Y, days, expert_wd=[1, 2, 5, 6, 7], 
                                                      expert_lags=[1, 2, 7, 14, 21], 
                                                      season_regr = ['DJF', 'MAM', 'JJA', 'SON'], 
                                                      no_lasthour = 1):
    S = Y.shape[1]
    forecast = np.repeat(np.nan, S)

    days_ext = days.append(pd.Series(days.iloc[-1] + pd.DateOffset(1),
                                     index=[len(days)]))
    
    month_ext = days_ext.dt.month
    
    SEASONS = np.array(['DJF', 'MAM', 'JJA', 'SON'])
    season = SEASONS[(month_ext // 3) % 4]
            
    # preparation of weekday dummies including the day to forecast
    weekdays_num = days_ext.dt.weekday+1  # 1 = Mon, 2 = Tue, ..., 7 = Sun
    WD = np.transpose([(weekdays_num == x) + 0 for x in expert_wd])
    SS = np.transpose([(season == x) + 0 for x in season_regr])
    SS = np.delete(SS, 3, axis = 1)
    
    # preparation of lags :
    def get_lagged(lag, Z):
        return np.concatenate((np.repeat(np.nan, lag), Z[:(len(Z) + 1-lag)]))

    coefs = np.empty((S, len(expert_wd)+len(expert_lags)+1 +len(season_regr) - 1 + no_lasthour + 6))
    
    XLastHour = pd.Series([np.nan])
    XLastHour = XLastHour.append(Y.iloc[:, S-1])

    for s in range(S):
        # prepare the Y vector
        YREG = Y.iloc[:, s].values

        # get lags
        XLAG = np.transpose([get_lagged(lag=lag, Z=YREG)
                            for lag in expert_lags])
        
        XHour = np.repeat(s, Y.shape[0]+1)
        XHour2 = np.repeat(s, Y.shape[0]+1) ** 2
        XHour3 = np.repeat(s, Y.shape[0]+1) ** 3
        XHour4 = np.repeat(s, Y.shape[0]+1) ** 4
        XHour5 = np.repeat(s, Y.shape[0]+1) ** 5
        XHour6 = np.repeat(s, Y.shape[0]+1) ** 6
        
        # combine to X matrix
        XREG = np.column_stack((np.ones(Y.shape[0]+1), WD, SS, XLAG, XLastHour, XHour, XHour2, XHour3, XHour4, XHour5, XHour6))

        act_index = ~ np.isnan(XREG).any(axis=1)
        act_index[-1] = False  # no NAs and no last obs
        model = LinearRegression(fit_intercept=False).fit(
            X=XREG[act_index], y=YREG[act_index[:-1]])

        forecast[s] = model.coef_ @ XREG[-1]
        coefs[s] = model.coef_

    regressor_names = ["intercept"]+[
        day_abbr[i-1]
        for i in expert_wd] + ["Winter", "Spring", "Summer"] + ["lag "+str(lag)
                             for lag in expert_lags] + ["LastHour"] + ["NoHour", "NoHour2", "NoHour3", "NoHour4", "NoHour5", "NoHour6"]

    coefs_df = pd.DataFrame(coefs, columns=regressor_names)

    return {"forecasts": forecast, "coefficients": coefs_df}


def forecast_mixexpert_sparse_addlag1421_addTueFri_plusLastHour_withSeason_addNoHour6(Y, days, 
                                                                expert_wd=[1, 2, 5, 6, 7], 
                                                                expert_lags=[1, 2, 7, 14, 21], 
                                                                season_regr = ['DJF', 'MAM', 'JJA', 'SON'], 
                                                                no_lasthour = 1):
    D = Y.shape[0]
    S = Y.shape[1]

    regressor_names = ["lag " + str(lag) for lag in expert_lags]
    # days vector including the day to forecast
    days_ext = days.append(
        pd.Series(days.iloc[-1] + pd.DateOffset(1), index=[len(days)])
    )
    
    SEASONS = np.array(['DJF', 'MAM', 'JJA', 'SON'])
    season = SEASONS[(days_ext.dt.month // 3) % 4]
    
    # preparation of weekday dummies including the day to forecast
    weekdays_num = days_ext.dt.weekday + 1  # 1 = Mon, 2 = Tue, ..., 7 = Sun
    WD = np.transpose([(weekdays_num == x) + 0 for x in expert_wd])
    
    SS = np.transpose([(season == x) + 0 for x in season_regr])
    SS = np.delete(SS, 3, axis = 1)
    
    XLastHour = pd.Series([np.nan])
    XLastHour = XLastHour.append(Y.iloc[:, S-1])
    
    def get_lagged(lag, Z):
        return np.concatenate((np.repeat(np.nan, lag), Z[: (len(Z) + 1 - lag)]))

    # list holding regressors that are constant
    XBREGlist = []
    # list holding regressors that vary across 0,...,S-1
    XREGlist = []
    # list holding corresponding dependend variable
    YREGlist = []

    for s in range(S):
        YREGlist.append(Y.iloc[:, s].values)

        XLAG = np.transpose([get_lagged(lag=lag, Z=YREGlist[s])
                            for lag in expert_lags])
        
        XHour = np.repeat(s, D+1)
        XHour2 = np.repeat(s, D+1) ** 2
        XHour3 = np.repeat(s, D+1) ** 3
        XHour4 = np.repeat(s, D+1) ** 4
        XHour5 = np.repeat(s, D+1) ** 5
        XHour6 = np.repeat(s, D+1) ** 6
        
        XBREGlist.append(XLAG)
        XREGlist.append(np.column_stack((np.ones(D + 1), WD, SS, XLastHour, XHour, XHour2, XHour3, XHour4, XHour5, XHour6)))

    regr = ["intercept"] + [day_abbr[i - 1] for i in expert_wd] + ["Winter", "Spring", "Summer"]+ ["LastHour"] + ["NoHour", "NoHour2", "NoHour3", "NoHour4", "NoHour5", "NoHour6" ]
    regressor_names += [
        regressor + "_s" + str(s) for s in range(S) for regressor in regr
    ]

    const_regr_n = np.concatenate(XBREGlist).shape[1]
    vary_regr_n = np.concatenate(XREGlist, axis=-1).shape[1]

    # matrix holding all regressors
    XREG = np.zeros((D * S, const_regr_n + vary_regr_n))
    XREG[:, :const_regr_n] = np.concatenate([array[:D] for array in XBREGlist])

    # regressor matrix out of sample used for forecasting
    XREGoos = np.zeros((S, const_regr_n + vary_regr_n))
    XREGoos[:, :const_regr_n] = np.concatenate(
        [array[D:] for array in XBREGlist])

    YREG = np.concatenate(YREGlist)

    for s in range(S):
        XREG[
            s * D: (s + 1) * D,
            (s * int(vary_regr_n / S) + const_regr_n): (
                (s + 1) * int(vary_regr_n / S) + const_regr_n
            ),
        ] = XREGlist[s][:D]
        XREGoos[
            s,
            (s * int(vary_regr_n / S) + const_regr_n): (
                (s + 1) * int(vary_regr_n / S) + const_regr_n
            ),
        ] = XREGlist[s][D:]

    # index without NA's
    act_index = ~np.isnan(XREG).any(axis=1)

    XREG_sparse = sparse.lil_matrix(XREG[act_index])

    model = LinearRegression(fit_intercept=False).fit(
        X=XREG_sparse, y=YREG[act_index])

    # compute forecast
    forecast = XREGoos @ model.coef_
    coefs_df = pd.DataFrame([model.coef_], columns=regressor_names)

    return {"forecasts": forecast, "coefficients": coefs_df}

def forecast_expert_REDADV_ext_advEstimation(dat, days, reg_names, Xlag1_min, Xlag1_max, wd=[1, 2, 5, 6, 7],
                        price_s_lags=[1, 2, 7, 14, 21], fuel_lags=[2],
                        season_regr = ['DJF', 'MAM', 'JJA', 'SON']):
    D = dat.shape[0]
    S = dat.shape[1]
    forecast = np.repeat(np.nan, S)

    days_ext = days.append(pd.Series(days.iloc[-1] + pd.DateOffset(1),
                                     index=[len(days)]))
    
    month_ext = days_ext.dt.month
    
    SEASONS = np.array(['DJF', 'MAM', 'JJA', 'SON'])
    season = SEASONS[(month_ext // 3) % 4]
    SS = np.transpose([(season == x) + 0 for x in season_regr])
    SS = np.delete(SS, 3, axis = 1)
    # preparation of weekday dummies including the day to forecast
    weekdays_num = days_ext.dt.weekday+1  # 1 = Mon, 2 = Tue, ..., 7 = Sun
    WD = np.transpose([(weekdays_num == x) + 0 for x in wd])
    
    
    # Names for subsetting:
    da_forecast_names = ["Load_DA"]

    fuel_names = ["EUA", "API2Coal", "BrentOil", "TTFGas"]

    # preparation of lags :
    def get_lagged(Z, lag):
        return np.concatenate((np.repeat(np.nan, lag), Z[:(len(Z) + 1-lag)]))

    # Lag of 2 as end of day data... at d-1
    mat_fuels = np.concatenate([np.apply_along_axis(
        get_lagged, 0, dat[:, 0, reg_names.isin(fuel_names)], lag=l)
        for l in fuel_lags], axis=-1)
    price_last = get_lagged(
        Z=dat[:, S-1, reg_names == "Price"][..., 0], lag=1)

    coefs = np.empty((S, len(wd)+len(price_s_lags) +
                     len(fuel_names)*len(fuel_lags)+len(da_forecast_names)+len(season_regr) -1  + 2 + 6 + 2))
    
   
    for s in range(S):
        # prepare the Y vector
        acty = dat[:, s, reg_names == "Price"][..., 0]
        
        XHour1 = np.repeat(s, D+1)
        XHour2 = np.repeat(s, D+1) ** 2
        XHour3 = np.repeat(s, D+1) ** 3
        XHour4 = np.repeat(s, D+1) ** 4
        XHour5 = np.repeat(s, D+1) ** 5
        XHour6 = np.repeat(s, D+1) ** 6
        
        # get lags
        mat_price_lags = np.transpose([get_lagged(lag=lag, Z=acty)
                                       for lag in price_s_lags])
        mat_da_forecasts = np.concatenate([np.full(
            (1, len(da_forecast_names)), np.nan), dat[:, s, reg_names.isin(
                da_forecast_names)]])

        # combine all regressors to a matrix
        regmat = np.column_stack(
            (np.append(acty, np.nan), Xlag1_min, Xlag1_max, WD, SS,
             mat_price_lags, price_last, mat_da_forecasts, mat_fuels))

        act_index = ~ np.isnan(regmat).any(axis=1)
        
        regmat_mean = np.mean(regmat[act_index], axis=0)
        regmat_std = np.std(regmat[act_index], axis=0)
        
        regmat_scaled = (regmat[act_index] - regmat_mean)/regmat_std
        regmat_scaled = np.column_stack((regmat_scaled, XHour1[act_index], XHour2[act_index], XHour3[act_index], XHour4[act_index], XHour5[act_index], XHour6[act_index]))
        
        regmat = np.column_stack((regmat, XHour1, XHour2, XHour3, XHour4, XHour5, XHour6))
        regmat_mean = np.append(regmat_mean, [0,0,0,0,0,0])
        regmat_std = np.append(regmat_std, [1,1,1,1,1,1])
        
        lambdas = 2**(np.linspace(0.5, -10, 100))
        
        nfold = 10
        mod0cv = linear_model.LassoCV(alphas=lambdas, cv=nfold).fit(
            X=regmat_scaled[:,1:], y=regmat_scaled[:,0])

        forecast[s] = np.hstack([1, (regmat[-1, 1:]-regmat_mean[1:])/regmat_std[1:]]) @ np.hstack(
                        [mod0cv.intercept_, mod0cv.coef_]) * regmat_std[0] + regmat_mean[0]
        coefs[s] = np.hstack([mod0cv.intercept_, mod0cv.coef_])

    regressor_names = ["intercept"]+ ["Xlag1_min", "Xlag1_max"] +[
        day_abbr[i-1] for i in wd] + ["Winter", "Spring", "Summer"] +["Price lag "+str(lag)
                                    for lag in price_s_lags]+[
        "Price last lag 1"]+da_forecast_names+[
        fuel+" lag "+str(lag) for lag in fuel_lags for fuel in fuel_names] +["NoHour", "NoHour2", "NoHour3", "NoHour4", "NoHour5", "NoHour6" ]

    coefs_df = pd.DataFrame(coefs, columns=regressor_names)

    return {"forecasts": forecast, "coefficients": coefs_df}


def forecast_expert_REDADV_ext(dat, days, reg_names, Xlag1_min, Xlag1_max, wd=[1, 2, 5, 6, 7],
                        price_s_lags=[1, 2, 7, 14, 21], fuel_lags=[2],
                        season_regr = ['DJF', 'MAM', 'JJA', 'SON']):
    D = dat.shape[0]
    S = dat.shape[1]
    forecast = np.repeat(np.nan, S)

    days_ext = days.append(pd.Series(days.iloc[-1] + pd.DateOffset(1),
                                     index=[len(days)]))
    
    month_ext = days_ext.dt.month
    
    SEASONS = np.array(['DJF', 'MAM', 'JJA', 'SON'])
    season = SEASONS[(month_ext // 3) % 4]
    SS = np.transpose([(season == x) + 0 for x in season_regr])
    SS = np.delete(SS, 3, axis = 1)
    # preparation of weekday dummies including the day to forecast
    weekdays_num = days_ext.dt.weekday+1  # 1 = Mon, 2 = Tue, ..., 7 = Sun
    WD = np.transpose([(weekdays_num == x) + 0 for x in wd])
    
    
    # Names for subsetting:
    da_forecast_names = ["Load_DA"]

    fuel_names = ["EUA", "API2Coal", "BrentOil", "TTFGas"]

    # preparation of lags :
    def get_lagged(Z, lag):
        return np.concatenate((np.repeat(np.nan, lag), Z[:(len(Z) + 1-lag)]))

    # Lag of 2 as end of day data... at d-1
    mat_fuels = np.concatenate([np.apply_along_axis(
        get_lagged, 0, dat[:, 0, reg_names.isin(fuel_names)], lag=l)
        for l in fuel_lags], axis=-1)
    price_last = get_lagged(
        Z=dat[:, S-1, reg_names == "Price"][..., 0], lag=1)

    coefs = np.empty((S, len(wd)+len(price_s_lags) +
                     len(fuel_names)*len(fuel_lags)+len(da_forecast_names)+len(season_regr) -1  + 2 + 6 + 2))
    
   
    for s in range(S):
        # prepare the Y vector
        acty = dat[:, s, reg_names == "Price"][..., 0]
        
        XHour1 = np.repeat(s, D+1)
        XHour2 = np.repeat(s, D+1) ** 2
        XHour3 = np.repeat(s, D+1) ** 3
        XHour4 = np.repeat(s, D+1) ** 4
        XHour5 = np.repeat(s, D+1) ** 5
        XHour6 = np.repeat(s, D+1) ** 6
        
        # get lags
        mat_price_lags = np.transpose([get_lagged(lag=lag, Z=acty)
                                       for lag in price_s_lags])
        mat_da_forecasts = np.concatenate([np.full(
            (1, len(da_forecast_names)), np.nan), dat[:, s, reg_names.isin(
                da_forecast_names)]])

        # combine all regressors to a matrix
        regmat = np.column_stack(
            (np.append(acty, np.nan),np.ones(acty.shape[0]+1), Xlag1_min, Xlag1_max, WD, SS,
             mat_price_lags, price_last, mat_da_forecasts, mat_fuels, XHour1, XHour2, XHour3, XHour4, XHour5, XHour6))
        
        act_index = ~ np.isnan(regmat).any(axis=1)

        model = LinearRegression(fit_intercept=False).fit(
            X=regmat[act_index, 1:], y=regmat[act_index, 0])

        # deal with singularities
        model.coef_[np.isnan(model.coef_)] = 0

        forecast[s] = model.coef_ @ regmat[-1, 1:]
        coefs[s] = model.coef_

    regressor_names = ["intercept"]+ ["Lag1_min", "Lag1_max"] + [
        day_abbr[i-1] for i in wd]  + ["Winter", "Spring", "Summer"] + ["Price lag "+str(lag)
                                    for lag in price_s_lags]+[
        "Price last lag 1"]+da_forecast_names+[
        fuel+" lag "+str(lag) for lag in fuel_lags for fuel in fuel_names] + ["NoHour1", "NoHour2", "NoHour3", "NoHour4", "NoHour5", "NoHour6" ]

    coefs_df = pd.DataFrame(coefs, columns=regressor_names)

    return {"forecasts": forecast, "coefficients": coefs_df}

