

# Copyright 2020 Â© Michal Narajewski, Florian Ziel
import pandas as pd
import numpy as np
from scipy.stats import t
from sklearn.linear_model import LinearRegression


def DST_trafo(X, Xtime, tz="CET"):
    """Converts a time series DataFrame to an DST-adjusted array

    The function takes a DataFrame of D*S rows and N columns and returns
    an array of shape (D,S,N) where D is the number of days, S the number
    of observations per day and N the number of variables. The function deals
    with the DST problem by averaging the additional hour in October and
    interpolating the missing hour in March.

    Parameters
    ----------
    X : DataFrame
        The time series DataFrame of shape (D*S,N) to be DST-adjusted.
    Xtime : datetime Series
        The series of length D*S containing UTC dates corresponding to the
        DataFrame X.
    tz : str
        The timezone to which the data needs to be adjusted to. The current
        implementation was not tested with other timezones than CET.

    Returns
    -------
    ndarray
        an ndarray of DST-adjusted variables of shape (D,S,N).
    """
    Xinit = X.values
    if len(Xinit.shape) == 1:
        Xinit = np.reshape(Xinit, (len(Xinit), 1))

    atime_init = pd.to_numeric(Xtime)
    freq = atime_init.diff().value_counts().idxmax()
    S = int(24*60*60 * 10**9 / freq)
    atime = pd.DataFrame(
        np.arange(start=atime_init.iloc[0], stop=atime_init.iloc[-1]+freq,
                  step=freq))
    idmatch = atime.reset_index().set_index(0).loc[atime_init, "index"].values
    X = np.empty((len(atime), Xinit.shape[1]))
    X[:] = np.nan
    X[idmatch] = Xinit

    new_time = Xtime.dt.tz_convert(tz).reset_index(drop=True)
    DLf = new_time.dt.strftime("%Y-%m-%d").unique()
    days = pd.Series(pd.to_datetime(DLf))

    # EUROPE
    DST_SPRING = pd.to_numeric(days.dt.strftime("%m%w")).eq(
        30) & pd.to_numeric(days.dt.strftime("%d")).ge(25)
    DST_FALL = pd.to_numeric(days.dt.strftime("%m%w")).eq(
        100) & pd.to_numeric(days.dt.strftime("%d")).ge(25)
    DST = ~(DST_SPRING | DST_FALL)

    time_start = new_time.iloc[range(
        S+int(S/24))].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    time_end = new_time.iloc[range(-S-int(S/24), 0)
                             ].dt.strftime("%Y-%m-%d %H:%M:%S %Z")

    Dlen = len(DLf)
    Shift = 2  # for CET

    X_dim = X.shape[1]

    Xout = np.empty((Dlen, S, X_dim))
    Xout[:] = np.nan

    k = 0
    # first entry:
    i_d = 0
    idx = time_start[time_start.str.contains(DLf[i_d])].index
    if DST[i_d]:
        Xout[i_d, S-1-idx[::-1], ] = X[range(k, len(idx)+k), ]
    elif DST_SPRING[i_d]:
        tmp = S-1-idx[::-1]
        # MARCH
        for i_S in range(len(idx)):
            if tmp[i_S] <= Shift * S/24-1:
                Xout[i_d, int(S-S/24 - len(idx) + i_S), ] = X[k+i_S, ]
            if tmp[i_S] == Shift * S/24-1:
                Xout[i_d, range(int(S-S/24 - len(idx) + i_S+1),
                                int(S-S/24 - len(idx) + i_S+1 + S/24)),
                     ] = X[[k+i_S, ]] + np.transpose(
                    np.atleast_2d(np.arange(1, int(S/24)+1)/(len(range(int(
                        S/24)))+1))).dot(X[[k+i_S+1, ]]-X[[k+i_S, ]])
            if tmp[i_S] > Shift * S/24-1:
                Xout[i_d, int(S-S/24 - len(idx) + i_S+S/24), ] = X[k+i_S, ]
    else:
        tmp = S-idx[::-1]
        # OCTOBER
        for i_S in range(len(idx)):
            if tmp[i_S] <= Shift * S/24-1:
                Xout[i_d, int(S+S/24 - len(idx) + i_S), ] = X[k+i_S, ]
            if tmp[i_S] in (Shift*S/24-1 + np.arange(1, int(S/24)+1)):
                Xout[i_d, int(S+S/24 - len(idx)+i_S), ] = 0.5 * \
                    (X[k+i_S, ] + X[int(k+i_S+S/24), ])
            if tmp[i_S] > (Shift+2) * S/24-1:
                Xout[i_d, int(S+S/24 - len(idx) + i_S-S/24), ] = X[k+i_S, ]
    k += len(idx)
    for i_d in range(1, len(DLf)-1):
        if DST[i_d]:
            idx = S
            Xout[i_d, range(idx), ] = X[range(k, k+idx), ]
        elif DST_SPRING[i_d]:
            idx = int(S-S/24)
            # MARCH
            for i_S in range(idx):
                if i_S <= Shift * S/24-1:
                    Xout[i_d, i_S, ] = X[k+i_S, ]
                if i_S == Shift * S/24-1:
                    Xout[i_d, range(int(i_S+1),
                                    int(i_S + 1 + S/24)),
                         ] = X[[k+i_S, ]] + np.transpose(
                        np.atleast_2d(np.arange(1, int(S/24)+1)/(len(range(int(
                            S/24)))+1))).dot(X[[k+i_S+1, ]]-X[[k+i_S, ]])
                if i_S > Shift * S/24-1:
                    Xout[i_d, int(i_S + S/24), ] = X[k+i_S, ]
        else:
            idx = int(S+S/24)
            # October
            for i_S in range(idx):
                if i_S <= Shift * S/24-1:
                    Xout[i_d, i_S, ] = X[k+i_S, ]
                if i_S in (Shift*S/24-1 + np.arange(1, int(S/24)+1)):
                    Xout[i_d, i_S, ] = 0.5*(X[k+i_S, ] + X[int(k+i_S+S/24), ])
                if i_S > (Shift+2) * S/24-1:
                    Xout[i_d, int(i_S-S/24), ] = X[k+i_S, ]
        k += idx
    # last
    i_d = len(DLf)-1
    idx = time_end[time_end.str.contains(DLf[i_d])].index
    if DST[i_d]:
        Xout[i_d, range(len(idx)), ] = X[range(k, k+len(idx)), ]
    elif DST_SPRING[i_d]:
        # MARCH
        for i_S in range(len(idx)):
            if i_S <= Shift * S/24-1:
                Xout[i_d, i_S, ] = X[k+i_S, ]
            if i_S == Shift * S/24-1:
                Xout[i_d, range(int(i_S+1),
                                int(i_S + 1 + S/24)), ] = X[[k+i_S, ]
                                                            ] + np.transpose(
                    np.atleast_2d(np.arange(1, int(S/24)+1)/(len(range(int(
                        S/24)))+1))).dot(X[[k+i_S+1, ]]-X[[k+i_S, ]])
            if i_S > Shift * S/24-1:
                Xout[i_d, int(i_S + S/24), ] = X[k+i_S, ]
    else:
        # OCTOBER
        for i_S in range(len(idx)):
            if i_S <= Shift * S/24-1:
                Xout[i_d, i_S, ] = X[k+i_S, ]
            if i_S in (Shift*S/24-1 + np.arange(1, int(S/24)+1)):
                Xout[i_d, i_S, ] = 0.5*(X[k+i_S, ] + X[int(k+i_S+S/24), ])
            if i_S > (Shift+2) * S/24-1:
                Xout[i_d, int(i_S-S/24), ] = X[k+i_S, ]
    return Xout


def get_pbas(Bindex, period=365.24, dK=365.24/6, order=4):
    """Estimates periodic B-splines to model the annual periodicity

    Parameters
    ----------
    Bindex : array_like of int
        The array of day numbers for which to estimate the B-splines.
    period : float
        The period of B-splines. By default set to 365.24.
    dK : float
        The equidistance distance used to calculate the knots.
    order : int
        The order of the B-splines. 3 indicates quadratic splines, 4 cubic etc.

    Returns
    -------
    ndarray
        an ndarray of estimated B-splines.
    """
    # ord=4 --> cubic splines
    # dK = equidistance distance
    # support will be 1:n
    n = len(Bindex)
    stp = dK
    x = np.arange(1, period)  # must be sorted!
    lb = x[0]
    ub = x[-1]
    knots = np.arange(lb, ub+stp, step=stp)
    degree = order-1
    Aknots = np.concatenate(
        (knots[0] - knots[-1] + knots[-1-degree:-1], knots,
         knots[-1] + knots[1:degree+1] - knots[0]))

    from bspline import Bspline
    bspl = Bspline(Aknots, degree)
    basisInterior = bspl.collmat(x)
    basisInteriorLeft = basisInterior[:, :degree]
    basisInteriorRight = basisInterior[:, -degree:]
    basis = np.column_stack(
        (basisInterior[:, degree:-degree],
         basisInteriorLeft+basisInteriorRight))
    ret = basis[np.array(Bindex % basis.shape[0], dtype="int"), :]
    return ret


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


def pcor(y, x, z):
    XREG = np.column_stack((np.ones(z.shape[0]), z))
    model_y = LinearRegression(fit_intercept=False).fit(X=XREG, y=y)
    model_x = LinearRegression(fit_intercept=False).fit(X=XREG, y=x)
    cor = np.corrcoef(y - model_y.predict(XREG),
                      x - model_x.predict(XREG))[0, 1]
    return cor


def hill(data, start=14, end=None, abline_y=None, ci=0.95, ax=None):
    """Hill estimator translation from R package evir::hill

    Plot the Hill estimate of the tail index of heavy-tailed data, or of an 
    associated quantile estimate.

    Parameters
    ----------
    data : array_like
        data vector
    start : int
        lowest number of order statistics at which to plot a point
    end : int, optional
        highest number of order statistics at which to plot a point
    abline_y : float, optional
        value to be plotted as horizontal straight line
    ci : float
        probability for asymptotic confidence band
    ax : Axes, optional
        the Axes in which to plot the estimator
    """
    ordered = np.sort(data)[::-1]
    ordered = ordered[ordered > 0]
    n = len(ordered)
    k = np.arange(n)+1
    loggs = np.log(ordered)
    avesumlog = np.cumsum(loggs)/k
    xihat = np.hstack([np.nan, (avesumlog-loggs)[1:]])
    alphahat = 1/xihat
    y = alphahat
    ses = y/np.sqrt(k)
    if end is None:
        end = n-1
    x = np.arange(np.min([end, len(data)-1]), start, -1)
    y = y[x]
    qq = norm.ppf(1 - (1-ci)/2)
    u = y + ses[x] * qq
    l = y - ses[x] * qq
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x, y, color='black', linewidth=1)
    ax.plot(x, u, color='red', linestyle='--', linewidth=1)
    ax.plot(x, l, color='red', linestyle='--', linewidth=1)
    if abline_y is not None:
        ax.axhline(abline_y, color='C0', linewidth=1)
    ax.set_ylabel('alpha (CI, p = '+str(ci)+")")
    ax.set_xlabel("Order Statistics")
