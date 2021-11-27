r"""
This file contains the functions for automatic methods for determining the growth rates.

1. mode fitting
2. maximum concentration
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from datetime import timedelta, datetime
from astropy import modeling
from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor

from .utils import num2time, time2num, mkdirs
from .visualize import psd2im


def get_SE(df, mask):
    r"""Get the start and end times (one-day or two-day)."""

    pos = np.where(mask)
    xmin, xmax = np.min(pos[0]), np.max(pos[0])

    start_time = df.index[xmin]
    end_time = df.index[xmax]
    return start_time, end_time

############################################
#              mode fitting
############################################
def multi_gaussians(x, *params):
    r"""
    This function builds a combination of several Gaussian curves to fit data.

    Parameters:
        params (list) -- a list of parameters for Gaussian curve fitting. [10, 0, 1, 3, 5, 2]. 
                         y1 = 10 * exp(-((x-0)/1)^2)
                         y2 = 3 * exp(-((x-5)/2)^2)
                         y = y1 + y2

    References:
        1. https://stackoverflow.com/questions/26902283/fit-multiple-gaussians-to-the-data-in-python
    """

    y = np.zeros_like(x)
    # print(len(params))
    for i in range(0, len(params), 3):
        # the amp is not the `1/sqrt(2*pi*sigma^2)`, but with a scale factor
        amp = params[i]
        mu = params[i+1]
        std = params[i+2]    # note that this is not the real std
        y += amp * np.exp(-((x-mu)/std)**2)
    return y


def fit_multi_gaussians(log_dps, arr):
    r"""
    fit multiple Gaussian distributions.
    
    The first choice is to fit multi log-norm curves, but since we use the log dps, 
    then it is equal to the log norm curves if the multi Gaussian curves are used.
    
    Parameters:
        log_dps (array) -- log dps
        arr     (array) -- psd at a specific timepoint

    References:
        1. https://www.researchgate.net/publication/227944272_Evaluation_of_an_automatic_algorithm_for_fitting_the_particle_number_size_distribution
        2. https://doi.org/10.1016/j.atmosenv.2006.03.053
    """
    
    # get rid of zeros and NaNs
    idx_arr = arr > 0

    log_dps = log_dps[idx_arr]
    arr = arr[idx_arr]
    try:
        # guess the values for curve fitting
        peak_idx, _ = find_peaks(arr)
        idx = [
            peak_idx[arr[peak_idx].argmin()],
            peak_idx[arr[peak_idx].argmax()],
            np.where(arr == np.median(arr[peak_idx]))[0].item()
        ]
    except:        
        idx = [
                0,
                len(arr)-1,
                int(len(arr)/2)
            ]
       

    params = []
    for ix in idx:
        params += [arr[ix], log_dps[ix], 1]
    
    try:
        popt, pcov = curve_fit(
            multi_gaussians, log_dps, arr, p0=params, maxfev=10000)

        dps = (np.exp(popt[1::3])*1e9).tolist()
        # print(dps)
        return [dp for dp in dps if (dp < 1000) and (dp > 2)]
    except:
        return None


def mask2minmaxdp(aligned_mask, dps):
    r"""convert the mask to the minimum and maximum dps for a specific timestamp.
    
    Parameters:
        aligned_mask (array) -- the detected mask (binary with 0s and 1s)
        dps (list)           -- a list containing dps (nm)
    
    Returns:
        a list, and each element in this list is a list containing two elements (minimum dp and maximum dp)

    References:
        https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function/43094244
    """
    arr = np.vstack(np.where(aligned_mask)).T

    return [[dps[item[0]], dps[item[-1]]]
            for item in np.split(arr[:, 1], np.unique(arr[:, 0], return_index=True)[1][1:])]


#########################################
#    maximum concentration
#########################################
def fit_gaussian(x, y):

    fitter = modeling.fitting.LevMarLSQFitter()
    #fitter = modeling.fitting.LinearLSQFitter()
    model = modeling.models.Gaussian1D()
    try:
        fitted_model = fitter(model, x, y)
        return fitted_model
    except Exception:
        return None


def get_max_time(arr1d, tm, time_resolution=10):

    # find nonzero values
    arr = arr1d[arr1d > 0]

    num = len(arr)

    if num == 0:
        return np.nan
    else:

        fitted_model = fit_gaussian(range(num), arr)
        if fitted_model is None:
            return np.nan
        else:
            mean_val = fitted_model.mean.value
            if mean_val > num or mean_val < 0:
                return np.nan
            else:
                arr1d_ = np.nan_to_num(arr1d)
                max_time = np.nonzero(arr1d_)[0][0] + mean_val
                dt64 = tm[int(max_time)]
                decimal = np.around(time_resolution*(max_time - int(max_time)))
                ts = (dt64 - np.datetime64('1970-01-01T00:00:00')) / \
                    np.timedelta64(1, 's')
                time = datetime.utcfromtimestamp(
                    ts) + timedelta(minutes=int(decimal))
                return time


def get_maxcon(df, mask, tm_res):

    values = df.values * mask
    tms = df.index
    dps = [float(dp) for dp in list(df.columns)]

    # find the indexes of the start and end times
    col_min = np.where((np.array(dps) >= 2.5) == True)[0][0]
    col_max = np.where((np.array(dps) <= 25.9) == True)[0][-1]

    # obtain the time (index) when max concentration occurs for each dp
    peak_con = np.array([[get_max_time(values[:, i], tms, tm_res), dps[i]]
                        for i in range(col_min, col_max+1)])

    # drop the nan values
    tm_dp = pd.DataFrame(peak_con).dropna(how='any').values
    tm_dp = tm_dp[tm_dp[:, 1]<=25.5, :]
    return tm_dp


def get_modefitting(df, mask):

    st, et = get_SE(df, mask)

    df_ = df[(df.index >= st) & (df.index <= et)]
    df_ = df_[[col for col in df_.columns if float(col) < 1000]]

    dps = [float(dp) for dp in list(df_.columns)]
    valid_dps = mask2minmaxdp(mask, dps)
    log_dps = np.array([np.log(dp*1e-9) for dp in dps])
    vals = df_.values
    tm_dp = []

    for i in range(vals.shape[0]):
        tm = df_.index[i]       
        fitted_dps = fit_multi_gaussians(log_dps, vals[i])
        if fitted_dps is not None:

            # check whether the dps are in the banana region
            fitted_dps = [dp for dp in fitted_dps if (
                dp <= valid_dps[i][-1]) and (dp >= valid_dps[i][0])]

            # find the fitted dps related to higher concentrations
            if len(fitted_dps) == 0:
                continue
            elif len(fitted_dps) == 1:
                tm_dp += [[tm, fitted_dps[0]]]
            else:
                cons = [vals[i, np.argmin(np.abs(np.array(dps)-dp))]
                        for dp in fitted_dps]
                dp = fitted_dps[np.argmax(np.array(cons))]
                # for dp in fitted_dps:
                tm_dp += [[tm, dp]]
    tm_dp = pd.DataFrame(tm_dp).dropna(how='any').values
    tm_dp = tm_dp[tm_dp[:, 1]<=25.5, :]

    return tm_dp


def get_size_gr(tm_dp, name='modefitting'):
    r"""
    Get the GR (slope * 60) and the intercept (for visualization)

    Parameters:
        tm_dp (2d array) -- the 1st column is the time points and the 2nd column is the dps (nm)
        name (str)       -- used as the keys for the return dictionary

    Parameters:
        a dictionary containing grs (3-10, 10-25, 3-25)
    """
    gr_dict = {}
    tms = tm_dp[:, 0]
    dt = str(tms[0].date())
    tms = np.array([time2num(tm) for tm in tms])   # convert the time points to numbers
    dps = tm_dp[:, 1]
    x_pred, y_pred = None, None

    for min_dp, max_dp in [[2.5, 10.5], [9.5, 25.5], [2.5, 25.5]]:
        idx = (dps >= min_dp) & (dps <= max_dp)
        if sum(idx) < 2:
            gr_dict.update({f'{name}_{int(min_dp)+1}_{int(max_dp)}':None})
                # , f'inter_{name}_{int(min_dp)+1}_{int(max_dp)}':None})
            continue
        if (min_dp == 2.5) and (max_dp == 25.5) and (name == 'modefitting'):
            fit_dps = tm_dp[:, 1]
            idx_list = [np.where((fit_dps >= dp_min) & (fit_dps < dp_min+1))[0].tolist()[:1]
                        for dp_min in np.arange(2.5, 25.5)]
            idx = []
            for item in idx_list:
                idx += item

            idx_arr = np.array(idx)
            tm_dp_ = tm_dp[idx_arr]
            tms_ = np.array([time2num(tm) for tm in tm_dp_[:, 0]])
            dps_ = tm_dp_[:, 1]
            slope, intercept = fit_GR(tms_, dps_, name)
        else:
            tms_ = tms[idx]
            dps_ = dps[idx]
            slope, intercept = fit_GR(tms_, dps_, name)
        
        if slope is not None:
            gr_dict.update({f'{name}_{int(min_dp)+1}_{int(max_dp)}': slope*60})
                    #    f'inter_{name}_{int(min_dp)+1}_{int(max_dp)}': intercept
            if (min_dp == 2.5) and (max_dp == 25.5):
                # plot the fitted curves
                num_min, num_max = (min_dp - intercept) / \
                    slope, (max_dp - intercept) / slope
                num_min, num_max = int(np.maximum(tms[0], num_min)), int(
                    np.minimum(tms[-1], num_max))
                x_num = np.arange(num_min, num_max)
                x_pred = np.array([num2time(dt, item) for item in x_num])
                y_pred = np.array([slope*item+intercept for item in x_num])
        else:
            gr_dict.update({f'{name}_{int(min_dp)+1}_{int(max_dp)}': None})
                            # f'inter_{name}_{int(min_dp)+1}_{int(max_dp)}': None})        

    return gr_dict, x_pred, y_pred


def fit_GR(tms, dps, name=None):
    r"""
    Fit the GR by the ransac, theil-sen, least-squre regression fitting.

    Parameters:
        tms (array)    -- time points (numbers)
        dps (array)    -- dps (unit: nm) 
    
    Returns:
        slope, intercept -- the gr (slope*60) and intercept (for visulization on the surface plots)
    """
    # thei = TheilSenRegressor(random_state=42)
    # thei.fit(tms.reshape(-1, 1), dps)
    # slope = thei.coef_[0]
    # intercept = thei.intercept_
   
    try:
        ransac = RANSACRegressor(random_state=42)
        ransac.fit(tms.reshape(-1, 1), dps)
        slope = ransac.estimator_.coef_[0]
        intercept = ransac.estimator_.intercept_
        # method = 'RANSAC'   
        # thei = TheilSenRegressor(random_state=42)
        # thei.fit(tms.reshape(-1, 1), dps)
        # slope = thei.coef_[0]
        # intercept = thei.intercept_
    except:
        ols = LinearRegression()
        ols.fit(tms.reshape(-1, 1), dps)
        slope = ols.coef_[0]
        intercept = ols.intercept_
        # method = 'LinearRegression'
    if slope <= 0:
        return None, None
    return slope, intercept


def get_GR(df, mask, tm_res=10, savefp=None, vmax=1e4):
    r"""
    Get the growth rates (GRs) through the mode fitting and maximum concentration methods.
    GRs: 3-7, 7-15, 15-25, 3-25 with different methods

    The automatic mode fitting method does not work well on some events. If you do want to use
    it, please uncomment related lines and check the detection results.

    Parameters:
        df (dataframe)  -- one-day data
        mask (array)    -- the detected mask
        tm_res (float)  -- the time resolution of the dataset

    Returns:
        gr_dict (dict)  -- dictionary containing the determined GRs
        x_pred (array)  -- time information for the visualization of GR 
        y_pred (array)  -- dps for the visualization of GR
    """
   
    maxcon_tm_dp = get_maxcon(df, mask, tm_res)
    # modefitting_tm_dp = get_modefitting(df, mask)
    maxcon_dict, maxcon_x_pred, maxcon_y_pred = get_size_gr(maxcon_tm_dp, 'maxcon')    
    # modefitting_dict, modefitting_x_pred, modefitting_y_pred = get_size_gr(modefitting_tm_dp, 'modefitting')
    # gr_dict = {**maxcon_dict, **modefitting_dict}
    if (savefp is not None) and (maxcon_x_pred is not None):
        mkdirs(os.path.join(savefp, 'maxcon'))
        # mkdirs(os.path.join(savefp, 'modefitting'))
        _ = psd2im(df, figsize=(12, 8), fit_data=[maxcon_tm_dp[:, 0], maxcon_tm_dp[:, 1]*1e-9], line_data=[
                   maxcon_x_pred, maxcon_y_pred*1e-9], savefp=os.path.join(savefp, 'maxcon'), vmax=vmax, dpi=100, use_cbar=True)
        # _ = psd2im(df, figsize=(12, 8), fit_data=[modefitting_tm_dp[:, 0], modefitting_tm_dp[:, 1]*1e-9], line_data=[
        #            modefitting_x_pred, modefitting_y_pred*1e-9], savefp=os.path.join(savefp, 'modefitting'), vmax=vmax, dpi=100, use_cbar=True)
    # return gr_dict
    return maxcon_dict


def get_GR_old(df, mask, dp_min=2.5, dp_max=25.5, savefp=None, tm_res=10, vmax=1e4):
    r"""
    Obtain the GRs.

    Parameters:
        df (dataframe)  -- one-day data
        mask (array)    -- the detected mask
        dp_min (float)  -- the start dp for the determination of GRs
        dp_max (float)  -- the end dp for the determination of GRs
        savefp (path)   -- save path
        tm_res (int)    -- the time resolution. Default is 10 minutes
        vmax (float)    -- maximum for normalization of colors
   
    Returns:
        GR (float)      -- automatically determined GR
        x_pred (array)  -- time information for the visualization of GR 
        y_pred (array)  -- dps for the visualization of GR
    """

    # get the date
    dt = str(df.index[0].date())

    # get the values of the banana-shape region
    values = df.values * mask

    # get the dps
    dps = [float(dp) for dp in list(df.columns)]

    # find the indexes of the start and end times
    col_min = np.where((np.array(dps) >= dp_min) == True)[0][0]
    col_max = np.where((np.array(dps) <= dp_max) == True)[0][-1]

    # obtain the time (index) when max concentration occurs for each dp
    peak_con = np.array([[get_max_time(values[:, i], df.index.values, tm_res), dps[i]]
                        for i in range(col_min, col_max+1)])

    # drop the nan values
    tm_con_mat = pd.DataFrame(peak_con).dropna(how='any').values

    # split the time and dps
    x_tm = np.array([time2num(item) for item in tm_con_mat[:, 0]])
    y_dp = tm_con_mat[:, 1]

    # if there are less than 2 valid time-dp points, return None
    if len(x_tm) < 2:
        return None, None, None

    # fit the time-dp relationship
    # the RANSAC fitting is applied and if failed, then ordinary linear fitting will be used
    try:
        ransac = RANSACRegressor(random_state=42)
        ransac.fit(x_tm.reshape(-1, 1), y_dp)
        slope = ransac.estimator_.coef_[0]
        intercept = ransac.estimator_.intercept_
        method = 'RANSAC'
    except Exception:
        ols = LinearRegression()
        ols.fit(x_tm.reshape(-1, 1), y_dp)
        slope = ols.coef_[0]
        intercept = ols.intercept_
        method = 'LinearRegression'

    # if the fitting slope less than 0, return None (negative GR)
    if slope <= 0:
        return None, None, None

    # get the time and dp pairs for visualization
    num_min, num_max = (dp_min - intercept) / \
        slope, (dp_max - intercept) / slope
    num_min, num_max = int(np.maximum(x_tm[0], num_min)), int(
        np.minimum(x_tm[-1], num_max))
    x_num = np.arange(num_min, num_max)
    x_pred = np.array([num2time(dt, item) for item in x_num])
    y_pred = np.array([slope*item+intercept for item in x_num])

    # save the fitting plot (time-dp)
    if savefp is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(tm_con_mat[:, 0], y_dp)
        my_fmt = dates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(my_fmt)
        ax.plot(x_pred, y_pred, 'r',
                label=f'Growth rate: {slope*60:.2f} ' + '$\mathrm{nm\;h}^{-1}$')
        ax.set_xlabel('Local time (h)', fontsize=12)
        ax.set_ylabel('Dp (nm)', fontsize=12)
        #ax.set_title(date, fontsize=ftsize+4)
        ax.legend(handlelength=0, loc=2)
        ax.text(0.8, 0.05, f'{method}', transform=ax.transAxes, fontsize=14)
        fig.savefig(os.path.join(savefp, f'{dt}_fit.png'), dpi=100)

        _ = psd2im(df, figsize=(12, 8), line_data=[
               x_pred, y_pred*1e-9], savefp=savefp, vmax=vmax, dpi=100)
    return slope*60, x_pred, y_pred
