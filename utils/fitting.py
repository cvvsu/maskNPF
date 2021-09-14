# -*- coding: utf-8 -*-
from astropy import modeling
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RANSACRegressor
from matplotlib import pyplot as plt
from matplotlib import dates

from datetime import timedelta, datetime


from .utils import psd2im


def time2num(tm):
    """Convert time to number.
    fit on numbers and then convert back the numbers to time
    """
    time = tm.time()
    minutes = time.hour * 60 + time.minute
    return minutes


def num2time(date, num):
    """Convert number to time.
    """
    year, month, day = date.split('-')
    hour = int(num / 60)
    minute = int(num - hour*60)
    return datetime(int(year), int(month), int(day), hour, minute)

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
    arr = arr1d[arr1d>0]

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
                ts = (dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
                time = datetime.utcfromtimestamp(ts) + timedelta(minutes = int(decimal))
                return time

def get_GR(df, mask, dp_min, dp_max, savefp=None, tm_res=10, vmax=1e4):
    """
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
    peak_con = np.array([[get_max_time(values[:, i], df.index.values, tm_res), dps[i]] for i in range(col_min, col_max+1)])
    
    # drop the nan values
    tm_con_mat = pd.DataFrame(peak_con).dropna(how='any').values
    
    # split the time and dps
    x_tm = np.array([time2num(item) for item in tm_con_mat[:, 0]])
    y_dp = tm_con_mat[:, 1]
    
    # if there are less than 2 valid time-dp points, return None
    if len(x_tm) < 2:
        return None
    
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
        return None
    
    # get the time and dp pairs for visualization
    num_min, num_max = (dp_min - intercept) / slope, (dp_max - intercept) / slope
    num_min, num_max = int(np.maximum(x_tm[0], num_min)), int(np.minimum(x_tm[-1], num_max))
    x_num = np.arange(num_min, num_max)
    x_pred = np.array([num2time(dt, item) for item in x_num])
    y_pred = np.array([slope*item+intercept for item in x_num])
    
    # save the fitting plot (time-dp)
    if savefp is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(tm_con_mat[:, 0], y_dp)
        my_fmt = dates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(my_fmt)
        ax.plot(x_pred, y_pred, 'r', label=f'Growth rate: {slope*60:.2f} ' + '$\mathrm{nm\;h}^{-1}$')
        ax.set_xlabel('Local time (h)', fontsize=12)
        ax.set_ylabel('Dp (nm)', fontsize=12)
        #ax.set_title(date, fontsize=ftsize+4)
        ax.legend(handlelength=0, loc=2)
        ax.text(0.8, 0.05, f'{method}', transform=ax.transAxes, fontsize=14)
        fig.savefig(os.path.join(savefp, f'{dt}_fit.png'), dpi=100)

        psd2im(df, figsize=(12, 8), fit_data=[x_pred, y_pred*1e-9], savefp=savefp, vmax=vmax, dpi=100)
    return slope*60, x_pred, y_pred

def get_SE(df, mask):
    """Get the start and end times (one-day or two-day)."""

    pos = np.where(mask)
    xmin, xmax = np.min(pos[0]), np.max(pos[0])

    start_time = df.index[xmin]
    end_time = df.index[xmax]
    return start_time, end_time
