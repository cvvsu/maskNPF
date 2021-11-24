import os
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from matplotlib import pyplot as plt
from matplotlib import colors
from PIL import Image


def mkdirs(path):
    r"""
    create a new folder named `path` if not exists.
    """
    path = f'{path}'
    if not os.path.exists(path):
        os.makedirs(path)


def get_next_day(current_day):
    r"""
    Get the next day given the current day.
    """
    return str(pd.to_datetime(current_day) + timedelta(days=1))[:10]


def reshape_mask(mask, shape):
    r"""reshape the mask to align the data
    
    Parameters:
        mask (array)  -- a detected mask
        shape (tuple) -- the shape of the data
        
    Returns:
        the aligned mask
    """
    mask = Image.fromarray(mask).resize(shape, Image.ANTIALIAS)
    mask = np.fliplr(np.array(mask).T)
    return mask.astype(int)


def convert_matlab_time(matlab_time):
    r"""
    convert the Matlab time to Python time
    """
    try:
        return (datetime.fromordinal(int(matlab_time)) + timedelta(days=matlab_time % 1) - timedelta(days=366)).date()
    except:
        return np.nan


def time2num(tm):
    r"""Convert time to number.
    fit on numbers and then convert back the numbers to time
    """
    time = tm.time()
    minutes = time.hour * 60 + time.minute
    return minutes


def num2time(date, num):
    r"""Convert number to time.
    """
    year, month, day = date.split('-')
    hour = int(num / 60)
    minute = int(num - hour*60)
    return datetime(int(year), int(month), int(day), hour, minute)











