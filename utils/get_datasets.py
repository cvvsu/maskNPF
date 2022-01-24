r"""
https://smear.avaa.csc.fi/

Please check the `Terms Of Use` before you use the code.
"""


import argparse
import os, glob 
import numpy as np 
import pandas as pd 
from datetime import datetime
from tqdm import tqdm


def get_message(parser, args):
    """https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/options/base_options.py#L88
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    return message


def get_args():
    parser = argparse.ArgumentParser(description='Download Datasets')
    parser.add_argument('--dataroot', metavar='DIR',
                        default='datasets', help='path to dataset')
    parser.add_argument('--station', default='varrio',
                        help='station that the dataset collected from')
    parser.add_argument('--start_year', default=2000,
                        type=int, help='start of the year')
    parser.add_argument('--end_year', default=2008, type=int,
                        help='end of the year')
    parser.add_argument('--merge_df', action='store_true', 
                        help='merge all the downloaded csv files')
    
    args = parser.parse_args()

    return args, get_message(parser, args)


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_names(names):
    names = [name.split('.')[-1] for name in names]
    names = [name.replace('d', '') for name in names]
    names = [round(float(name)*1e-3, 2) for name in names]
    return names, sorted(names)


if __name__=='__main__':

    opt, msg = get_args()
    print(msg)

    savefp = os.path.join(opt.dataroot, opt.station)
    mkdirs(savefp)

    start_year, end_year = 'start_year', 'end_year'
    station_dict = {'hyytiala': 'HYY', 'varrio': 'VAR', 'kumpula': 'KUM'}

    base_url = 'https://smear-backend.rahtiapp.fi/search/timeseries/csv?tablevariable='\
    'VAR_DMPS.d112e1&tablevariable=VAR_DMPS.d100e1&tablevariable=VAR_DMPS.d126e1&tablevariable='\
    'VAR_DMPS.d141e1&tablevariable=VAR_DMPS.d158e1&tablevariable=VAR_DMPS.d178e1&tablevariable='\
    'VAR_DMPS.d200e1&tablevariable=VAR_DMPS.d224e1&tablevariable=VAR_DMPS.d251e1&tablevariable='\
    'VAR_DMPS.d282e1&tablevariable=VAR_DMPS.d316e1&tablevariable=VAR_DMPS.d355e1&tablevariable='\
    'VAR_DMPS.d398e1&tablevariable=VAR_DMPS.d447e1&tablevariable=VAR_DMPS.d501e1&tablevariable='\
    'VAR_DMPS.d562e1&tablevariable=VAR_DMPS.d631e1&tablevariable=VAR_DMPS.d708e1&tablevariable='\
    'VAR_DMPS.d794e1&tablevariable=VAR_DMPS.d891e1&tablevariable=VAR_DMPS.d100e2&tablevariable='\
    'VAR_DMPS.d112e2&tablevariable=VAR_DMPS.d126e2&tablevariable=VAR_DMPS.d141e2&tablevariable='\
    'VAR_DMPS.d158e2&tablevariable=VAR_DMPS.d178e2&tablevariable=VAR_DMPS.d200e2&tablevariable='\
    'VAR_DMPS.d100e4&tablevariable=VAR_DMPS.d891e3&tablevariable=VAR_DMPS.d794e3&tablevariable='\
    'VAR_DMPS.d708e3&tablevariable=VAR_DMPS.d631e3&tablevariable=VAR_DMPS.d562e3&tablevariable='\
    'VAR_DMPS.d501e3&tablevariable=VAR_DMPS.d447e3&tablevariable=VAR_DMPS.d398e3&tablevariable='\
    'VAR_DMPS.d355e3&tablevariable=VAR_DMPS.d316e3&tablevariable=VAR_DMPS.d282e3&tablevariable='\
    'VAR_DMPS.d251e3&tablevariable=VAR_DMPS.d224e3&tablevariable=VAR_DMPS.d200e3&tablevariable='\
    'VAR_DMPS.d178e3&tablevariable=VAR_DMPS.d158e3&tablevariable=VAR_DMPS.d141e3&tablevariable='\
    'VAR_DMPS.d126e3&tablevariable=VAR_DMPS.d112e3&tablevariable=VAR_DMPS.d100e3&tablevariable='\
    'VAR_DMPS.d891e2&tablevariable=VAR_DMPS.d794e2&tablevariable=VAR_DMPS.d708e2&tablevariable='\
    'VAR_DMPS.d631e2&tablevariable=VAR_DMPS.d562e2&tablevariable=VAR_DMPS.d501e2&tablevariable='\
    'VAR_DMPS.d447e2&tablevariable=VAR_DMPS.d398e2&tablevariable=VAR_DMPS.d355e2&tablevariable='\
    'VAR_DMPS.d316e2&tablevariable=VAR_DMPS.d282e2&tablevariable=VAR_DMPS.d251e2&tablevariable='\
    f'VAR_DMPS.d224e2&from={start_year}-01-01T00%3A00%3A00.000&to={end_year}-12-31T23%3A59%3A59.999&quality=ANY&aggregation=NONE&interval=1'
    

    base_url = base_url.replace('VAR', station_dict[opt.station])

    err_year = []

    for year in tqdm(range(opt.start_year, opt.end_year + 1)):
        url = base_url.replace(start_year, str(year)).replace(end_year, str(year))
        try:
            df = pd.read_csv(url, index_col='time', parse_dates={'time': [
                            0, 1, 2, 3, 4, 5]}, date_parser=lambda x: datetime.strptime(x, '%Y %m %d %H %M %S'))

            names, new_names = process_names(df.columns)
            # print(names, new_names)
            df.columns = names 
            df = df[new_names]

            # df = df.dropna(how='all', axis='columns')

            df[df < 0] = 0

            df.to_csv(os.path.join(savefp, f'{year}.csv'))
        except Exception:
            err_year.append(year)
            continue
    if len(err_year) > 0:
        print(f'Data in years that cannot be downloaded: {err_year}. Please have a manual check.')

    if opt.merge_df:
        dfs = []
        for file in glob.glob(savefp+'/*.csv'):
            dfs.append(pd.read_csv(file))
        dfs = pd.concat(dfs)
        dfs = dfs.drop_duplicates()
        dfs.to_csv(os.path.join(savefp, f'{opt.station}.csv'), index=False)

        

    
    
    
