import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors
from datetime import timedelta, datetime

from .utils import mkdirs


def psd2im(df,
           ax=None,
           fig=None,
           mask=None,
           savefp=None,
           dpi=600,
           n_xticks=5,
           figsize=(1.6, 1.2),
           show_figure=True,
           use_title=False,
           fit_data=None,
           line_data=None,
           index=None,
           vmax=None,
           lcolor='white',
           lwidth=3,
           use_xaxis=True,
           use_xlabel=False,
           use_yaxis=True,
           use_cbar=False,
           ftsize=16
           ):
    r"""Draw single or multiple surface plots.

    Parameters:
        df (dataframe)     --  particle size distribution data for one day or multiple days
        ax (ax)            --  specify the ax to visualize the psd. If not specified, a new one will be created
        fig (fig)          --  the whole figure
        mask (array)       --  numpy array with the same shape as the input psds
        savefp (str)       --  path for storing the figures
        dpi (int)          --  default is 600
        n_xticks (int)     --  how many ticklabels shown on the x-axis
        figsize (tuple)    --  used only if a new ax is created
        show_figure (bool) --  clear all the figures if drawing many surface plots
        use_title (bool)   --  use the date as the title for the psd
        fit_data (list)    --  the fitted time points and related Dps
        line_data (list)   --  the lines to show the determined GRs
        index (int)        --  there many be more than one masks detected for one day's psd
        vmax (float|none)  --  color scale for visualization, default is None.
        lcolor (str)       --  color for visualizing the GR
        lwidth (int)       --  linewidth
        use_xaxis (bool)   --  whether to draw the x-axis
        use_yaxis (bool)   --  whether to draw the y-axis
        use_cbar (bool)    --  whether to use the colorbar
        ftsize (int)       --  fontsize for plotting
    
    """

    # get the psd data
    dfc = df.copy(deep=True)    # get a copy version
    df_min = np.nanmin(dfc.replace(0, np.nan))    # find the minimul value
    # use the minimul value to replace the na values
    dfc.fillna(df_min, inplace=True)
    dfc[dfc == 0] = df_min               # use the minimul value to replace 0
    dfc = dfc.replace(0, df_min)

    # use the dynamic vmax
    if vmax is None:
        max_val = np.nanmax(dfc.values)
        # check the number of digits
        n_digits = len(str(int(max_val)))        
        vmax = np.power(10, n_digits)      

    values = dfc.values.T if mask is None else (
        dfc.values*mask).T   # values for visualization
    dps = [float(dp)*1e-9 for dp in list(dfc.columns)]    # Dps
    tm = dfc.index.values    # time points

    # check how many days of data to be shown
    whole_dates = np.unique([item.date() for item in df.index])
    num_days = (whole_dates[-1] - whole_dates[0]).days + 1

    # once the ax is none, create a new one
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(tm,    # time points
                       dps,    # particle sizes
                       values,  # distribution
                       norm=colors.LogNorm(vmin=1e1, vmax=vmax),
                       cmap='jet',
                       shading='auto')

    # add the fitted line for determinating the GRs
    if fit_data is not None:
        ax.scatter(fit_data[0], fit_data[1], c='k', s=15, marker='o')
    
    if line_data is not None:
        ax.plot(line_data[0], line_data[1], c=lcolor, linewidth=lwidth)

    # use the log scale for y-axis
    ax.set_yscale('log')

    # get title
    title = str(whole_dates[0]) if num_days <= 1 else str(
        whole_dates[0])+'_'+str(whole_dates[-1])

    # add index
    if index is not None:
        title = title + f' {index}'

    # add the title on the figure
    if use_title:
        ax.set_title(title, fontsize=ftsize+4)

    # add y-axis
    if use_yaxis:
        ax.set_ylabel('$\mathrm{D_p}$ (m)', fontsize=ftsize+2)
    else:
        ax.get_yaxis().set_visible(False)

    # add x-axis
    if use_xaxis:
        xtick = [datetime(sdate.year, sdate.month, sdate.day) + timedelta(i/(n_xticks-1)) for sdate in whole_dates
                 for i in range(n_xticks-1)] + [datetime(whole_dates[-1].year, whole_dates[-1].month, whole_dates[-1].day)+timedelta(1)]
        xtick_labels = ['00:00', '06:00', '12:00',
                        '18:00'] * num_days + ['00:00']
        ax.set_xticks(xtick)
        ax.set_xticklabels(xtick_labels)
    else:
        ax.get_xaxis().set_visible(False)

    if use_xlabel:
        ax.set_xlabel('Local time (h)', fontsize=ftsize+2)

    # add colorbar
    if use_cbar:
        # here fig is the default input for subplots
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(
            'dN/dlog$\mathrm{D_p} (\mathrm{cm}^{-3})$', fontsize=ftsize+2)

    # to avoid the black edges
    if (not use_xaxis) and (not use_yaxis):
        ax.set_axis_off()

    # save the currect figure
    if (savefp is not None) and (not use_xaxis) and (not use_yaxis):
        fig.savefig(os.path.join(savefp, title + '.png'),
                    bbox_inches='tight', pad_inches=0, dpi=dpi)
    elif (savefp is not None) and (use_xaxis or use_yaxis):
        fig.savefig(os.path.join(savefp, title + '.png'),
                    bbox_inches='tight', pad_inches=0.1, dpi=dpi)

    if not show_figure:
        plt.cla()
        plt.clf()
        plt.close('all')
    return im


def draw_subplots(dfs,
                  names,
                  nrows=3,
                  ncols=3,
                  savefp=None,
                  savename='test',
                  dpi=600,
                  GRs=None,
                  vmaxs=[1e4],
                  use_title=False,
                  indexes=None,
                  cbar='single',
                  lcolor='white',
                  lwidth=3,
                  texts=None,
                  ftsize=16
                  ):
    r"""Draw subplots with one colorbar.

    Parameters:
        dfs (dataframe)    --  pandas dataframe or list of dataframes
        names (list)       --  a list of days ['1999-01-01', '2000-01-01']
        nrows (int)        --  number of rows
        ncols (int)        --  number of columns
        savefp (str)       --  path to save the figure
        savename (str)     --  name of the figure to save
        dpi (int)          --  dpi for saving the figures (both in `png` and `pdf` formats)
        GRs (list)         --  visualizing the GRs
        vmaxs (int list)   --  color scale for visualization
        use_title (bool)   --  add title on each subplot
        indexes (str list) --  indexed for each subplot adding to the title (suffix)
        cbar (str)         --  ['single' | 'multirow' | 'all' | 'none']
        texts (str array)  --  an array containing letters 'a', 'b', ...
        ftsize (int)       --  fontsize for plotting
    """
    assert len(names) == nrows * ncols

    if GRs is not None:
        assert len(GRs) == len(names)

    fig, axes = plt.subplots(nrows, ncols, figsize=(
        ncols*6, nrows*4), constrained_layout=True)

    use_cbar = True if cbar == 'all' else False

    if len(vmaxs) == 1:
        vmaxs = vmaxs * nrows

    cnt = 0
    for i in range(nrows):
        for j in range(ncols):

            vmax = vmaxs[i]

            index = None if indexes == None else indexes[cnt]
            use_xlabel = True if i == nrows-1 else False
            GR = None if GRs is None else GRs[cnt]

            if isinstance(dfs, list):
                df = dfs[cnt]
            elif isinstance(dfs, pd.DataFrame):
                df = dfs[names[cnt]]
            else:
                raise ValueError('Unknown input type.')

            im = psd2im(df,
                        ax=axes[i, j],
                        fig=fig,
                        use_cbar=use_cbar,
                        use_xlabel=use_xlabel,
                        vmax=vmax,
                        dpi=dpi,
                        use_title=use_title,
                        index=index,
                        lcolor=lcolor,
                        lwidth=lwidth,
                        line_data=GR,
                        ftsize=ftsize
                        )
            cnt += 1

            if texts is not None:
                axes[i, j].text(
                    0.05, 0.92, f'({texts[i, j]})', transform=axes[i, j].transAxes, fontsize=ftsize+6, color='white')

        if cbar == 'multirow':
            #fig.tight_layout()
            cb = fig.colorbar(im, ax=axes[i, :], pad=0.01)
            cb.set_label(
                'dN/dlog$\mathrm{D_p} (\mathrm{cm}^{-3})$', fontsize=ftsize)

    if cbar == 'single':
        fig.tight_layout()
        im = plt.gca().get_children()[0]
        plt.subplots_adjust(right=0.9)
        cax = plt.axes([0.92, 0.1, 0.02, 0.85])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(
            'dN/dlog$\mathrm{D_p} (\mathrm{cm}^{-3})$', fontsize=ftsize)

    if savefp is not None:
        mkdirs(savefp)
        fig.savefig(os.path.join(savefp, savename+'.png'),
                    dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        fig.savefig(os.path.join(savefp, savename+'.pdf'),
                    bbox_inches='tight', pad_inches=0.1)
