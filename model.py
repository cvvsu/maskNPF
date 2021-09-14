import os, glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
from PIL import Image


from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

import warnings
warnings.filterwarnings("ignore")

import torch
from torchvision import transforms
from utils.utils import get_next_day, mkdirs, psd2im
from utils.networks import get_instance_segmentation_model
from utils.utils import reshape_mask
from utils.fitting import get_GR, get_SE


class NPFDetection(object):
    """Class for NPF detection."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cpu_count = os.cpu_count() // 2 + 1
        self.dataroot = os.path.join(opt.dataroot, opt.station)
        self.station = opt.station
        self.vmax = opt.vmax
        self.tm_res = opt.time_res
        self.df = pd.read_csv(os.path.join(self.dataroot, self.station+'.csv'), parse_dates=[0], index_col=0)         
        self.days = sorted(np.unique(self.df.index.date.astype(str)).tolist())
        print(f'There are {len(self.days)} days of data to be processed.')        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.key_index = 0

    def draw_one_day_images(self):
        """Draw NPF images with one-day unit"""
        self.savefp = os.path.join(self.dataroot, 'images', 'one_day')
        mkdirs(self.savefp)
        self.dimg = 1
        
        with Pool(self.cpu_count) as p:
            p.map(self.draw_image, self.days)

    def draw_two_day_images(self):
        """Draw NPF images with two-day unit"""
        self.savefp = os.path.join(self.dataroot, 'images', 'two_day')
        mkdirs(self.savefp)
        self.dimg = 2
        with Pool(self.cpu_count) as p:
            p.map(self.draw_image, self.days)

    def draw_image(self, day):
        """Draw an NPF image"""
        if self.dimg == 1:
            if not os.path.exists(os.path.join(self.savefp, day+'.png')):
                try:
                    psd2im(self.df.loc[day], use_xaxis=False, use_yaxis=False, vmax=self.vmax, savefp=self.savefp, show_figure=False)
                except Exception:
                    print(f'Cannot draw the NPF image for current day {day}.')
        elif self.dimg == 2:
            day_ = get_next_day(day)
            if day_ in self.days and not os.path.exists(os.path.join(self.savefp, day+'_'+day_+'.png')):
                try:
                    psd2im(self.df.loc[day:day_], use_xaxis=False, use_yaxis=False, vmax=self.vmax, savefp=self.savefp, show_figure=False)
                except Exception:
                    print(f'Cannot draw the NPF image for current day {day}_{day_}.')

    def detect_one_day_masks(self):
        """Detect masks for one-day NPF images"""
        self.load_model()
        size = (self.opt.im_size, self.opt.im_size)
        res = {}
        for im_path in glob.glob(os.path.join(self.dataroot, 'images/one_day')+'/*.png'):
            mask = self.detect_mask(im_path, size)
            if mask is not None:
                res.update(mask)
        print(f'Detected {len(res)} one-day masks whose scores are higher than {self.opt.scores:.2f}.')
        savefp = os.path.join(self.dataroot, 'masks')
        mkdirs(savefp)
        np.save(os.path.join(savefp, 'one_day.npy'), res)

    def detect_two_day_masks(self):
        """Detect masks for two-day NPF images"""
        self.load_model()
        size = (self.opt.im_size*2, self.opt.im_size)
        res = {}
        for im_path in glob.glob(os.path.join(self.dataroot, 'images/two_day')+'/*.png'):
            mask = self.detect_mask(im_path, size)
            if mask is not None:
                res.update(mask)
        print(f'Detected {len(res)} two-day masks whose scores are higher than {self.opt.scores:.2f}.')
        savefp = os.path.join(self.dataroot, 'masks')
        mkdirs(savefp)
        np.save(os.path.join(savefp, 'two_day.npy'), res)

    def load_model(self):
        # load the pre-trained Mask R-CNN model
        self.model = get_instance_segmentation_model()
        self.model.load_state_dict(torch.load(f'{self.opt.ckpt_dir}/{self.opt.model_name}'))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect_mask(self, im_path, size):
        """Detect valid masks for NPF images"""
        # get mask
        im = Image.open(im_path).convert('RGB').resize(size, Image.ANTIALIAS)
        ts = transforms.ToTensor()(im)
        out = self.model([ts.to(self.device)])[0]
        if len(out['scores']) == 0:
            return None
        else:
            idx_bool = out['scores'].cpu().numpy() >= self.opt.scores
            index = [i for i, item in enumerate(idx_bool) if item]
            if len(index) == 0:
                return None
            else:
                masks = out['masks'][index].squeeze(1).cpu().numpy() >= self.opt.mask_thres
                day = im_path.split(os.sep)[-1].split('.')[0].split('_')[0]
                return {day: masks}

    def visualize_masks(self):
        self.masks_oneday = np.load(os.path.join(self.dataroot, 'masks', 'one_day.npy'), allow_pickle=True).tolist()
        self.masks_twoday = np.load(os.path.join(self.dataroot, 'masks', 'two_day.npy'), allow_pickle=True).tolist()
        self.keys = sorted(list(self.masks_oneday.keys()))
    
        self.keys_ = sorted(list(self.masks_twoday.keys()))
        self.len_keys = len(self.keys)

        self.win = tk.Tk()
        self.win.title('NPF Detection')
        self.fig = Figure(dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win)
        graph_widget = self.canvas.get_tk_widget()
        graph_widget.grid(row=0, column=0, rowspan=2, columnspan=4, ipadx=200, sticky = tk.NW)

        self.fig1 = Figure(dpi=100)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.win)
        graph_widget1 = self.canvas1.get_tk_widget()
        graph_widget1.grid(row=2, column=0, rowspan=2, columnspan=4, ipadx=200, sticky = tk.NW)

        tk.Label(self.win, text='Select the one-day mask (select only one mask currently)').grid(row=0, column=5, columnspan=5, ipadx=50)
        tk.Label(self.win, text='Select the two-day mask (select only one mask currently)').grid(row=2, column=5, columnspan=5, ipadx=50)

        self.plot_next()
        tk.Button(self.win,text="Prev",command=self.plot_prev).grid(row=5,column=3, columnspan=5, sticky=tk.W)
        tk.Button(self.win,text="Next",command=self.plot_next).grid(row=5,column=7, columnspan=5, sticky=tk.W)

        self.win.mainloop()

    def plot(self):
        self.fig.clear()
        self.fig1.clear()
        self.key = self.keys[self.key_index]
        self.visualize_oneday_mask(self.fig, self.key)

        if self.key in self.keys_:
            self.visualize_twoday_mask(self.fig1, self.key)

        self.canvas.draw_idle()
        self.canvas1.draw_idle()

    def plot_prev(self):
        self.plot()
        self.key_index -= 1
        tk.Label(self.win, text=f'{self.key_index}/{self.len_keys}', fg='blue').grid(row=4, column=7, ipadx=50)
        if self.key_index < 0:
            tk.messagebox.showerror(title='Warning', message='You are at the begining, please click the Next button.')

    def plot_next(self):
        self.plot()
        self.key_index += 1
        tk.Label(self.win, text=f'{self.key_index}/{self.len_keys}', fg='blue').grid(row=4, column=7, ipadx=50)
        if self.key_index == self.len_keys - 1:
            tk.messagebox.showinfo(title='Warning', message='Good job! All masks have been checked!')

    def visualize_oneday_mask(self, fig, day):
        masks = self.masks_oneday[day]
        num_masks = masks.shape[0]
        ax = fig.add_subplot(1, num_masks+1, 1)

        im = Image.open(os.path.join(self.dataroot, 'images/one_day', day+'.png'))
        im = im.resize((self.opt.im_size, self.opt.im_size), Image.ANTIALIAS)
        ax.imshow(np.array(im))
        ax.set_title(day)
        ax.axis('off')

        # plot masks
        for i in range(masks.shape[0]):
            ax = fig.add_subplot(1, num_masks+1, i+2)
            ax.imshow(masks[i], cmap='gray')
            ax.set_title(f'mask {i}')
            ax.axis('off')

        for i in range(5):
            ck_btn = tk.Checkbutton(self.win, text=f'one-day mask {i}')
            ck_btn.grid(row=1, column=5+i, ipadx=10, ipady=5)
            ck_btn.config(command=lambda btn=ck_btn:self.save_mask(btn))

    def visualize_twoday_mask(self, fig, day):
        day_ = get_next_day(day)
        masks_ = self.masks_twoday[day]
        num_masks = masks_.shape[0]
        ax = fig.add_subplot(1, num_masks+1, 1)
        im_ = Image.open(os.path.join(self.dataroot, 'images/two_day', day+'_'+day_+'.png'))
        im_ = im_.resize((self.opt.im_size*2, self.opt.im_size), Image.ANTIALIAS)
        ax.imshow(np.array(im_))
        ax.set_title(day+'_'+day_)
        ax.axis('off')

        for i in range(masks_.shape[0]):
            ax = fig.add_subplot(1, num_masks+1, i+2)
            ax.imshow(masks_[i], cmap='gray')
            ax.set_title(f'mask {i}')
            ax.axis('off')

        for i in range(5):
            ck_btn_ = tk.Checkbutton(self.win, text=f'two-day mask {i}')
            ck_btn_.grid(row=3, column=5+i, ipadx=10, ipady=5)
            ck_btn_.config(command=lambda btn=ck_btn_:self.save_mask(btn))

    def save_mask(self, btn):
        text = btn.cget('text')
        idx = int(text[-1])
        if 'one-day' in text:
            savefp = os.path.join(self.dataroot, 'masks/one_day')
            mkdirs(savefp)
            np.save(os.path.join(savefp, f'{self.key}.npy'), self.masks_oneday[self.key][idx])
        elif 'two-day' in text:
            savefp = os.path.join(self.dataroot, 'masks/two_day')
            mkdirs(savefp)
            np.save(os.path.join(savefp, f'{self.key}.npy'), self.masks_twoday[self.key][idx])

    def get_GRs(self, df, mask):
        mask = reshape_mask(mask, df.shape)
        pos = np.where(mask)
        ymin = np.min(pos[1])
        ymax = np.max(pos[1])
        dps = [float(dp) for dp in df.columns]
        dp_start = dps[ymin]
        dp_end = dps[ymax]
        s_tm, e_tm = get_SE(df, mask)

        gr_3_10, _, _ = get_GR(df, mask, 2.75, 10, savefp=None, tm_res=self.tm_res, vmax=self.vmax)
        gr_10_25, _, _ = get_GR(df, mask, 10, 25.2, savefp=None, tm_res=self.tm_res, vmax=self.vmax)
        savefp = os.path.join(self.dataroot, 'GR/3_25')
        mkdirs(savefp)
        gr_3_25, _, _ = get_GR(df, mask, 2.75, 25.2, savefp=savefp, tm_res=self.tm_res, vmax=self.vmax)
        gr_mask, _, _ = get_GR(df, mask, dp_start, dp_end, savefp=None, tm_res=self.tm_res, vmax=self.vmax)
        return s_tm, e_tm, gr_3_10, gr_10_25, gr_3_25, gr_mask

    def get_SE_GR(self, day):
        df = self.df.loc[day]
        mask = np.load(os.path.join(self.dataroot, 'masks/one_day', day+'.npy'), allow_pickle=True)
        try:
            s_tm, e_tm, gr_3_10, gr_10_25, gr_3_25, gr_mask = self.get_GRs(df, mask)
        except:
            print(day)
            return

        try:
            mask_ = np.load(os.path.join(self.dataroot, 'masks/two_day', day+'.npy'), allow_pickle=True)
            df_ = self.df.loc[day:get_next_day(day)]
            mask_ = reshape_mask(mask_, df_.shape)
            st, et = get_SE(df_, mask_)
        except:
            st, et = s_tm, e_tm
        #return s_tm, e_tm, st, et, gr_3_10, gr_10_25, gr_3_25
        save_dict = {'date':[day],
                     'GR (3-10)': [gr_3_10],
                     'GR (10-25)': [gr_10_25],
                     'GR (3-25)': [gr_3_25],
                     'GR_mask': [gr_mask],
                     'ST (one)': [s_tm],
                     'ET (one)': [e_tm],
                     'ST (two)': [st],
                     'ET (two)': [et]}
        pd.DataFrame(save_dict).to_csv(os.path.join(self.savefp, f'{day}.csv'), index=False)

    def save_SE_GR(self):
        files = glob.glob(os.path.join(self.dataroot, 'masks/one_day')+'/*.npy')
        daysh = [file.split(os.sep)[-1].split('.')[0] for file in files]
        
        grs = glob.glob(os.path.join(self.dataroot, 'GR')+'/*.csv')
        days_1 = [gr.split(os.sep)[-1].split('.')[0] for gr in grs]
        
        days = [item for item in daysh if item not in days_1]
        
        self.savefp = os.path.join(self.dataroot, 'GR')
        mkdirs(self.savefp)
        x_len = len(days) // 10
        for i in range(11):
            days_ =  days[i*x_len:(i+1)*x_len]
            with Pool(self.cpu_count) as p:
                p.map(self.get_SE_GR, days_)
