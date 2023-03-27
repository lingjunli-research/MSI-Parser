# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:18:44 2023

@author: lawashburn
"""
from tkinter import *
from PIL import ImageTk, Image,ImageOps
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
# import sys
# from itertools import islice
# import subprocess
# from subprocess import Popen, PIPE
# from textwrap import dedent
# from tkinter import messagebox
# from tkinter import filedialog
# import os
# import pickle
# from tkinter import ttk
# import threading
# import traceback
# import time
# import psutil
import webbrowser
# from PIL import Image,ImageTk
# from pdf2image import convert_from_path
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLParser import getionimage
import matplotlib.pyplot as plt
import numpy as np
import csv
# import matplotlib.pyplot as plt
# import numpy as np
# import plotly.express as px
# from plotly.subplots import make_subplots
import pandas as pd
# import cv2
# import copy
# import csv
# from IPython.display import display
# from skimage import feature
# from skimage import measure
# from skimage.io import imread
from scipy import ndimage
# import kaleido
# from pyimzml.ImzMLParser import ImzMLParser
# from pyimzml.ImzMLParser import getionimage
# from scipy import signal
# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
# import numpy as np
# import plotly.express as px
# from plotly.subplots import make_subplots
# import pandas as pd
# import cv2
# import copy
# import csv
# from IPython.display import display
# from skimage import feature
# from skimage import measure
# from skimage.io import imread
# from pyimzml.ImzMLParser import ImzMLParser
# from pyimzml.ImzMLParser import getionimage
# from scipy import signal
#import matplotlib.pyplot as plt
# from matplotlib import colors
# import numpy as np
# import cv2
# from scipy.optimize import dual_annealing
# import os
# import sys
# from skimage.measure import regionprops
# from skimage.filters import median, gaussian, threshold_local
# from skimage.feature import peak_local_max
# from skimage.segmentation import watershed, relabel_sequential
# from skimage.morphology import disk, label
# from scipy import ndimage
# import xlsxwriter
# from sklearn.cluster import KMeans, DBSCAN
# import random
# import scipy
# import scipy.ndimage
# from PIL import Image
# from numpy import asarray
# from scipy import stats as st
from scipy.ndimage import median_filter
from cycler import cycler
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from umap import UMAP
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import numpy as np
# from scipy.signal import freqz
import seaborn as sns
# import matplotlib
#from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
# from matplotlib.axes import Subplot

### GUI appearance settings ###
Font_tuple = ("Corbel Light", 20)
root = Tk()
root.title('MSI Parser')
root.iconbitmap(r"LiClaw.ico")
root.geometry('1200x800')

##Variable storage
input_optical_text = StringVar()
input_maldi_text = StringVar()
mz_settings_var = StringVar()
mz_bkgd_settings_var = StringVar()
min_cell_area = StringVar()
max_cell_area = StringVar()
mz_choice = StringVar()
tolerance_choice =  StringVar()
z_choice = StringVar()
intensity_thresh_choice = StringVar()
resolution_choice = StringVar()
primary_background_mz = StringVar()
cell_to_remove = StringVar()
output_folder_path = StringVar()

##Definition storage
def openweb_liweb():
    new = 1
    url = "https://www.lilabs.org/resources"
    webbrowser.open(url,new=new)

def openweb_git():
    new = 1
    url = "https://github.com/lingjunli-research"
    webbrowser.open(url,new=new)

def openweb_user_manual():
    new = 1
    url = "https://docs.google.com/document/d/e/2PACX-1vRKyqvEpRbcrYHWTq1CLRImNfC6f_gxaXnKgH2I_ZX_E-kSA2PvUiy4d8kMddS2B8PcEwsLAngMcjvg/pub"
    webbrowser.open(url,new=new)
    
def set_optical_image_field():
    path_optical = askopenfilename(filetypes=[("Image Files",("*.png","*.jpeg","*.TIF"))]) 
    input_optical_text.set(path_optical)
    img=Image.open(path_optical) # read the image file
    img=img.resize((497,355)) # new width & height
    img=ImageTk.PhotoImage(img)
    e1 =Label(optical_image_frame)
    e1.pack(side=TOP)
    e1.image = img # keep a reference! by attaching it to a widget attribute
    e1['image']=img # Show Image  

def set_maldi_image_field():
    path_maldi = askopenfilename(filetypes=[("MSI Files",("*.imzML"))]) 
    input_maldi_text.set(maldi_optical)

def set_output_folder_field():
    path_out_dir = path_out_dir = filedialog.askdirectory() 
    output_folder_path.set(path_out_dir)

def check_cell_mz_selection():
    str_cell_select = mz_settings_var.get()
    list_str_cell_select = str_cell_select.split(',')
    float_list_str_cell_select = []
    for a in list_str_cell_select:
        float_list_str_cell_select.append(float(a))
    print(float_list_str_cell_select)

def parse_imzml(path):
    #import MSI data
    f = ImzMLParser(path) #ibd path required to parse
    return f
#plt.rcParams['axes.prop_cycle'] = cycler('color', plt.get_cmap('Dark2').colors)
background_array_storage = []
cell_array_storage = []
  
def change_pic(photo1):
    vlabel.configure(image=photo1)
    
def check_cell_mz(path,string_mz_list,mz,tol,z,parsed_imzml):    #check_cell_mz
    string_split_mz_list = string_mz_list.split(",")
    mz_list = []
    for k in string_split_mz_list:
        mz_list.append(float(k))
    primary_img = getionimage(parsed_imzml, mz, tol, z,reduce_func=sum) # img stored as 2D numpy array
    #plt.colorbar()
    #plt.set_cmap('viridis')
    for mz in mz_list:
        ind = mz_list.index(mz)
        img = getionimage(parsed_imzml, mz, tol, z,reduce_func=sum) # img stored as 2D numpy array
        plt.imshow(img,  interpolation='nearest')
        primary_img = np.add(img,primary_img)
    plt.clf()
    plt.imshow(primary_img,  interpolation='nearest')
    plt.colorbar()
    plt.savefig('maldi_added.png')
    img=Image.open('maldi_added.png') # read the image file
    root.img=ImageTk.PhotoImage(img)
    change_pic(root.img)
    return primary_img
    # e1.pack(side=TOP)
    # e1 = changepic('maldi_added.png')
    # #e1.image = img # keep a reference! by attaching it to a widget attribute
    # e1['image']=img # Show Image  

def check_background_mz(path,string_mz_list,mz,tol,z,parsed_imzml):    #check_cell_mz
    # path = input_maldi_text.get()
    # string_mz_list = mz_bkgd_settings_var.get()
    string_split_mz_list = string_mz_list.split(",")
    mz_list = []
    for k in string_split_mz_list:
        mz_list.append(float(k))
    # mz = float(primary_background_mz.get())
    # tol = float(tolerance_choice.get())
    # z = int(z_choice.get())
    # parsed_imzml = parse_imzml(path)
    primary_img2 = getionimage(parsed_imzml, mz, tol, z,reduce_func=sum) # img stored as 2D numpy array
    for mz in mz_list:
        ind = mz_list.index(mz)
        img = getionimage(parsed_imzml, mz, tol, z,reduce_func=sum) # img stored as 2D numpy array
        #plt.imshow(img,  interpolation='nearest')
        primary_img = np.add(img,primary_img2)
    plt.clf()
    plt.imshow(primary_img2,  interpolation='nearest')
    plt.colorbar()
    plt.savefig('background_added.png')
    img=Image.open('background_added.png') # read the image file
    root.img=ImageTk.PhotoImage(img)
    change_pic(root.img)
    background_array_storage = primary_img
    return primary_img
    # e1.pack(side=TOP)
    # e1.image = img # keep a reference! by attaching it to a widget attribute
    # e1['image']=img # Show Image 

def binarize_cells(foreground_img,bkgd_img):
    average_bkrd = np.average(bkgd_img)
    std_bkrd = np.std(bkgd_img)
    bkrd_intensity = (average_bkrd+std_bkrd)*6
    foreground_img[foreground_img <= bkrd_intensity] = 0
    foreground_img[foreground_img > 0] = 1
    plt.clf()
    plt.imshow(foreground_img, interpolation='nearest')
    plt.colorbar()
    plt.savefig('binarized_cells.png')
    img_png=Image.open('binarized_cells.png') # read the image file
    root.img_png=ImageTk.PhotoImage(img_png)
    change_pic(root.img_png)
    return foreground_img

def remove_cellular_holes(binarized_img):
    cellular_holeless_image = ndimage.binary_fill_holes(binarized_img).astype(int) #removes any holes in the interior of a cell
    plt.clf()
    plt.imshow(cellular_holeless_image, interpolation='nearest')
    plt.colorbar()
    plt.savefig('cell_fill_holes.png')
    img_png=Image.open('cell_fill_holes.png') # read the image file
    root.img_png=ImageTk.PhotoImage(img_png)
    change_pic(root.img_png)
    return cellular_holeless_image

def apply_median_filter_cells(holes_filled):
    filtered_array = median_filter(holes_filled, size=2) #remove any "speckles" or regions of just one or two pixels
    plt.clf()
    plt.imshow(filtered_array, interpolation='nearest')
    plt.colorbar()
    plt.savefig('median_filtered_cells.png')
    img_png=Image.open('median_filtered_cells.png') # read the image file
    root.img_png=ImageTk.PhotoImage(img_png)
    change_pic(root.img_png)
    return filtered_array

def find_cell (img,i):
    # find cell areas - pixel with intensity >= given i
    # intensities of non-cell pixels are set to 0, return as 2D numpy array
    img_cell = img
    img_cell[img_cell<i] = 0
    return(img_cell)

def dfs(img_cell):
    # depth first search used to find (x,y) of each cell
    # return a list of (x,y) coordinates for each cell
    # first set cell=1 and non-cell=0
    grid = img_cell
    grid[grid != 0]=1
    seen = set()
    l_all = list()
    for r0, row in enumerate(grid):
        for c0, val in enumerate(row):
            l = list()
            if val and (r0, c0) not in seen:
                stack = [(r0, c0)]
                seen.add((r0, c0))
                l.append((r0,c0))
                while stack:
                    r, c = stack.pop()
                    for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                        if (0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] and (nr, nc) not in seen):
                            stack.append((nr, nc))
                            seen.add((nr, nc))
                            l.append((nr,nc))
            l_all.append(l)
    return list(filter(None, l_all))

def initial_define_find_cells(filtered_array):
    # determine if each pixel is a cell or not
    img_cell = find_cell(filtered_array,1)
    # split to each cell area
    cc = dfs(img_cell)
    cell_localization_df = pd.DataFrame(columns=['x','y','cell_num'])
    i = 1 # cell number
    for l in cc:
        for x,y in l:
            cell_localization_df.loc[len(cell_localization_df.index)] = [y, x, i] 
        i = i+1
    cell_localization_df['t'] = cell_localization_df['cell_num'].astype('category')
    plt.clf()
    # sns.scatterplot('x', 'y', data=cell_localization_df, hue='t',linewidth=0, palette="deep")
    # plt.legend(loc='upper left',ncol=4, title="Cell #",bbox_to_anchor=(1, 1),fontsize='small',title_fontsize='medium')
    x_values_filtered = cell_localization_df['x'].values.tolist()
    y_values_filtered = cell_localization_df['y'].values.tolist()
    t_values_filtered = cell_localization_df['t'].values.tolist()

    t_values_filtered_str = []
    for a in t_values_filtered:
        t_values_filtered_str.append(str(a))

    sns.scatterplot(x=x_values_filtered, y=y_values_filtered, hue=t_values_filtered_str,linewidth=0, palette='husl')
    #sns.set_palette("pastel")
    plt.legend(loc='upper left',ncol=2, title="Cell #",bbox_to_anchor=(1, 1),fontsize='small',title_fontsize='medium')
    # filtered_cells_clean2 = cell_localization_df.copy()
    # x_values2 = filtered_cells_clean2['x'].values.tolist()
    # y_values2 = filtered_cells_clean2['y'].values.tolist()
    # color_values2 = filtered_cells_clean2['t'].values.tolist()
    # print(len(color_values2))
    # print(len(x_values2))
    # print(len(y_values2))
    # fig, ax = plt.subplots()
    # scatter2 = ax.scatter(x=x_values2, y=y_values2, c=color_values2)
    # legend12 = ax.legend(*scatter2.legend_elements(),
    #                 loc="lower left", title="Cell #",bbox_to_anchor=(1, 0))
    # ax.add_artist(legend12)
    
    
    plt.savefig('inital_cells_define.png', bbox_inches = 'tight')
    plt.clf()
    img_png=Image.open('inital_cells_define.png') # read the image file
    img_png = ImageOps.contain(img_png, (500,400))
    #img_png = ImageOps.contain(img_png, (200,100))
    root.img_png=ImageTk.PhotoImage(img_png)
    change_pic(root.img_png)
    return cell_localization_df

def area_filter_cells(min_cell_pixels,max_cell_pixels,filtered_array):
    # determine if each pixel is a cell or not
    img_cell = find_cell(filtered_array,1)
    # split to each cell area
    cc = dfs(img_cell)
    cell_localization_df = pd.DataFrame(columns=['x','y','cell_num'])
    i = 1 # cell number
    for l in cc:
        for x,y in l:
            cell_localization_df.loc[len(cell_localization_df.index)] = [y, x, i] 
        i = i+1
    cell_localization_df['t'] = cell_localization_df['cell_num'].astype('category')
    print(len(cell_localization_df))
    # min_cell_pixels = int(min_cell_area.get())
    # max_cell_pixels = int(max_cell_area.get())
    
    # cell_localization_df = initial_define_find_cells()
    
    cell_numbers = cell_localization_df['t'].values.tolist()
    cell_number_no_dups = []
    for aa in cell_numbers:
        if aa not in cell_number_no_dups:
            cell_number_no_dups.append(aa)
    
    filtered_cells = pd.DataFrame()  
    for bb in cell_number_no_dups:
        cell_local_filter = cell_localization_df[cell_localization_df['t'] == bb]
        if len(cell_local_filter) >=min_cell_pixels:
            if len(cell_local_filter) <=max_cell_pixels:
                filtered_cells = pd.concat([filtered_cells,cell_local_filter])
            else:
                pass
        else:
            pass
    filtered_cells['g'] = filtered_cells['t'].astype('category') 
    print(len(filtered_cells))
    plt.clf()
    filtered_cells_clean = filtered_cells.copy()
    # ax = filtered_cells_clean.plot(x='x',y='y',kind='scatter',figsize=(10,10))
    # #filtered_cells_clean[['x','y','t']].apply(lambda x: ax.text(*x),axis=1)

    x_values_filtered = filtered_cells_clean['x'].values.tolist()
    y_values_filtered = filtered_cells_clean['y'].values.tolist()
    t_values_filtered = filtered_cells_clean['t'].values.tolist()

    t_values_filtered_str = []
    for a in t_values_filtered:
        t_values_filtered_str.append(str(a))

    sns.scatterplot(x=x_values_filtered, y=y_values_filtered, hue=t_values_filtered_str,linewidth=0, palette='husl')
    #sns.set_palette("pastel")
    plt.legend(loc='upper left',ncol=2, title="Cell #",bbox_to_anchor=(1, 1),fontsize='small',title_fontsize='medium')
    #plt.legend()
    plt.savefig('area_filtered_cells.png', bbox_inches = 'tight')
    plt.show()
    img_png=Image.open('area_filtered_cells.png') # read the image file
    img_png = ImageOps.contain(img_png, (500,400))
    
    root.img_png=ImageTk.PhotoImage(img_png)
    
    change_pic(root.img_png)
    plt.clf()
    return filtered_cells_clean
    
def remove_cell_by_number(cells_to_remove,filtered_cells2):
    
    cells_to_remove = cells_to_remove.split(",")
    cells_IDs = []
    for x in cells_to_remove:
        cells_IDs.append(int(x))

    if len(cells_IDs)>0:
        filtered_cells2 = filtered_cells2[~filtered_cells2['t'].isin(cells_IDs)]
    else:
        pass

    filtered_cells2['i'] = filtered_cells2['t'].astype('category')    
    plt.clf()
    
    x_values = filtered_cells2['x'].values.tolist()
    y_values = filtered_cells2['y'].values.tolist()
    t_values = filtered_cells2['i'].values.tolist()
    
    t_values_str = []
    for m in t_values:
        t_values_str.append(str(m))
    
    sns.scatterplot(x=x_values, y=y_values, hue=t_values_str,linewidth=0, palette="husl")
    plt.legend(loc='upper left',ncol=2, title="Cell #",bbox_to_anchor=(1, 1),fontsize='small',title_fontsize='medium')
    #ax2.legend(loc='upper left',ncol=4, title="Cell #",bbox_to_anchor=(1, 1),fontsize='small',title_fontsize='medium')
    
    plt.savefig('manual_filtered_cells.png', bbox_inches = 'tight')
    img_png=Image.open('manual_filtered_cells.png') # read the image file
    img_png = ImageOps.contain(img_png, (500,400))
    
    root.img_png=ImageTk.PhotoImage(img_png)
    
    change_pic(root.img_png)
    return filtered_cells2

def start_cell_mz():
    path = input_maldi_text.get()
    string_mz_list = mz_settings_var.get()
    mz = float(mz_choice.get())
    tol = float(tolerance_choice.get())
    z = int(z_choice.get())
    parsed_imzml = parse_imzml(path)
    check_cell_mz_results = check_cell_mz(path,string_mz_list,mz,tol,z,parsed_imzml)

def start_background_mz():
    path = input_maldi_text.get()
    string_mz_list = mz_bkgd_settings_var.get()
    mz = float(primary_background_mz.get())
    tol = float(tolerance_choice.get())
    z = int(z_choice.get())
    parsed_imzml = parse_imzml(path)
    check_background_mz_results = check_background_mz(path,string_mz_list,mz,tol,z,parsed_imzml)    

def start_binarized_cell_search():
    path = input_maldi_text.get()
    cell_string_mz_list = mz_settings_var.get()
    cell_mz = float(mz_choice.get())
    bkgd_string_mz_list = mz_bkgd_settings_var.get()
    bkgd_mz = float(primary_background_mz.get())
    tol = float(tolerance_choice.get())
    z = int(z_choice.get())
    parsed_imzml = parse_imzml(path)
    foreground_img = check_cell_mz(path,cell_string_mz_list,cell_mz,tol,z,parsed_imzml)    
    bkgd_img = check_background_mz(path,bkgd_string_mz_list,bkgd_mz,tol,z,parsed_imzml) 
    binarized_cells = binarize_cells(foreground_img,bkgd_img)
    
def start_fill_holes_search():
    path = input_maldi_text.get()
    cell_string_mz_list = mz_settings_var.get()
    cell_mz = float(mz_choice.get())
    bkgd_string_mz_list = mz_bkgd_settings_var.get()
    bkgd_mz = float(primary_background_mz.get())
    tol = float(tolerance_choice.get())
    z = int(z_choice.get())
    parsed_imzml = parse_imzml(path)
    foreground_img = check_cell_mz(path,cell_string_mz_list,cell_mz,tol,z,parsed_imzml)    
    bkgd_img = check_background_mz(path,bkgd_string_mz_list,bkgd_mz,tol,z,parsed_imzml) 
    binarized_img = binarize_cells(foreground_img,bkgd_img)
    holy_image = remove_cellular_holes(binarized_img)
    
def start_median_filter_search():
    path = input_maldi_text.get()
    cell_string_mz_list = mz_settings_var.get()
    cell_mz = float(mz_choice.get())
    bkgd_string_mz_list = mz_bkgd_settings_var.get()
    bkgd_mz = float(primary_background_mz.get())
    tol = float(tolerance_choice.get())
    z = int(z_choice.get())
    parsed_imzml = parse_imzml(path)
    foreground_img = check_cell_mz(path,cell_string_mz_list,cell_mz,tol,z,parsed_imzml)    
    bkgd_img = check_background_mz(path,bkgd_string_mz_list,bkgd_mz,tol,z,parsed_imzml) 
    binarized_img = binarize_cells(foreground_img,bkgd_img)
    holes_filled = remove_cellular_holes(binarized_img)
    median_applied = apply_median_filter_cells(holes_filled)

def start_cell_define_inital_search():
    path = input_maldi_text.get()
    cell_string_mz_list = mz_settings_var.get()
    cell_mz = float(mz_choice.get())
    bkgd_string_mz_list = mz_bkgd_settings_var.get()
    bkgd_mz = float(primary_background_mz.get())
    tol = float(tolerance_choice.get())
    z = int(z_choice.get())
    parsed_imzml = parse_imzml(path)
    foreground_img = check_cell_mz(path,cell_string_mz_list,cell_mz,tol,z,parsed_imzml)    
    bkgd_img = check_background_mz(path,bkgd_string_mz_list,bkgd_mz,tol,z,parsed_imzml) 
    binarized_img = binarize_cells(foreground_img,bkgd_img)
    holes_filled = remove_cellular_holes(binarized_img)
    filtered_array = apply_median_filter_cells(holes_filled)
    cell_localization_df = initial_define_find_cells(filtered_array)

def start_cell_define_area_filter_search():
    path = input_maldi_text.get()
    cell_string_mz_list = mz_settings_var.get()
    cell_mz = float(mz_choice.get())
    bkgd_string_mz_list = mz_bkgd_settings_var.get()
    bkgd_mz = float(primary_background_mz.get())
    tol = float(tolerance_choice.get())
    z = int(z_choice.get())
    parsed_imzml = parse_imzml(path)
    foreground_img = check_cell_mz(path,cell_string_mz_list,cell_mz,tol,z,parsed_imzml)    
    bkgd_img = check_background_mz(path,bkgd_string_mz_list,bkgd_mz,tol,z,parsed_imzml) 
    binarized_img = binarize_cells(foreground_img,bkgd_img)
    holes_filled = remove_cellular_holes(binarized_img)
    filtered_array = apply_median_filter_cells(holes_filled)
    min_cell_pixels = int(min_cell_area.get())
    max_cell_pixels = int(max_cell_area.get())
    filtered_cells = area_filter_cells(min_cell_pixels,max_cell_pixels,filtered_array)

def start_cell_remove_specifics_search():
    path = input_maldi_text.get()
    cell_string_mz_list = mz_settings_var.get()
    cell_mz = float(mz_choice.get())
    bkgd_string_mz_list = mz_bkgd_settings_var.get()
    bkgd_mz = float(primary_background_mz.get())
    tol = float(tolerance_choice.get())
    z = int(z_choice.get())
    parsed_imzml = parse_imzml(path)
    foreground_img = check_cell_mz(path,cell_string_mz_list,cell_mz,tol,z,parsed_imzml)    
    bkgd_img = check_background_mz(path,bkgd_string_mz_list,bkgd_mz,tol,z,parsed_imzml) 
    binarized_img = binarize_cells(foreground_img,bkgd_img)
    holes_filled = remove_cellular_holes(binarized_img)
    filtered_array = apply_median_filter_cells(holes_filled)
    min_cell_pixels = int(min_cell_area.get())
    max_cell_pixels = int(max_cell_area.get())
    filtered_cells2 = area_filter_cells(min_cell_pixels,max_cell_pixels,filtered_array)
    cells_to_remove = cell_to_remove.get()
    remove_cell_by_number(cells_to_remove,filtered_cells2)

def mzbin(mz,intensity,start,end,resolution):
    # mz is an array of m/z values of a mass spectrum at some pixel
    # intensity is an array of corresponding intensities
    # start and end are the lower and upper bound m/z
    # resolution defines res of mass binning
    # returns result_mz: if a bin has edges 0.1-0.2, returns its center 0.15
    # result_intensity: the average intensity of every value falls into that bin
    step=(end-start)/resolution
    i=start
    result_mz=[]
    result_intensity=[]
    while i<end:
        avg_intensity=0
        count=0
        for j in range(len(mz)):
            if mz[j]<i+step and mz[j]>=i:
                avg_intensity+=intensity[j]
                count+=1 
        if count==0:
            result_intensity.append(0)
        else:
            result_intensity.append(avg_intensity/count)
        #print(avg_intensity)
        result_mz.append((i+i+step)/2)
        i+=step
    return result_mz,result_intensity
    
def export_all_results():
    
    output_path = output_folder_path.get()
    path = input_maldi_text.get()
    cell_string_mz_list = mz_settings_var.get()
    cell_mz = float(mz_choice.get())
    bkgd_string_mz_list = mz_bkgd_settings_var.get()
    bkgd_mz = float(primary_background_mz.get())
    tol = float(tolerance_choice.get())
    z = int(z_choice.get())
    parsed_imzml = parse_imzml(path)
    foreground_img = check_cell_mz(path,cell_string_mz_list,cell_mz,tol,z,parsed_imzml)    
    bkgd_img = check_background_mz(path,bkgd_string_mz_list,bkgd_mz,tol,z,parsed_imzml) 
    binarized_img = binarize_cells(foreground_img,bkgd_img)
    holes_filled = remove_cellular_holes(binarized_img)
    filtered_array = apply_median_filter_cells(holes_filled)
    min_cell_pixels = int(min_cell_area.get())
    max_cell_pixels = int(max_cell_area.get())
    filtered_cells = area_filter_cells(min_cell_pixels,max_cell_pixels,filtered_array)
    cells_to_remove = cell_to_remove.get()
    if len(cells_to_remove)>0:
        filtered_cells2 = remove_cell_by_number(cells_to_remove,filtered_cells)
    else:
        filtered_cells2 = filtered_cells
    
    filtered_cells2["Identifier"] = filtered_cells2['x'].astype(str) +"_"+ filtered_cells2["y"].astype(str)

    cell_IDs = filtered_cells2['Identifier'].values.tolist()
    
    cell_ID_report = []
    cell_ID_count = []
    
    for cc in cell_IDs:
        cell_count = cell_IDs.count(cc)
        cell_ID_report.append(cc)
        cell_ID_count.append(cell_count)
        
    cell_count_report = pd.DataFrame()
    cell_count_report['ID'] = cell_ID_report
    cell_count_report['Count'] = cell_ID_count
    
    filtered_cells2['coordinate count'] = cell_ID_count
    filtered_cells2 = filtered_cells2[filtered_cells2['coordinate count'] == 1]

    filtered_cells2['t'] = filtered_cells2['cell_num'].astype('category')    
    # dict - {x,y: index}
    # a dictionary of pixel information
    coord_dict = {}
    for i in range(len(f.coordinates)):
        coord_dict[(f.coordinates[i][0],f.coordinates[i][1])]= i
    final_cell_list_dups = filtered_cells2['t'].values.tolist()

    final_cell_list = []
    for l in final_cell_list_dups:
        if l not in final_cell_list:
            final_cell_list.append(l)
        else:
            pass
    final_report = pd.DataFrame()
    for m in final_cell_list:
        # the following is to extract the mean mass spectrum of the cell #1
        cell_one=filtered_cells2.loc[filtered_cells2['t'] == m]
    
        # do mass binning for each pixel in a cell, then calculate the average mass spectrum
        total_intensity=0
        output_intensity=[]
        output_mz=[]
        for i in range(len(cell_one)):
            try:
                xx=cell_one.iloc[i]['x']
                yy=cell_one.iloc[i]['y']
                index=coord_dict[(xx+1,yy+1)]
                mz, intensity = ImzMLParser.getspectrum(f,index) # retrieve mass spectrum at a pixel
                sum_intensity=sum(intensity)
                total_intensity+=sum_intensity
                result_mz,result_intensity=mzbin(mz, intensity, 50,600,7000)
                # print(result_mz,result_intensity)
                # print(sum_intensity)
                # calculate weighted average: sum(intensity of each cell * total intensity)/sum(total intensity)
                result_intensity=np.asarray(result_intensity)*sum_intensity 
                output_intensity.append(result_intensity)
                output_mz=result_mz
            except Exception:
                pass
        output=np.sum(output_intensity,axis=0)/total_intensity
    
    
        plt.plot(output_mz,output)
        plt.title("Cell #: "+str(m))
        file_name = output_path + "\\cell_number_" + str(m) + ".png" 
        plt.savefig(file_name)
        
        output_df = pd.DataFrame()
        output_df['m/z'] = output_mz
        output_df[("Cell #: " + str(m))] = output
        #output_df['cell_number'] = m
        if len(final_report)>0:
            final_report = final_report.merge(output_df,on='m/z',how='outer')
        else:
            final_report = pd.concat([final_report,output_df])
        
        plt.close()
    file_path = output_path + '\\avg_spectra_all.csv'
    with open(file_path,'w',newline='') as filec:
                writerc = csv.writer(filec)
                final_report.to_csv(filec,index=False)
 
##Layout

menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
# filemenu.add_command(label="New", command=donothing)
# filemenu.add_command(label="Open", command=donothing)
# filemenu.add_command(label="Save", command=donothing)
#filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)

helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="Li Lab Website", command=openweb_liweb)
helpmenu.add_command(label="Li Lab GitHub", command=openweb_git)
helpmenu.add_command(label="User manual", command=openweb_user_manual)
menubar.add_cascade(label="Help", menu=helpmenu)

toolmenu = Menu(menubar, tearoff=0)
toolmenu.add_command(label="Step evaluate tool")
menubar.add_cascade(label="Tools", menu=toolmenu)

root.config(menu=menubar)

main_frame = Canvas(root, width= 1200, height= 800)
main_frame.pack(side=TOP)

optical_entry_frame = Canvas(main_frame, width= 600, height= 800)
optical_entry_frame.pack(side=LEFT)

optical_text_frame = Canvas(optical_entry_frame, width= 600, height= 100)
optical_text_frame.pack(side=TOP)

optical_entry_text_frame = Canvas(optical_text_frame, width= 103, height= 50)
optical_entry_text_frame.pack(side=LEFT)
optical_entry_text_frame.create_text(53, 25, text="Optical image", fill="#2F4FAA", font=(Font_tuple,8),justify=CENTER)
optical_entry_text_frame.pack()
optical_entry_choice_frame = Canvas(optical_text_frame, width= 200, height= 50)
optical_entry_choice_frame.pack(side=LEFT)
optical_choice_browse_entry = Entry(optical_entry_choice_frame, textvariable = input_optical_text, width = 63)
optical_choice_browse_entry.pack(side=LEFT)
optical_choice_browse = Button(optical_entry_choice_frame, text = "Browse", command = set_optical_image_field)
optical_choice_browse.pack(side=RIGHT)

optical_image_frame = Canvas(optical_entry_frame, width= 600, height= 750)
optical_image_frame.pack(side=TOP)

maldi_entry_frame = Canvas(main_frame, width= 600, height= 50)
maldi_entry_frame.pack(side=RIGHT)

maldi_path_frame = Canvas(maldi_entry_frame, width= 600, height= 50)
maldi_path_frame.pack(side=TOP)

general_settings_frame = Canvas(maldi_entry_frame, width= 600, height= 50)
general_settings_frame.pack(side=TOP)

general_settings_frame2 = Canvas(maldi_entry_frame, width= 600, height= 50)
general_settings_frame2.pack(side=TOP)

cell_mz_select_frame = Canvas(maldi_entry_frame, width= 600, height= 50)
cell_mz_select_frame.pack(side=TOP)

background_mz_select_frame = Canvas(maldi_entry_frame, width= 600, height= 50)
background_mz_select_frame.pack(side=TOP)

filtering_frame = Canvas(maldi_entry_frame, width= 600, height= 50)
filtering_frame.pack(side=TOP)

pixel_select_frame = Canvas(maldi_entry_frame, width= 600, height= 50)
pixel_select_frame.pack(side=TOP)

intensity_select_frame = Canvas(maldi_entry_frame, width= 600, height= 50)
intensity_select_frame.pack(side=TOP)

maldi_preview_frame = Canvas(maldi_entry_frame, width= 600, height= 500)
maldi_preview_frame.pack(side=TOP)
photo = "LiClaw_small.png"
root.photo = ImageTk.PhotoImage(Image.open(photo))
#root.photo = root.photo.resize(5,5)
vlabel=Label(maldi_preview_frame,image=root.photo)
vlabel.pack()

avg_spectra_export_frame = Canvas(maldi_entry_frame, width= 600, height= 50)
avg_spectra_export_frame.pack(side=TOP)

maldi_entry_text_frame = Canvas(maldi_path_frame, width= 103, height= 50)
maldi_entry_text_frame.pack(side=LEFT)
maldi_entry_text_frame.create_text(53, 25, text="MSI Path (.imzML)", fill="#2F4FAA", font=(Font_tuple,8),justify=CENTER)
maldi_entry_text_frame.pack(side=LEFT)
maldi_entry_choice_frame = Canvas(maldi_path_frame, width= 200, height= 50)
maldi_entry_choice_frame.pack(side=LEFT)
maldi_choice_browse_entry = Entry(maldi_entry_choice_frame, textvariable = input_maldi_text, width = 70)
maldi_choice_browse_entry.insert(END, r"C:\Users\lawashburn\Documents\MSI_test\test_data_from_Hua\small_dataset_pos_20230216\psc_cell_ctrl_3-05-2022_smallarea021623.imzML")
maldi_choice_browse_entry.pack(side=LEFT)
maldi_choice_browse = Button(maldi_entry_choice_frame, text = "Browse", command = set_maldi_image_field)
maldi_choice_browse.pack(side=LEFT)
### M/Z settings row ###
mz_text_frame = Canvas(general_settings_frame, width= 100, height= 50)
mz_text_frame.pack(side=LEFT)
mz_text_frame.create_text(53, 25, text="m/z:", fill="#2F4FAA", font=(Font_tuple,8),justify=CENTER)
mz_text_frame.pack(side=LEFT)

mz_text_entry_frame = Canvas(general_settings_frame, width= 100, height= 50)
mz_text_entry_frame.pack(side=LEFT)
mz_text_entry = Entry(mz_text_entry_frame, textvariable = mz_choice, width = 10)
mz_text_entry.insert(END,'810.6052')
mz_text_entry.pack(side=LEFT)

mz_tol_text_frame = Canvas(general_settings_frame, width= 100, height= 50)
mz_tol_text_frame.pack(side=LEFT)
mz_tol_text_frame.create_text(53, 25, text="tolerance:", fill="#2F4FAA", font=(Font_tuple,8),justify=CENTER)
mz_tol_text_frame.pack(side=LEFT)

mz_tol_entry_frame = Canvas(general_settings_frame, width= 100, height= 50)
mz_tol_entry_frame.pack(side=LEFT)
mz_tol_entry = Entry(mz_tol_entry_frame, textvariable = tolerance_choice, width = 10)
mz_tol_entry.insert(END,'0.1')
mz_tol_entry.pack(side=LEFT)

mz_z_text_frame = Canvas(general_settings_frame, width= 100, height= 50)
mz_z_text_frame.pack(side=LEFT)
mz_z_text_frame.create_text(53, 25, text="z:", fill="#2F4FAA", font=(Font_tuple,8),justify=CENTER)
mz_z_text_frame.pack(side=LEFT)

mz_z_entry_frame = Canvas(general_settings_frame, width= 100, height= 50)
mz_z_entry_frame.pack(side=LEFT)
mz_z_entry = Entry(mz_z_entry_frame, textvariable = z_choice, width = 10)
mz_z_entry.insert(END,'1')
mz_z_entry.pack(side=LEFT)

mz_apply_button_frame = Canvas(general_settings_frame, width= 100, height= 50)
mz_apply_button_frame.pack(side=LEFT)
mz_apply_button = Button(mz_apply_button_frame, text = "Apply")
mz_apply_button.pack(side=LEFT)
###

### M/Z settings row2 ###
intensity_text_frame = Canvas(general_settings_frame2, width= 100, height= 50)
intensity_text_frame.pack(side=LEFT)
intensity_text_frame.create_text(53, 25, text="Intensity\nthreshold:", fill="#2F4FAA", font=(Font_tuple,8),justify=CENTER)
intensity_text_frame.pack(side=LEFT)

intensity_entry_frame = Canvas(general_settings_frame2, width= 100, height= 50)
intensity_entry_frame.pack(side=LEFT)
intensity_entry = Entry(intensity_entry_frame, textvariable = intensity_thresh_choice, width = 10)
intensity_entry.insert(END,'2000')
intensity_entry.pack(side=LEFT)

resolution_text_frame = Canvas(general_settings_frame2, width= 100, height= 50)
resolution_text_frame.pack(side=LEFT)
resolution_text_frame.create_text(53, 25, text="Resolution:", fill="#2F4FAA", font=(Font_tuple,8),justify=CENTER)
resolution_text_frame.pack(side=LEFT)

resolution_entry_frame = Canvas(general_settings_frame2, width= 100, height= 50)
resolution_entry_frame.pack(side=LEFT)
resolution_entry = Entry(resolution_entry_frame, textvariable = resolution_choice, width = 10)
resolution_entry.insert(END,'7000')
resolution_entry.pack(side=LEFT)

primary_bkgd_text_frame = Canvas(general_settings_frame2, width= 100, height= 50)
primary_bkgd_text_frame.pack(side=LEFT)
primary_bkgd_text_frame.create_text(53, 25, text="Primary Background\nm/z:", fill="#2F4FAA", font=(Font_tuple,8),justify=CENTER)
primary_bkgd_text_frame.pack(side=LEFT)

primary_bkgd_entry_frame = Canvas(general_settings_frame2, width= 100, height= 50)
primary_bkgd_entry_frame.pack(side=LEFT)
primary_bkgd_entry = Entry(primary_bkgd_entry_frame, textvariable = primary_background_mz, width = 10)
primary_bkgd_entry.insert(END,'580.4977')
primary_bkgd_entry.pack(side=LEFT)



mz_apply_button_frame = Canvas(general_settings_frame2, width= 100, height= 50)
mz_apply_button_frame.pack(side=LEFT)
mz_apply_button = Button(mz_apply_button_frame, text = "Apply")
mz_apply_button.pack(side=LEFT)
###

mz_settings_entry_text_frame = Canvas(cell_mz_select_frame, width= 100, height= 50)
mz_settings_entry_text_frame.pack(side=LEFT)
mz_settings_entry_text_frame.create_text(37, 25, text="Cell m/z", fill="#2F4FAA", font=(Font_tuple,8),justify=CENTER)
mz_settings_entry_text_frame.pack(side=LEFT)
mz_settings_browse_frame = Canvas(cell_mz_select_frame, width= 400, height= 50)
mz_settings_browse_frame.pack(side=LEFT)
mz_settings_browse_entry = Entry(mz_settings_browse_frame, textvariable = mz_settings_var, width = 70)
mz_settings_browse_entry.insert(END,'810.6052,632.6295,604.5808,760.5623,808.5812,788.5988,786.5904')
mz_settings_browse_entry.pack(side=LEFT)
mz_settings_check_choice_frame = Canvas(cell_mz_select_frame, width= 100, height= 50)
mz_settings_check_choice_frame.pack(side=LEFT)
maldi_choice_cell_check = Button(mz_settings_check_choice_frame, text = "Check", command=start_cell_mz)
maldi_choice_cell_check.pack(side=LEFT)

bkgd_mz_settings_entry_text_frame = Canvas(background_mz_select_frame, width= 100, height= 50)
bkgd_mz_settings_entry_text_frame.pack(side=LEFT)
bkgd_mz_settings_entry_text_frame.create_text(37, 25, text="Background\nm/z", fill="#2F4FAA", font=(Font_tuple,8),justify=CENTER)
bkgd_mz_settings_entry_text_frame.pack(side=LEFT)
bkgd_mz_settings_browse_frame = Canvas(background_mz_select_frame, width= 400, height= 50)
bkgd_mz_settings_browse_frame.pack(side=LEFT)
bkgd_mz_settings_browse_entry = Entry(bkgd_mz_settings_browse_frame, textvariable = mz_bkgd_settings_var, width = 70)
bkgd_mz_settings_browse_entry.pack(side=LEFT)
bkgd_mz_settings_browse_entry.insert(END,'444.1061,614.1936,613.181,615.1925,595.178,596.1608,565.1754')
bkgd_mz_settings_check_choice_frame = Canvas(background_mz_select_frame, width= 100, height= 50)
bkgd_mz_settings_check_choice_frame.pack(side=LEFT)
maldi_choice_bkgd_check = Button(bkgd_mz_settings_check_choice_frame, text = "Check", command=start_background_mz)
maldi_choice_bkgd_check.pack(side=LEFT)
##row 4 functions ###
remove_background_frame = Canvas(filtering_frame, width= 100, height= 50)
remove_background_frame.pack(side=LEFT)
remove_background_button = Button(remove_background_frame, text = "Remove background", command=start_binarized_cell_search)
remove_background_button.pack(side=LEFT)

fill_holes_frame = Canvas(filtering_frame, width= 100, height= 50)
fill_holes_frame.pack(side=LEFT)
fill_holes_button = Button(fill_holes_frame, text = "Fill holes", command=start_fill_holes_search)
fill_holes_button.pack(side=LEFT)

remove_speckles_frame = Canvas(filtering_frame, width= 100, height= 50)
remove_speckles_frame.pack(side=LEFT)
remove_speckles_button = Button(remove_speckles_frame, text = "Remove speckles", command=start_median_filter_search)
remove_speckles_button.pack(side=LEFT)

define_cells_frame = Canvas(filtering_frame, width= 100, height= 50)
define_cells_frame.pack(side=LEFT)
define_cells_button = Button(define_cells_frame, text = "Define cells", command=start_cell_define_inital_search)
define_cells_button.pack(side=LEFT)

###
pixel_text_frame = Canvas(pixel_select_frame, width= 100, height= 50)
pixel_text_frame.pack(side=LEFT)
pixel_text_frame.create_text(37, 25, text="Cell Area\nRange", fill="#2F4FAA", font=(Font_tuple,8),justify=CENTER)
pixel_text_frame.pack(side=LEFT)
min_pixel_entry = Canvas(pixel_select_frame, width= 100, height= 50)
min_pixel_entry.pack(side=LEFT)
min_pixel_entry_bar = Entry(min_pixel_entry, textvariable = min_cell_area, width = 10)
min_pixel_entry_bar.insert(END,'50')
min_pixel_entry_bar.pack(side=LEFT)
dash_text_frame = Canvas(pixel_select_frame, width= 100, height= 50)
dash_text_frame.pack(side=LEFT)
dash_text_frame.create_text(37, 25, text="-", fill="#2F4FAA", font=(Font_tuple,20),justify=CENTER)
dash_text_frame.pack(side=LEFT)
max_pixel_entry = Canvas(pixel_select_frame, width= 100, height= 50)
max_pixel_entry.pack(side=LEFT)
max_pixel_entry_bar = Entry(max_pixel_entry, textvariable = max_cell_area, width = 10)
max_pixel_entry_bar.insert(END,'300')
max_pixel_entry_bar.pack(side=LEFT)
pixel_check_frame = Canvas(pixel_select_frame, width= 100, height= 50)
pixel_check_frame.pack(side=LEFT)
pixel_range_check = Button(pixel_check_frame, text = "Check",command = start_cell_define_area_filter_search)
pixel_range_check.pack(side=LEFT)

###

cell_removal_text_frame = Canvas(intensity_select_frame, width= 150, height= 50)
cell_removal_text_frame.pack(side=LEFT)
cell_removal_text_frame.create_text(80, 25, text="Remove Cell by ID (optional)", fill="#2F4FAA", font=(Font_tuple,8),justify=CENTER)
cell_removal_text_frame.pack(side=LEFT)

cell_removal_entry_frame = Canvas(intensity_select_frame, width= 100, height= 50)
cell_removal_entry_frame.pack(side=LEFT)
cell_removal_entry_bar = Entry(cell_removal_entry_frame, textvariable = cell_to_remove, width = 25)
cell_removal_entry_bar.pack(side=LEFT)

cell_removal_apply_frame = Canvas(intensity_select_frame, width= 100, height= 50)
cell_removal_apply_frame.pack(side=LEFT)
pixel_range_apply = Button(cell_removal_apply_frame, text = "Apply",command=start_cell_remove_specifics_search)
pixel_range_apply.pack(side=LEFT)

export_folder_choice_frame = Canvas(avg_spectra_export_frame, width= 65, height= 50)
export_folder_choice_frame.pack(side=LEFT)
export_folder_choice_text_frame = Canvas(export_folder_choice_frame, width= 150, height= 50)
export_folder_choice_text_frame.pack(side=LEFT)
export_folder_choice_text_frame.create_text(80, 25, text="Export results", fill="#2F4FAA", font=(Font_tuple,8),justify=CENTER)
export_folder_choice_text_frame.pack(side=LEFT)

export_folder_browse_entry = Entry(export_folder_choice_frame, textvariable = output_folder_path, width = 50)
#maldi_choice_browse_entry.insert(END, r"C:\Users\lawashburn\Documents\MSI_test\test_data_from_Hua\small_dataset_pos_20230216\psc_cell_ctrl_3-05-2022_smallarea021623.imzML")
export_folder_browse_entry.pack(side=LEFT)
export_folder_browse = Button(avg_spectra_export_frame, text = "Browse", command = set_output_folder_field)
export_folder_browse.pack(side=LEFT)
export_apply_button = Button(avg_spectra_export_frame, text = "Export",command=export_all_results)
export_apply_button.pack(side=LEFT)

root.mainloop()