# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 09:23:59 2023

@author: lawashburn
"""

from tkinter import *
from PIL import ImageTk, Image,ImageOps
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
import webbrowser
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLParser import getionimage
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from scipy import ndimage
from scipy.ndimage import median_filter
from cycler import cycler
import seaborn as sns

def check_cell_mz_selection():
    str_cell_select = mz_settings_var.get()
    list_str_cell_select = str_cell_select.split(',')
    float_list_str_cell_select = []
    for a in list_str_cell_select:
        float_list_str_cell_select.append(float(a))

def parse_imzml(path):
    #import MSI data
    f = ImzMLParser(path) #ibd path required to parse
    return f
def check_cell_mz(path,string_mz_list,mz,tol,z,parsed_imzml):    #check_cell_mz
    string_split_mz_list = string_mz_list.split(",")
    mz_list = []
    for k in string_split_mz_list:
        mz_list.append(float(k))
    primary_img = getionimage(parsed_imzml, mz, tol, z,reduce_func=sum) # img stored as 2D numpy array
    for mz in mz_list:
        ind = mz_list.index(mz)
        img = getionimage(parsed_imzml, mz, tol, z,reduce_func=sum) # img stored as 2D numpy array
        plt.imshow(img,  interpolation='nearest')
        primary_img = np.add(img,primary_img)
    plt.clf()
    plt.imshow(primary_img,  interpolation='nearest')
    plt.colorbar()
    plt.savefig('maldi_added.png')
    # img=Image.open('maldi_added.png') # read the image file
    # root.img=ImageTk.PhotoImage(img)
    # change_pic(root.img)
    return primary_img
def check_background_mz(path,string_mz_list,mz,tol,z,parsed_imzml):    #check_cell_mz
    # path = input_maldi_text.get()
    # string_mz_list = mz_bkgd_settings_var.get()
    string_split_mz_list = string_mz_list.split(",")
    mz_list = []
    for k in string_split_mz_list:
        mz_list.append(float(k))
    primary_img2 = getionimage(parsed_imzml, mz, tol, z,reduce_func=sum) # img stored as 2D numpy array
    for mz in mz_list:
        ind = mz_list.index(mz)
        img = getionimage(parsed_imzml, mz, tol, z,reduce_func=sum) # img stored as 2D numpy array
        primary_img = np.add(img,primary_img2)
    plt.clf()
    plt.imshow(primary_img2,  interpolation='nearest')
    plt.colorbar()
    plt.savefig('background_added.png')
    # img=Image.open('background_added.png') # read the image file
    # root.img=ImageTk.PhotoImage(img)
    # change_pic(root.img)
    background_array_storage = primary_img
    return primary_img

def make_plot(fig,title):
    from matplotlib_interactor_test import maldi_plot_app
    maldi_plot_app(fig,title)

def binarize_cells(step,foreground_img,bkgd_img):
    average_bkrd = np.average(bkgd_img)
    std_bkrd = np.std(bkgd_img)
    bkrd_intensity = (average_bkrd+std_bkrd)*step
    foreground_img[foreground_img <= bkrd_intensity] = 0
    foreground_img[foreground_img > 0] = 1
    plt.clf()
    fig = plt.figure(figsize = (6, 4))
    plt.imshow(foreground_img, interpolation='nearest')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('assets\\binarized_cells.png')
    return foreground_img
    #make_plot(fig)
def binarize_cells_w_pop(array,title):
    plt.clf()
    fig = plt.figure(figsize = (6, 4))
    plt.imshow(array, interpolation='nearest')
    plt.colorbar()
    plt.tight_layout()
    make_plot(fig,title)

def remove_cellular_holes(binarized_img):
    cellular_holeless_image = ndimage.binary_fill_holes(binarized_img).astype(int) #removes any holes in the interior of a cell
    plt.clf()
    plt.imshow(cellular_holeless_image, interpolation='nearest')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('assets\\cell_fill_holes.png')
    # img_png=Image.open('cell_fill_holes.png') # read the image file
    # root.img_png=ImageTk.PhotoImage(img_png)
    # change_pic(root.img_png)
    return cellular_holeless_image

def apply_median_filter_cells(filter_size,holes_filled):
    filtered_array = median_filter(holes_filled, size=filter_size) #remove any "speckles" or regions of just one or two pixels
    plt.clf()
    plt.imshow(filtered_array, interpolation='nearest')
    plt.colorbar()
    plt.savefig('assets\\median_filtered_cells.png')
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
    #print(len(cell_localization_df))
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
    plt.clf()
    filtered_cells_clean = filtered_cells.copy()
    x_values_filtered = filtered_cells_clean['x'].values.tolist()
    y_values_filtered = filtered_cells_clean['y'].values.tolist()
    t_values_filtered = filtered_cells_clean['t'].values.tolist()
    t_values_filtered_str = []
    for a in t_values_filtered:
        t_values_filtered_str.append(str(a))

    sns.scatterplot(x=x_values_filtered, y=y_values_filtered, hue=t_values_filtered_str,linewidth=0, palette='husl')
    plt.gca().invert_yaxis()
    plt.legend(loc='upper left',ncol=2, title="Cell #",bbox_to_anchor=(1, 1),fontsize='small',title_fontsize='medium')
    #plt.legend()
    plt.savefig('assets\\area_filtered_cells.png', bbox_inches = 'tight')
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
    plt.gca().invert_yaxis()
    plt.legend(loc='upper left',ncol=2, title="Cell #",bbox_to_anchor=(1, 1),fontsize='small',title_fontsize='medium')
    plt.savefig('assets\\manual_filtered_cells.png', bbox_inches = 'tight')
    return filtered_cells2
def mzbin(mz,intensity,start,end,resolution):
    # mz is an array of m/z values of a mass spectrum at some pixel
    # intensity is an array of corresponding intensities
    # start and end are the lower and upper bound m/z
    # resolution defines res of mass binning
    # returns result_mz: if a bin has edges 0.1-0.2, returns its center 0.15
    # result_intensity: the average intensity of every value falls into that bin
    result_mz=[]
    result_intensity=[]
    # step=(end-start)/resolution
    sort_mz=[x for x,_ in sorted(zip(mz,intensity))]
    sort_inten=[y for _,y in sorted(zip(mz,intensity))]
    HIST_BINS = np.linspace(start, end, resolution)
    n, m = np.histogram(sort_mz, HIST_BINS)
    counter=0
    for i in range(len(n)):
        hist_val=sort_inten[counter:counter+n[i]]
        if len(hist_val)==0:
            result_intensity.append(0)
        else:
            avg=np.average(hist_val)
            result_intensity.append(avg)

        result_mz.append((m[i]+m[i+1])/2)
        counter+=n[i]
    return result_mz,result_intensity
def export_spectra(filtered_cells2,path,weighted_avg,output_spectra_path,min_mz,max_mz,resolution):
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

    f = parse_imzml(path)
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
    if weighted_avg==1:    
        final_report = pd.DataFrame()
        for m in final_cell_list:
            cell=filtered_cells2.loc[filtered_cells2['t'] == m]
            # do mass binning for each pixel in a cell, then calculate the average mass spectrum
            total_intensity=0
            output_intensity=[]
            output_mz=[]
            for i in range(len(cell)):
                try:
                    xx=cell.iloc[i]['x']
                    yy=cell.iloc[i]['y']
                    index=coord_dict[(xx+1,yy+1)]
                    mz, intensity = ImzMLParser.getspectrum(f,index) # retrieve mass spectrum at a pixel
                    sum_intensity=sum(intensity)
                    total_intensity+=sum_intensity
                    result_mz,result_intensity=mzbin(mz, intensity, min_mz,max_mz,resolution)
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
            file_name = output_spectra_path + "cell_number_" + str(m) + ".png" 
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
        file_path = output_spectra_path + 'avg_spectra_all.csv'
        with open(file_path,'w',newline='') as filec:
                    writerc = csv.writer(filec)
                    final_report.to_csv(filec,index=False)
    
    if weighted_avg==0:
        final_report = pd.DataFrame()
        for m in final_cell_list:
            cell_one=filtered_cells2.loc[filtered_cells2['t'] == m]

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
                    
                    result_mz,result_intensity=mzbin(mz, intensity,min_mz,max_mz,resolution)

                    # result_intensity=np.asarray(result_intensity)*sum_intensity 
                    output_intensity.append(result_intensity)
                    output_mz=(result_mz)
                except Exception:
                    pass
            
            output=np.sum(output_intensity,axis=0)/total_intensity
            plt.plot(output_mz,output)
            plt.title("Cell #: "+str(m))
            file_name = output_spectra_path + "\\cell_number_" + str(m) + ".png" 
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
        file_path = output_spectra_path + '\\avg_spectra_all.csv'
        with open(file_path,'w',newline='') as filec:
                    writerc = csv.writer(filec)
                    final_report.to_csv(filec,index=False)

def check_cell_mz(path,string_split_mz_list,mz,tol,z,parsed_imzml):    #check_cell_mz
    mz_list = []
    for k in string_split_mz_list:
        mz_list.append(float(k))
    primary_img = getionimage(parsed_imzml, mz, tol, z,reduce_func=sum) # img stored as 2D numpy array
    for mz in mz_list:
        ind = mz_list.index(mz)
        img = getionimage(parsed_imzml, mz, tol, z,reduce_func=sum) # img stored as 2D numpy array
        plt.imshow(img,  interpolation='nearest')
        primary_img = np.add(img,primary_img)
        plt_path = str(ind) + '_img.png'
        plt.savefig(plt_path)
    plt.clf()
    plt.imshow(primary_img,  interpolation='nearest')
    plt.colorbar()
    plt.savefig('assets\\maldi_added.png')
    return primary_img

def check_background_mz(path,string_split_mz_list,mz,tol,z,parsed_imzml):    #check_cell_mz
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
        plt.imshow(img,  interpolation='nearest')
        primary_img = np.add(img,primary_img)
        plt_path = str(ind) + '_img.png'
        plt.savefig(plt_path)
    plt.clf()
    plt.imshow(primary_img2,  interpolation='nearest')
    plt.colorbar()
    plt.savefig('assets\\background_added.png')
    return primary_img