# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 16:39:28 2023

@author: lawashburn
"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import cv2
import copy
import csv
from IPython.display import display
from skimage import feature
from skimage import measure
from skimage.io import imread
from scipy import ndimage
import kaleido
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLParser import getionimage
from scipy import signal
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import cv2
import copy
import csv
from IPython.display import display
from skimage import feature
from skimage import measure
from skimage.io import imread
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLParser import getionimage
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import cv2
from scipy.optimize import dual_annealing
import os
import sys
from skimage.measure import regionprops
from skimage.filters import median, gaussian, threshold_local
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, relabel_sequential
from skimage.morphology import disk, label
from scipy import ndimage
import xlsxwriter
from sklearn.cluster import KMeans, DBSCAN
import random
import scipy
import scipy.ndimage
from PIL import Image
from numpy import asarray
from scipy import stats as st
from scipy.ndimage import median_filter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

path = r"C:\Users\lawashburn\Documents\MSI_test\test_data_from_Hua\small_dataset_pos_20230216\psc_cell_ctrl_3-05-2022_smallarea021623.imzML"
mz = 810.6052
mz_tolerance = 10
z_value = 1
mz_list = [810.6052,632.6295,604.5808,760.5623,808.5812,788.5988,786.5904]
#mz_list = [810.6052,632.6295,604.5808]

def parse_imzml(path):
    #import MSI data
    f = ImzMLParser(path) #ibd path required to parse
    return f

def check_cell_mz(path,mz_list, mz, mz_tolerance, z_value):
    parsed_imzml = parse_imzml(path)
    primary_img = getionimage(parsed_imzml, mz, tol=mz_tolerance, z=z_value,reduce_func=sum) # img stored as 2D numpy array
    for mz in mz_list:
        ind = mz_list.index(mz)
        img = getionimage(parsed_imzml, mz, tol=mz_tolerance, z=z_value,reduce_func=sum) # img stored as 2D numpy array
        plt.imshow(img,  interpolation='nearest')
        primary_img = np.add(img,primary_img)
    plt.colorbar()
    plt.show()
    return primary_img
    

results = check_cell_mz(path,mz_list, mz, mz_tolerance, z_value)
#%%



# parsed_imzml = parse_imzml(path)
# # for mz in mz_list:
# img = getionimage(parsed_imzml, mz, tol=mz_tolerance, z=z_value,reduce_func=sum) # img stored as 2D numpy array
# plt.imshow(img,  interpolation='nearest')
# plt.show()

# for mz in background_mz_list:
#     img = getionimage(f, mz, tol=mz_tolerance, z=z_value,reduce_func=sum) # img stored as 2D numpy array
#     fig = px.imshow(img,title='Step 1: Combined Ion Plot')
#     fig_path = output_background_img_path + "\\" + str(mz) + ".png"
#     fig.write_image(fig_path) 
# img = getionimage(f, mz, tol=mz_tolerance, z=z_value,reduce_func=sum) # img stored as 2D numpy array
# for value in mz_list:
#     mz_img = getionimage(f, value, tol=mz_tolerance, z=z_value,reduce_func=sum) # img stored as 2D numpy array
#     img = np.add(img,mz_img)
# fig = px.imshow(img,title='Step 1: Combined Ion Plot')