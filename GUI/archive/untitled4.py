# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:18:44 2023

@author: lawashburn
"""
from tkinter import *
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
import sys
from itertools import islice
import subprocess
from subprocess import Popen, PIPE
from textwrap import dedent
from tkinter import messagebox
from tkinter import filedialog
import os
import pickle
from tkinter import ttk
import threading
import traceback
import time
import psutil
import webbrowser
from PIL import Image,ImageTk
from pdf2image import convert_from_path

### GUI appearance settings ###
Font_tuple = ("Corbel Light", 20)
root = Tk()
root.title('Neuropeptide Database Search')
root.iconbitmap(r"hypep_icon.ico")
root.geometry('1500x800')

##Variable storage
input_optical_text = StringVar()
input_maldi_text = StringVar()
mz_settings_var = StringVar()
mz_bkgd_settings_var = StringVar()

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
    path_optical = askopenfilename(filetypes=[("Image Files",("*.png","*.jpeg"))]) 
    input_optical_text.set(path_optical)

    img=Image.open(path_optical) # read the image file
    img=img.resize((500,500)) # new width & height
    img=ImageTk.PhotoImage(img)
    e1 =Label(optical_image_frame)
    e1.pack(side=TOP)
    e1.image = img # keep a reference! by attaching it to a widget attribute
    e1['image']=img # Show Image  

def set_maldi_image_field():
    path_maldi = askopenfilename(filetypes=[("MSI Files",("*.imzML"))]) 
    input_maldi_text.set(maldi_optical)
    
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

main_frame = Canvas(root, width= 800, height= 800)
main_frame.pack(side=TOP)

optical_entry_frame = Canvas(main_frame, width= 200, height= 55)
optical_entry_frame.pack(side=LEFT)
optical_entry_text_frame = Canvas(optical_entry_frame, width= 103, height= 50)
optical_entry_text_frame.pack(side=LEFT)
optical_entry_text_frame.create_text(53, 25, text="Optical image", fill="#2F4FAA", font=(Font_tuple,8),justify=CENTER)
optical_entry_text_frame.pack()
optical_entry_choice_frame = Canvas(optical_entry_frame, width= 200, height= 50)
optical_entry_choice_frame.pack(side=LEFT)
optical_choice_browse_entry = Entry(optical_entry_choice_frame, textvariable = input_optical_text, width = 63)
optical_choice_browse_entry.pack(side=LEFT)
optical_choice_browse = Button(optical_entry_choice_frame, text = "Browse", command = set_optical_image_field)
optical_choice_browse.pack(side=RIGHT)
optical_image_frame = Canvas(optical_entry_frame, width= 200, height= 55)
optical_image_frame.pack(side=BOTTOM)

maldi_entry_frame = Canvas(main_frame, width= 200, height= 55)
maldi_entry_frame.pack(side=LEFT)

maldi_path_frame = Canvas(maldi_entry_frame, width= 200, height= 55)
maldi_path_frame.pack(side=TOP)

cell_mz_select_frame = Canvas(maldi_entry_frame, width= 200, height= 55)
cell_mz_select_frame.pack(side=TOP)

background_mz_select_frame = Canvas(maldi_entry_frame, width= 200, height= 55)
background_mz_select_frame.pack(side=TOP)

maldi_entry_text_frame = Canvas(maldi_path_frame, width= 103, height= 50)
maldi_entry_text_frame.pack(side=TOP)
maldi_entry_text_frame.create_text(53, 25, text="MSI Path (.imzML)", fill="#2F4FAA", font=(Font_tuple,8),justify=CENTER)
maldi_entry_text_frame.pack(side=LEFT)
maldi_entry_choice_frame = Canvas(maldi_path_frame, width= 200, height= 50)
maldi_entry_choice_frame.pack(side=LEFT)
maldi_choice_browse_entry = Entry(maldi_entry_choice_frame, textvariable = input_maldi_text, width = 70)
maldi_choice_browse_entry.pack(side=LEFT)
maldi_choice_browse = Button(maldi_entry_choice_frame, text = "Browse", command = set_maldi_image_field)
maldi_choice_browse.pack(side=LEFT)

mz_settings_frame = Canvas(cell_mz_select_frame, width= 650, height= 50)
mz_settings_frame.pack(side=TOP)
mz_settings_entry_text_frame = Canvas(mz_settings_frame, width= 103, height= 50)
mz_settings_entry_text_frame.pack(side=LEFT)
mz_settings_entry_text_frame.create_text(37, 25, text="Cell m/z", fill="#2F4FAA", font=(Font_tuple,8),justify=LEFT)
mz_settings_entry_text_frame.pack(side=LEFT)
mz_settings_entry_choice_frame = Canvas(mz_settings_frame, width= 300, height= 50)
mz_settings_entry_choice_frame.pack(side=RIGHT)
mz_settings_browse_frame = Canvas(mz_settings_entry_choice_frame, width= 300, height= 50)
mz_settings_browse_frame.pack(side=TOP)
mz_settings_browse_entry = Entry(mz_settings_browse_frame, textvariable = mz_settings_var, width = 70)
mz_settings_browse_entry.pack(side=LEFT)
maldi_choice_browse = Button(mz_settings_browse_frame, text = "Check")
maldi_choice_browse.pack(side=LEFT)

mz_bkgd_settings_frame = Canvas(background_mz_select_frame, width= 650, height= 50)
mz_bkgd_settings_frame.pack(side=TOP)
mz_bkgd_settings_entry_text_frame = Canvas(mz_bkgd_settings_frame, width= 103, height= 50)
mz_bkgd_settings_entry_text_frame.pack(side=LEFT)
mz_bkgd_settings_entry_text_frame.create_text(37, 25, text="Background\nm/z", fill="#2F4FAA", font=(Font_tuple,8),justify=LEFT)
mz_bkgd_settings_entry_text_frame.pack(side=LEFT)
mz_bkgd_settings_entry_choice_frame = Canvas(mz_bkgd_settings_frame, width= 300, height= 50)
mz_bkgd_settings_entry_choice_frame.pack(side=RIGHT)
mz_bkgd_settings_browse_frame = Canvas(mz_bkgd_settings_entry_choice_frame, width= 300, height= 50)
mz_bkgd_settings_browse_frame.pack(side=TOP)
mz_bkgd_settings_browse_entry = Entry(mz_bkgd_settings_browse_frame, textvariable = mz_bkgd_settings_var, width = 70)
mz_bkgd_settings_browse_entry.pack(side=LEFT)
maldi_bkgd_choice_browse = Button(mz_bkgd_settings_browse_frame, text = "Check")
maldi_bkgd_choice_browse.pack(side=LEFT)

root.mainloop()
