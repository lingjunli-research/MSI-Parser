# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:55:28 2023

@author: lawashburn
"""

from tkinter.filedialog import askopenfilename
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from PIL import ImageTk, Image
#from zoom_test2 import CanvasImage

##define window

window = Tk()
#window.geometry("1400x408")
frame = Frame(window, width=600, height=400)
frame.pack()
frame.place(anchor='center', relx=0.5, rely=0.5)

def show_fig1():

    img = ImageTk.PhotoImage(Image.open("assets\\microscope_small.png"))

    # Create a Label Widget to display the text or Image
    label = Label(frame, image = img)
    label.pack()
    
# def show_fig2():

#     img2 = ImageTk.PhotoImage(Image.open("assets\\tims_small.png"))
#     label2 = Label(frame, image = img2)
#     label2.pack()

button_1 = Button(
    text='figure1',
    borderwidth=0,
    highlightthickness=0,
    command=show_fig1,
    relief="flat"
)
button_1.pack()
# button_2 = Button(
#     text='figure2',
#     borderwidth=0,
#     highlightthickness=0,
#     command=show_fig2,
#     relief="flat"
# )
# button_2.pack()
# #
window.mainloop()