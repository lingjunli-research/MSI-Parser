# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:21:56 2023

@author: lawashburn
"""

from tkinter import *
from PIL import ImageTk, Image
# set up the tkinter window

def ChangeImage():
    
    if counter < len(image_list) - 1:
        counter += 1
    else:
        counter = 0
    imageLabel.config(image=image_list[counter])
    infoLabel.config(text="Image " + str(counter + 1) + " of " + str(len(image_list)))

def make_gallary():
    global counter
    root = Tk()
    root.title("MOO ICT Python/Tkinter Image Viewer")
    root.geometry("610x430")
    #root.iconbitmap("images/icon.ico")
    # set up the images
    image1 = ImageTk.PhotoImage(Image.open(r"C:\Users\lawashburn\Documents\MSI_test\gui_test\20230411\cell_number_14.png").resize((600, 350)))
    image2 = ImageTk.PhotoImage(Image.open(r"C:\Users\lawashburn\Documents\MSI_test\gui_test\20230411\cell_number_42.png").resize((600, 350)))
    image3 = ImageTk.PhotoImage(Image.open(r"C:\Users\lawashburn\Documents\MSI_test\gui_test\20230411\cell_number_38.png").resize((600, 350)))
    image4 = ImageTk.PhotoImage(Image.open(r"C:\Users\lawashburn\Documents\MSI_test\gui_test\20230411\cell_number_3.png").resize((600, 350)))
    image5 = ImageTk.PhotoImage(Image.open(r"C:\Users\lawashburn\Documents\MSI_test\gui_test\20230411\cell_number_14.png").resize((600, 350)))
    # add them to the list
    image_list = [image1, image2, image3, image4, image5]
    # counter integer
    counter = 0
    # change image function
    
    # set up the components
    imageLabel = Label(root, image=image1)
    infoLabel = Label(root, text="Image 1 of 5", font="Helvetica, 20")
    button = Button(root, text="Change", width=20, height=2, bg="purple", fg="white", command=ChangeImage)
    # display the components
    imageLabel.pack()
    infoLabel.pack()
    button.pack(side="bottom", pady=3)
    # run the main loop
    root.mainloop()
make_gallary()