
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path
from tkinter.filedialog import askopenfilename
from tkinter import *
from PIL import ImageTk, Image,ImageOps
import PIL
import backend
import threading
from backend import *
#from zoom_test2 import CanvasImage

##define window
Font_tuple = ("Corbel Light", 20)
window = Tk()
window.title('MSI Parser')
window.iconbitmap(r"assets\LiClaw.ico")

window.geometry("1400x408")
window.configure(bg = "#FFFFFF")


###

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")
##Variable storage
# input_optical_text = StringVar()
# input_maldi_text = StringVar()
# mz_settings_var = StringVar()
# mz_bkgd_settings_var = StringVar()
# min_cell_area = StringVar()
# max_cell_area = StringVar()
# mz_choice = StringVar()
# tolerance_choice =  StringVar()
# z_choice = StringVar()
# intensity_thresh_choice = StringVar()
# resolution_choice = StringVar()
# primary_background_mz = StringVar()
# cell_to_remove = StringVar()
# output_folder_path = StringVar()

maldi_path_storage = StringVar()
optical_path_storage = StringVar()
export_path_storage = StringVar()
cell_mz_list_storage = StringVar()
bgrd_mz_list_storage = StringVar()
mz_value_storage = StringVar()
bgrd_mz_value_storage = StringVar()
resolution_storage = StringVar()
intensity_storage = StringVar()
charge_storage = StringVar()
tolerance_storage = StringVar()
median_size_storage = StringVar()
bin_step_size_storage = StringVar()
min_cell_size_storage = StringVar()
max_cell_size_storage = StringVar()
cell_remove_storage = StringVar()
latest_step_var = StringVar()

##Definition storage
def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)
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



def determine_optical_frame():
    value = optical_path_storage.get()
    if len(value) > 1:
        path_to_show = value
    else:
        path_to_show = 'microscope_small.png'
    return path_to_show

def set_optical_image_field():

    path_optical = askopenfilename(filetypes=[("Image Files",("*.png","*.jpeg","*.TIF"))]) 
    optical_path_storage.set(path_optical)

    photo_choice = determine_optical_frame()
    
    microscope_image1 = Image.open(
        relative_to_assets(photo_choice))
    w, h = microscope_image1.size
    if w>360:
        ratio = 360/w
        print(ratio)
    elif h>308:
        ratio = 360/h
    else:
        ratio=1 
    new_width = int(w*ratio)
    new_height = int(h*ratio)
    microscope_image2=microscope_image1.resize((new_width,new_height),Image.ANTIALIAS)
    im1 = microscope_image2.save('assets\\optical_test2.png')
    microscope_image = PhotoImage(
        file=relative_to_assets("optical_test2.png"))
    canvas.create_rectangle(
        465.0,
        44.0,
        825.0,
        352.0,
        fill="#B19BB3",
        outline="")
    image_1 = canvas.create_image(
        645.0,
        158.0,
        image=microscope_image
    ).place()

def start_binarized_cell_search_w_pop(array):
    # path = maldi_path_storage.get()
    # cell_string_mz_list = cell_mz_list_storage.get()
    # cell_mz = float(mz_value_storage.get())
    # bkgd_string_mz_list = bgrd_mz_list_storage.get()
    # bkgd_mz = float(bgrd_mz_value_storage.get())
    # tol = float(tolerance_storage.get())
    # z = int(charge_storage.get())
    # parsed_imzml = parse_imzml(path)
    # foreground_img = check_cell_mz(path,cell_string_mz_list,cell_mz,tol,z,parsed_imzml)    
    # bkgd_img = check_background_mz(path,bkgd_string_mz_list,bkgd_mz,tol,z,parsed_imzml) 
    binarized_cells = binarize_cells_w_pop(array)


def execute_maldi_pop(array):
    start_binarized_cell_search_w_pop(array)

def determine_maldi_frame():
    value = latest_step_var.get()
    if value == 'binarized':
        path_to_show = 'binarized_cells.png'
    else:
        path_to_show = 'tims_small.png'
    print(path_to_show)
    return path_to_show

def set_maldi_image_field(image_path):
    determine_maldi_frame()
    button_image_1100 = PhotoImage(
        file=relative_to_assets("button_110.png"))
    button_1100 = Button(
        image=button_image_1100,
        borderwidth=0,
        highlightthickness=0,
        command='',
        relief="flat",
        bg='#8C6D8F'
    )
    button_1100.place(
        x=1050.0,
        y=300.0,
        width=55.0,
        height=17.0
    )
#execute_maldi_pop(array)
    maldi_image_path = determine_maldi_frame()
    maldi_image1 = Image.open(
        relative_to_assets(maldi_image_path))
    w, h = maldi_image1.size
    if w>360:
        ratio = 360/w
    elif h>308:
        ratio = 360/h
    else:
        ratio=1 
    new_width = int(w*ratio)
    new_height = int(h*ratio)
    
    maldi_image2=maldi_image1.resize((new_width,new_height),Image.ANTIALIAS)
    im1 = maldi_image2.save('assets\\maldi_image_resize.png')
    maldi_image = PhotoImage(
        file=relative_to_assets("maldi_image_resize.png"))
    canvas.create_rectangle(
        875.0,
        44.0,
        1280.0,
        300.0,
        outline="")
    image_1 = canvas.create_image(
        1075.0,
        175.0,
        image=maldi_image
    ).place()

def optical_pop_out():
    from optical_zoom_window import app_call
    filename_path = determine_optical_frame()  # place path to your image here
    app_call(filename_path)

def start_binarized_cell_search():
    latest_step_var.set('binarized')
    # path = maldi_path_storage.get()
    # cell_string_mz_list = cell_mz_list_storage.get()
    # cell_mz = float(mz_value_storage.get())
    # bkgd_string_mz_list = bgrd_mz_list_storage.get()
    # bkgd_mz = float(bgrd_mz_value_storage.get())
    # tol = float(tolerance_storage.get())
    # z = int(charge_storage.get())
    # parsed_imzml = parse_imzml(path)
    # foreground_img = check_cell_mz(path,cell_string_mz_list,cell_mz,tol,z,parsed_imzml)    
    # bkgd_img = check_background_mz(path,bkgd_string_mz_list,bkgd_mz,tol,z,parsed_imzml) 
    # binarized_cells = binarize_cells(foreground_img,bkgd_img)
    set_maldi_image_field('binarized_cells.png')

menubar = Menu(window)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Exit", command=window.quit)
menubar.add_cascade(label="File", menu=filemenu)
helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="Li Lab Website", command=openweb_liweb)
helpmenu.add_command(label="Li Lab GitHub", command=openweb_git)
helpmenu.add_command(label="User manual", command=openweb_user_manual)
menubar.add_cascade(label="Help", menu=helpmenu)
toolmenu = Menu(menubar, tearoff=0)
toolmenu.add_command(label="Step evaluate tool")
menubar.add_cascade(label="Tools", menu=toolmenu)

window.config(menu=menubar)


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 408,
    width = 1403,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    1289.0,
    0.0,
    1402.0,
    408.0,
    fill="#5F4A61",
    outline="")

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command = lambda: start_binarized_cell_search(),
    relief="flat"
)
button_1.place(
    x=1292.0,
    y=90.0,
    width=107.0,
    height=33.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_2 clicked"),
    relief="flat"
)
button_2.place(
    x=1293.0,
    y=135.0,
    width=107.0,
    height=40.0
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_3 clicked"),
    relief="flat"
)
button_3.place(
    x=1293.0,
    y=179.0,
    width=107.0,
    height=40.0
)

button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_4 clicked"),
    relief="flat"
)
button_4.place(
    x=1292.0,
    y=225.0,
    width=107.0,
    height=40.0
)

button_image_5 = PhotoImage(
    file=relative_to_assets("button_5.png"))
button_5 = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_5 clicked"),
    relief="flat"
)
button_5.place(
    x=1293.0,
    y=269.0,
    width=107.0,
    height=40.0
)

canvas.create_rectangle(
    0.0,
    0.0,
    430.0,
    408.0,
    fill="#D5C9D6",
    outline="")

canvas.create_text(
    165.0,
    8.0,
    anchor="nw",
    text="Analysis Settings",
    fill="#000000",
    font=("HammersmithOne Regular", 16 * -1,'bold')
)

# entry_image_1 = PhotoImage(
#     file=relative_to_assets("entry_1.png"))
# entry_bg_1 = canvas.create_image(
#     243.0,
#     73.5,
#     image=entry_image_1
# )
entry_1 = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0,
    textvariable=maldi_path_storage
)
entry_1.place(
    x=120.0,
    y=65.0,
    width=246.0,
    height=15.0
)
entry_1.insert(END,r"C:\Users\lawashburn\Documents\MSI_test\test_data_from_Hua\small_dataset_pos_20230216\psc_cell_ctrl_3-05-2022_smallarea021623.imzML")
button_image_6 = PhotoImage(
    file=relative_to_assets("button_6.png"))
button_6 = Button(
    image=button_image_6,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_6 clicked"),
    relief="flat",
    bg='#D5C9D6'
)
button_6.place(
    x=367.0,
    y=65.0,
    width=55.0,
    height=17.0
)

canvas.create_text(
    38.0,
    60.0,
    anchor="nw",
    text="Spectral file\n(.imzML)",
    fill="#000000",
    font=("Inter", 12 * -1),
    justify='center'
)

# entry_image_2 = PhotoImage(
#     file=relative_to_assets("entry_2.png"))
# entry_bg_2 = canvas.create_image(
#     240.0,
#     385.5,
#     image=entry_image_2
# )
entry_2 = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0,
    textvariable = export_path_storage
)
entry_2.place(
    x=117.0,
    y=377.0,
    width=246.0,
    height=15.0
)
entry_2.insert(END,r"C:\Users\lawashburn\Documents\MSI_test\gui_test\20230411")
button_image_7 = PhotoImage(
    file=relative_to_assets("button_7.png"))
button_7 = Button(
    image=button_image_7,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_7 clicked"),
    relief="flat",
    bg='#D5C9D6'
)
button_7.place(
    x=364.0,
    y=377.0,
    width=55.0,
    height=17.0
)

canvas.create_text(
    34.0,
    378.0,
    anchor="nw",
    text="Export folder",
    fill="#000000",
    font=("Inter", 12 * -1)
)

button_image_8 = PhotoImage(
    file=relative_to_assets("button_8.png"))
button_8 = Button(
    image=button_image_8,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_8 clicked"),
    relief="flat",
    bg='#D5C9D6'
)
button_8.place(
    x=367.0,
    y=241.0,
    width=55.0,
    height=17.0
)

# entry_image_3 = PhotoImage(
#     file=relative_to_assets("entry_3.png"))
# entry_bg_3 = canvas.create_image(
#     225.0,
#     249.5,
#     image=entry_image_3
# )
entry_3 = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0,
    textvariable = cell_mz_list_storage
)
entry_3.insert(END,'810.6052,632.6295,604.5808,760.5623,808.5812,788.5988,786.5904')
entry_3.place(
    x=87.0,
    y=241.0,
    width=276.0,
    height=15.0
)

canvas.create_text(
    17.0,
    236.0,
    anchor="nw",
    text="Cell m/z\nlist",
    fill="#000000",
    font=("Inter", 12 * -1),
    justify='center'
)

button_image_9 = PhotoImage(
    file=relative_to_assets("button_9.png"))
button_9 = Button(
    image=button_image_9,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_9 clicked"),
    relief="flat",
    bg='#D5C9D6'
)
button_9.place(
    x=367.0,
    y=279.0,
    width=55.0,
    height=17.0
)

# entry_image_4 = PhotoImage(
#     file=relative_to_assets("entry_4.png"))
# entry_bg_4 = canvas.create_image(
#     225.0,
#     287.5,
#     image=entry_image_4
# )
entry_4 = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0,
    textvariable = bgrd_mz_list_storage
)
entry_4.insert(END,'444.1061,614.1936,613.181,615.1925,595.178,596.1608,565.1754')
entry_4.place(
    x=87.0,
    y=279.0,
    width=276.0,
    height=15.0
)

canvas.create_text(
    9.0,
    275.0,
    anchor="nw",
    text="Background \nm/z list",
    fill="#000000",
    font=("Inter", 12 * -1),
    justify='center'
)

# entry_image_5 = PhotoImage(
#     file=relative_to_assets("entry_5.png"))
# entry_bg_5 = canvas.create_image(
#     119.5,
#     131.5,
#     image=entry_image_5
# )
entry_5 = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0,
    textvariable = mz_value_storage
)
entry_5.insert(END,'810.6052')
entry_5.place(
    x=95.0,
    y=123.0,
    width=49.0,
    height=15.0
)

canvas.create_text(
    40.0,
    123.0,
    anchor="nw",
    text="m/z",
    fill="#000000",
    font=("Inter", 12 * -1)
)

# entry_image_6 = PhotoImage(
#     file=relative_to_assets("entry_6.png"))
# entry_bg_6 = canvas.create_image(
#     397.5,
#     175.5,
#     image=entry_image_6
# )
entry_6 = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0,
    textvariable = bgrd_mz_value_storage
)
entry_6.insert(END,'580.4977')
entry_6.place(
    x=373.0,
    y=167.0,
    width=49.0,
    height=15.0
)

canvas.create_text(
    295.0,
    163.0,
    anchor="nw",
    text="background\nm/z",
    fill="#000000",
    font=("Inter", 12 * -1),
    justify='center'
)

# entry_image_7 = PhotoImage(
#     file=relative_to_assets("entry_7.png"))
# entry_bg_7 = canvas.create_image(
#     258.5,
#     175.5,
#     image=entry_image_7
# )
entry_7 = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0,
    textvariable = resolution_storage
)
entry_7.insert(END,'7000')
entry_7.place(
    x=234.0,
    y=167.0,
    width=49.0,
    height=15.0
)

canvas.create_text(
    163.0,
    168.0,
    anchor="nw",
    text="resolution",
    fill="#000000",
    font=("Inter", 12 * -1)
)

# entry_image_8 = PhotoImage(
#     file=relative_to_assets("entry_8.png"))
# entry_bg_8 = canvas.create_image(
#     119.5,
#     175.5,
#     image=entry_image_8
# )
entry_8 = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0,
    textvariable = intensity_storage
)
entry_8.insert(END,'2000')
entry_8.place(
    x=95.0,
    y=167.0,
    width=49.0,
    height=15.0
)

canvas.create_text(
    24.0,
    163.0,
    anchor="nw",
    text="Intensity\nthreshold",
    fill="#000000",
    font=("Inter", 12 * -1),
    justify='center'
)

# entry_image_9 = PhotoImage(
#     file=relative_to_assets("entry_9.png"))
# entry_bg_9 = canvas.create_image(
#     397.5,
#     131.5,
#     image=entry_image_9
# )
entry_9 = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0,
    textvariable = charge_storage
)
entry_9.place(
    x=373.0,
    y=123.0,
    width=49.0,
    height=15.0
)
entry_9.insert(END,'1')
canvas.create_text(
    325.0,
    123.0,
    anchor="nw",
    text="z",
    fill="#000000",
    font=("Inter", 12 * -1)
)

# entry_image_10 = PhotoImage(
#     file=relative_to_assets("entry_10.png"))
# entry_bg_10 = canvas.create_image(
#     258.5,
#     131.5,
#     image=entry_image_10
# )
entry_10 = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0,
    textvariable = tolerance_storage
)
entry_10.place(
    x=234.0,
    y=123.0,
    width=49.0,
    height=15.0
)
entry_10.insert(END,'0.1')
canvas.create_text(
    165.0,
    123.0,
    anchor="nw",
    text="tolerance",
    fill="#000000",
    font=("Inter", 12 * -1)
)

entry_13 = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0,
    textvariable = median_size_storage
)
entry_13.place(
    x=150.0,
    y=205.0,
    width=49.0,
    height=15.0
)
entry_13.insert(END,'2')
canvas.create_text(
    75.0,
    200.0,
    anchor="nw",
    text="median filter\nsize",
    fill="#000000",
    font=("Inter", 12 * -1),
    justify='center'
)

entry_14 = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0,
    textvariable = bin_step_size_storage
)
entry_14.place(
    x=300.0,
    y=205.0,
    width=49.0,
    height=15.0
)
entry_14.insert(END,'6')
canvas.create_text(
    245.0,
    200.0,
    anchor="nw",
    text="binarize\nstep size",
    fill="#000000",
    font=("Inter", 12 * -1),
    justify='center'
)

# entry_image_11 = PhotoImage(
#     file=relative_to_assets("entry_11.png"))
# entry_bg_11 = canvas.create_image(
#     212.5,
#     336.5,
#     image=entry_image_11
# )
entry_11 = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0,
    textvariable = min_cell_size_storage
)
entry_11.place(
    x=188.0,
    y=328.0,
    width=49.0,
    height=15.0
)
entry_11.insert(END,'50')
# entry_image_12 = PhotoImage(
#     file=relative_to_assets("entry_12.png"))
# entry_bg_12 = canvas.create_image(
#     289.5,
#     336.5,
#     image=entry_image_12
# )
entry_12 = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0,
    textvariable = max_cell_size_storage
)
entry_12.place(
    x=265.0,
    y=328.0,
    width=49.0,
    height=15.0
)
entry_12.insert(END,'300')
canvas.create_text(
    95.0,
    328.0,
    anchor="nw",
    text="Cell area range",
    fill="#000000",
    font=("Inter", 12 * -1)
)

canvas.create_text(
    242.0,
    310.0,
    anchor="nw",
    text="-",
    fill="#000000",
    font=("Inter", 40 * -1)
)

canvas.create_rectangle(
    430.0,
    0.0,
    860.0,
    408.0,
    fill="#B19BB3",
    outline="")

canvas.create_text(
    600.0,
    8.0,
    anchor="nw",
    text="Optical Viewer",
    fill="#000000",
    font=("HammersmithOne Regular", 16 * -1,'bold')
)

# entry_image_13 = PhotoImage(
#     file=relative_to_assets("entry_13.png"))
# entry_bg_13 = canvas.create_image(
#     658.0,
#     385.5,
#     image=entry_image_13
# )
entry_13 = Entry(
    bd=0,
    bg="#FFFFFF",
    highlightthickness=0,
    textvariable=optical_path_storage
)
entry_13.place(
    x=500.0,
    y=377.0,
    width=246.0,
    height=15.0
)

button_image_10 = PhotoImage(
    file=relative_to_assets("button_10.png"))
button_10 = Button(
    image=button_image_10,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: set_optical_image_field(),
    relief="flat",
    bg='#B19BB3'
)
button_10.place(
    x=740.0,
    y=377.0,
    width=55.0,
    height=17.0
)



canvas.create_text(
    450.0,
    360.0,
    anchor="nw",
    text="Select\noptical\nimage",
    fill="#000000",
    font=("Inter", 12 * -1),
    justify='center'
)

microscope_image = PhotoImage(
    file=relative_to_assets("microscope_small.png"))

canvas.create_rectangle(
    465.0,
    44.0,
    825.0,
    352.0,
    fill="#B19BB3",
    outline="")

image_1 = canvas.create_image(
    645.0,
    158.0,
    image=microscope_image
)


canvas.create_text(
    525.0,
    275.0,
    anchor="nw",
    text="Image will display upon selection",
    fill="#000000",
    font=("Inter", 16 * -1)
)

canvas.create_rectangle(
    860.0,
    0.0,
    1290.0,
    408.0,
    fill="#8C6D8F",
    outline="")

canvas.create_text(
    1015.0,
    8.0,
    anchor="nw",
    text="MALDI Image Viewer",
    fill="#000000",
    font=("HammersmithOne Regular", 16 * -1,'bold')
)

# entry_image_14 = PhotoImage(
#     file=relative_to_assets("entry_14.png"))
# entry_bg_14 = canvas.create_image(
#     1088.0,
#     385.5,
#     image=entry_image_14
# )
entry_14 = Entry(
    bd=0,
    bg="#F9F8F9",
    highlightthickness=0,
    textvariable = cell_remove_storage
)
entry_14.place(
    x=940.0,
    y=377.0,
    width=246.0,
    height=15.0
)

button_image_11 = PhotoImage(
    file=relative_to_assets("button_11.png"))
button_11 = Button(
    image=button_image_11,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_11 clicked"),
    relief="flat",
    bg='#8C6D8F'
)
button_11.place(
    x=1180.0,
    y=377.0,
    width=55.0,
    height=17.0
)

button_image_110 = PhotoImage(
    file=relative_to_assets("button_110.png"))
button_110 = Button(
    image=button_image_110,
    borderwidth=0,
    highlightthickness=0,
    command=optical_pop_out,
    relief="flat",
    bg='#B19BB3'
)
button_110.place(
    x=793.0,
    y=377.0,
    width=55.0,
    height=17.0
)

canvas.create_text(
    885.0,
    360.0,
    anchor="nw",
    text="Remove\ncell by\nnumber",
    fill="#000000",
    font=("Inter", 12 * -1),
    justify='center'
)
tims_image = PhotoImage(
    file=relative_to_assets("tims_small.png"))

canvas.create_rectangle(
    895.0,
    44.0,
    1255.0,
    352.0,
    fill="#8C6D8F",
    outline="")

image_2 = canvas.create_image(
    1075.0,
    158.0,
    image=tims_image
)

canvas.create_text(
    960.0,
    275.0,
    anchor="nw",
    text="Image will display upon selection",
    fill="#000000",
    font=("Inter", 16 * -1)
)

canvas.create_text(
    1312.0,
    8.0,
    anchor="nw",
    text="Workflow",
    fill="#000000",
    font=("HammersmithOne Regular", 16 * -1,'bold')
) #1238.0
window.resizable(False, False)
window.mainloop()
