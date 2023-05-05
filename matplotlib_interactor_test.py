import tkinter

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np

def maldi_plot_app(plot,title):
    root = tkinter.Tk()
    root.wm_title(title)
    root.iconbitmap(r"assets\LiClaw.ico")
    canvas = FigureCanvasTkAgg(plot, master=root)  # A tk.DrawingArea.
    toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
    
    tkinter.mainloop()