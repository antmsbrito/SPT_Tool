import tkinter as tk
import numpy as np
import pandas as pd

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

# Class inheriting toplevel windows from tk
class DrawingEllipses(tk.Toplevel):

    def __init__(self, rawdata):
        super().__init__() # toplevel init

        # Configuring window
        self.wm_title("Drawing Ellipses")
        self.title("Drawing Ellipses")
        self.geometry("600x600")

        # To store clicks
        self.clicks = []

        # Store current canvas to update the image
        self.canvas = None

        # Store raw data, xml file path and image objects
        self.rawdata = rawdata

        # Store currently plotted TRACK
        self.current_track = 0

        # Window has two frames
        self.init_plot()
        self.init_buttons()

    def init_plot(self):
        frame_plot = tk.Frame(self)
        frame_plot.pack(fill='both', expand=True)

        fig = Figure()
        # FIRST GRAPH
        canvas = FigureCanvasTkAgg(fig, master=frame_plot)
        self.canvas = canvas

        image = self.rawdata[self.current_track].imageobject
        x = np.array(self.rawdata[self.current_track].x) / 0.08
        y = np.array(self.rawdata[self.current_track].y) / 0.08

        ax = fig.add_subplot()
        ax.imshow(image, cmap="binary")
        ax.plot(x,y, color='r')
        ax.set_xlabel("x coordinates")
        ax.set_ylabel("y coordinates")
        ax.set_xlim((np.average(x)-25,np.average(x)+25))
        ax.set_ylim((np.average(y)-25, np.average(y)+25))

        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        # Toolbar
        toolbar = NavigationToolbar2Tk(canvas, frame_plot)
        toolbar.update()
        # Event handler
        canvas.mpl_connect("button_press_event", self.clickGraph)

    def clickGraph(self, event):
        if event.inaxes is not None:
            print(event.xdata, event.ydata)
        else:
            print("Outside")

    def init_buttons(self):

        frame_buttons = tk.Frame(self)
        frame_buttons.pack(fill='x')
        ALLDONE_button = tk.Button(master=frame_buttons, text="Next track", command=self.nexttrack)
        ALLDONE_button.pack(side='left', fill='x', expand=True)
        UNDO_button = tk.Button(master=frame_buttons, text="Undo", command=self.undo)
        UNDO_button.pack(side='left', fill='x', expand=True)
        QUIT_button = tk.Button(master=frame_buttons, text="QUIT", command=self.quit)
        QUIT_button.pack(side='left', fill='x', expand=True)

    def nexttrack(self):
        pass

    def undo(self):
        pass
