"""
SPT_TOOL
@author António Brito
ITQB-UNL BCB 2021
"""

import tkinter as tk
import numpy as np

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib import patches

from tracks import *


# Class inheriting toplevel windows from tk
class DrawingEllipses(tk.Toplevel):

    def __init__(self, rawdata):
        super().__init__()  # toplevel init

        # Configuring window
        self.wm_title("Drawing Ellipses")
        self.title("Drawing Ellipses")
        self.geometry("600x600")

        # To store clicks
        self.x_clicks = []
        self.y_clicks = []

        # Store current canvas to update the image
        self.canvas = None

        # Store raw data, xml file path and image objects
        self.rawdata = rawdata

        # To store ellipses
        self.x0 = None
        self.y0 = None
        self.minor = None
        self.major = None
        self.angle = None
        self.elidict = [None] * len(rawdata)

        # Store currently plotted TRACK
        self.current_track = 0

        # Final result
        self.track_classes = None
        self.rejects = None

        # Window has two frames
        self.init_plot()
        self.init_buttons()

        tk.messagebox.showinfo(title="IMPORTANT", message="ALWAYS DRAW THE MAJOR AXIS FIRST")

    def init_plot(self):
        frame_plot = tk.Frame(self)
        frame_plot.pack(fill='both', expand=True)

        fig = Figure()
        # FIRST GRAPH
        self.canvas = FigureCanvasTkAgg(fig, master=frame_plot)

        image = self.rawdata[self.current_track].imageobject
        x = np.array(self.rawdata[self.current_track].x) / 0.08  # TODO UM TO NM
        y = np.array(self.rawdata[self.current_track].y) / 0.08  # TODO UM TO NM

        ax = fig.add_subplot()
        ax.imshow(image, cmap='gray')
        ax.plot(x, y, color='r')
        ax.set_xlabel("x coordinates (px)")
        ax.set_ylabel("y coordinates (px)")
        ax.set_xlim((np.average(x) - 30, np.average(x) + 30))
        ax.set_ylim((np.average(y) - 30, np.average(y) + 30))

        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, frame_plot)
        toolbar.update()

        # Event handler
        self.canvas.mpl_connect("button_press_event", self.clickGraph)

    def clickGraph(self, event):
        if event.inaxes is not None:
            print(event.xdata, event.ydata)
            ax = self.canvas.figure.axes[0]

            xclick = event.xdata
            yclick = event.ydata

            self.x_clicks.append(xclick)
            self.y_clicks.append(yclick)

            # First two points are major axis
            if len(self.x_clicks) == 1:
                ax.scatter(self.x_clicks[0], self.y_clicks[0], facecolors='none', edgecolors='k')
                self.canvas.draw()

            if len(self.x_clicks) == 2:
                self.major = np.sqrt(
                    (self.x_clicks[0] - self.x_clicks[1]) ** 2 + (self.y_clicks[0] - self.y_clicks[1]) ** 2)
                self.x0 = np.mean(self.x_clicks)
                self.y0 = np.mean(self.y_clicks)
                yangle = np.max(np.array(self.y_clicks) - self.y0)
                xangle = np.array(self.x_clicks[np.argmax(np.array(self.y_clicks) - self.y0)]) - self.x0
                self.angle = np.arctan2(yangle, xangle)

                ax.scatter(self.x_clicks[1], self.y_clicks[1], facecolors='none', edgecolors='k')
                ax.plot(self.x_clicks, self.y_clicks, '--k')
                ax.scatter(self.x0, self.y0, facecolors='none', edgecolors='k')
                self.canvas.draw()

            if len(self.x_clicks) == 3:
                self.minor = np.sqrt((self.x_clicks[2] - self.x0) ** 2 + (self.y_clicks[2] - self.y0) ** 2) * 2

                if self.minor > self.major:
                    self.minor, self.major = self.major, self.minor

                # TODO recheck angle for the 1000 time just to be sure
                eli = patches.Ellipse((self.x0, self.y0), self.major, self.minor, np.rad2deg(self.angle), fill=False,
                                      edgecolor='black', alpha=0.3)
                ax.add_patch(eli)
                ax.scatter(self.x_clicks[2], self.y_clicks[2], facecolors='none', edgecolors='k')
                self.canvas.draw()
            else:
                pass
        else:
            print("Outside")

    def init_buttons(self):

        frame_buttons = tk.Frame(self)
        frame_buttons.pack(fill='x')

        ALLDONE_button = tk.Button(master=frame_buttons, text="Next track", command=self.nexttrack)
        ALLDONE_button.pack(side='left', fill='x', expand=True)

        UNDO_button = tk.Button(master=frame_buttons, text="Undo", command=self.undo)
        UNDO_button.pack(side='left', fill='x', expand=True)

        IGNORE_BUTTON = tk.Button(master=frame_buttons, text="Discart", command=self.discard)
        IGNORE_BUTTON.pack(side='left', fill='x', expand=True)

        QUIT_button = tk.Button(master=frame_buttons, text="QUIT", command=self.destroy)
        QUIT_button.pack(side='left', fill='x', expand=True)

    def nexttrack(self):

        if not self.x_clicks or not len(self.x_clicks) == 3:
            return 0

        self.x_clicks = []
        self.y_clicks = []

        #  Check if we have more tracks other wise finish up the gui
        if self.rawdata[-1] == self.rawdata[self.current_track]:
            self.elidict[self.current_track] = {'x0': self.x0 * 0.08, 'y0': self.y0 * 0.08, 'major': self.major * 0.08,
                                                'minor': self.minor * 0.08, 'angle': np.rad2deg(self.angle)}
            self.finishup()
        else:
            self.elidict[self.current_track] = {'x0': self.x0 * 0.08, 'y0': self.y0 * 0.08, 'major': self.major * 0.08,
                                                'minor': self.minor * 0.08, 'angle': np.rad2deg(self.angle)}
            self.current_track += 1
            self.redraw_graph()

    def discard(self):

        self.x_clicks = []
        self.y_clicks = []

        if self.rawdata[-1] == self.rawdata[self.current_track]:
            self.elidict[self.current_track] = None
            self.finishup()
        else:
            self.elidict[self.current_track] = None
            self.current_track += 1
            self.redraw_graph()

    def undo(self):
        if not self.x_clicks and self.current_track > 0:
            self.current_track -= 1
            self.redraw_graph()
        else:
            self.x_clicks = []
            self.y_clicks = []
            self.redraw_graph()

    def redraw_graph(self):
        ax = self.canvas.figure.axes[0]

        ax.lines = []
        ax.images = []
        ax.patches = []
        ax.collections = []

        image = self.rawdata[self.current_track].imageobject
        x = np.array(self.rawdata[self.current_track].x) / 0.08
        y = np.array(self.rawdata[self.current_track].y) / 0.08
        ax.imshow(image, cmap="gray")
        ax.plot(x, y, color='r')
        ax.set_xlim((np.average(x) - 30, np.average(x) + 30))
        ax.set_ylim((np.average(y) - 30, np.average(y) + 30))

        self.canvas.draw()

    def finishup(self):
        self.track_classes = TrackV2.generatetrack_ellipse(self.rawdata, self.elidict)
        self.rejects = [r for i, r in enumerate(self.rawdata) if not self.elidict[i]]
        self.quit()
        self.destroy()

