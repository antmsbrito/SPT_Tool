"""
SPT_TOOL
@author António Brito
ITQB-UNL BCB 2021
"""

import tkinter as tk

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from PIL import Image, ImageEnhance

from tracks import *


# Class inheriting toplevel windows from tk
class DrawingEllipses(tk.Toplevel):

    def __init__(self, rawdata):
        super().__init__()  # toplevel init

        # Configuring window
        self.wm_title("Drawing Ellipses")
        self.title("Drawing Ellipses")
        self.geometry("700x700")

        self.pxmax = tk.IntVar()
        self.pxmin = tk.IntVar()

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
        self.init_sliders()
        self.init_plot()
        self.init_buttons()


        tk.messagebox.showinfo(title="IMPORTANT", message="ALWAYS DRAW THE MAJOR AXIS FIRST")

    def init_sliders(self):
        frame_slider = tk.Frame(self)
        frame_slider.pack(side=tk.RIGHT)

        max_slider = tk.Scale(master=frame_slider, command=self.update_max, from_=0, to=2 ** 16, variable=self.pxmax, orient=tk.VERTICAL, label="Max", length=200)
        max_slider.pack(side=tk.TOP)

        min_slider = tk.Scale(master=frame_slider, command=self.update_min, from_=0, to=2 ** 16, variable=self.pxmin, orient=tk.VERTICAL, label="Min", length=200)
        # min_slider.pack(side=tk.BOTTOM)

    def update_min(self, _):
        self.redraw_graph(pxmax=self.pxmax.get(), pxmin=self.pxmin.get())

    def update_max(self, _):
        self.redraw_graph(pxmax=self.pxmax.get(), pxmin=self.pxmin.get())

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
        # ax.plot(x, y, color='r')
        ax.set_xlabel("x coordinates (px)")
        ax.set_ylabel("y coordinates (px)")
        ax.set_xlim((np.average(x) - 30, np.average(x) + 30))
        ax.set_ylim((np.average(y) - 30, np.average(y) + 30))

        cumulative_disp = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(cumulative_disp.min(), cumulative_disp.max())
        lc = LineCollection(segments, cmap='rainbow', norm=norm)
        lc.set_array(cumulative_disp)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)

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

        IGNORE_BUTTON = tk.Button(master=frame_buttons, text="Discard", command=self.discard)
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

    def redraw_graph(self, pxmin=0, pxmax=2 ** 16):
        print(pxmax)
        ax = self.canvas.figure.axes[0]

        ax.lines = []
        ax.images = []
        ax.patches = []
        ax.collections = []

        image = self.rawdata[self.current_track].imageobject
        x = np.array(self.rawdata[self.current_track].x) / 0.08
        y = np.array(self.rawdata[self.current_track].y) / 0.08
        ax.imshow(image, cmap="gray", vmin=pxmin, vmax=pxmax)
        # ax.plot(x, y, color='r')
        ax.set_xlim((np.average(x) - 30, np.average(x) + 30))
        ax.set_ylim((np.average(y) - 30, np.average(y) + 30))

        cumulative_disp = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(cumulative_disp.min(), cumulative_disp.max())
        lc = LineCollection(segments, cmap='rainbow', norm=norm)
        lc.set_array(cumulative_disp)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)


        self.canvas.draw()

    def finishup(self):
        for idx, tr in enumerate(self.rawdata):
            self.rawdata[idx].ellipse = self.elidict[idx]
        self.track_classes = [r for i, r in enumerate(self.rawdata) if self.elidict[i]]
        self.rejects = [r for i, r in enumerate(self.rawdata) if not self.elidict[i]]
        self.quit()
        self.destroy()
