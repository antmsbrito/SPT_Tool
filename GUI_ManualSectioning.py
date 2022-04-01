"""
SPT_TOOL
@author António Brito
ITQB-UNL BCB 2021
"""

import tkinter as tk
import numpy as np

from scipy.stats import linregress

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# Class that inherits root window class from tk
class ManualSectioning(tk.Toplevel):
    def __init__(self, tracks):
        super().__init__()  # init of tk.Tk()

        # End product, velocity of track sections
        self.section_velocity = []

        #  Configure root window
        self.wm_title("Manual sectioning")
        self.title("Manual sectioning")
        self.geometry('600x600')

        # Store clicks
        self.clicks = []

        # Store current plotted track
        self.current_track = 0

        # Store all tracks
        self.alltracks = tracks

        # Store canvas for updating
        self.canvas = None

        # final dictionary for all clicks
        self.clickdata = {}

        # Window has two frames
        self.init_plot()
        self.init_buttons()

    def init_plot(self):
        frame_plot = tk.Frame(self)
        frame_plot.pack(fill='both', expand=True)

        fig = Figure()

        # INSERT FIRST GRAPH
        canvas = FigureCanvasTkAgg(fig, master=frame_plot)
        self.canvas = canvas

        SR = self.alltracks[self.current_track].samplerate
        y = self.alltracks[self.current_track].unwrapped
        x = np.array(range(len(y))) * SR
        smoothedy = smoothing(y, int((len(y) * 10) // 100))
        smoothedx = np.array(range(len(smoothedy))) * SR + 0.05*x[-1]

        ax = fig.add_subplot()
        ax.plot(x, y, color='r')
        ax.plot(smoothedx, smoothedy, color='k')
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Position (mm)")
        ax.set_title(self.alltracks[self.current_track].name)

        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        # INSERT MATPLOTLIB TOOLBAR
        toolbar = NavigationToolbar2Tk(canvas, frame_plot)
        toolbar.update()
        # INSERT CLICKING EVENT HANDLER
        canvas.mpl_connect("button_press_event", self.clickGraph)

    def init_buttons(self):

        frame_buttons = tk.Frame(self)
        frame_buttons.pack(fill='x')
        ALLDONE_button = tk.Button(master=frame_buttons, text="Next track", command=self.nexttrack)
        ALLDONE_button.pack(side='left', fill='x', expand=True)
        UNDO_button = tk.Button(master=frame_buttons, text="Undo", command=self.undo)
        UNDO_button.pack(side='left', fill='x', expand=True)
        QUIT_button = tk.Button(master=frame_buttons, text="QUIT", command=self.quit)
        QUIT_button.pack(side='left', fill='x', expand=True)

    def clickGraph(self, event):
        if event.inaxes is not None:
            ax = self.canvas.figure.axes[0]
            ax.axvline(x=event.xdata, color='r')
            self.canvas.draw()
            self.clicks.append(event.xdata)
        else:
            print("OUTSIDE")

    def nexttrack(self):
        # Store previous data:
        self.clickdata[self.alltracks[self.current_track].name] = self.clicks

        # reset clicks
        self.clicks = []

        # Delete graph
        ax = self.canvas.figure.axes[0]
        ax.lines = []
        if len(self.alltracks) == self.current_track + 1:
            self.finishup()
        else:
            # Next graph
            self.current_track += 1

            SR = self.alltracks[self.current_track].samplerate
            y = self.alltracks[self.current_track].unwrapped
            x = np.array(range(len(y))) * SR
            smoothedy = smoothing(y, int((len(y) * 10) // 100))
            smoothedx = np.array(range(len(smoothedy))) * SR + 0.05 * x[-1]

            ax.plot(x, y, color='r')
            ax.plot(smoothedx, smoothedy, color='k')
            ax.set_title(self.alltracks[self.current_track].name)
            self.canvas.draw()

    def undo(self):
        if not self.clicks:
            print("No lines")
        else:
            ax = self.canvas.figure.axes[0]
            ax.lines.pop(-1)
            self.canvas.draw()
            self.clicks.pop(-1)

    def finishup(self):
        # Everything will be done in the bg
        self.quit()
        self.destroy()

        for idx, tr in enumerate(self.alltracks):
            delimiter = self.findclosest_idx(self.clickdata[tr.name], tr)
            self.alltracks[idx].manual_sections = delimiter

            SR = tr.samplerate
            rawy = tr.unwrapped
            rawx = np.array(range(len(rawy))) * SR
            y = smoothing(rawy, int((len(rawy) * 10) // 100))
            x = np.array(range(len(y))) * SR + 0.05 * rawx[-1]

            if not delimiter.size:
                m, b = self.slope(x, y)
                self.alltracks[idx].manual_velo.append(np.abs(m)*1000)
                self.section_velocity.append(m*1000)
                # plt.plot(x, x * m + b)
            else:
                m, b = self.slope(x[0:delimiter[0]], y[0:delimiter[0]])
                # plt.plot(x[0:delimiter[0]], x[0:delimiter[0]] * m + b)
                self.section_velocity.append(m*1000)
                self.alltracks[idx].manual_velo.append(np.abs(m)*1000)
                for idx2, d in enumerate(delimiter):
                    if d == delimiter[-1]:
                        m, b = self.slope(x[d:-1], y[d:-1])
                        # plt.plot(x[d:-1], x[d:-1] * m + b)
                        self.section_velocity.append(m*1000)
                        self.alltracks[idx].manual_velo.append(np.abs(m)*1000)
                    else:
                        nextd = delimiter[idx2 + 1]
                        m, b = self.slope(x[d:nextd], y[d:nextd])
                        self.section_velocity.append(m*1000)
                        self.alltracks[idx].manual_velo.append(np.abs(m)*1000)
                        # plt.plot(x[d:next], x[d:next] * m + b)
            # plt.show()

        #save histogram
        self.section_velocity = np.abs(self.section_velocity)

    @staticmethod
    def findclosest_idx(clickarray, trackobject):
        # Click data is not real data; transform click data into indices
        if not clickarray:
            return np.array([])
        indexes = []
        SR = trackobject.samplerate
        y = trackobject.unwrapped
        x = np.array(range(len(y))) * SR
        for value in clickarray:
            idx = (np.abs(x - value)).argmin()
            indexes.append(idx)
        return np.sort(indexes)

    @staticmethod
    def slope(x, y):
        s, b, r_value, p_value, std_err = linregress(x, y)
        return s, b

    @staticmethod
    def smoothing(unsmoothed, windowsize):
        smoothed = np.convolve(unsmoothed, np.ones(windowsize)/windowsize, mode='valid')
        return smoothed

if __name__ == '__main__':
    app = ManualSectioning()
    app.mainloop()
