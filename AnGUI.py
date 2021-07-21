import tkinter as tk
from Analysis import *

from scipy.stats import linregress

import numpy as np
import pandas as pd

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from tracks import Track

from GUI_ManualSectioning import ManualSectioning

# Class that inherits root window class from tk
class analysisGUI(tk.Tk):
    def __init__(self, tracks):
        super().__init__()  # init of tk.Tk

        self.TrackList = tracks

        self.wm_title("Analysis")
        self.title("Analysis")

        self.TrackList = tracks

        self.manual_var = tk.IntVar()
        self.minmax_var = tk.IntVar()
        self.finite_var = tk.IntVar()
        self.displacement_var = tk.IntVar()

        self.manual_velo_array = 0
        self.minmax_velo_array = 0
        self.finite_velo_array = 0
        self.displacement_velo_array = 0

        self.init_options()
        self.init_stats()

    def init_options(self):
        options_frame = tk.Frame(self)
        options_frame.pack(fill='both', expand=True, side='left')

        a_label = tk.Label(master=options_frame, text="Please choose the analysis", wraplength=300, justify='center')
        a_label.pack(side='top', fill='both')

        check_manual = tk.Checkbutton(master=options_frame, text="Manual Sectioning", variable=self.manual_var,
                                      onvalue=1, offvalue=0)
        check_manual.pack(anchor='w')
        check_minmax = tk.Checkbutton(master=options_frame, text="MinMax Sectioning", variable=self.minmax_var,
                                      onvalue=1, offvalue=0)
        check_minmax.pack(anchor='w')
        check_finite = tk.Checkbutton(master=options_frame, text="Finite Differences", variable=self.finite_var,
                                      onvalue=1, offvalue=0)
        check_finite.pack(anchor='w')
        check_displacement = tk.Checkbutton(master=options_frame, text="Real Displacement",
                                            variable=self.displacement_var, onvalue=1, offvalue=0)
        check_displacement.pack(anchor='w')

        analysis_button = tk.Button(master=options_frame, text="Start analysis", command=self.analyze)
        analysis_button.pack(fill='x', expand=True)

    def init_stats(self):
        stats_frame = tk.Frame(self)
        stats_frame.pack(fill='both', expand=True, side='right')

        stats_label = tk.Label(master=stats_frame, text="Stats", wraplength=300, justify='center')
        stats_label.pack(side='top', fill='both')

    def analyze(self):
        if self.manual_var.get():
            TL = ManualSectioning(self.TrackList)
            TL.grab_set()
            self.manual_velo_array = TL.section_velocity

        if self.minmax_var.get():
            self.minmax_velo_array = minmax(self.TrackList)

        if self.finite_var.get():
            self.finite_velo_array = finite(self.TrackList)

        if self.displacement_var.get():
            self.displacement_velo_array = displacement(self.TrackList)

    def report(self, *args):
        for array in args:
            if array:
                pass
