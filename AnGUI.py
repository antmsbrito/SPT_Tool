"""
SPT_Tool 2021
Antonio Brito @ BCB lab ITQB
"""

import tkinter as tk

import os

import pandas as pd
from datetime import date

import numpy as np

from tracks import Track

from GUI_ManualSectioning import ManualSectioning

from ReportBuilder import html_summary, npy_builder, makeimage

class analysisGUI(tk.Tk):
    """
    Class that inherits root window class from tk. This GUI window shows the varied 
    """

    def __init__(self, tracks):
        super().__init__()  # init of tk.Tk

        self.TrackList = tracks

        self.wm_title("Analysis")
        self.title("Analysis")
        self.geometry("150x150")

        self.TrackList = tracks

        self.manual_var = tk.IntVar()
        self.minmax_var = tk.IntVar()
        self.finite_var = tk.IntVar()
        self.displacement_var = tk.IntVar()

        self.manual_velo_array = np.nan
        self.minmax_velo_array = np.nan
        self.finite_velo_array = np.nan
        self.displacement_velo_array = np.nan

        self.savepath = tk.filedialog.askdirectory(initialdir="C:", title="Please select where to save the data")
        self.savepath = os.path.join(self.savepath, rf"SPT_{date.today().strftime('%d_%m_%Y')}")
        os.makedirs(self.savepath, exist_ok=True)

        self.init_options()
        # self.init_stats()

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

        if not self.manual_var.get() and not self.minmax_var.get() and not self.finite_var.get() and not self.displacement_var.get():
            tk.messagebox.showerror(title="Error", message="Please select at least one option")
            return

        if self.manual_var.get():
            print("Manual...")
            TL = ManualSectioning(self.TrackList)
            TL.grab_set()
            self.wait_window(TL)
            self.TrackList = TL.alltracks

        # What to save?
        # 1 - array of Track objects (.npy) to reload for reanalysis or comparison between conditions DONE
        npy_builder(self.TrackList, self.savepath)
        # 4 - general html report DONE #TODO improve it
        html_summary(self.TrackList,self.savepath, self.manual_var.get(), self.minmax_var.get(), self.finite_var.get(), self.displacement_var.get())

        tk.messagebox.showinfo(title="All done!", message="All done! Check folder for full report data.")
        self.destroy()
        exit()

    def build_xlsx(self):
        with pd.ExcelWriter(f"{self.savepath}\\TrackData.xlsx") as writer:
            for tr in self.TrackList:
                df = pd.DataFrame({'xtrack': tr.xtrack, 'ytrack': tr.ytrack, 'zellipse': tr.ztrack})
                df.to_excel(writer, sheet_name=tr.designator)

