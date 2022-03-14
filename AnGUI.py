"""
SPT_Tool 2021
Antonio Brito @ BCB lab ITQB
"""

import tkinter as tk

import os
from datetime import date

from tracks import *
from GUI_ManualSectioning import ManualSectioning
from ReportBuilder import html_summary, npy_builder


class analysisGUI(tk.Tk):
    """
    Class that inherits root window class from tk. This GUI window shows the varied 
    """

    def __init__(self, tracks, rejected_tracks):
        super().__init__()  # init of tk.Tk

        self.TrackList = tracks

        self.wm_title("Analysis")
        self.title("Analysis")
        self.geometry("150x150")

        self.TrackList = tracks
        self.RejectedTracks = rejected_tracks

        self.manual_var = tk.IntVar()

        self.manual_velo_array = np.nan
        self.minmax_velo_array = np.nan

        self.savepath = tk.filedialog.askdirectory(initialdir="C:", title="Please select where to save the data")
        self.savepath = os.path.join(self.savepath, rf"SPT_{date.today().strftime('%d_%m_%Y')}")
        os.makedirs(self.savepath, exist_ok=True)

        self.init_options()


    def init_options(self):
        options_frame = tk.Frame(self)
        options_frame.pack(fill='both', expand=True, side='left')

        a_label = tk.Label(master=options_frame, text="Please choose the analysis", wraplength=300, justify='center')
        a_label.pack(side='top', fill='both')

        check_manual = tk.Checkbutton(master=options_frame, text="Manual Sectioning", variable=self.manual_var,
                                      onvalue=1, offvalue=0)
        check_manual.pack(anchor='w')

        analysis_button = tk.Button(master=options_frame, text="Start analysis", command=self.analyze)
        analysis_button.pack(fill='x', expand=True)


    def analyze(self):
        if self.manual_var.get():
            print("Manual...")
            TL = ManualSectioning(self.TrackList)
            TL.grab_set()
            self.wait_window(TL)
            self.TrackList = TL.alltracks

        # What to save?
        # 1 - array of Track objects (.npy) to reload for reanalysis or comparison between condition (for legacy purposes)
        npy_builder(self.TrackList, self.RejectedTracks, self.savepath)
        # 2 - general html report DONE #TODO improve it
        html_summary(self.TrackList, self.RejectedTracks, self.savepath, self.manual_var.get())
        # 3 - csv file with results
        #csv_results() # TODO add csv
        # 4 -

        tk.messagebox.showinfo(title="All done!", message="All done! Check folder for full report data.")
        self.destroy()
        exit()
