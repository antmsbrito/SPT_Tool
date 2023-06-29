"""
SPT_Tool 2021
Antonio Brito @ BCB lab ITQB
"""

import os
import time
from datetime import date
import tkinter as tk
from multiprocessing import Pool

from tracks import *
from GUI_ManualSectioning import ManualSectioning
from Analysis import minmax
from ReportBuilder import makeimage, npy_builder, hd5_dump, csv_dump

from tqdm import tqdm

class analysisGUI(tk.Tk):
    """
    Class that inherits root window class from tk
    """

    def __init__(self, tracks, rejected_tracks, npyname):
        super().__init__()  # init of tk.Tk

        self.TrackList = tracks

        self.wm_title("Analysis")
        self.title("Analysis")
        self.geometry("150x150")

        self.TrackList = tracks
        self.RejectedTracks = rejected_tracks

        self.manual_var = tk.IntVar()
        self.breakpointvar = tk.IntVar()
        self.makeimagesvar = tk.IntVar()

        self.manual_velo_array = np.nan
        self.minmax_velo_array = np.nan

        self.savepath = tk.filedialog.askdirectory(initialdir="C:", title="Please select where to save the data")
        self.savepath = os.path.join(self.savepath, npyname)
        os.makedirs(self.savepath, exist_ok=False)

        self.init_options()

    def init_options(self):
        options_frame = tk.Frame(self)
        options_frame.pack(fill='both', expand=True, side='left')

        a_label = tk.Label(master=options_frame, text="Please choose the analysis", wraplength=300, justify='center')
        a_label.pack(side='top', fill='both')

        check_manual = tk.Checkbutton(master=options_frame, text="Manual Sectioning", variable=self.manual_var,
                                      onvalue=1, offvalue=0)
        check_manual.pack(anchor='w')

        IMAGES_tick = tk.Checkbutton(master=options_frame, text="Optimize breakpoints", variable=self.breakpointvar,
                                     onvalue=1,
                                     offvalue=0)
        IMAGES_tick.pack(anchor='w')

        IMAGES_tick = tk.Checkbutton(master=options_frame, text="Make Images", variable=self.makeimagesvar, onvalue=1,
                                     offvalue=0)
        IMAGES_tick.pack(anchor='w')

        analysis_button = tk.Button(master=options_frame, text="Start analysis", command=self.analyze)
        analysis_button.pack(fill='x', expand=True)

    def analyze(self):
        if self.manual_var.get() == 1:
            print("Manual...")
            TL = ManualSectioning(self.TrackList)
            TL.grab_set()
            self.wait_window(TL)
            self.TrackList = TL.alltracks

        if self.breakpointvar.get() == 1:
            print(f"Optimizing breakpoint locations...")
            print(f"Using {os.cpu_count()} cpu cores")
            start = time.time()

            # Optimizations
            with Pool(processes=os.cpu_count()) as pool:
                iterable = tuple(tqdm(pool.imap(minmax, self.TrackList), total=len(self.TrackList)))

            # Save results in another loop
            # not really efficient but has to be this way to have 'pretty' progress bar with tqdm
            # Overhead shouldnt be that bad
            for idx, out in enumerate(iterable):
                tr = self.TrackList[idx]
                tr.bruteforce_velo, tr.bruteforce_phi, tr.muggeo_velo, tr.muggeo_phi, tr.muggeo_params = out

            end = time.time()
            print(f"Breakpoint optimization in {end-start:.2f} seconds")

            # Not parallelized
            """
            for tr in self.TrackList:
                tr.minmax_velo, tr.minmax_sections = minmax(tr)
            """

        if self.manual_var.get() == 1 or self.TrackList[0].manual_velo:
            manualboolean = True
        else:
            manualboolean = False

        # What to save?
        # 1 - array of Track objects (.npy) to reload for reanalysis
        npy_builder(self.TrackList, self.RejectedTracks, self.savepath)

        # 2 - csv file with results
        csv_dump(self.TrackList, self.savepath)

        # 3 - hd5 dump
        # hd5_dump(self.TrackList, self.RejectedTracks, self.savepath)

        # 4 - images
        if self.makeimagesvar.get() == 1:
            makeimage(self.TrackList, self.savepath, manualboolean)

        tk.messagebox.showinfo(title="All done!", message="All done! Check folder for full report data.")
        self.destroy()
        exit()
