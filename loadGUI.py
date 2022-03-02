"""
SPT_TOOL
@author António Brito
ITQB-UNL BCB 2021
"""

import os
import tkinter as tk
from datetime import date

import numpy as np
from tracks import *
from ReportBuilder import html_summary, html_comparison, makeimage, npy_builder


# Class that inherits root window class from tk
class loadGUI(tk.Tk):

    def __init__(self):
        super().__init__()  # init of tk.Tk

        self.wm_title("Load data")
        self.title("Load data")
        self.geometry('350x75')

        self.numberofnpy = tk.IntVar()
        self.LabelText = tk.StringVar()
        self.LabelText.set(f"{self.numberofnpy.get()} files loaded")

        self.TrackObjects = []
        self.filenames = []

        self.makeimagesvar = tk.IntVar()

        self.init_input()
        self.init_output()

    def init_input(self):
        frame_input = tk.Frame(self)
        frame_input.pack(fill='both', expand=True, side='left')

        NPY_button = tk.Button(master=frame_input, text="Load .npy file", command=self.loadnpy)
        NPY_button.pack(fill='x', expand=True)

        ANALYZE_button = tk.Button(master=frame_input, text="Analyze .npy files", command=self.analyze)
        ANALYZE_button.pack(fill='x', expand=True)

        IMAGES_tick = tk.Checkbutton(master=frame_input, text="Make Images", variable=self.makeimagesvar, onvalue=1, offvalue=0)
        IMAGES_tick.pack(anchor='w')

    def init_output(self):
        frame_output = tk.Frame(self)
        frame_output.pack(fill='both', expand=True, side='right')

        status_text = tk.Label(master=frame_output, textvariable=self.LabelText)
        status_text.pack(side='top', fill='both')

    def loadnpy(self):

        npy = tk.filedialog.askopenfilename(initialdir="C:", title="Select .npy file to load")
        if not npy[-3:] == "npy":
            tk.messagebox.showerror(title="NPY", message="File extension must be .npy")
        else:
            self.numberofnpy.set(self.numberofnpy.get() + 1)
            self.LabelText.set(f"{self.numberofnpy.get()} files loaded")
            objs = np.load(npy, allow_pickle=True)
            if isinstance(objs[0], Track):
                newObjects = [TrackV2(t.image, t.xtrack, t.ytrack, t.samplerate, t.designator, t.ellipse) for t in objs]
                self.TrackObjects.append(newObjects)
            else:
                newObjects = [TrackV2(t.imageobject, t.x, t.y, t.samplerate, t.name, t.ellipse) for t in objs]
                self.TrackObjects.append(newObjects)
            self.filenames.append(npy)

    def analyze(self):

        if not self.numberofnpy.get():
            tk.messagebox.showerror(title="NPY", message="No file loaded!")
        else:
            savepath = tk.filedialog.askdirectory(initialdir="C:", title="Please select where to save the data")
            savepath = os.path.join(savepath, rf"SPT_{date.today().strftime('%d_%m_%Y')}_reanalysis")
            os.makedirs(savepath, exist_ok=True)

        if self.numberofnpy.get() == 1:
            manual = True if self.TrackObjects[0][0].manual_velo else False
            html_summary(self.TrackObjects[0], [], savepath, manual)
            if self.makeimagesvar:
                makeimage(self.TrackObjects[0], savepath, manual)
            npy_builder(self.TrackObjects[0], None, savepath)
            self.destroy()
            exit()
        else:
            all_arr = np.array([])
            for obj in self.TrackObjects:
                all_arr = np.append(all_arr, obj)
            manual = True if all_arr[0].manual_velo else False
            html_summary(all_arr, [], savepath, manual)
            if self.makeimagesvar:
                makeimage(all_arr, savepath, manual)
            npy_builder(all_arr, None, savepath)
            self.destroy()
            exit()


if __name__ == '__main__':
    pass
