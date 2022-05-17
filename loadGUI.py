"""
SPT_TOOL
@author António Brito
ITQB-UNL BCB 2021
"""

import os
import json
import tkinter as tk
from datetime import date
from PIL import Image

import h5py
import numpy as np
from tracks import *
from ReportBuilder import makeimage, npy_builder, hd5_dump
from AnGUI import analysisGUI




# Class that inherits root window class from tk
class loadNPY(tk.Tk):

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

        self.init_input()
        self.init_output()

    def init_input(self):
        frame_input = tk.Frame(self)
        frame_input.pack(fill='both', expand=True, side='left')

        NPY_button = tk.Button(master=frame_input, text="Load .npy file", command=self.loadnpy)
        NPY_button.pack(fill='x', expand=True)

        ANALYZE_button = tk.Button(master=frame_input, text="Analyze .npy files", command=self.analyze)
        ANALYZE_button.pack(fill='x', expand=True)

    def init_output(self):
        frame_output = tk.Frame(self)
        frame_output.pack(fill='both', expand=True, side='right')

        status_text = tk.Label(master=frame_output, textvariable=self.LabelText)
        status_text.pack(side='top', fill='both')

    def loadnpy(self):

        npy = tk.filedialog.askopenfilename(initialdir="C:", title="Select .npy file to load")

        while not npy[-3:] == "npy":
            tk.messagebox.showerror(title="NPY", message="File extension must be .npy")
            npy = tk.filedialog.askopenfilename(initialdir="C:", title="Select .npy file to load")

        self.numberofnpy.set(self.numberofnpy.get() + 1)
        self.LabelText.set(f"{self.numberofnpy.get()} files loaded")
        self.filenames.append(npy)

        old_objs = np.load(npy, allow_pickle=True)

        newobjs = []
        for idx, t in enumerate(old_objs):
            newobjs.append(TrackV2(t.imageobject, t.x, t.y, t.samplerate, t.name, t.ellipse))
            if hasattr(t, 'bruteforce_velo'):
                newobjs[-1].bruteforce_velo = t.bruteforce_velo
                newobjs[-1].bruteforce_phi = t.bruteforce_phi

            if hasattr(t, 'muggeo_velo'):
                newobjs[-1].muggeo_velo = t.muggeo_velo
                newobjs[-1].muggeo_phi = t.muggeo_phi
                newobjs[-1].muggeo_params = t.muggeo_params

            if hasattr(t, 'manual_velo'):
                newobjs[-1].manual_velo = t.manual_velo
                try:
                    newobjs[-1].manual_phi = t.manual_phi
                except AttributeError: # For legacy purposes
                    newobjs[-1].manual_phi = t.manual_sections

        self.TrackObjects.append(newobjs)

    def analyze(self):

        if not self.numberofnpy.get():
            tk.messagebox.showerror(title="NPY", message="No file loaded!")

        if self.numberofnpy.get() == 1:
            self.destroy()
            analysisapp = analysisGUI(self.TrackObjects[0], [], self.filenames[-1].split('/')[-1][:-4])
            analysisapp.mainloop()
        else:
            all_arr = np.array([])
            for obj in self.TrackObjects:
                all_arr = np.append(all_arr, obj)
            self.destroy()
            analysisapp = analysisGUI(self.FinalTracks, [])
            analysisapp.mainloop()


class loadHD5(tk.Tk):
    def __init__(self):
        super().__init__()  # init of tk.Tk

        self.wm_title("Load data")
        self.title("Load data")
        self.geometry('350x75')

        self.numberofh5 = tk.IntVar()
        self.LabelText = tk.StringVar()
        self.LabelText.set(f"{self.numberofh5.get()} files loaded")

        self.TrackObjects = []
        self.filenames = []

        self.previousvar = tk.IntVar()

        self.init_input()
        self.init_output()

    def init_input(self):
        frame_input = tk.Frame(self)
        frame_input.pack(fill='both', expand=True, side='left')

        NPY_button = tk.Button(master=frame_input, text="Load .h5 file", command=self.loadh5)
        NPY_button.pack(fill='x', expand=True)

        ANALYZE_button = tk.Button(master=frame_input, text="Analyze .h5 files", command=self.analyze)
        ANALYZE_button.pack(fill='x', expand=True)

        PREVIOUSDATA_tick = tk.Checkbutton(master=frame_input, text="Is there previous data?",
                                           variable=self.previousvar, onvalue=1, offvalue=0)
        PREVIOUSDATA_tick.pack(anchor='w')

    def init_output(self):
        frame_output = tk.Frame(self)
        frame_output.pack(fill='both', expand=True, side='right')

        status_text = tk.Label(master=frame_output, textvariable=self.LabelText)
        status_text.pack(side='top', fill='both')

    def loadh5(self):
        h5file = tk.filedialog.askopenfilename(initialdir="C:", title="Select .h5 file to load")
        if not h5file[-2:] == "h5":
            tk.messagebox.showerror(title="H5", message="File extension must be .h5")
        else:
            self.numberofh5.set(self.numberofh5.get() + 1)
            self.LabelText.set(f"{self.numberofh5.get()} files loaded")
            self.filenames.append(h5file)

            hf = h5py.File(h5file, 'r')
            tracks = hf['tracks']

            for key in tracks:
                x = tracks[key].get('x')[:]
                y = tracks[key].get('y')[:]
                samplerate = tracks[key].get('samplerate')[:][0]
                image = Image.fromarray(tracks[key].get('image')[:])
                ellipse = json.loads(tracks[key].get('ellipse').asstr()[()])
                name = tracks[key].get('name').asstr()[()]
                manual_sections = tracks[key].get('manual_sections')[:]

                if self.previousvar.get() == 1:
                    mvelo = np.array(tracks[key].get('minmax_velo')[:], dtype=int)
                    msection = np.array(tracks[key].get('minmax_sections')[:], dtype=int)

                    self.TrackObjects.append(TrackV2(image, x, y, samplerate, name, ellipse, (mvelo, msection)))
                    self.TrackObjects[-1].manual_sections = manual_sections
                else:
                    self.TrackObjects.append(TrackV2(image, x, y, samplerate, name, ellipse))
                    self.TrackObjects[-1].manual_sections = manual_sections

    def analyze(self):
        if not self.numberofh5.get():
            tk.messagebox.showerror(title="h5", message="No file loaded!")
            return 0

        self.destroy()
        analysisapp = analysisGUI(self.TrackObjects, [])
        analysisapp.mainloop()


if __name__ == '__main__':
    pass
