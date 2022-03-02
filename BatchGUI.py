import tkinter as tk
import os

from tracks import *

from datetime import date

from ReportBuilder import html_summary, html_comparison, makeimage, npy_builder


# Class that inherits root window class from tk
class batchGUI(tk.Tk):
    def __init__(self):
        super().__init__()  # init of tk.Tk

        # List of track related objects
        self.TrackObjects = []
        self.FinalTracks = None

        folder = tk.filedialog.askdirectory(initialdir="C:", title="Please choose folder")

        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith("Dump.npy"):
                    self.load(file)

        self.analyze()


    def load(self, npy):
        objs = np.load(npy, allow_pickle=True)
        if isinstance(objs[0], Track):
            newObjects = [TrackV2(t.image, t.xtrack, t.ytrack, t.samplerate, t.designator, t.ellipse) for t in objs]
            self.TrackObjects.append(newObjects)
        else:
            newObjects = [TrackV2(t.imageobject, t.x, t.y, t.samplerate, t.name, t.ellipse) for t in objs]
            self.TrackObjects.append(newObjects)


    def analyze(self):

        savepath = tk.filedialog.askdirectory(initialdir="C:", title="Please select where to save the data")
        savepath = os.path.join(savepath, rf"SPT_{date.today().strftime('%d_%m_%Y')}_reanalysis")
        os.makedirs(savepath, exist_ok=True)

        if self.numberofnpy.get() == 1:
            manual = True if self.TrackObjects[0][0].manual_velo else False
            html_summary(self.TrackObjects[0], [], savepath, manual)
            npy_builder(self.TrackObjects[0], None, savepath)
            self.destroy()
            exit()
        else:
            all_arr = np.array([])
            for obj in self.TrackObjects:
                all_arr = np.append(all_arr, obj)
            manual = True if all_arr[0].manual_velo else False
            html_summary(all_arr, [], savepath, manual)
            npy_builder(all_arr, None, savepath)
            self.destroy()
            exit()

