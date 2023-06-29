import os
import tkinter as tk
from datetime import date


from tracks import *
from ReportBuilder import npy_builder, csv_dump


# Class that inherits root window class from tk
class batchNPY(tk.Tk):

    def __init__(self):
        super().__init__()  # init of tk.Tk

        # List of track related objects
        self.TrackObjects = []
        self.FinalTracks = None

        folder = tk.filedialog.askdirectory(initialdir="C:", title="Please choose folder")

        #load one by one
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith("Dump.npy"):
                    self.load(root+os.sep+file)

        self.analyze()


    def load(self, npy):
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
                newobjs[-1].manual_phi = t.manual_phi

        return self.TrackObjects.append(np.array(newobjs))

    def analyze(self):

        savepath = tk.filedialog.askdirectory(initialdir="C:", title="Please select where to save the data")
        savepath = os.path.join(savepath, rf"SPT_{date.today().strftime('%d_%m_%Y')}_batch")
        os.makedirs(savepath, exist_ok=True)

        all_arr = np.array([]) 
        for obj in self.TrackObjects:
            all_arr = np.append(all_arr, obj)

        npy_builder(all_arr, [], savepath)
        csv_dump(all_arr, None, savepath)
        self.destroy()
        exit()

