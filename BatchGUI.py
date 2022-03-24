import os
import tkinter as tk
from datetime import date


from tracks import *
from ReportBuilder import html_summary, makeimage, npy_builder, hd5_dump, csv_dump


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
            csv_dump(self.TrackObjects[0], None, savepath, manual)
            hd5_dump(self.TrackObjects[0], [], savepath)
            self.destroy()
            exit()
        else:
            all_arr = np.array([]) # TODO check if this for loop works for both branches
            for obj in self.TrackObjects:
                all_arr = np.append(all_arr, obj)
            manual = True if all_arr[0].manual_velo else False
            html_summary(all_arr, [], savepath, manual)
            npy_builder(all_arr, None, savepath)
            hd5_dump(all_arr, [], savepath)
            csv_dump(all_arr, None, savepath, manual)
            self.destroy()
            exit()

# Class that inherits root window class from tk
class batchHD5(tk.Tk):

    def __init__(self):
        super().__init__()  # init of tk.Tk
        # TODO build batch hd5's