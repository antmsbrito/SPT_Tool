import tkinter as tk
import os
import numpy as np

from tracks import *

from PIL import Image

from AnGUI import analysisGUI
from DrawingGUI import DrawingEllipses

# Class that inherits root window class from tk
class batchGUI(tk.Tk):
    def __init__(self):
        super().__init__()  # init of tk.Tk

        # Configure root window
        self.wm_title("IO Window")
        self.title("IO Window")
        self.geometry('350x75')

        # List of track related objects
        self.TrackList = []
        self.FinalTracks = None

        folder = tk.filedialog.askdirectory(initialdir="C:", title="Please choose folder")

        for root, dirs, files in os.walk(folder):
            if 'XML' in root or 'xml' in root:
                for file in files:
                    if file.endswith(".xml"):
                        imgfile = os.path.join(root, file[:-4] + ".tif")
                        if not os.path.exists(imgfile):
                            print(f"OOPS, the following xml file does not have an image counterpart \n {file} \n")
                        else:
                            self.load(os.path.join(root, file), imgfile)

        self.drawing_button()

    def load(self, xml, image):
        imgobj = Image.open(image)
        self.TrackList = np.append(self.TrackList, TrackV2.generator_xml(xml, imgobj))
        print(len(self.TrackList), xml)

    def drawing_button(self):

        drawing_window = DrawingEllipses(self.TrackList)
        drawing_window.grab_set()
        self.wait_window(drawing_window)

        self.FinalTracks = drawing_window.track_classes
        self.destroy()
        analysisapp = analysisGUI(self.FinalTracks, drawing_window.rejects)
        analysisapp.mainloop()
