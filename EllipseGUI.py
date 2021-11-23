"""
SPT_TOOL
@author António Brito
ITQB-UNL BCB 2021
"""

import tkinter as tk
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

from DrawingGUI import DrawingEllipses
from AnGUI import analysisGUI


# Class that inherits root window class from tk
class ellipseGUI(tk.Tk):
    def __init__(self):
        super().__init__()  # init of tk.Tk

        # Configure root window
        self.wm_title("IO Window")
        self.title("IO Window")
        self.geometry('350x75')

        # List of track related objects
        self.TrackList = []
        self.NumberOfImages = tk.IntVar()

        # Text for number of tracks loaded and respective track name
        self.LabelText = tk.StringVar()
        self.LabelText.set(f"{self.NumberOfImages.get()} files loaded")

        # Main window has two frames
        # left side for input related stuff; right for output and
        self.init_input()
        self.init_output()

    def init_input(self):
        frame_input = tk.Frame(self)
        frame_input.pack(fill='both', expand=True, side='left')

        XML_button = tk.Button(master=frame_input, text="Load Trackmate .xml file", command=self.load_xml)
        XML_button.pack(fill='x')

        ANALYSIS_button = tk.Button(master=frame_input, text="Draw Ellipses", command=self.drawing_button)
        ANALYSIS_button.pack(fill='x', expand=True)

    def init_output(self):
        frame_output = tk.Frame(self)
        frame_output.pack(fill='both', expand=True, side='right')

        status_text = tk.Label(master=frame_output, textvariable=self.LabelText)
        status_text.pack(side='top', fill='both')

    def load_xml(self):

        xml = tk.filedialog.askopenfilename(initialdir="C:", title="Select Trackmate xml file")
        if not xml[-3:] == "xml":
            tk.messagebox.showerror(title="XML", message="File extension must be .xml")
        else:
            tk.messagebox.showinfo(title="Load image", message="Please load the corresponding image file")
            image = []
            while not image:
                image = self.load_image(xml)

        self.TrackList = np.append(self.TrackList, PrecursorTrackObject.generator(xml, image))
        self.NumberOfImages.set(len(self.TrackList))
        self.LabelText.set(f"{self.NumberOfImages.get()} images loaded")

    def load_image(self, xmlpath):
        imgdir = xmlpath
        imgpath = tk.filedialog.askopenfilename(initialdir=imgdir, title="Select image file")
        im = Image.open(imgpath)
        return im

    def drawing_button(self):

        drawing_window = DrawingEllipses(self.TrackList)
        drawing_window.grab_set()
        self.wait_window(drawing_window)

        self.FinalTracks = drawing_window.track_classes
        self.destroy()
        analysisapp = analysisGUI(self.FinalTracks, drawing_window.rejects)
        analysisapp.mainloop()


class PrecursorTrackObject:

    def __init__(self, im, x, y, samplerate, xmlhandle):
        self.imageobject = im
        self.x = x
        self.y = y
        self.xml = xmlhandle
        self.sr = samplerate

    @classmethod
    def generator(cls, xmlfile, image):

        classlist = []
        root = ET.parse(xmlfile).getroot()
        srate = root.attrib['frameInterval']
        counter = 0
        for children in root:
            tempx = []
            tempy = []
            for grandchildren in children:
                tempx.append(float(grandchildren.attrib['x']))  # list of x coords
                tempy.append(float(grandchildren.attrib['y']))  # list of y coords
            classlist.append(cls(image, tempx, tempy, srate, xmlfile))
            counter += 1

        return classlist


if __name__ == '__main__':
    app = ellipseGUI()
    app.mainloop()
