import tkinter as tk
import numpy as np

from tracks import Track

from PIL import Image
from matplotlib import pyplot as plt


# Class that inherits root window class from tk
class ellipseGUI(tk.Tk):
    def __init__(self):
        super().__init__()  # init of tk.Tk

        # Configure root window
        self.wm_title("IO Window")
        self.title("IO Window")
        self.geometry('600x600')

        # List of track related objects
        self.TrackList = []
        self.NumberOfTracks = tk.IntVar()
        self.CurrentTrack = tk.StringVar()
        self.CurrentTrack.set("")

        # Text for number of tracks loaded and respective track name
        self.LabelText = tk.StringVar()
        self.LabelText.set(f"{self.NumberOfTracks.get()} tracks loaded")

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
        track_text = tk.Label(master=frame_output, textvariable=self.CurrentTrack,wraplength=300, justify="center")
        track_text.pack(side='bottom', fill='both')

    def load_xml(self):

        xml = tk.filedialog.askopenfilename(initialdir="C:", title="Select Trackmate xml file")
        if not xml[-3:] == "xml":
            tk.messagebox.showerror(title="XML", message="File extension must be .xml")
        else:
            tk.messagebox.showinfo(title="Load image", message="Please load the corresponding image file")
            image = []
            while not image:
                image = self.load_image(xml)

            self.TrackList = np.append(self.TrackList, [1])
            self.NumberOfTracks.set(len(self.TrackList))
            self.LabelText.set(f"{self.NumberOfTracks.get()} tracks loaded")

    def load_image(self, xmlpath):
        imgdir = xmlpath
        imgpath = tk.filedialog.askopenfilename(initialdir=imgdir, title="Select image file")

        im = Image.open(imgpath)
        return im

    def drawing_button(self):
        pass

if __name__ == '__main__':
    app = ellipseGUI()
    app.mainloop()
