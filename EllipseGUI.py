import tkinter as tk
import numpy as np

from tracksv2 import Track


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

        # Text for number of tracks loaded
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

        # QUIT_button = tk.Button(master=frame_input, text="Quit", command=self.quit)
        # QUIT_button.pack(side='bottom', fill='x')

    def init_output(self):
        frame_output = tk.Frame(self)
        frame_output.pack(fill='both', expand=True, side='right')

        status_text = tk.Label(master=frame_output, textvariable=self.LabelText)
        status_text.pack(side='top', fill='both')

        ELLIPSE_button = tk.Button(master=frame_output, text="Draw Ellipses", command=self.draw_ellipse)
        ELLIPSE_button.pack(side='top', fill='both')

        OUTPUT_button = tk.Button(master=frame_output, text="Output", command=self.output)
        OUTPUT_button.pack(side='top', fill='both')

    def load_xml(self):

        xml = tk.filedialog.askopenfilename(initialdir="C:", title="Select Trackmate xml file")

        if not xml[-3:] == "xml":
            print("File extension not supported")
        else:
            self.TrackList = np.append(self.TrackList, Track.generatetrack(xml))
            self.NumberOfTracks.set(len(self.TrackList))
            self.LabelText.set(f"{self.NumberOfTracks.get()} tracks loaded")

    def draw_ellipse(self):
        if self.NumberOfTracks.get() == 0:
            print("No tracks!")
        else:
            newWindow = tk.Toplevel()
            newWindow.title("Graph")
            newWindow.geometry("600x600")
            newWindow.grab_set()

    def output(self):
        pass


if __name__ == '__main__':
    app = ellipseGUI()
    app.mainloop()
