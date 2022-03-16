"""
SPT_Tool 2021
Antonio Brito @ BCB lab ITQB
"""

import tkinter as tk
from EllipseGUI import ellipseGUI
from CSVGUI import csvGUI
from loadGUI import loadHD5, loadNPY
from BatchGUI import batchHD5, batchNPY


# Class that inherits root window class from tk
class ChooseWindow(tk.Tk):
    """
    GUI window responsible for the choosing what input method to use
     - .csv predrawn ellipse from fiji
     - draw ellipses in-situ
     - load previous analyzed data to compare or reanalyze
     - batch load npys for several analysis
    """
    def __init__(self):
        super().__init__()  # init of tk.Tk

        self.wm_title("Methods")
        self.title("Methods")
        self.geometry('250x180')

        status_text = tk.Label(master=self, text="Choose how to input ellipse data")
        status_text.pack(side='top', fill='both')

        CSV_button = tk.Button(master=self, text="Input ellipse CSV", command=self.csv)
        CSV_button.pack(side='top', fill='both')

        ELLIPSE_button = tk.Button(master=self, text="Draw Ellipses (xml + tif)", command=self.ellipse)
        ELLIPSE_button.pack(side='top', fill='both')

        LOADhd5_button = tk.Button(master=self, text="Load .h5 data", command=self.loadhd5)
        LOADhd5_button.pack(side='top', fill='both')

        BATCHhd5_button = tk.Button(master=self, text="Batch analysis (load several .h5 data)", command=self.batchhd5)
        BATCHhd5_button.pack(side='top', fill='both')

        LOAD_button = tk.Button(master=self, text="Load .npy data (.npy [Legacy])", command=self.loadnpy)
        LOAD_button.pack(side='top', fill='both')

        BATCH_button = tk.Button(master=self, text="Batch analysis (load several .npy [Legacy])", command=self.batchnpy)
        BATCH_button.pack(side='top', fill='both')


    def csv(self):
        """
        Open next GUI window for inputting xml and csv data
        """
        self.destroy()
        gui = csvGUI()
        gui.mainloop()

    def ellipse(self):
        """
        Open next GUI window for inputting xml and drawing ellipse
        """
        self.destroy()
        gui = ellipseGUI()
        gui.mainloop()

    def loadhd5(self):
        """
        Open next GUI window for inputting one or more .npy files of previously loaded data
        """
        self.destroy()
        gui = loadHD5()
        gui.mainloop()

    def batchhd5(self):
        """
        Opens next GUI window for batch analysis of several conditions
        """

        self.destroy()
        gui = batchHD5()
        gui.mainloop()

    def loadnpy(self):
        """
        Open next GUI window for inputting one or more .npy files of previously loaded data
        """
        self.destroy()
        gui = loadNPY()
        gui.mainloop()

    def batchnpy(self):
        """
        Opens next GUI window for batch analysis of several conditions
        """

        self.destroy()
        gui = batchNPY()
        gui.mainloop()



if __name__ == '__main__':
    app = ChooseWindow()
    app.mainloop()
