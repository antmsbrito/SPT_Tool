"""
SPT_Tool 2021
Antonio Brito @ BCB lab ITQB
"""

import tkinter as tk
from EllipseGUI import ellipseGUI
from CSVGUI import csvGUI
from loadGUI import loadGUI
from BatchGUI import batchGUI


# Class that inherits root window class from tk
class ChooseWindow(tk.Tk):
    """
    GUI window responsible for the choosing what input method to use
     - .csv predrawn ellipse from fiji
     - draw ellipses in-situ
     - load previous analyzed data to compare or reanalyze
    """
    def __init__(self):
        super().__init__()  # init of tk.Tk

        self.wm_title("Methods")
        self.title("Methods")
        self.geometry('225x125')

        status_text = tk.Label(master=self, text="Choose how to input ellipse data")
        status_text.pack(side='top', fill='both')

        CSV_button = tk.Button(master=self, text="Input CSV", command=self.csv)
        CSV_button.pack(side='top', fill='both')

        ELLIPSE_button = tk.Button(master=self, text="Draw Ellipses", command=self.ellipse)
        ELLIPSE_button.pack(side='top', fill='both')

        LOAD_button = tk.Button(master=self, text="Load .npy data", command=self.loadnpy)
        LOAD_button.pack(side='top', fill='both')

        BATCH_button = tk.Button(master=self, text="Batch analysis", command=self.batch)
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

    def loadnpy(self):
        """
        Open next GUI window for inputting one or more .npy files of previously loaded data
        """
        self.destroy()
        gui = loadGUI()
        gui.mainloop()

    def batch(self):
        """
        Opens next GUI window for batch analysis of several conditions
        """

        self.destroy()
        gui = batchGUI()
        gui.mainloop()


if __name__ == '__main__':
    app = ChooseWindow()
    app.mainloop()
