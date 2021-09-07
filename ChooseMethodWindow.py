import tkinter as tk
from EllipseGUI import ellipseGUI
from CSVGUI import csvGUI

# Class that inherits root window class from tk
class ChooseWindow(tk.Tk):
    def __init__(self):
        super().__init__()  # init of tk.Tk

        self.wm_title("Method")
        self.title("Method")

        status_text = tk.Label(master=self, text="Choose how to input ellipse data")
        status_text.pack(side='top', fill='both')

        CSV_button = tk.Button(master=self, text="Input CSV", command=self.csv)
        CSV_button.pack(side='top', fill='both')

        ELLIPSE_button = tk.Button(master=self, text="Draw Ellipses", command=self.ellipse)
        ELLIPSE_button.pack(side='top', fill='both')

    def csv(self):
        self.destroy()
        gui = csvGUI()
        gui.mainloop()

    def ellipse(self):
        self.destroy()
        gui = ellipseGUI()
        gui.mainloop()

if __name__ == '__main__':
    app = ChooseWindow()
    app.mainloop()
