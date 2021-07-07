import tkinter as tk
from tracks import Track


# Class that inherits root window class from tk
class IOWindow(tk.Tk):
    def __init__(self):
        super().__init__() # init of tk.Tk

        # Configure root window
        self.wm_title("IO Window")
        self.title("IO Window")
        self.geometry('600x600')

        # Main window has two frames
        self.init_input()
        self.init_output()

        # List of track objects
        self.TrackList = []

    def init_input(self):
        frame_input = tk.Frame(self)
        frame_input.pack(fill='both', expand=True)

    def init_output(self):
        frame_output = tk.Frame(self)
        frame_output.pack(fill='both', expand=True)

if __name__ == '__main__':
    app = IOWindow()
    app.mainloop()