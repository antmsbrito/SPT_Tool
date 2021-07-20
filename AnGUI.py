import tkinter as tk

# Class that inherits root window class from tk
class analysisGUI(tk.Tk):
    def __init__(self, tracks):
        super().__init__()  # init of tk.Tk

        self.TrackList = tracks

        self.wm_title("Analysis")
        self.title("Analysis")

        self.init_options()
        self.init_stats()

        self.TrackList = tracks

    def init_options(self):
        options_frame = tk.Frame(self)
        options_frame.pack(fill='both', expand=True, side='left')

        manual_button = tk.Button(master=options_frame, text='', command=self.)
        manual_button.pack(fill='x', expand=True)

        minmax_button = tk.Button(master=options_frame, text='', command=self.)
        minmax_button.pack(fill='x', expand=True)

        finite_button = tk.Button(master=options_frame, text='', command=self.)
        finite_button.pack(fill='x', expand=True)

        displacement_button = tk.Button(master=options_frame, text='', command=self.)
        displacement_button.pack(fill='x', expand=True)

    def init_stats(self):
        stats_frame = tk.Frame(self)
        stats_frame.pack(fill='both', expand=True, side='right')

    