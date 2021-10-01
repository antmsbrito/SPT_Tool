import tkinter as tk

import os

from jinja2 import Template
from datetime import datetime, date
from matplotlib import pyplot as plt
import base64
from io import BytesIO

import numpy as np

from tracks import Track


from GUI_ManualSectioning import ManualSectioning


# Class that inherits root window class from tk
class analysisGUI(tk.Tk):
    def __init__(self, tracks):
        super().__init__()  # init of tk.Tk

        self.TrackList = tracks

        self.wm_title("Analysis")
        self.title("Analysis")
        self.geometry("150x150")

        self.TrackList = tracks

        self.manual_var = tk.IntVar()
        self.minmax_var = tk.IntVar()
        self.finite_var = tk.IntVar()
        self.displacement_var = tk.IntVar()

        self.manual_velo_array = np.nan
        self.minmax_velo_array = np.nan
        self.finite_velo_array = np.nan
        self.displacement_velo_array = np.nan

        self.savepath = tk.filedialog.askdirectory(initialdir="C:", title="Please select where to save the data")
        self.savepath = os.path.join(self.savepath, rf"SPT_{date.today().strftime('%d_%m_%Y')}")
        os.makedirs(self.savepath, exist_ok=True)

        self.init_options()
        # self.init_stats()

    def init_options(self):
        options_frame = tk.Frame(self)
        options_frame.pack(fill='both', expand=True, side='left')

        a_label = tk.Label(master=options_frame, text="Please choose the analysis", wraplength=300, justify='center')
        a_label.pack(side='top', fill='both')

        check_manual = tk.Checkbutton(master=options_frame, text="Manual Sectioning", variable=self.manual_var,
                                      onvalue=1, offvalue=0)
        check_manual.pack(anchor='w')

        check_minmax = tk.Checkbutton(master=options_frame, text="MinMax Sectioning", variable=self.minmax_var,
                                      onvalue=1, offvalue=0)
        check_minmax.pack(anchor='w')

        check_finite = tk.Checkbutton(master=options_frame, text="Finite Differences", variable=self.finite_var,
                                      onvalue=1, offvalue=0)
        check_finite.pack(anchor='w')

        check_displacement = tk.Checkbutton(master=options_frame, text="Real Displacement",
                                            variable=self.displacement_var, onvalue=1, offvalue=0)
        check_displacement.pack(anchor='w')

        analysis_button = tk.Button(master=options_frame, text="Start analysis", command=self.analyze)
        analysis_button.pack(fill='x', expand=True)

    def init_stats(self):
        stats_frame = tk.Frame(self)
        stats_frame.pack(fill='both', expand=True, side='right')

        stats_label = tk.Label(master=stats_frame, text="Stats", wraplength=300, justify='center')
        stats_label.pack(side='top', fill='both')

    def analyze(self):
        fig, ax = plt.subplots()
        print("Starting analysis")
        if self.manual_var.get():
            print("Manual...")
            TL = ManualSectioning(self.TrackList)
            TL.grab_set()
            self.wait_window(TL)
            self.manual_velo_array = TL.section_velocity * 1000
            n, bins, patches = plt.hist(x=self.manual_velo_array, bins='auto', density=True, alpha=0.1)
            plt.plot(self.buildhistogram(bins), n, 'k', linewidth=1, label="Manual Sectioning")

        if self.minmax_var.get():
            print("MinMax...")
            self.minmax_velo_array = self.build_property_array(self.TrackList, 'minmax')
            n, bins, patches = plt.hist(x=self.minmax_velo_array, bins='auto', density=True, alpha=0.1)
            plt.plot(self.buildhistogram(bins), n, 'b', linewidth=1, label="MinMax Sectioning")

        if self.finite_var.get():
            print("Finite...")
            self.finite_velo_array = self.build_property_array(self.TrackList, 'finitediff')
            n, bins, patches = plt.hist(x=self.finite_velo_array, bins='auto', density=True, alpha=0.1)
            plt.plot(self.buildhistogram(bins), n, 'r', linewidth=1, label="Finite Differences")

        if self.displacement_var.get():
            print("Displacement...")
            self.displacement_velo_array = self.build_property_array(self.TrackList, 'disp')
            n, bins, patches = plt.hist(x=self.displacement_velo_array, bins='auto', density=True, alpha=0.1)
            plt.plot(self.buildhistogram(bins), n, 'g', linewidth=1, label="Displacement")

        print("Building histogram")
        plt.xlabel('Velocity (nm/s)')
        plt.ylabel('PDF')
        plt.legend()
        plt.tight_layout()
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

        # Display parameters
        meanfd = [np.mean(i.finitediff) for i in self.TrackList]
        meanminmax = [np.mean(i.minmax) for i in self.TrackList]
        meandisp = [np.mean(i.disp) for i in self.TrackList]
        tracklength = [len(i.xtrack) for i in self.TrackList]
        diameter = [i.ellipse['major']*1000 for i in self.TrackList]

        number_of_tracks = len(meanfd)
        average_track_length = np.mean(tracklength)
        average_total_2d_disp = np.mean([np.sqrt((i.xtrack[-1]-i.xtrack[0])**2+(i.ytrack[-1]-i.ytrack[0])**2) for i in self.TrackList])
        average_speed_2d = np.mean([i.cumvelonoz for i in self.TrackList])


        fig, ax = plt.subplots()
        plt.scatter(tracklength, meandisp,c='g',label="Displacement")
        plt.scatter(tracklength, meanminmax, c='b', label="MinMax")
        plt.scatter(tracklength, meanfd, c='r', label="Finite Differences")
        plt.xlabel("Track Length")
        plt.ylabel("Average Velocity per track (nm/s)")
        plt.legend()
        plt.tight_layout
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        enconded_tracklength = base64.b64encode((tmpfile.getvalue())).decode('utf8')

        fig, ax = plt.subplots()
        plt.scatter(diameter, meandisp,c='g',label="Displacement")
        plt.scatter(diameter, meanminmax, c='b', label="MinMax")
        plt.scatter(diameter, meanfd, c='r', label="Finite Differences")
        plt.xlabel("Major axis of ellipse (nm)")
        plt.ylabel("Average Velocity per track (nm/s)")
        plt.legend()
        plt.tight_layout
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        enconded_diameter = base64.b64encode((tmpfile.getvalue())).decode('utf8')

        print("Building .html...")

        report_dict = {
            "finite_vel": np.mean(self.finite_velo_array),
            "finite_std": np.std(self.finite_velo_array),
            "finite_med": np.median(self.finite_velo_array),
            "finite_n": len(self.finite_velo_array),
            "minmax_vel": np.mean(self.minmax_velo_array),
            "minmax_std": np.std(self.minmax_velo_array),
            "minmax_med": np.median(self.minmax_velo_array),
            "minmax_n": len(self.minmax_velo_array),
            "manual_vel": np.mean(self.manual_velo_array),
            "manual_std": np.std(self.manual_velo_array),
            "manual_med": np.median(self.manual_velo_array),
            "manual_n": len(self.manual_velo_array),
            "disp_vel": np.mean(self.displacement_velo_array),
            "disp_std": np.std(self.displacement_velo_array),
            "disp_med": np.median(self.displacement_velo_array),
            "disp_n": len(self.displacement_velo_array),
            "enconded_hist": encoded,
            "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "enconded_diameter": enconded_diameter,
            "enconded_tracklength":enconded_tracklength,
            "number_of_tracks":number_of_tracks,
            "average_track_length":average_track_length,
            "average_total_2d_disp":average_total_2d_disp,
            "average_speed_2d":average_speed_2d}

        with open(r"template.html", 'r') as f:
            template = Template(f.read())

        with open(os.path.join(self.savepath, "Summary.html"), 'w+') as f:
            f.write(template.render(report_dict))

        tk.messagebox.showinfo(title="All done!", message="Check for the .html file for full report")

        self.destroy()

    @staticmethod
    def buildhistogram(bins):
        centerbins = []
        for idx, bini in enumerate(bins):
            if bini == bins[-1]:
                continue
            else:
                centerbins.append((bins[idx + 1] + bins[idx]) / 2)
        return centerbins

    @staticmethod
    def build_property_array(trackobj,prop):
        arr = []
        for tr in trackobj:
            arr = np.append(arr, getattr(tr,prop))
        return arr