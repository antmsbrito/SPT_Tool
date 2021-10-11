import os

from tracks import Track

import numpy as np

from jinja2 import Template
from datetime import datetime, date
from matplotlib import pyplot as plt
import base64
from io import BytesIO


# From an array of track objects build the report folder
# Input: array of tracks and folder path
# Output: folder with
#                     1 npy dump
#                     2 html summary
#                     more to come...


def build_property_array(trackobj, prop):
    arr = []
    for tr in trackobj:
        arr = np.append(arr, getattr(tr, prop))
    return arr


def buildhistogram(bins):
    centerbins = []
    for idx, bini in enumerate(bins):
        if bini == bins[-1]:
            continue
        else:
            centerbins.append((bins[idx + 1] + bins[idx]) / 2)
    return centerbins


def html_summary(tracklist, savepath, MANUALbool, MINMAXbool, FINITEbool, DISPbool):
    if not MANUALbool and not MINMAXbool and not FINITEbool and not DISPbool:
        return 0

    manual_array = 0
    minmax_array = 0
    finite_array = 0
    disp_array = 0

    fig, ax = plt.subplots()

    if MANUALbool:
        manual_array = build_property_array(tracklist, 'manual')
        n, bins, patches = plt.hist(x=manual_array, bins='auto', density=True, alpha=0.1)
        plt.plot(buildhistogram(bins), n, 'k', linewidth=1, label="Manual Sectioning")

    if MINMAXbool:
        minmax_array = build_property_array(tracklist, 'minmax')
        n, bins, patches = plt.hist(x=minmax_array, bins='auto', density=True, alpha=0.1)
        plt.plot(buildhistogram(bins), n, 'b', linewidth=1, label="MinMax Sectioning")

    if FINITEbool:
        finite_array = build_property_array(tracklist, 'finitediff')
        n, bins, patches = plt.hist(x=finite_array, bins='auto', density=True, alpha=0.1)
        plt.plot(buildhistogram(bins), n, 'r', linewidth=1, label="Finite Sectioning")

    if DISPbool:
        disp_array = build_property_array(tracklist, 'disp')
        n, bins, patches = plt.hist(x=disp_array, bins='auto', density=True, alpha=0.1)
        plt.plot(buildhistogram(bins), n, 'g', linewidth=1, label="Displacement")

    plt.xlabel('Velocity (nm/s)')
    plt.ylabel('PDF')
    plt.legend()
    plt.tight_layout()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    # Other important parameters
    if MANUALbool or tracklist[0].manual:
        meanman = [np.mean(i.manual) for i in tracklist]
    meanfd = [np.mean(i.finitediff) for i in tracklist]
    meanminmax = [np.mean(i.minmax) for i in tracklist]
    meandisp = [np.mean(i.disp) for i in tracklist]
    tracklength = [len(i.xtrack) for i in tracklist]
    diameter = [i.ellipse['major'] * 1000 for i in tracklist]
    angle = [np.rad2deg(np.cos(i.ellipse['minor'] / i.ellipse['major'])) for i in tracklist]
    number_of_tracks = len(meanfd)
    average_track_length = np.mean(tracklength)
    average_total_2d_disp = np.mean(
        [np.sqrt((i.xtrack[-1] - i.xtrack[0]) ** 2 + (i.ytrack[-1] - i.ytrack[0]) ** 2)*1000 for i in tracklist])
    average_speed_2d = np.mean([i.cumvelonoz for i in tracklist])*1000

    fig, ax = plt.subplots()
    if MANUALbool or tracklist[0].manual:
        plt.scatter(tracklength, meanman, c='k', label="Manual")
    plt.scatter(tracklength, meandisp, c='g', label="Displacement")
    plt.scatter(tracklength, meanminmax, c='b', label="MinMax")
    plt.scatter(tracklength, meanfd, c='r', label="Finite Differences")
    plt.xlabel("Track Length")
    plt.ylabel("Average Velocity per track (nm/s)")
    plt.legend()
    plt.tight_layout()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    enconded_tracklength = base64.b64encode((tmpfile.getvalue())).decode('utf8')

    fig, ax = plt.subplots()
    if MANUALbool or tracklist[0].manual:
        plt.scatter(diameter, meanman, c='k', label="Manual")
    plt.scatter(diameter, meandisp, c='g', label="Displacement")
    plt.scatter(diameter, meanminmax, c='b', label="MinMax")
    plt.scatter(diameter, meanfd, c='r', label="Finite Differences")
    plt.xlabel("Major axis of ellipse (nm)")
    plt.ylabel("Average Velocity per track (nm/s)")
    plt.legend()
    plt.tight_layout()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    enconded_diameter = base64.b64encode((tmpfile.getvalue())).decode('utf8')

    fig, ax = plt.subplots()
    if MANUALbool or tracklist[0].manual:
        plt.scatter(angle, meanman, c='k', label="Manual")
    plt.scatter(angle, meandisp, c='g', label="Displacement")
    plt.scatter(angle, meanminmax, c='b', label="MinMax")
    plt.scatter(angle, meanfd, c='r', label="Finite Differences")
    plt.xlabel("Angle to the microscopy plane (deg)")
    plt.ylabel("Average Velocity per track (nm/s)")
    plt.legend()
    plt.tight_layout()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    enconded_angle = base64.b64encode((tmpfile.getvalue())).decode('utf8')

    report_dict = {
        "finite_vel": np.mean(finite_array),
        "finite_std": np.std(finite_array),
        "finite_med": np.median(finite_array),
        "finite_n": len(finite_array) if isinstance(finite_array, (list, tuple, np.ndarray)) else 0,
        "minmax_vel": np.mean(minmax_array),
        "minmax_std": np.std(minmax_array),
        "minmax_med": np.median(minmax_array),
        "minmax_n": len(minmax_array) if isinstance(minmax_array, (list, tuple, np.ndarray)) else 0,
        "manual_vel": np.mean(manual_array),
        "manual_std": np.std(manual_array),
        "manual_med": np.median(manual_array),
        "manual_n": len(manual_array) if isinstance(manual_array, (list, tuple, np.ndarray)) else 0,
        "disp_vel": np.mean(disp_array),
        "disp_std": np.std(disp_array),
        "disp_med": np.median(disp_array),
        "disp_n": len(disp_array) if isinstance(disp_array, (list, tuple, np.ndarray)) else 0,
        "enconded_hist": encoded,
        "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "enconded_diameter": enconded_diameter,
        "enconded_tracklength": enconded_tracklength,
        "enconded_angle": enconded_angle,
        "number_of_tracks": number_of_tracks,
        "average_track_length": average_track_length,
        "average_total_2d_disp": average_total_2d_disp,
        "average_speed_2d": average_speed_2d}

    with open(r"templates/Summary_Template.html", 'r') as f:
        template = Template(f.read())

    with open(os.path.join(savepath, "Summary.html"), 'w+') as f:
        f.write(template.render(report_dict))


def npy_builder(tracklist, savepath):
    np.save(f"{savepath}\\DataDump.npy", tracklist)
    return
