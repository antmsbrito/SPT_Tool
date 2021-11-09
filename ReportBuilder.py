import os

import pandas as pd

from tracks import Track

import numpy as np

from jinja2 import Template
from datetime import datetime, date
from matplotlib import pyplot as plt
from matplotlib import patches
import seaborn as sns

sns.set_theme()
sns.set_style("white")
sns.set_style("ticks")

import base64
from io import BytesIO

from scipy.stats import mannwhitneyu

import pathlib

# From an array of track objects build the report folder
# Input: array of tracks and folder path
# Output: folder with
#                     1 npy dump
#                     2 html summary
#                     more to come...


def build_property_array_old(trackobj, prop):
    if prop == "disp":
        arr = []
        for tr in trackobj:
            arr = np.append(arr, np.mean(getattr(tr, prop)))
        return arr
    else:
        arr = []
        for tr in trackobj:
            arr = np.append(arr, getattr(tr, prop))
        return arr

def build_property_array(trackobj, prop):
    if prop == "minmax":
        arr = []
        l = []
        for tr in trackobj:
            l.append(len(getattr(tr, prop)))
            arr = np.append(arr, getattr(tr, prop))
        print(np.mean(l))
        return arr
    else:
        arr = []
        for tr in trackobj:
            arr = np.append(arr, np.array(np.mean(getattr(tr, prop))))
        return arr




def buildhistogram(bins):
    centerbins = []
    for idx, bini in enumerate(bins):
        if bini == bins[-1]:
            continue
        else:
            centerbins.append((bins[idx + 1] + bins[idx]) / 2)
    return centerbins


def html_summary(tracklist, rejects, savepath, MANUALbool=False, MINMAXbool=True, FINITEbool=True, DISPbool=True):
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
    angle = [np.rad2deg(np.arccos(i.ellipse['minor'] / i.ellipse['major'])) for i in tracklist]
    number_of_tracks = len(meanfd)
    average_track_length = np.mean(tracklength)
    average_total_2d_disp = np.mean(
        [np.sqrt((i.xtrack[-1] - i.xtrack[0]) ** 2 + (i.ytrack[-1] - i.ytrack[0]) ** 2) * 1000 for i in tracklist])
    average_speed_2d = np.mean([i.cumvelonoz for i in tracklist]) * 1000

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
        "number_of_tracks": f'{number_of_tracks} ({number_of_tracks/(number_of_tracks+len(rejects)):0.2f}%)',
        "average_track_length": average_track_length,
        "average_total_2d_disp": average_total_2d_disp,
        "average_speed_2d": average_speed_2d}

    with open(r"templates/Summary_Template.html", 'r') as f:
        template = Template(f.read())

    with open(os.path.join(savepath, "Summary.html"), 'w+') as f:
        f.write(template.render(report_dict))


def html_comparison(listoffiles, savepath):
    filenames = [i[0].designator[:-2] for i in listoffiles]

    manual = [build_property_array(file, 'manual') for file in listoffiles]
    displacement = [build_property_array(file, 'disp') for file in listoffiles]
    minmax = [build_property_array(file, 'minmax') for file in listoffiles]
    finite = [build_property_array(file, 'finitediff') for file in listoffiles]

    fig, ax = plt.subplots()
    finitedata = pd.DataFrame({'Velocity frequency (nm/s)': list(finite[0]) + list(finite[1]),
                               'Condition': ['A'] * len(finite[0]) + ['B'] * len(finite[1])})
    ax = sns.violinplot(x='Condition', y='Velocity frequency (nm/s)', data=finitedata)
    plt.tight_layout()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    enc_finite = base64.b64encode((tmpfile.getvalue())).decode('utf8')

    fig, ax = plt.subplots()
    minmaxdata = pd.DataFrame({'Velocity frequency (nm/s)': list(minmax[0]) + list(minmax[1]),
                               'Condition': ['A'] * len(minmax[0]) + ['B'] * len(minmax[1])})
    ax = sns.violinplot(x='Condition', y='Velocity frequency (nm/s)', data=minmaxdata)
    plt.tight_layout()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    enc_minmax = base64.b64encode((tmpfile.getvalue())).decode('utf8')

    fig, ax = plt.subplots()
    manualdata = pd.DataFrame({'Velocity frequency (nm/s)': list(manual[0]) + list(manual[1]),
                               'Condition': ['A'] * len(manual[0]) + ['B'] * len(manual[1])})
    ax = sns.violinplot(x='Condition', y='Velocity frequency (nm/s)', data=manualdata)
    plt.tight_layout()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    enc_manual = base64.b64encode((tmpfile.getvalue())).decode('utf8')

    fig, ax = plt.subplots()
    dispdata = pd.DataFrame({'Velocity frequency (nm/s)': list(displacement[0]) + list(displacement[1]),
                             'Condition': ['A'] * len(displacement[0]) + ['B'] * len(displacement[1])})
    ax = sns.violinplot(x='Condition', y='Velocity frequency (nm/s)', data=dispdata)
    plt.tight_layout()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    enc_disp = base64.b64encode((tmpfile.getvalue())).decode('utf8')

    report_dict = {'number_of_files': len(listoffiles),
                   'files': filenames,
                   'mannManual': np.nan if not len(manual[0]) or not len(manual[1]) else mannwhitneyu(manual[0], manual[1]).pvalue,
                   'mannDisp': mannwhitneyu(displacement[0], displacement[1]).pvalue,
                   'mannMinmax': mannwhitneyu(minmax[0], minmax[1]).pvalue,
                   'mannFinite': mannwhitneyu(finite[0], finite[1]).pvalue,
                   'enc_disp': enc_disp,
                   'enc_manual': enc_manual,
                   'enc_minmax': enc_minmax,
                   'enc_finite': enc_finite,
                   'date': datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

    with open(r"templates/Comparison_Template.html", 'r') as f:
        template = Template(f.read())

    with open(os.path.join(savepath, "Summary.html"), 'w+') as f:
        f.write(template.render(report_dict))


def npy_builder(tracklist, rejects, savepath):
    np.save(f"{savepath}\\DataDump.npy", tracklist)
    np.save(f"{savepath}\\RejectedTracks.npy", rejects)
    return


def makeimage(tracklist, savepath, MANUALbool=False, MINMAXbool=True, FINITEbool=True, DISPbool=True):
    """This makes for each track an image with two plots:
            Track overlaid with raw image (IF POSSIBLE)
            Histogram with axvline of where it belongs (or maybe violin plot it)"""

    for tr in tracklist:

        plots = np.count_nonzero([MANUALbool, MINMAXbool, FINITEbool, DISPbool])
        currentplot = plots + 1
        fig = plt.figure(figsize=(16, 9))

        ax1 = fig.add_subplot(2, int(plots), (1, int(plots)))
        if tr.image:
            ax1.imshow(tr.image, cmap='gray')
        ax1.plot(tr.xtrack / 0.08, tr.ytrack / 0.08, color='r', label="Track")
        eli = patches.Ellipse((tr.ellipse['x0'] / 0.08, tr.ellipse['y0'] / 0.08), tr.ellipse['major'] / 0.08,
                              tr.ellipse['minor'] / 0.08, tr.ellipse['angle'], fill=False,
                              edgecolor='black', alpha=0.3)
        ax1.add_patch(eli)
        ax1.set_xlabel("x coordinates (px)")
        ax1.set_ylabel("y coordinates (px)")
        ax1.set_xlim((np.average(tr.xtrack / 0.08) - 30, np.average(tr.xtrack / 0.08) + 30))
        ax1.set_ylim((np.average(tr.ytrack / 0.08) - 30, np.average(tr.ytrack / 0.08) + 30))
        ax1.set_aspect('equal')
        ax1.legend()

        if MANUALbool:
            ax2 = fig.add_subplot(2, int(plots), currentplot)
            currentplot += 1
            manual_array = build_property_array(tracklist, 'manual')
            n, bins, pat = ax2.hist(x=manual_array, bins='auto', density=True, alpha=0.2)
            ax2.plot(buildhistogram(bins), n, 'k', linewidth=1, label="Manual Sectioning")
            ax2.vlines(x=tr.manual, colors='k', ymin=0, ymax=np.max(n), alpha=0.4)
            ax2.set_xlabel('Velocity (nm/s)')
            ax2.set_ylabel('PDF')
            ax2.legend()

        if MINMAXbool:
            ax3 = fig.add_subplot(2, int(plots), currentplot)
            currentplot += 1
            minmax_array = build_property_array(tracklist, 'minmax')
            n, bins, pat = ax3.hist(x=minmax_array, bins='auto', density=True, alpha=0.2)
            ax3.plot(buildhistogram(bins), n, 'b', linewidth=1, label="MinMax Sectioning")
            ax3.vlines(x=tr.minmax, colors='b', ymin=0, ymax=np.max(n), alpha=0.4)
            ax3.set_xlabel('Velocity (nm/s)')
            ax3.set_ylabel('PDF')
            ax3.legend()

        if FINITEbool:
            ax4 = fig.add_subplot(2, int(plots), currentplot)
            currentplot += 1
            finite_array = build_property_array(tracklist, 'finitediff')
            n, bins, pat = ax4.hist(x=finite_array, bins='auto', density=True, alpha=0.2)
            ax4.plot(buildhistogram(bins), n, 'r', linewidth=1, label="Finite Differences")
            ax4.vlines(x=tr.finitediff, ymin=0, ymax=np.max(n), colors='r', alpha=0.4)
            ax4.set_xlabel('Velocity (nm/s)')
            ax4.set_ylabel('PDF')
            ax4.legend()

        if DISPbool:
            ax5 = fig.add_subplot(2, int(plots), currentplot)
            currentplot += 1
            disp_array = build_property_array(tracklist, 'disp')
            n, bins, pat = ax5.hist(x=disp_array, bins='auto', density=True, alpha=0.2)
            ax5.plot(buildhistogram(bins), n, 'g', linewidth=1, label="Displacement")
            ax5.vlines(x=tr.disp, ymin=0, ymax=np.max(n), colors='g', alpha=0.4)
            ax5.set_xlabel('Velocity (nm/s)')
            ax5.set_ylabel('PDF')
            ax5.legend()

        plt.tight_layout()

        #TODO
        try:
            name = tr.designator.split('\\')[-2] + '_' + tr.designator.split('\\')[-1] + '.jpeg'
        except IndexError:
            name = tr.designator + '.jpeg'

        sv = os.path.join(savepath, name)
        #checkfolder = ''.join(savepath.split('/')[:-1])
        #pathlib.Path(checkfolder).mkdir(parents=True, exist_ok=True)
        fig.savefig(sv)
