"""
SPT_TOOL
@author António Brito
ITQB-UNL BCB 2021
"""

import base64
import os
from datetime import datetime
from io import BytesIO

import pandas as pd
import seaborn as sns
from jinja2 import Template
from matplotlib import patches
from matplotlib import pyplot as plt

sns.set_theme()
sns.set_style("white")
sns.set_style("ticks")

from scipy.stats import mannwhitneyu

from Analysis import *
from tracks import *


def build_property_array(trackobj, prop):
    if prop == "minmax_velo" or prop == "manual_velo":
        arr = []
        l = []
        for tr in trackobj:
            l.append(len(getattr(tr, prop)))
            arr = np.append(arr, getattr(tr, prop))
        # print(np.mean(l), prop)
        return arr
    else:
        raise RuntimeError(f"ERROR: no attribute called {prop}")


def buildhistogram(bins):
    centerbins = []
    for idx, bini in enumerate(bins):
        if bini == bins[-1]:
            continue
        else:
            centerbins.append((bins[idx + 1] + bins[idx]) / 2)
    return centerbins


def html_summary(tracklist, rejects, savepath, manualBool):
    manual_array = 0
    minmax_array = 0

    fig, ax = plt.subplots()

    if manualBool:
        manual_array = build_property_array(tracklist, 'manual_velo')
        n, bins, patches = plt.hist(x=manual_array, bins='auto', density=True, alpha=0.1)
        plt.plot(buildhistogram(bins), n, 'k', linewidth=1, label="Manual Sectioning")

    minmax_array = build_property_array(tracklist, 'minmax_velo')
    n, bins, patches = plt.hist(x=minmax_array, bins='auto', density=True, alpha=0.1)
    plt.plot(buildhistogram(bins), n, 'b', linewidth=1, label="MinMax Sectioning")

    plt.xlabel('Velocity (nm/s)')
    plt.ylabel('PDF')
    plt.legend()
    plt.tight_layout()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    # Other important parameters
    if manualBool:
        meanman = [np.mean(i.manual_velo) for i in tracklist]
    meanminmax = [np.mean(i.minmax_velo) for i in tracklist]
    tracklength = [len(i.x) for i in tracklist]
    diameter = [i.ellipse['major'] * 1000 for i in tracklist]
    angle = [np.rad2deg(np.arccos(i.ellipse['minor'] / i.ellipse['major'])) for i in tracklist]
    average_track_length = np.mean(tracklength)
    average_total_2d_disp = np.mean(
        [np.sqrt((i.x[-1] - i.x[0]) ** 2 + (i.y[-1] - i.y[0]) ** 2) * 1000 for i in tracklist])
    average_speed_2d = np.mean([i.twodspeed for i in tracklist]) * 1000

    fig, ax = plt.subplots()
    if manualBool:
        plt.scatter(tracklength, meanman, c='k', label="Manual")
    plt.scatter(tracklength, meanminmax, c='b', label="MinMax")
    plt.xlabel("Track Length")
    plt.ylabel("Average Velocity per track (nm/s)")
    plt.legend()
    plt.tight_layout()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    enconded_tracklength = base64.b64encode((tmpfile.getvalue())).decode('utf8')

    fig, ax = plt.subplots()
    if manualBool:
        plt.scatter(diameter, meanman, c='k', label="Manual")
    plt.scatter(diameter, meanminmax, c='b', label="MinMax")
    plt.xlabel("Major axis of ellipse (nm)")
    plt.ylabel("Average Velocity per track (nm/s)")
    plt.legend()
    plt.tight_layout()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    enconded_diameter = base64.b64encode((tmpfile.getvalue())).decode('utf8')

    fig, ax = plt.subplots()
    if manualBool:
        plt.scatter(angle, meanman, c='k', label="Manual")
    plt.scatter(angle, meanminmax, c='b', label="MinMax")
    plt.xlabel("Angle to the microscopy plane (deg)")
    plt.ylabel("Average Velocity per track (nm/s)")
    plt.legend()
    plt.tight_layout()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    enconded_angle = base64.b64encode((tmpfile.getvalue())).decode('utf8')

    report_dict = {
        "minmax_vel": np.mean(minmax_array),
        "minmax_std": np.std(minmax_array),
        "minmax_med": np.median(minmax_array),
        "minmax_n": len(minmax_array) if isinstance(minmax_array, (list, tuple, np.ndarray)) else 0,
        "manual_vel": np.mean(manual_array),
        "manual_std": np.std(manual_array),
        "manual_med": np.median(manual_array),
        "manual_n": len(manual_array) if isinstance(manual_array, (list, tuple, np.ndarray)) else 0,
        "enconded_hist": encoded,
        "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "enconded_diameter": enconded_diameter,
        "enconded_tracklength": enconded_tracklength,
        "enconded_angle": enconded_angle,
        "number_of_tracks": f'{len(tracklist)} ({len(tracklist) / (len(tracklist) + len(rejects)):0.2f}%)',
        "average_track_length": average_track_length,
        "average_total_2d_disp": average_total_2d_disp,
        "average_speed_2d": average_speed_2d}

    with open(r"templates/Summary_Template.html", 'r') as f:
        template = Template(f.read())

    with open(os.path.join(savepath, "Summary.html"), 'w+') as f:
        f.write(template.render(report_dict))


def html_comparison(listoffiles, savepath):
    filenames = [i[0].designator[:-2] for i in listoffiles]

    manual = [build_property_array(file, 'manual_velo') for file in listoffiles]
    minmax = [build_property_array(file, 'minmax_velo') for file in listoffiles]

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

    report_dict = {'number_of_files': len(listoffiles),
                   'files': filenames,
                   'mannManual': np.nan if not len(manual[0]) or not len(manual[1]) else mannwhitneyu(manual[0],
                                                                                                      manual[1]).pvalue,
                   'mannMinmax': mannwhitneyu(minmax[0], minmax[1]).pvalue,
                   'enc_manual': enc_manual,
                   'enc_minmax': enc_minmax,
                   'date': datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

    with open(r"templates/Comparison_Template.html", 'r') as f:
        template = Template(f.read())

    with open(os.path.join(savepath, "Summary.html"), 'w+') as f:
        f.write(template.render(report_dict))


def npy_builder(tracklist, rejects, savepath):
    np.save(f"{savepath}\\DataDump.npy", tracklist)
    np.save(f"{savepath}\\RejectedTracks.npy", rejects)
    return


def makeimage(tracklist, savepath, MANUALbool):
    """Image of each track"""

    for tr in tracklist:

        fig = plt.figure(figsize=(16, 9))

        ax1 = fig.add_subplot(2, 3, (1,2))

        if tr.imageobject:
            ax1.imshow(tr.imageobject, cmap='gray')
        ax1.plot(tr.x / 0.08, tr.y / 0.08, color='b', label="Track")

        ax1.plot(tr.xellipse / 0.08, tr.yellipse / 0.08, color='r', label="Ellipse Points")

        eli = patches.Ellipse((tr.ellipse['x0'] / 0.08, tr.ellipse['y0'] / 0.08), tr.ellipse['major'] / 0.08,
                              tr.ellipse['minor'] / 0.08, tr.ellipse['angle'], fill=False,
                              edgecolor='black', alpha=0.3)
        ax1.add_patch(eli)
        ax1.set_xlabel("x coordinates (px)")
        ax1.set_ylabel("y coordinates (px)")
        ax1.set_xlim((np.average(tr.x / 0.08) - 20, np.average(tr.x / 0.08) + 20))
        ax1.set_ylim((np.average(tr.y / 0.08) - 20, np.average(tr.y / 0.08) + 20))
        ax1.set_aspect('equal')
        ax1.legend()

        ax2 = fig.add_subplot(2, 3, 3)
        xaxis = np.linspace(1, len(tr.unwrapped) * tr.samplerate, len(tr.unwrapped))
        ax2.plot(xaxis, (tr.unwrapped - tr.unwrapped[0]) * 1000, label="Original")
        sm = smoothing(tr.unwrapped, int((len(tr.unwrapped) * 20) // 100))
        smoothedxaxis = np.linspace(1, len(sm) * tr.samplerate, len(sm))
        smoothedxaxis += tr.samplerate * (len(xaxis) - len(smoothedxaxis)) / 2
        ax2.plot(smoothedxaxis, (sm - sm[0]) * 1000, label="Smoothed")
        delimeters = findallpeaks(sm)
        if delimeters.size:
            ax2.vlines(x=smoothedxaxis[delimeters], ymin=0, ymax=(sm[delimeters] - sm[0]) * 1000, colors='r', alpha=1)
        ax2.set_xlabel("Time (sec)")
        ax2.set_ylabel("Unwrapped trajectory (nm)")
        ax2.legend()

        if MANUALbool:
            ax4 = fig.add_subplot(2, 3, 4)
            manual_array = build_property_array(tracklist, 'manual_velo')
            n, bins, pat = ax4.hist(x=manual_array, bins='auto', density=True, alpha=0.2)
            ax4.plot(buildhistogram(bins), n, 'k', linewidth=1, label="Manual Sectioning")
            ax4.vlines(x=tr.manual_velo, colors='k', ymin=0, ymax=np.max(n), alpha=0.4)
            ax4.set_xlabel('Velocity (nm/s)')
            ax4.set_ylabel('PDF')
            ax4.set_xlim((0, 30))
            ax4.legend()

        ax5 = fig.add_subplot(2, 3, 5)
        minmax_array = build_property_array(tracklist, 'minmax_velo')
        n, bins, pat = ax5.hist(x=minmax_array, bins='auto', density=True, alpha=0.2)
        ax5.plot(buildhistogram(bins), n, 'b', linewidth=1, label="MinMax Sectioning")
        ax5.vlines(x=tr.minmax_velo, colors='b', ymin=0, ymax=np.max(n), alpha=0.4)
        ax5.set_xlabel('Velocity (nm/s)')
        ax5.set_ylabel('PDF')
        ax5.set_xlim((0, 30))
        ax5.legend()

        plt.tight_layout()

        try:
            name = tr.name.split('/')[-2] + '_' + tr.name.split('/')[-1] + '.jpeg'
        except IndexError:
            name = tr.name + '.jpeg'
        sv = os.path.join(savepath, name)
        fig.savefig(sv)
        plt.close(fig)
