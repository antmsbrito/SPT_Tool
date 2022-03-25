"""
SPT_TOOL
@author António Brito
ITQB-UNL BCB 2021
"""

import os
import h5py
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

sns.set_theme()
sns.set_style("white")
sns.set_style("ticks")

from tracks import *

def buildhistogram(bins):
    centerbins = []
    for idx, bini in enumerate(bins):
        if bini == bins[-1]:
            continue
        else:
            centerbins.append((bins[idx + 1] + bins[idx]) / 2)
    return centerbins


def npy_builder(tracklist, rejects, savepath):
    np.save(f"{savepath}\\DataDump.npy", tracklist)
    np.save(f"{savepath}\\RejectedTracks.npy", rejects)
    return


def hd5_dump(tracklist, rejects, savepath):
    hf = h5py.File(f'{savepath}\\DataDump.h5', 'w')

    track_group = hf.create_group('tracks')
    rejects_group = hf.create_group('rejects')

    dt = h5py.special_dtype(vlen=str)

    for idx, tr in enumerate(tracklist):
        track_subfolder = hf.create_group(f'tracks/{tr.name}_{idx}')

        track_subfolder.create_dataset('x', data=tr.x)  # np array
        track_subfolder.create_dataset('y', data=tr.y)  # np array
        track_subfolder.create_dataset('samplerate', data=np.array([tr.samplerate]))  # float
        track_subfolder.create_dataset('image', data=np.array(tr.imageobject))  # np array
        track_subfolder.create_dataset('name', data=tr.name, dtype=dt)  # string
        track_subfolder.create_dataset('ellipse', data=json.dumps(tr.ellipse), dtype=dt)  # string
        track_subfolder.create_dataset('manual_sections', data=np.array(tr.manual_sections))  # np array
        track_subfolder.create_dataset('minmax_sections', data=np.array(tr.minmax_sections))  # np array
        track_subfolder.create_dataset('minmax_velo', data=np.array(tr.minmax_velo))  # np array

    for idx, tr in enumerate(rejects):
        rejects_subfolder = hf.create_group(f'rejects/{tr.name}_{idx}')
        rejects_subfolder.create_dataset('x', data=tr.x)  # np array
        rejects_subfolder.create_dataset('y', data=tr.y)  # np array
        rejects_subfolder.create_dataset('samplerate', data=np.array([tr.samplerate]))  # float
        rejects_subfolder.create_dataset('image', data=np.array(tr.imageobject))  # np array
        rejects_subfolder.create_dataset('name', data=tr.name, dtype=dt)
        rejects_subfolder.create_dataset('ellipse', data=json.dumps(tr.ellipse), dtype=dt)

    hf.close()

    return


def makeimage(tracklist, savepath, MANUALbool):
    """Image of each track"""
    matplotlib.use('Agg')
    for tr in tracklist:

        fig = plt.figure(figsize=(16, 9))

        ax1 = fig.add_subplot(2, 3, 1)
        if tr.imageobject:
            ax1.imshow(tr.imageobject, cmap='gray')
        ax1.plot(tr.x / 0.08, tr.y / 0.08, color='b', label="Track")

        xeli, yeli = tr.xellipse / 0.08, tr.yellipse / 0.08
        # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
        cumulative_disp = np.cumsum(np.sqrt(np.diff(xeli * 0.08 * 1000) ** 2 + np.diff(yeli * 0.08 * 1000) ** 2))
        points = np.array([xeli, yeli]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(cumulative_disp.min(), cumulative_disp.max())
        lc = LineCollection(segments, cmap='rainbow', norm=norm)
        lc.set_array(cumulative_disp)
        lc.set_linewidth(2)
        line = ax1.add_collection(lc)
        # fig.colorbar(line, ax=ax1, label="Total distance traveled (nm)")
        # ax1.plot(tr.xellipse / 0.08, tr.yellipse / 0.08, color='r', label="Ellipse Points")

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

        ax2 = fig.add_subplot(2, 3, 2)
        xaxis = np.array(range(len(tr.unwrapped))) * tr.samplerate
        ax2.plot(xaxis, (tr.unwrapped - tr.unwrapped[0]) * 1000, label="Raw data")
        ax2.vlines(x=xaxis[tr.minmax_sections], ymin=0, ymax=(tr.unwrapped[tr.minmax_sections] - tr.unwrapped[0]) * 1000, colors='r')
        ax2.set_xlabel("Time (sec)")
        ax2.set_ylabel("Unwrapped trajectory (nm)")
        ax2.legend()

        ax3 = fig.add_subplot(2, 3, 3)
        xaxis = np.linspace(1, len(tr.unwrapped) * tr.samplerate, len(tr.unwrapped))
        yaxis = np.linalg.norm(tr.xypairs - tr.xy_ellipse, axis=1) * 1000
        ax3.plot(xaxis, yaxis)
        average_dist = np.mean(yaxis)
        ax3.axhline(y=average_dist, label="Average")
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Distance to the ellipse (nm)')
        ax3.legend()

        if MANUALbool:
            ax4 = fig.add_subplot(2, 3, 4)
            manual_array = np.hstack([tr.manual_velo for tr in tracklist])
            n, bins, pat = ax4.hist(x=manual_array, bins='auto', density=True, alpha=0.2)
            ax4.plot(buildhistogram(bins), n, 'k', linewidth=1, label="Manual Sectioning")
            ax4.vlines(x=tr.manual_velo, colors='k', ymin=0, ymax=np.max(n), alpha=0.4)
            ax4.set_xlabel('Velocity (nm/s)')
            ax4.set_ylabel('PDF')
            ax4.set_xlim((0, 30))
            ax4.legend()
        else:
            ax4 = fig.add_subplot(2, 3, 4)


        ax5 = fig.add_subplot(2, 3, 5)
        minmax_array = np.hstack([tr.minmax_velo for tr in tracklist])
        n, bins, pat = ax5.hist(x=minmax_array, bins='auto', density=True, alpha=0.2)
        ax5.plot(buildhistogram(bins), n, 'b', linewidth=1, label="MinMax Sectioning")
        ax5.vlines(x=tr.minmax_velo, colors='b', ymin=0, ymax=np.max(n), alpha=0.4)
        ax5.set_xlabel('Velocity (nm/s)')
        ax5.set_ylabel('PDF')
        ax5.set_xlim((0, 30))
        ax5.legend()

        ax6 = fig.add_subplot(2, 3, 6)
        avgminmax = np.mean(tr.minmax_velo)
        avgmanual = np.mean(tr.manual_velo) if not np.array(tr.manual_velo).size == 0 else 0.0
        rawtxt = '\n'.join((f'MinMax average = {avgminmax:.2f} nm/s',
                            f'Manual average = {avgmanual:.2f} nm/s',
                            f'Average distance to ellipse {average_dist:.2f} nm',
                            f'Total distance traveled {cumulative_disp[-1]:.2f} nm',
                            f'Total displacement {np.sqrt((xeli[-1] - xeli[0]) ** 2 + (yeli[-1] - yeli[0]) ** 2) * 0.08 * 1000:.2f} nm'))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax6.text(0, 0, rawtxt, fontsize=20, bbox=props)
        ax6.set_ylim((0, 0.4))
        ax6.axis('off')

        plt.tight_layout()

        try:
            name = tr.name.split(os.sep)[-2] + '_' + tr.name.split(os.sep)[-1] + '.jpeg'
        except IndexError:
            name = tr.name + '.jpeg'
        sv = os.path.join(savepath, name)
        fig.savefig(sv)
        plt.close(fig)
        plt.close('all')


def csv_dump(tracklist, rejects, savepath, manualBool):
    return 0
