import os
from tkinter import Tk
from tkinter import filedialog

from ipywidgets import Dropdown, interactive, fixed, Accordion, IntSlider
from IPython.display import display
from matplotlib import pyplot as plt

from tracks import *


def FileLoader(folder, root):
    
    display(f"Loading {folder}...")
    
    folder = os.path.join(root, folder)
    filedump = os.path.join(folder, "DataDump.npy")
    Tracks = readfile(filedump)
    
    display(f"Loaded a total of {len(Tracks)} tracks")
    
    _ = showStats(Tracks)

    return Tracks
    
    
def readfile(f):
    if f[:-4] == ".npy":
        return 0
    old_objs = np.load(f, allow_pickle=True)

    newobjs = []
    for idx, t in enumerate(old_objs):
        newobjs.append(TrackV2(t.imageobject, t.x, t.y, t.samplerate, t.name, t.ellipse))
        if hasattr(t, 'bruteforce_velo'):
            newobjs[-1].bruteforce_velo = t.bruteforce_velo
            newobjs[-1].bruteforce_phi = t.bruteforce_phi

        if hasattr(t, 'muggeo_velo'):
            newobjs[-1].muggeo_velo = t.muggeo_velo
            newobjs[-1].muggeo_phi = t.muggeo_phi
            newobjs[-1].muggeo_params = t.muggeo_params

        if hasattr(t, 'manual_velo'):
            newobjs[-1].manual_velo = t.manual_velo
            newobjs[-1].manual_sections = t.manual_sections
    
    return newobjs 

def showStats(tr):
    angles = np.rad2deg(np.arccos([i.ellipse['minor']/i.ellipse['major'] for i in tr]))

    plt.figure()
    plt.hist(angles, density=False)
    plt.xlabel("Angle in degrees")
    plt.ylabel("Frequency")
    plt.show()
    
    return 0 

def Update_Graphs(angle_threshold, all_tracks, angles):
    print(
        f"Less than {angle_threshold} deg means that minor axis <= {max(0, np.cos(np.deg2rad(angle_threshold))) * 100:.0f}% of major ")
    
    lowangle_tracks = all_tracks[angles <= angle_threshold]
    
    print(
        f"n = {len(lowangle_tracks)} (out of {len(all_tracks)} ({(len(lowangle_tracks) / len(all_tracks)) * 100:.1f}%))\n\n")

    displacement_velo = [np.average(tr.disp_velo) for tr in lowangle_tracks] 
    brute_velo = [tr.bruteforce_velo for tr in lowangle_tracks]
    brute_sections = [len(tr.bruteforce_phi) for tr in lowangle_tracks]
    mug_velo = [tr.muggeo_velo for tr in lowangle_tracks]
    mug_sections = [len(tr.muggeo_phi) for tr in lowangle_tracks]
    
    
    print(f"Displacement: {np.average(displacement_velo):.2f} +- {np.std(displacement_velo):.2f} nm/s \n")
    print(f"Sectioned: {np.average(np.hstack(brute_velo)):.2f} +- {np.std(np.hstack(brute_velo)):.2f} nm/s \n")
    print(f"Average number of sections of {np.average(brute_sections):.2f}")
    print(f"Sectioned: {np.average(np.hstack(mug_velo)):.2f} +- {np.std(np.hstack(mug_velo)):.2f} nm/s \n")
    print(f"Average number of sections of {np.average(mug_sections):.2f}")

    lowangles = np.rad2deg(np.arccos([i.ellipse['minor'] / i.ellipse['major'] for i in lowangle_tracks]))
    
    plt.figure()
    plt.hist(displacement_velo, label="Displacement", alpha=0.5, density=True)
    plt.hist(np.hstack(brute_velo), label="Brute force", alpha=0.5, density=True)
    plt.hist(np.hstack(mug_velo), label="Muggeo et al", alpha=0.5, density=True)
    plt.ylabel("PDF")
    plt.xlabel("Velocity (nm/s)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.subplots()
    plt.scatter(x=lowangles, y=displacement_velo, label="Displacement")
    plt.scatter(x=lowangles, y=[np.average(i) for i in brute_velo], label="Brute force", c='k', marker='*')
    plt.scatter(x=lowangles, y=[np.average(i) for i in mug_velo], label="Muggeo et al", c='r', marker='+')
    plt.ylabel("Average Velocity (nm/s)")
    plt.xlabel("Angle (deg)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.scatter(x=lowangles, y=[np.std(i) for i in brute_velo], label="Brute force", c='k', marker='*')
    plt.scatter(x=lowangles, y=[np.std(i) for i in mug_velo], label="Muggeo et al", c='r', marker='+')
    plt.ylabel("Standard deviation (nm/s)")
    plt.xlabel("Angle (deg)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    major_axis = [tr.ellipse['major']*1000 for tr in lowangle_tracks]
    plt.scatter(x=major_axis, y=displacement_velo, label="Displacement")
    plt.scatter(x=major_axis, y=[np.average(i) for i in brute_velo], label="Brute force", c='k', marker='*')
    plt.scatter(x=major_axis, y=[np.average(i) for i in mug_velo], label="Muggeo et al", c='r', marker='+')
    plt.ylabel("Average Velocity (nm/s)")
    plt.xlabel("Length of major axis (nm))")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(x=lowangles, y=brute_sections, label="Brute force", c='k', marker='*')
    plt.scatter(x=lowangles, y=mug_sections, label="Muggeo et al", c='r', marker='+')
    plt.ylabel("Number of sections")
    plt.xlabel("Angle (deg)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(x=[len(tr.unwrapped) for tr in lowangle_tracks], y=displacement_velo, label="Displacement")
    plt.scatter(x=[len(tr.unwrapped) for tr in lowangle_tracks], y=[np.average(i) for i in brute_velo], label="Brute force", c='k', marker='*')
    plt.scatter(x=[len(tr.unwrapped) for tr in lowangle_tracks], y=[np.average(i) for i in mug_velo], label="Muggeo et al", c='r', marker='+')
    plt.xlabel("Length of track")
    plt.ylabel("Velocity (nm/s)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    
