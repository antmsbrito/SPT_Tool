import os
from functools import cache, partial
from tkinter import Tk
from tkinter import filedialog

import pandas as pd
from ipywidgets import Dropdown, interactive, fixed, Accordion, IntSlider, SelectMultiple, BoundedIntText, Button, Output, IntRangeSlider
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from tracks import *


def FileLoader(folder, root):
    
    display(f"Loading {folder}...")
    
    folder = os.path.join(root, folder)
    filedump = os.path.join(folder, "DataDump.npy")
    Tracks = readfile(filedump)
    
    display(f"Loaded a total of {len(Tracks)} tracks")

    return Tracks
    
@cache
def readfile(f):
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
            newobjs[-1].manual_phi = t.manual_phi
    
    return np.array(newobjs)


def loadall(rootdirectory, output, ev):
    with output:
        for f in os.listdir(rootdirectory):
            FileLoader(f, rootdirectory)


def showStats(tr):
    angles = np.rad2deg(np.arccos([i.ellipse['minor']/i.ellipse['major'] for i in tr]))

    plt.figure()
    plt.hist(angles, density=False)
    plt.xlabel("Angle in degrees")
    plt.ylabel("Frequency")
    plt.xlim((0,90))
    plt.show()
    
    return 0 

def Update_Graphs(angle_threshold, major_threshold, all_tracks):
    
    all_angles = np.rad2deg(np.arccos([i.ellipse['minor'] / i.ellipse['major'] for i in all_tracks]))
    all_diameter = np.array([i.ellipse['major']*1000 for i in all_tracks])
    
    filtered_tracks = []
    for idx,tr in enumerate(all_tracks):
        if (all_angles[idx]<angle_threshold[1] and all_angles[idx]>angle_threshold[0]) and (all_diameter[idx]<major_threshold[1] and all_diameter[idx]>major_threshold[0]):
            filtered_tracks.append(tr)
    filtered_tracks = np.array(filtered_tracks)
    
    print(
        f"n = {len(filtered_tracks)} (out of {len(all_tracks)} ({(len(filtered_tracks) / len(all_tracks)) * 100:.1f}%))\n\n")

    angles = np.rad2deg(np.arccos([i.ellipse['minor'] / i.ellipse['major'] for i in filtered_tracks]))
    diameter = np.array([i.ellipse['major']*1000 for i in filtered_tracks])
    
    displacement_velo = [np.average(tr.disp_velo) for tr in filtered_tracks]
    brute_velo = [tr.bruteforce_velo for tr in filtered_tracks]
    brute_sections = [len(tr.bruteforce_phi) for tr in filtered_tracks]
    mug_velo = [tr.muggeo_velo for tr in filtered_tracks]
    mug_sections = [len(tr.muggeo_phi) for tr in filtered_tracks]
    
    print(f"Displacement: {np.nanmean(displacement_velo):.2f} +- {np.nanstd(displacement_velo):.2f} nm/s \n")
    print(f"Brute force sectioning: {np.average(np.hstack(brute_velo)):.2f} +- {np.std(np.hstack(brute_velo)):.2f} nm/s")
    print(f"\t Average number of sections of {np.average(brute_sections):.2f} \n")
    print(f"Muggeo et al sectioning: {np.nanmean(np.hstack(mug_velo)):.2f} +- {np.nanstd(np.hstack(mug_velo)):.2f} nm/s")
    print(f"\t Average number of sections of {np.nanmean(mug_sections):.2f} \n")

    
    fig = plt.figure()
    gs = GridSpec(2,1, figure=fig, heigth_ratios=(1), width_ratios=(1,1))
    
    ax1 = fig.add_subplot(gs[2,0], adjustable='datalim', aspect='equal')
    ax1.hist(displacement_velo, label="Displacement", alpha=0.5, density=True)
    ax1.hist(np.hstack(brute_velo), label="Brute force", alpha=0.5, density=True)
    ax1.hist(np.hstack(mug_velo), label="Muggeo et al", alpha=0.5, density=True)
    ax1.set_ylabel("PDF")
    ax1.set_xlabel("Velocity (nm/s)")
    ax1.legend()
    
    ax2 = fig.add_subplot(gs[1,0], adjustable='datalim', aspect='equal')
    ax2.hist(displacement_velo, label="Displacement", alpha=0.5, density=True)
    ax2.hist(np.hstack(brute_velo), label="Brute force", alpha=0.5, density=True)
    ax2.hist(np.hstack(mug_velo), label="Muggeo et al", alpha=0.5, density=True)
    ax2.set_ylabel("PDF")
    ax2.set_xlabel("Velocity (nm/s)")
    ax2.set_xscale('log')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    plt.subplots()
    plt.scatter(x=angles, y=displacement_velo, label="Displacement")
    plt.scatter(x=angles, y=[np.average(i) for i in brute_velo], label="Brute force", c='k', marker='*')
    plt.scatter(x=angles, y=[np.average(i) for i in mug_velo], label="Muggeo et al", c='r', marker='+')
    plt.ylabel("Average Velocity (nm/s)")
    plt.xlabel("Angle (deg)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.scatter(x=angles, y=[np.std(i) for i in brute_velo], label="Brute force", c='k', marker='*')
    plt.scatter(x=angles, y=[np.std(i) for i in mug_velo], label="Muggeo et al", c='r', marker='+')
    plt.ylabel("Standard deviation (nm/s)")
    plt.xlabel("Angle (deg)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.scatter(x=angles, y=brute_sections, label="Brute force", c='k', marker='*')
    plt.scatter(x=angles, y=mug_sections, label="Muggeo et al", c='r', marker='+')
    plt.ylabel("Number of sections")
    plt.xlabel("Angle (deg)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(x=diameter, y=displacement_velo, label="Displacement")
    plt.scatter(x=diameter, y=[np.average(i) for i in brute_velo], label="Brute force", c='k', marker='*')
    plt.scatter(x=diameter, y=[np.average(i) for i in mug_velo], label="Muggeo et al", c='r', marker='+')
    plt.ylabel("Average Velocity (nm/s)")
    plt.xlabel("Diameter (nm))")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(x=diameter, y=brute_sections, label="Brute force", c='k', marker='*')
    plt.scatter(x=diameter, y=mug_sections, label="Muggeo et al", c='r', marker='+')
    plt.ylabel("Number of sections")
    plt.xlabel("Diameter (nm)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(x=[len(tr.unwrapped) for tr in filtered_tracks], y=displacement_velo, label="Displacement")
    plt.scatter(x=[len(tr.unwrapped) for tr in filtered_tracks], y=[np.average(i) for i in brute_velo], label="Brute force", c='k', marker='*')
    plt.scatter(x=[len(tr.unwrapped) for tr in filtered_tracks], y=[np.average(i) for i in mug_velo], label="Muggeo et al", c='r', marker='+')
    plt.xlabel("Length of track (#)")
    plt.ylabel("Velocity (nm/s)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def ViolinComparison(conditions, root, anglethresh):
    
    if len(conditions) == 0:
        print("Select file(s) to load")
        return 
    
    print("Please wait...")
    
    anglethresh = int(anglethresh)
    final_brute = pd.DataFrame({'Condition' : [], 'Velocity (nm/s)':[], 'Protein':[]})
    final_mug = pd.DataFrame({'Condition' : [], 'Velocity (nm/s)':[], 'Protein':[]})
    final_disp = pd.DataFrame({'Condition' : [], 'Velocity (nm/s)':[], 'Protein':[]})

    analyzed = []
    # Iterate through conditions
    for c in conditions:
        
        if c in analyzed:
            print(f"{c} already analyzed")
            continue 
            
        # Initialize velocity dicts
        databrute = {}
        datamug = {}
        datadisp = {}
        
        # Load divIB AND ftsW
        # To facilitate and make code more readable when you load a condition 
        # make it so ftsW is always first 
        if 'ftsW' in c:
            hue = c.replace('ftsW', 'divIB')
        elif 'divIB' in c:
            c, hue = c.replace('divIB', 'ftsW'), c
            
        conditionname = c.replace('_ftsW','')
        
        # readfile function is lru cached
        print(f"Loading {c}")
        ftswc = readfile(os.path.join(os.path.join(root,c), "DataDump.npy"))
        print(f"Loading {hue}")
        divibc = readfile(os.path.join(os.path.join(root,hue), "DataDump.npy"))
        analyzed.append(c)
        analyzed.append(hue)

        print(f"Filtering tracks...")
        anglearr = np.rad2deg(np.arccos([i.ellipse['minor']/i.ellipse['major'] for i in ftswc]))
        filtered_ftsw = ftswc[anglearr <= anglethresh]
        anglearr = np.rad2deg(np.arccos([i.ellipse['minor']/i.ellipse['major'] for i in divibc]))
        filtered_divib = divibc[anglearr <= anglethresh]

        disp_ftsw = [np.average(tr.disp_velo) for tr in filtered_ftsw] 
        brute_ftsw = [tr.bruteforce_velo for tr in filtered_ftsw]
        mug_ftsw = [tr.muggeo_velo for tr in filtered_ftsw]
        
        disp_divib = [np.average(tr.disp_velo) for tr in filtered_divib] 
        brute_divib = [tr.bruteforce_velo for tr in filtered_divib]
        mug_divib = [tr.muggeo_velo for tr in filtered_divib]


        print(f"Saving velocities...")
        databrute['Velocity (nm/s)'] = np.hstack([np.hstack(brute_ftsw), np.hstack(brute_divib)])
        databrute['Condition'] = [conditionname] * len(databrute['Velocity (nm/s)'])
        databrute['Protein'] = ['FtsW'] * len(np.hstack(brute_ftsw)) + ['DivIB'] * len(np.hstack(brute_divib))
        final_brute = pd.concat([final_brute, pd.DataFrame(data=databrute)])

        datamug['Velocity (nm/s)'] = np.hstack([np.hstack(mug_ftsw), np.hstack(mug_divib)])
        datamug['Condition'] = [conditionname] * len(datamug['Velocity (nm/s)'])
        datamug['Protein'] = ['FtsW'] * len(np.hstack(mug_ftsw)) + ['DivIB'] * len(np.hstack(mug_divib))
        final_mug = pd.concat([final_mug, pd.DataFrame(datamug)])

        datadisp['Velocity (nm/s)'] = np.hstack([np.hstack(disp_ftsw), np.hstack(disp_divib)])
        datadisp['Condition'] = [conditionname] * len(datadisp['Velocity (nm/s)'])
        datadisp['Protein'] = ['FtsW'] * len(np.hstack(disp_ftsw)) + ['DivIB'] * len(np.hstack(disp_divib))
        final_disp = pd.concat([final_disp, pd.DataFrame(datadisp)])
        
    print("Done!")
    
    final_all = pd.concat([final_brute, final_mug, final_disp], keys=["Brute Force", "Muggeo et al", "Displacement"])
    final_all['Method'] = final_all.index.get_level_values(0)
    
    
    g = sns.catplot(x="Condition", y="Velocity (nm/s)", hue="Protein", col='Method', data=final_all, kind="violin", split=False, cut=0, scale='count', scale_hue='False', inner='quartile', palette='Set2', col_wrap=2, sharex=False, aspect=1.5)
    plt.ylim((0,30))
    plt.show()
    
    return final_all