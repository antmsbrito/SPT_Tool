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

from scipy.optimize import minimize 

from tracks import *


def gaussian(v, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((v-mu)/sigma)**2)

def twogaussian(v, p1, mu1, sig1, mu2, sig2):
    return gaussian(v, mu1, sig1)*p1 + gaussian(v, mu2, sig2)*(1-p1)

def lognormal(v, mu, sigma):
    return (1/(v*sigma*np.sqrt(2*np.pi))) * np.exp(-(np.log(v)-mu)**2/(2*sigma**2))

def twolognormal(v, p1, mu1, sig1, mu2, sig2):
    return lognormal(v, mu1, sig1)*p1 + lognormal(v, mu2, sig2)*(1-p1)

def residues(x, *args):
    
    x_data = args[0]
    y_data = args[1]
    func = args[2]
    
    return np.sum((y_data - func(x_data, *x))**2)
    

def build_bin_centers(bins):
    centerbins = []
    for idx, bini in enumerate(bins):
        if bini == bins[-1]:
            continue
        else:
            centerbins.append((bins[idx + 1] + bins[idx]) / 2)
    return np.array(centerbins)


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
    
    plt.close('all')
    
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

    
    ##############################################################################################################################################
    fig = plt.figure("Velocity Distributions", figsize=(10,5))
    gs = GridSpec(1,2, figure=fig, width_ratios=(1,1))
    
    ax1 = fig.add_subplot(gs[0,0])
    ax1.hist(displacement_velo, label="Displacement", alpha=0.5, density=True, bins='auto')
    pdf, bins, _ = ax1.hist(np.hstack(brute_velo), label="Brute force", alpha=0.5, density=True, bins='auto')
    ax1.hist(np.hstack(mug_velo), label="Muggeo et al", alpha=0.5, density=True, bins='auto')
    ax1.set_ylabel("Probability Density Function")
    ax1.set_xlabel("Velocity (nm/s)")
    ax1.legend()
    
    ax2 = fig.add_subplot(gs[0,1])
    ax2.hist(displacement_velo, label="Displacement", alpha=0.5, density=True, bins='auto')
    ax2.hist(np.hstack(brute_velo), label="Brute force", alpha=0.5, density=True, bins='auto')
    ax2.hist(np.hstack(mug_velo), label="Muggeo et al", alpha=0.5, density=True, bins='auto')
    ax2.set_ylabel("Probability Density Function")
    ax2.set_xlabel("Velocity (nm/s)")
    ax2.set_xscale('log')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    ##############################################################################################################################################
    
    # FITTING ON BRUTE_FORCE
    cbins = build_bin_centers(bins)
    
    # ONE GAUSSIAN FITTING
    one_g = minimize(fun=residues, x0=[10,4], args=(cbins, pdf, gaussian), bounds=((0, np.inf), (0, np.inf)))
    # TWO GAUSSIAN FITTING
    two_g = minimize(fun=residues, x0=[0.5,10,4,2,2], args=(cbins, pdf, twogaussian), bounds=((0, 1),(0, np.inf),(0, np.inf),(0, np.inf),(0, np.inf)))
    # ONE LOG-NORMAL FITTING
    one_lg = minimize(fun=residues, x0=[1,0.5], args=(cbins, pdf, lognormal), bounds=((-np.inf, np.inf), (1e-5, np.inf)))
    # TWO LOG-NORMAL FITTING
    two_lg = minimize(fun=residues, x0=[0.5,1,0.5,2,0.5], args=(cbins, pdf, twolognormal), bounds=((0, 1),(-np.inf, np.inf),(1e-5, np.inf),(-np.inf, np.inf),(1e-5, np.inf)))
    
    fig = plt.figure("Fitted Velocity Distributions", figsize=(15,10))
    gs = GridSpec(2,2, figure=fig, width_ratios=(1,1), height_ratios=(1,0.2))
    
    fig.suptitle(f"Angle={angle_threshold}º, Diameter={major_threshold} nm ")
    
    ax1 = fig.add_subplot(gs[0,0])
    ax1.hist(np.hstack(brute_velo), label="Brute force", alpha=0.1, density=True, bins='auto', color='k')
    ax1.plot(cbins, pdf, '--k')
    ax1.plot(cbins, gaussian(cbins,*one_g.x), label='Unimodal Normal')
    ax1.plot(cbins, twogaussian(cbins, *two_g.x), label='Bimodal Normal')
    ax1.set_ylabel("Probability Density Function")
    ax1.set_xlabel("Velocity (nm/s)")
    ax1.set_title("Gaussians")
    ax1.legend()
    
    ax2 = fig.add_subplot(gs[0,1])
    ax2.hist(np.hstack(brute_velo), label="Brute force", alpha=0.1, density=True, bins='auto', color='k')
    ax2.plot(cbins, pdf, '--k')
    ax2.plot(cbins, lognormal(cbins,*one_lg.x), label='Unimodal LogNormal')
    ax2.plot(cbins, twolognormal(cbins, *two_lg.x), label='Bimodal LogNormal')
    ax2.set_ylabel("Probability Density Function")
    ax2.set_xlabel("Velocity (nm/s)")
    ax2.set_title("Log-Normals")
    ax2.legend()
    
    ax3 = fig.add_subplot(gs[1,0])
    onegstr = f"One Gaussian: $\mu$={one_g.x[0]:.2f}, $\sigma=${one_g.x[1]:.2f}, SSE={residues(one_g.x,cbins,pdf,gaussian):.2e}"
    twogstr = f"Two Gaussian: {100*two_g.x[0]:.2f}% $\mu_1$={two_g.x[1]:.2f}, $\sigma_1$={two_g.x[2]:.2f} \n   {100-100*two_g.x[0]:.2f}% $\mu_2$={two_g.x[3]:.2f}, $\sigma_2$={two_g.x[4]:.2f} \n  SSE={residues(two_g.x,cbins,pdf,twogaussian):.2e}"
    ax3.text(0.5,0.7, onegstr, size=12, ha="center", va="center", bbox=dict(boxstyle="round",ec='k',fc='tab:grey'), wrap=False)
    ax3.text(0.5,0.3, twogstr, size=12, ha="center", va="center", bbox=dict(boxstyle="round",ec='k',fc='tab:grey'), wrap=False)
    ax3.axis('off')
    
    ax4=fig.add_subplot(gs[1,1])
    onelgstr = f"One LogNormal: $\mu$={one_lg.x[0]:.2f}, $\sigma=${one_lg.x[1]:.2f}, SSE={residues(one_lg.x,cbins,pdf,lognormal):.2e}"
    twolgstr = f"Two LogNormal: {100*two_lg.x[0]:.2f}% $\mu_1$={two_lg.x[1]:.2f}, $\sigma_1$={two_lg.x[2]:.2f} \n    {100-100*two_lg.x[0]:.2f}% $\mu_2$={two_lg.x[3]:.2f}, $\sigma_2$={two_lg.x[4]:.2f} \n  SSE={residues(two_lg.x,cbins,pdf,twolognormal):.2e}"
    ax4.text(0.5,0.7, onelgstr, size=12, ha="center", va="center", bbox=dict(boxstyle="round",ec='k',fc='tab:grey'), wrap=False)
    ax4.text(0.5,0.3, twolgstr, size=12, ha="center", va="center", bbox=dict(boxstyle="round",ec='k',fc='tab:grey'), wrap=False)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"One Gaussian: mu={one_g.x[0]:.2f}, sigma={one_g.x[1]:.2f}, SSE={residues(one_g.x,cbins,pdf,gaussian):.2e}")
    print(f"Two Gaussian: {100*two_g.x[0]:.2f}% mu={two_g.x[1]:.2f}, sigma={two_g.x[2]:.2f} \n",
     f"             {100-100*two_g.x[0]:.2f}% mu={two_g.x[3]:.2f}, sigma={two_g.x[4]:.2f} \n", 
     f"             SSE={residues(two_g.x,cbins,pdf,twogaussian):.2e}")
    
    print(f"One LogNormal: mu={one_lg.x[0]:.2f}, sigma={one_lg.x[1]:.2f}, SSE={residues(one_lg.x,cbins,pdf,lognormal):.2e}")
    print(f"Two LogNormal: {100*two_lg.x[0]:.2f}% mu={two_lg.x[1]:.2f}, sigma={two_lg.x[2]:.2f} \n",
     f"              {100-100*two_lg.x[0]:.2f}% mu={two_lg.x[3]:.2f}, sigma={two_lg.x[4]:.2f} \n",
     f"              SSE={residues(two_lg.x,cbins,pdf,twolognormal):.2e}")
    
    
    
    ##############################################################################################################################################
    fig2 = plt.figure("Effect of angle", figsize=(10,10))
    gs = GridSpec(2,2, figure=fig2, width_ratios=(1,1), height_ratios=(1,1))
    
    ax1 = fig2.add_subplot(gs[0,0])
    ax1.scatter(x=angles, y=displacement_velo, label="Displacement")
    ax1.scatter(x=angles, y=[np.average(i) for i in brute_velo], label="Brute force", c='k', marker='*')
    ax1.scatter(x=angles, y=[np.average(i) for i in mug_velo], label="Muggeo et al", c='r', marker='+')
    ax1.set_ylabel("Average Velocity (nm/s)")
    ax1.set_xlabel("Angle (deg)")
    ax1.legend()
    
    ax2 = fig2.add_subplot(gs[0,1])
    ax2.scatter(x=angles, y=[np.std(i) for i in brute_velo], label="Brute force", c='k', marker='*')
    ax2.scatter(x=angles, y=[np.std(i) for i in mug_velo], label="Muggeo et al", c='r', marker='+')
    ax2.set_ylabel("Standard deviation (nm/s)")
    ax2.set_xlabel("Angle (deg)")
    ax2.legend()
    
    ax3 = fig2.add_subplot(gs[1,0])
    ax3.scatter(x=angles, y=brute_sections, label="Brute force", c='k', marker='*')
    ax3.scatter(x=angles, y=mug_sections, label="Muggeo et al", c='r', marker='+')
    ax3.set_ylabel("Number of sections")
    ax3.set_xlabel("Angle (deg)")
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    ##############################################################################################################################################
    
    ##############################################################################################################################################
    fig3 = plt.figure("Effect of diameter", figsize=(10,10))
    gs = GridSpec(2,2, figure=fig3, width_ratios=(1,1), height_ratios=(1,1))
    
    ax1 = fig3.add_subplot(gs[0,0])
    ax1.scatter(x=diameter, y=displacement_velo, label="Displacement")
    ax1.scatter(x=diameter, y=[np.average(i) for i in brute_velo], label="Brute force", c='k', marker='*')
    ax1.scatter(x=diameter, y=[np.average(i) for i in mug_velo], label="Muggeo et al", c='r', marker='+')
    ax1.set_ylabel("Average Velocity (nm/s)")
    ax1.set_xlabel("Diameter (nm)")
    ax1.legend()
    
    ax2 = fig3.add_subplot(gs[0,1])
    ax2.scatter(x=diameter, y=[np.std(i) for i in brute_velo], label="Brute force", c='k', marker='*')
    ax2.scatter(x=diameter, y=[np.std(i) for i in mug_velo], label="Muggeo et al", c='r', marker='+')
    ax2.set_ylabel("Standard deviation (nm/s)")
    ax2.set_xlabel("Diameter (nm)")
    ax2.legend()
    
    ax3 = fig3.add_subplot(gs[1,0])
    ax3.scatter(x=diameter, y=brute_sections, label="Brute force", c='k', marker='*')
    ax3.scatter(x=diameter, y=mug_sections, label="Muggeo et al", c='r', marker='+')
    ax3.set_ylabel("Number of sections")
    ax3.set_xlabel("Diameter (nm)")
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    ##############################################################################################################################################

    ##############################################################################################################################################
    fig4 = plt.figure("Effect of track length", figsize=(10,10))
    gs = GridSpec(2,2, figure=fig4, width_ratios=(1,1), height_ratios=(1,1))
    
    ax1 = fig4.add_subplot(gs[0,0])
    ax1.scatter(x=[len(tr.unwrapped) for tr in filtered_tracks], y=displacement_velo, label="Displacement")
    ax1.scatter(x=[len(tr.unwrapped) for tr in filtered_tracks], y=[np.average(i) for i in brute_velo], label="Brute force", c='k', marker='*')
    ax1.scatter(x=[len(tr.unwrapped) for tr in filtered_tracks], y=[np.average(i) for i in mug_velo], label="Muggeo et al", c='r', marker='+')
    ax1.set_xlabel("Length of track (#)")
    ax1.set_ylabel("Velocity (nm/s)")
    ax1.legend()
    
    ax2 = fig4.add_subplot(gs[0,1])
    ax2.scatter(x=[len(tr.unwrapped) for tr in filtered_tracks], y=[np.std(i) for i in brute_velo], label="Brute force", c='k', marker='*')
    ax2.scatter(x=[len(tr.unwrapped) for tr in filtered_tracks], y=[np.std(i) for i in mug_velo], label="Muggeo et al", c='r', marker='+')
    ax2.set_xlabel("Length of track (#)")
    ax2.set_ylabel("Standard deviation (nm/s)")
    ax2.legend()
    
    ax3 = fig4.add_subplot(gs[1,0])
    ax3.scatter(x=[len(tr.unwrapped) for tr in filtered_tracks], y=brute_sections, label="Brute force", c='k', marker='*')
    ax3.scatter(x=[len(tr.unwrapped) for tr in filtered_tracks], y=mug_sections, label="Muggeo et al", c='r', marker='+')
    ax3.set_xlabel("Length of track (#)")
    ax3.set_ylabel("Number of sections")
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    ##############################################################################################################################################
    

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