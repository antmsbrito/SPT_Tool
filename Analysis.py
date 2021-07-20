from tracks import Track
from scipy.signal import find_peaks
from scipy.stats import linregress
import numpy as np


def findallpeaks(y):
    peaks1, _ = find_peaks(y)
    peaks2, _ = find_peaks(y*-1)
    peaks = np.union1d(peaks2, peaks1)

    grady = np.gradient(y)
    peaks3, _ = find_peaks(grady)
    peaks4, _ = find_peaks(grady * -1)
    inflexion = np.union1d(peaks3, peaks4)

    allpeaks = np.union1d(peaks, inflexion)

    return allpeaks

def slope(x, y):
    s, o, r_value, p_value, std_err = linregress(x, y)
    return s, o

def displacement(tracks):
    velo = []
    for tr in tracks:
        xcoord = np.diff(tr.xtrack)
        ycoord = np.diff(tr.ytrack)
        zcoord = np.diff(tr.ztrack)
        displacement = np.sqrt(xcoord ** 2 + ycoord ** 2 + zcoord ** 2)
        velo = np.append(velo, displacement / tr.samplerate)

    return velo

def minmax(tracks):
    section_velocity = []
    for tr in tracks:
        x = tr.timeaxis
        y = tr.smoothedtrajectory
        delimiter = findallpeaks(y)

        if not delimiter.size: # not empty
            m, b = slope(x,y)
            section_velocity.append(m)
        else:
            m, b = slope(x[0:delimiter[0]], y[0:delimiter[0]])
            section_velocity.append(m)
            for idx, d in enumerate(delimiter):
                if d == delimiter[-1]:
                    m, b = slope(x[d:-1], y[d:-1])
                    section_velocity.append(m)
                else:
                    nextd = delimiter[idx + 1]
                    m, b = slope(x[d:nextd], y[d:nextd])
                    section_velocity.append(m)
                    
    section_velocity = np.abs(section_velocity)

    return section_velocity

def finite(tracks):
    velo = []
    for tr in tracks:
        if True:
            x = tr.timeaxis
            y = tr.smoothedtrajectory
            velo = np.append(velo, np.abs(np.gradient(y,x)))
    return velo

