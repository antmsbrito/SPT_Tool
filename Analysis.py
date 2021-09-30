
from scipy.signal import find_peaks
from scipy.stats import linregress
import numpy as np


def smoothing(unsmoothed, windowsize):
    # Helper function for displacement method
    # Moving average smoothing
    # Convolves with a array ones
    smoothed = np.convolve(unsmoothed, np.ones(windowsize) / windowsize, mode='valid')
    return smoothed


def findallpeaks(y):
    """
    Helper function for the minmax method

    Calculates all minimums,maximums and inflexion points of an array. Since they are the delimiters for
    slope calculations they ignore points at the start, end and consecutive points.

    """
    peaks1, _ = find_peaks(y)
    peaks2, _ = find_peaks(y * -1)
    peaks = np.union1d(peaks2, peaks1)

    grady = np.gradient(y)
    peaks3, _ = find_peaks(grady)
    peaks4, _ = find_peaks(grady * -1)
    inflexion = np.union1d(peaks3, peaks4)

    allpeaks = np.union1d(peaks, inflexion)
    allpeaks = np.sort(allpeaks)

    # If peak is at the start or at the end just ignore them
    if allpeaks[0] == 1:
        allpeaks = allpeaks[1:]
    if allpeaks[-1] == len(y) - 2:
        allpeaks = allpeaks[:-1]

    final_peaks = []
    # If peaks are consecutive ignore the smaller one
    for idx, d in enumerate(allpeaks):
        try:
            nd = allpeaks[idx + 1]
        except IndexError:
            continue
        if nd - d == 1:
            continue
        else:
            final_peaks.append(d)

    return np.array(final_peaks)


def slope(x, y):
    # Helper function for minmax method, calculates the slope
    s, o, r_value, p_value, std_err = linregress(x, y)
    return s, o


def displacement(track):
    """
    Displacement method. For a track calculates the displacement between each point in 3D
    The velocity is calculated by dividing each displacement by the sample rate and smoothing everything
    by a 30% window moving average.
    """

    velo = []
    xcoord = np.diff(track.xtrack)
    ycoord = np.diff(track.ytrack)
    zcoord = np.diff(track.ztrack)
    displacement = np.sqrt(xcoord ** 2 + ycoord ** 2 + zcoord ** 2)

    # In reality we should be looking to regions of flatness
    # Plateaus of slope zero which indicate constant velocity

    window = int((len(displacement) * 30) // 100)
    velo = smoothing(displacement / track.samplerate, window)

    return velo * 1000


def minmax(track):
    """
    Minmax method
    Given a set of delimiters (min, max and inflexion points), calculates the slopes between those delimiters
    """

    section_velocity = []
    y = track.smoothedtrajectory
    x = np.array(range(len(y))) * track.samplerate
    delimiter = findallpeaks(y)

    if not delimiter.size:  # is empty
        m, b = slope(x, y)
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

    return section_velocity * 1000


def finite(track):
    """
    Finite differences method
    Self explanatory, Central differences 2nd order, 1st order for the boundaries
    """
    velo = []
    y = track.smoothedtrajectory
    x = np.array(range(len(y))) * track.samplerate
    velo = np.append(velo, np.abs(np.gradient(y, x)))

    return velo * 1000
