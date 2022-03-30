"""
SPT_Tool 2021
Antonio Brito @ BCB lab ITQB
"""

from scipy.signal import savgol_filter
from scipy.stats import linregress
from scipy.optimize import brute
import numpy as np


def smoothing(unsmoothed, windowsize):
    """
    Helper function for all methods
    Moving average leveraging 1d convolution (with array of ones)
    """
    smoothed = np.convolve(unsmoothed, np.ones(windowsize) / windowsize, mode='valid')
    return smoothed


def slope_and_mse(x, y, Rbool=False):
    """
    Helper function for minmax method
    Calculates the slope of a line given x and y coordinates
    """
    s, o, r_value, p_value, std_err = linregress(x, y)
    ypred = s * x + o
    mse = np.average((y - ypred) ** 2)

    if Rbool:
        return s, mse, r_value
    else:
        return s, mse


def objective_function(x, *args):
    xcoord = args[0]
    ycoord = args[1]

    # Make sure x is a list of integers (indexes of breakpoints)
    x = [int(r) for r in x]

    # Solution must have with UNIQUE indexes
    if not len(np.unique(x)) == len(x):
        return np.inf

    # Indexes MUST be sorted
    # https://stackoverflow.com/questions/3755136/pythonic-way-to-check-if-a-list-is-sorted-or-not
    if not all(x[i] <= x[i + 1] for i in range(len(x) - 1)):
        return np.inf

    # Breakpoints must be far apart (here 2 points in between each breakpoint)
    if np.any(np.diff(x, prepend=0, append=len(xcoord) - 1) < 4):
        return np.inf

    # Measure mean squared displacement between sections
    _, mse = breakpoint_regression(xcoord, ycoord, x)

    # sanity check, number of sections = number of breakpoints + 1
    assert len(mse) == len(x) + 1

    # mean of the mean squared errors of each section?
    # try median? how to penalize for number of sections?
    # Maybe try sqrt(mse) => um of each trackpoint => 0.04um localization error is acceptable?
    # Right now= weighted average by the length of each section
    avg_mse = np.average(mse, weights=np.diff(x, prepend=0, append=len(xcoord) - 1))

    return avg_mse


def bruteforce(x, y):
    all_velos = []
    opt_results = []

    ysmooth = y  # savgol_filter(y, 5, 2)  # arbitrary for now

    # Check 0 breakpoints since it might be better
    v, e = slope_and_mse(x, y)
    all_velos.append(v)
    opt_results.append({'fval': e, 'x': [0, -1]})

    # Check 1-4 breakpoints
    # Calculating breakpoint on a smoothed trajectory
    # Velocity calculations are on the real trajectory
    for sec_n in range(1, 4):
        optimum = brute(objective_function, ranges=(slice(0, len(x) - 1, 1),) * sec_n, args=(x, ysmooth))

        # Make sure the result is a list of ints
        if isinstance(optimum, float):
            optimum = [optimum]
        optimum = [int(r) for r in optimum]

        vv, e1 = breakpoint_regression(x, y, optimum)
        all_velos.append(vv)

        e2 = objective_function(optimum, x, y)
        # Sanity check
        assert np.average(e1, weights=np.diff(optimum, prepend=0, append=len(x) - 1)) == e2

        opt_results.append({'fval': e2, 'x': optimum})

    # Analyze all number of breakpoints and choose the min error
    errors = [r['fval'] for r in opt_results]
    velocity = all_velos[np.argmin(errors)]
    sections = [r['x'] for r in opt_results][np.argmin(errors)]

    return velocity, errors[np.argmin(errors)], sections


def minmax(track):
    ycoordinate = track.unwrapped
    xcoordinate = np.array(range(len(ycoordinate))) * track.samplerate

    # Test no sectioning
    velob4, errorb4, rsquared = slope_and_mse(xcoordinate, ycoordinate, RBool=True)
    if errorb4 < 0.05 or rsquared**2 >= 0.9 or True:
        # error is ok!
        return np.abs([velob4]) * 1000, []
    else:
        # brute force
        veloaf, erroraf, finaldelimiters = bruteforce(xcoordinate, ycoordinate)
        return veloaf, finaldelimiters


def breakpoint_regression(x, y, delimiter):
    """
    Given a set of delimiters (min, max and inflexion points), calculates the slopes
    and mean squared errors between those delimiters
    """

    section_velocity = []
    section_mse = []

    if delimiter == [0, -1]:  # is empty or delimiters are beginning and end
        ve, er = slope_and_mse(x, y)
        section_velocity.append(ve)
        section_mse.append(er)
    else:
        ve, er = slope_and_mse(x[0:delimiter[0]], y[0:delimiter[0]])
        section_velocity.append(ve)
        section_mse.append(er)
        for idx, d in enumerate(delimiter):
            if d == delimiter[-1]:
                ve, er = slope_and_mse(x[d:-1], y[d:-1])
                section_velocity.append(ve)
                section_mse.append(er)
            else:
                nextd = delimiter[idx + 1]
                ve, er = slope_and_mse(x[d:nextd], y[d:nextd])
                section_velocity.append(ve)
                section_mse.append(er)

    section_velocity = np.abs(section_velocity) * 1000
    section_mse = np.array(section_mse)

    return section_velocity, section_mse


def displacement(track):
    """
    Displacement method. For a given track calculates the displacement between each point in 3D
    The velocity is calculated by dividing each displacement by the sample rate and smoothing everything
    by a 30% window moving average.
    """

    xcoord = np.diff(track.x)
    ycoord = np.diff(track.y)
    zcoord = np.diff(track.z)
    displacement_ = np.sqrt(xcoord ** 2 + ycoord ** 2 + zcoord ** 2)

    # In reality we should be looking to regions of flatness
    # Plateaus of slope zero which indicate constant velocity

    velo = smoothing(displacement_ / track.samplerate, int((len(displacement_) * 30) // 100))

    return velo * 1000
