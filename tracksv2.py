import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from scipy.optimize import minimize, NonlinearConstraint
from scipy.signal import butter, filtfilt
import os
from matplotlib import pyplot as plt
from matplotlib import patches


class Track:
    def __init__(self, trackx, tracky, samplerate, trackname):

        self.designator = trackname
        self.samplerate = float(samplerate)

        self.xtrack = np.array(trackx)
        self.ytrack = np.array(tracky)
        self.xypairs = np.array([[xc, yc] for xc, yc in zip(self.xtrack, self.ytrack)])
        self.timeaxis = np.arange(0, len(self.xtrack), 1) * self.samplerate

        self.ellipse = None
        self.ellipsepoints = None

        self.ztrack = None
        self.unwrappedtrajectory = None
        self.smoothedtrajectory = None
        self.instavelo = None
        self.ellipse_error = None

    def __repr__(self):
        return self.designator

    @classmethod
    def generatetrack(cls, xmlfile):

        classlist = []
        root = ET.parse(xmlfile).getroot()
        srate = root.attrib['frameInterval']
        counter = 0
        for children in root:
            tempx = []
            tempy = []
            for grandchildren in children:
                tempx.append(float(grandchildren.attrib['x']))  # list of x coords
                tempy.append(float(grandchildren.attrib['y']))  # list of y coords

            classlist.append(cls(tempx, tempy, srate, str(xmlfile).split('/')[-1][:-4] + f"_{counter}"))
            counter += 1

        return classlist

    def closest_ellipse_point(self):
        """Given a track and its closest ellipse finds the ellipse points closest to each
        of the track points
        It minimizes distance to the ellipse given a constrained optimization algorithm
        """
        cpoints = []

        # Ellipse center
        x0 = self.ellipse['x0']
        y0 = self.ellipse['y0']
        # Angle to the x axis in radians
        theta = np.deg2rad(self.ellipse['angle'])
        # 'Radius' aka half of the distance of the major and minor axis
        a = self.ellipse['major'] / 2
        b = self.ellipse['minor'] / 2

        for xpoint, ypoint in self.xypairs:
            # All ellipses obey to this equation==1; It is easily proven just take a regular ellipse equation and
            # substitute x and y for (x-x0) * rotationmatrix and simplify
            ellipsefunction = lambda inp: ((inp[0] - x0) * np.cos(theta) + (inp[1] - y0) * np.sin(theta)) ** 2 / (
                a) ** 2 + ((inp[0] - x0) * np.sin(theta) - (inp[1] - y0) * np.cos(theta)) ** 2 / (b) ** 2
            nlc = NonlinearConstraint(ellipsefunction, 0.999,
                                      1.001)  # Ellipse function MUST be equal to 1 (some tolerance)

            initialguess = (
                x0, y0)  # Initial guess for the algorithm, in lieu of a good guess just take the initial point

            # Call to the minimization algorithm, two methods are available: SLSQP (quadratic, quick but more unstable);
            # COBYLA (linear but more stable), trust-constr (gradient descent with barrier methods)
            # In this problem trust-constr seems to yield better results
            result = minimize(self.objective_function, x0=initialguess, args=(xpoint, ypoint), method='COBYLA',
                              constraints=nlc)

            # TODO Add exception clauses in case optimization is not a success
            #  TODO add options to minimization algorithm
            # TODO sometimes the algorithm skips a solution especially when using higher order methods (
            #   SLSQP) check alternatives
            # TODO Profile and optimize above code; banach fixed point iteration may
            #    prove faster and easier to implement but harder to explain to biologists

            cpoints.append(result.x)

        return cpoints

    def closest_ellipse(self, elidict):
        """Given a track array find closest ellipse
        Inputs are a track array and a elipse dict
        Returns the key of ellipse dict that corresponds to the closest ellipse"""
        track = self.xypairs
        xmean = np.mean([coord[0] for coord in track])
        ymean = np.mean([coord[1] for coord in track])
        ellipsedistance = np.array([(elidict[elikey]["x0"] - xmean) ** 2 for elikey in elidict]) + np.array(
            [(elidict[elikey]["y0"] - ymean) ** 2 for elikey in elidict])
        closestellipse = elidict[str(np.argmin(ellipsedistance))]  # todo check if correct
        return closestellipse

    def unwrapper(self):

        # center referencial
        x = self.xtrack - self.ellipse['x0']
        y = self.ytrack - self.ellipse['y0']
        z = self.ztrack - 0

        angles_to_x = np.arctan2(y, x)

        for idx, val in enumerate(angles_to_x):
            if 0 <= val <= np.pi:
                continue
            else:
                angles_to_x[idx] = np.pi + (np.pi + val)

        rawperimeter = angles_to_x * self.ellipse['major'] / 2

        turns = 0
        perimeter = []

        # Wrap turns
        for idx, val in enumerate(angles_to_x):
            if idx == 0:
                perimeter.append(val * self.ellipse['major'] / 2)
            else:
                prevval = angles_to_x[idx - 1]
                if 0 < val < np.pi / 2 and 1.5 * np.pi < prevval < 2 * np.pi and np.abs(
                        val - prevval) * (self.ellipse['major'] / 2) / self.samplerate > 30 / 1000:
                    turns += 2 * np.pi
                    perimeter.append(val * (self.ellipse['major'] / 2) + turns * (self.ellipse['major'] / 2))
                elif 1.5 * np.pi < val < 2 * np.pi and 0 < prevval < np.pi / 2 and np.abs(
                        val - prevval) * (self.ellipse['major'] / 2) / self.samplerate > 30 / 1000:
                    turns -= 2 * np.pi
                    perimeter.append(val * (self.ellipse['major'] / 2) + turns * (self.ellipse['major'] / 2))
                else:
                    perimeter.append(val * (self.ellipse['major'] / 2) + turns * (self.ellipse['major'] / 2))

        perimeter = np.array(perimeter)
        rawperimeter = np.array(rawperimeter)

        return perimeter

    def calculatez(self):
        radius = self.ellipse['major'] / 2

        zcoord = []
        counter = 1
        for idx, pair in enumerate(self.ellipsepoints):
            xdistancevector = pair[0] - self.ellipse['x0']
            ydistancevector = pair[1] - self.ellipse['y0']
            distance = np.linalg.norm([xdistancevector, ydistancevector])
            sqrarg = radius ** 2 - distance ** 2
            if sqrarg < 0:
                sqrarg = 0
                counter = -1
            temporaryZ = np.array(np.sqrt(sqrarg))
            zcoord.append(temporaryZ * counter)

        return zcoord

    def smoothing(self):
        b, a = butter(6, 0.125, btype='lowpass')
        # w, h = freqz(b, a, worN=2000) # for frequency response if needed
        fperi = filtfilt(b, a, self.unwrappedtrajectory, method="gust")
        return fperi

    @staticmethod
    def objective_function(inp, *args):
        """
        Objective function subject to minimization - Euclidean distance between two points
        inputs are the points that belong to the ellipse
        *args are the static point
        """
        xellipse = inp[0]
        yellipse = inp[1]
        return np.sqrt((xellipse - args[0]) ** 2 + (yellipse - args[1]) ** 2)


if __name__ == '__main__':
    pass
