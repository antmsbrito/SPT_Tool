"""
SPT_TOOL
@author António Brito
ITQB-UNL BCB 2021
"""

import math
import pandas as pd
import xml.etree.ElementTree as ET
from scipy.optimize import minimize, NonlinearConstraint

from Analysis import *


class TrackV2:

    def __init__(self, im, x, y, samplerate, name, ellipse=None):
        self.imageobject = im
        self.x = np.array(x)
        self.y = np.array(y)
        self.xypairs = np.array([[xc, yc] for xc, yc in zip(self.x, self.y)])

        self.name = name
        self.samplerate = float(samplerate)

        self.twodspeed = np.sum(np.sqrt(np.diff(self.x) ** 2 + np.diff(self.y) ** 2)) / (len(np.diff(self.y)) * self.samplerate)

        self._ellipse = ellipse
        self.xy_ellipse = None
        self.xellipse = None
        self.yellipse = None
        self.z = None
        self.unwrapped = None
        self.minmax_velo = None

        self.manual_sections = []
        self.manual_velo = []

    @property
    def ellipse(self):
        return self._ellipse

    @ellipse.setter
    def ellipse(self, new_value):
        self._ellipse = new_value
        if new_value:
            self.update()
        else:
            return

    @classmethod
    def generator_xml(cls, xmlfile, image):

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
            classlist.append(cls(image, tempx, tempy, srate, str(xmlfile).split('/')[-1][:-4] + f"_{counter}"))
            counter += 1
        return classlist

    @classmethod
    def generator_csv(cls, xmlfile, csvfile):

        # ellipse
        ellipses_data = pd.read_csv(csvfile, sep=",")
        ellipsesdict = {}
        for i in list(ellipses_data.index.values):
            ellipsesdict[str(i)] = {}
            ellipsesdict[str(i)]["x0"] = ellipses_data.iloc[i]["X"]
            ellipsesdict[str(i)]["y0"] = ellipses_data.iloc[i]["Y"]
            ellipsesdict[str(i)]["major"] = ellipses_data.iloc[i]["Major"]
            ellipsesdict[str(i)]["minor"] = ellipses_data.iloc[i]["Minor"]
            # Transform the imagej angle into angle to x axis (90º to -90º)
            ellipsesdict[str(i)]["angle"] = -ellipses_data.iloc[i]["Angle"] if ellipses_data.iloc[i]["Angle"] < 90 else \
                180 - ellipses_data.iloc[i]["Angle"]

        # xml
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

            xmean = np.mean(tempx)
            ymean = np.mean(tempy)
            ellipse_distance = np.array(
                [(ellipsesdict[elikey]["x0"] - xmean) ** 2 for elikey in ellipsesdict]) + np.array(
                [(ellipsesdict[elikey]["y0"] - ymean) ** 2 for elikey in ellipsesdict])
            closestellipse = ellipsesdict[str(np.argmin(ellipse_distance))]  # todo check if correct

            classlist.append(
                cls(None, tempx, tempy, srate, str(xmlfile).split('/')[-1][:-4] + f"_{counter}", closestellipse))
            counter += 1

        return classlist

    def update(self):
        self.xy_ellipse = self.closest_ellipse_points()
        self.xellipse, self.yellipse = np.array(list(zip(*self.xy_ellipse)))
        self.z = self.calculatez()
        self.unwrapped = self.unwrapper()
        self.minmax_velo = minmax(self)
        self.manual_sections = None if self.manual_sections is None else self.manual_sections
        self.manual_velo = None if self.manual_sections is None else self.manual_sections

    def closest_ellipse_points(self):

        # Ellipse center, semi axes and angle to x axis
        center = np.array([self._ellipse['x0'], self._ellipse['y0']])
        smmajor = self._ellipse['major'] / 2
        smminor = self._ellipse['minor'] / 2
        ang = np.deg2rad(self._ellipse['angle'])

        # Translate and rotate ellipse to be centered at (0,0) with major axis horizontal
        translated = [x-center for x in self.xypairs]
        rotated_and_translated = [self.rot2d(-1*ang).dot(t) for t in translated]

        # Solve closest point problem
        ellipsepoints = [self.solve(smmajor, smminor, point) for point in rotated_and_translated]

        # Undo rotation and translation
        return np.array([self.rot2d(ang).dot(p)+center for p in ellipsepoints])

    def unwrapper(self):

        # center referencial
        x = self.xellipse - self._ellipse['x0']
        y = self.yellipse - self._ellipse['y0']
        z = self.z - 0

        angles_to_x = np.arctan2(y, x)

        for idx, val in enumerate(angles_to_x):
            if 0 <= val <= np.pi:
                continue
            else:
                angles_to_x[idx] = np.pi + (np.pi + val)

        rawperimeter = angles_to_x * self._ellipse['major'] / 2

        turns = 0
        perimeter = []

        # Wrap turns
        for idx, val in enumerate(angles_to_x):
            if idx == 0:
                perimeter.append(val * self._ellipse['major'] / 2)
            else:
                prevval = angles_to_x[idx - 1]
                if 0 < val < np.pi / 2 and 1.5 * np.pi < prevval < 2 * np.pi:
                    turns += 2 * np.pi
                    perimeter.append(val * (self._ellipse['major'] / 2) + turns * (self._ellipse['major'] / 2))
                elif 1.5 * np.pi < val < 2 * np.pi and 0 < prevval < np.pi / 2:
                    turns -= 2 * np.pi
                    perimeter.append(val * (self._ellipse['major'] / 2) + turns * (self._ellipse['major'] / 2))
                else:
                    perimeter.append(val * (self._ellipse['major'] / 2) + turns * (self._ellipse['major'] / 2))

        perimeter = np.array(perimeter)
        rawperimeter = np.array(rawperimeter)

        return perimeter

    def calculatez(self):
        radius = self._ellipse['major'] / 2

        zcoord = []
        counter = 1
        for idx, pair in enumerate(self.xy_ellipse):
            xdistancevector = pair[0] - self._ellipse['x0']
            ydistancevector = pair[1] - self._ellipse['y0']
            distance = np.linalg.norm([xdistancevector, ydistancevector])
            sqrarg = radius ** 2 - distance ** 2
            if sqrarg < 0:
                sqrarg = 0
                counter = -1
            temporaryZ = np.array(np.sqrt(sqrarg))
            zcoord.append(temporaryZ * counter)

        return np.array(zcoord)

    @staticmethod
    def solve(semi_major, semi_minor, p):
        """
        These cryptic (yet genius) lines of code were taken verbatim from
        https://wet-robots.ghost.io/simple-method-for-distance-to-ellipse/
        This implements a method to calculate the ellipse point that is closen to a given point p
        It works by using the evolute of the ellipse (ex, ey the center of curvature) to locally approximate
        the ellipse to a circle! Then we just iterate the method 3-4 times to yield the final point.
        The current implementation NEEDS and ellipse centered at (0,0) with horizontal major axis.
        """
        px = abs(p[0])
        py = abs(p[1])

        t = math.pi / 4

        a = semi_major
        b = semi_minor

        for x in range(0, 3):
            x = a * math.cos(t)
            y = b * math.sin(t)

            ex = (a * a - b * b) * math.cos(t) ** 3 / a
            ey = (b * b - a * a) * math.sin(t) ** 3 / b

            rx = x - ex
            ry = y - ey

            qx = px - ex
            qy = py - ey

            r = math.hypot(ry, rx)
            q = math.hypot(qy, qx)

            delta_c = r * math.asin((rx * qy - ry * qx) / (r * q))
            delta_t = delta_c / math.sqrt(a * a + b * b - x * x - y * y)

            t += delta_t
            t = min(math.pi / 2, max(0, t))

        return math.copysign(x, p[0]), math.copysign(y, p[1])

    @staticmethod
    def rot2d(angle):
        s, c = np.sin(angle), np.cos(angle)
        return np.array([[c, -s], [s, c]])


class Track:
    def __init__(self, ellipse, trackx, tracky, samplerate, trackname, image):

        # Name based on file
        self.designator = trackname

        # Trackmate output
        self.samplerate = float(samplerate)

        # Track position directly off trackmate xml
        self.xtrack = np.array(trackx)
        self.ytrack = np.array(tracky)
        self.xypairs = np.array([[xc, yc] for xc, yc in zip(self.xtrack, self.ytrack)])

        # This assumes that every track HAS to have an ellipse
        self.ellipse = ellipse

        # ezrA image if it exists / is provided
        self.image = image

        # Coordinates of the closest ellipse points for each track coordinate pair (x,y)
        self.ellipsepoints = self.closest_ellipse_point()
        self.xellipse, self.yellipse = np.array(list(zip(*self.ellipsepoints)))

        # Calculate Z position based upon ellipse and the projected circle of said ellipse
        self.ztrack = np.array(self.calculatez())

        # Unwrap the 2D trajectory of the real x's and y's around the projected circle
        self.unwrappedtrajectory = np.array(self.unwrapper())

        # To check magnitude for debugging purposes not used at all
        self.cumvelowithz = np.sum(
            np.sqrt(np.diff(self.xtrack) ** 2 + np.diff(self.ytrack) ** 2 + np.diff(self.ztrack) ** 2)) / (
                                    len(np.diff(self.ytrack)) * self.samplerate)
        self.cumvelonoz = np.sum(np.sqrt(np.diff(self.xtrack) ** 2 + np.diff(self.ytrack) ** 2)) / (
                len(np.diff(self.ytrack)) * self.samplerate)

        # Smoothing the unwrapped trajectory with a 20% window moving average
        self.smoothedtrajectory = self.smoothing()

        # For quality control purposes, currently only serves to check to remove tracks with no ellipses
        self.ellipse_error = np.mean(np.linalg.norm(self.xypairs - self.ellipsepoints, axis=1) * 1000)

        # No need to calculate a time axis. All our metrics are time invariant
        # Better to leave the calculation for later in a case by case, since smoothing may introduce arrays
        # with less elements
        # In a case by case basis the samplerate is enough to define a suitable timeaxis
        # self.timeaxis = np.linspace(1, len(self.unwrappedtrajectory) * self.samplerate, len(self.unwrappedtrajectory))
        # in case of smoothing just translate the arrays over by half the amount of the difference of points

        self.minmax = minmax(self)
        self.manual = []

    def __repr__(self):
        return self.designator

    @classmethod
    def generatetrackfromcsv(cls, xmlfile, csvfile):

        ellipses_data = pd.read_csv(csvfile, sep=",")
        ellipsesdict = {}
        # todo check these indexes
        for i in list(ellipses_data.index.values):
            ellipsesdict[str(i)] = {}
            ellipsesdict[str(i)]["x0"] = ellipses_data.iloc[i]["X"]
            ellipsesdict[str(i)]["y0"] = ellipses_data.iloc[i]["Y"]
            # Check measurements on fiji
            # ellipsesdict[str(i)]["bx"] = ellipses_data.i  loc[i]["BX"]
            # ellipsesdict[str(i)]["by"] = ellipses_data.iloc[i]["BY"]
            # ellipsesdict[str(i)]["width"] = ellipses_data.iloc[i]["Width"]
            # ellipsesdict[str(i)]["height"] = ellipses_data.iloc[i]["Height"]
            ellipsesdict[str(i)]["major"] = ellipses_data.iloc[i]["Major"]
            ellipsesdict[str(i)]["minor"] = ellipses_data.iloc[i]["Minor"]
            # Transform the imagej angle into angle to x axis (90º to -90º)
            ellipsesdict[str(i)]["angle"] = -ellipses_data.iloc[i]["Angle"] if ellipses_data.iloc[i]["Angle"] < 90 else \
                180 - ellipses_data.iloc[i]["Angle"]

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

            xmean = np.mean(tempx)
            ymean = np.mean(tempy)
            ellipsedistance = np.array(
                [(ellipsesdict[elikey]["x0"] - xmean) ** 2 for elikey in ellipsesdict]) + np.array(
                [(ellipsesdict[elikey]["y0"] - ymean) ** 2 for elikey in ellipsesdict])
            closestellipse = ellipsesdict[str(np.argmin(ellipsedistance))]  # todo check if correct

            classlist.append(
                cls(closestellipse, tempx, tempy, srate, str(xmlfile).split('/')[-1][:-4] + f"_{counter}", None))
            counter += 1

        return classlist

    @classmethod
    def generatetrack_ellipse(cls, precursorobjectlist, ellipse):
        classlist = []

        for idx, obj in enumerate(precursorobjectlist):
            eli = ellipse[idx]
            if eli:
                classlist.append(
                    cls(eli, obj.x, obj.y, obj.sr, str(obj.xml).split('/')[-1][:-4] + f"_{idx}", obj.imageobject))
            else:
                pass

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

            # Call to the minimization algorithm, several methods are available: SLSQP (quadratic, quick but more unstable);
            # COBYLA (linear but more stable), trust-constr (gradient descent with barrier methods)
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

    def unwrapper(self):

        # center referencial
        x = self.xellipse - self.ellipse['x0']
        y = self.yellipse - self.ellipse['y0']
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
        windowsize = int((len(self.unwrappedtrajectory) * 20) // 100)
        fperi = np.convolve(self.unwrappedtrajectory, np.ones(windowsize) / windowsize, mode='valid')
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
