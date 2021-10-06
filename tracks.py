import pandas as pd
import xml.etree.ElementTree as ET
from scipy.optimize import minimize, NonlinearConstraint

from Analysis import *


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

        self.finitediff = finite(self)
        self.disp = displacement(self)
        self.minmax = minmax(self)
        self.manual = None

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
            classlist.append(
                cls(eli, obj.x, obj.y, obj.sr, str(obj.xml).split('/')[-1][:-4] + f"_{idx}", obj.imageobject))

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
