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


class Track:

    def __init__(self, im, x, y, samplerate, name, ellipse=None):

        # Raw data
        self.imageobject = im
        self.x = np.array(x)
        self.y = np.array(y)
        self.xypairs = np.array([np.array([xc, yc]) for xc, yc in zip(self.x, self.y)])
        self.name = name
        self.samplerate = float(samplerate)
        self.timeaxis = np.array(range(len(x))) * samplerate

        # Two dimensional statistics
        self.twodspeed = np.sum(np.sqrt(np.diff(self.x) ** 2 + np.diff(self.y) ** 2)) / (
                len(np.diff(self.y)) * self.samplerate)
        self.twodposition = np.array([np.sqrt(np.square(p[0]) + np.square(p[1])) for p in self.xypairs])
        self.msd = self.msd_calc(self.xypairs)
        self.msd_alpha, _ = slope_and_mse(np.log10(np.arange(1, 20) * 3), np.log10(self.msd[1:20]))

        # Ellipse and 3D stats
        self._ellipse = ellipse
        self.xy_ellipse = None
        self.xellipse = None
        self.yellipse = None
        self.z = None
        self.unwrapped = None

        # Velocities and other data
        # Initialize it always as an empty list or dicts
        # Load method SHOULD check if data existed previously
        self.bruteforce_velo = []
        self.bruteforce_phi = []

        self.muggeo_velo = []
        self.muggeo_phi = []
        self.muggeo_params = {}

        self.manual_sections = []
        self.manual_velo = []

        self.disp_velo = []

        if self._ellipse is not None:
            self.update()

    @property
    def ellipse(self):
        return self._ellipse

    # This allows me to update the measurements ONLY when the ellipse value is assigned to anything except None
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

    def update(self):
        # Called if ellipse attr is set or if loaded with a non None ellipse attr
        # Calculates all 3D stuff plus displacement velocity which is fast to do

        self.xy_ellipse = self.closest_ellipse_points()
        self.xellipse, self.yellipse = np.array(list(zip(*self.xy_ellipse)))
        self.z = self.calculatez()
        self.unwrapped = self.unwrapper()

        self.disp_velo = displacement(self)

    def closest_ellipse_points(self):
        # Ellipse center, semi axes and angle to x axis
        center = np.array([self._ellipse['x0'], self._ellipse['y0']])
        smmajor = self._ellipse['major'] / 2
        smminor = self._ellipse['minor'] / 2
        ang = np.deg2rad(self._ellipse['angle'])

        # Translate and rotate ellipse to be centered at (0,0) with major axis horizontal
        translated = [x - center for x in self.xypairs]
        rotated_and_translated = [self.rot2d(-1 * ang).dot(t) for t in translated]

        # Solve closest point problem
        ellipsepoints = [self.solve(smmajor, smminor, point) for point in rotated_and_translated]

        # Undo rotation and translation
        return np.array([self.rot2d(ang).dot(p) + center for p in ellipsepoints])

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
    def msd_calc(r):
        shifts = np.arange(len(r))
        msds = np.zeros(shifts.size)

        for i, shift in enumerate(shifts):
            diffs = r[:-shift if shift else None] - r[shift:]
            sqdist = np.square(diffs).sum(axis=1)
            msds[i] = sqdist.mean()

        return msds

    @staticmethod
    def solve(semi_major, semi_minor, p):
        """
        These cryptic (yet genius) lines of code were taken verbatim from
        https://wet-robots.ghost.io/simple-method-for-distance-to-ellipse/
        This implements a method to calculate the ellipse point that is closer to a given point p
        It works by using the evolute of the ellipse (ex, ey the center of curvature) to locally approximate
        the ellipse to a circle! Then we just iterate the method 3-4 times to yield the final point.
        The current implementation NEEDS an ellipse centered at (0,0) with horizontal major axis.
        """
        # center
        px = abs(p[0])
        py = abs(p[1])

        # constant
        t = math.pi / 4

        # axis, a is horizontal
        a = semi_major
        b = semi_minor

        # 3 iterations
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

