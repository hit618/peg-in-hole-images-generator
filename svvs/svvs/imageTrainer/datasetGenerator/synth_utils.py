import argparse
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from typing import Union
import cv2
from transform3d import Transform

deg = math.pi / 180
COLORS = {
    'r': (255, 0, 0, 255),
    'g': (0, 255, 0, 255),
    'b': (0, 0, 255, 255),
    'k': (0, 0, 0, 255),
    'w': (255, 255, 255, 255),
}
# parser = argparse.ArgumentParser()
# parser.add_argument("--frame-start", type=int, default=0)
# parser.add_argument("--frame-end", type=int, default=1000)

def hyperSphereVolumeSampler(d, r=1.):
    while True:
        p = np.random.uniform(-r, r, d)
        if np.linalg.norm(p) <= r:
            return p


def hyperSphereSurfaceSampler(d, r=1.):
    p = hyperSphereVolumeSampler(d)
    return p / np.linalg.norm(p) * r

def pegPosSampler(xyRange, zRange):
    # xy = hyperSphereVolumeSampler(2, 0.03)
    # z = np.random.uniform(0.305, 0.355)
    xy = hyperSphereVolumeSampler(2, xyRange)
    z = np.random.uniform(zRange[0], zRange[1])
    return (*xy, z)

def pegPosSampler2(xRange, yRange, zRange):
    # xy = hyperSphereVolumeSampler(2, 0.03)
    # z = np.random.uniform(0.305, 0.355)
    x = np.random.uniform(xRange[0], xRange[1])
    y = np.random.uniform(yRange[0], yRange[1])
    z = np.random.uniform(zRange[0], zRange[1])
    return (x, y, z)


def pegRotSampler(range,express='euler'):
    rotvec = hyperSphereSurfaceSampler(3, np.random.uniform(0, range * deg))
    if express=='euler':
        return Rotation.from_rotvec(rotvec).as_euler('xyz')
    elif express=='quat':
        return Rotation.from_rotvec(rotvec).as_quat()
    return Rotation.from_rotvec(rotvec).as_euler('xyz')

def draw_points(img, points, c: Union[str, tuple] = 'r'):
    if isinstance(c, str):
        c = COLORS[c]
    for i, p in enumerate(points):
        cv2.drawMarker(img, tuple(p[::-1]), c, cv2.MARKER_TILTED_CROSS, 10, 1, cv2.LINE_AA)


def load_rand_overlay(overlay_img_fps, random_state: np.random.RandomState = np.random,
                      size=None, imread=cv2.IMREAD_COLOR, angle=None):
    fp = random_state.choice(overlay_img_fps)
    img = cv2.imread(str(fp), imread)
    if size is not None:
        if angle is None:
            angle = np.random.uniform(low=np.pi / 2, high=3* np.pi / 2)
        cx, cy = img.shape[0] / 2, img.shape[1] / 2
        M = (
                    Transform(p=(cx, cy, 0)).matrix @
                    Transform(rotvec=(0, 0, angle)).matrix @
                    Transform(p=(-cx, -cy, 0)).matrix
            )[:2, (0, 1, 3)]
        img = cv2.warpAffine(img, M, (img.shape[0], img.shape[1]), borderValue=(255, 255, 255))
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return img


def overlay_composite(img, overlay_img_fps, random_state: np.random.RandomState = np.random, max_alpha=0.6):
    h, w = img.shape[:2]
    # angle = np.random.uniform(low=np.pi - 0.2, high=np.pi + 0.2)
    angle = np.random.uniform(low=np.pi / 2, high= 3 * np.pi / 2)
    overlay = load_rand_overlay(overlay_img_fps, random_state, size=(w, h),angle=angle)
    mask = load_rand_overlay(overlay_img_fps, random_state, size=(w, h), imread=cv2.IMREAD_GRAYSCALE,angle=angle)
    mask_f = mask.astype(np.float32)[..., None] * (max_alpha / 255)
    comp = img.astype(np.float32) * (1 - mask_f) + overlay.astype(np.float32) * mask_f
    # comp = img.astype(np.float32) + overlay.astype(np.float32) * mask_f
    comp = comp.round().astype(np.uint8)
    return comp, overlay, mask


def heatmap(sigma, w, h, points, d=3):  # efficient version of heatmap naive
    s = int(sigma * d)  # assumes that values further away than sigma * d are insignificant
    hm = np.zeros((h, w))
    for x, y in points:
        _x, _y = int(round(x)), int(round(y))
        xmi, xma = max(0, _x - s), min(w, _x + s)
        ymi, yma = max(0, _y - s), min(h, _y + s)
        _h, _w = yma - ymi, xma - xmi
        if _h > 0 and _w > 0:
            X, Y = np.arange(_w).reshape(1, _w), np.arange(_h).reshape(_h, 1)
            _hm = (x - xmi - X) ** 2 + (y - ymi - Y) ** 2
            _hm = np.exp(-_hm / (2 * sigma ** 2))
            hm[ymi:yma, xmi:xma] = np.maximum(hm[ymi:yma, xmi:xma], _hm)
    return hm

# def resetPeg(sim, objectHandle):
#
#     position = list(pegPosSampler())
#     print(position)
#     sim.setObjectPosition(objectHandle, position, sim.handle_world)
#     quat = list(pegRotSampler(express='quat'))
#     print(quat)
#     sim.setObjectQuaternion(objectHandle, quat, sim.handle_world)

