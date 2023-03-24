"""Source code: https://github.com/ysnan/NBD_KerUnc/blob/master/utils/imtools.py"""

import numpy as np
from scipy.ndimage import filters


def cconv_np(data, ker):
    return filters.convolve(data, ker, mode='wrap')


def fspecial(type: str, *args):
    dtype = np.float32
    if type == 'average':
        siz = (args[0],args[0])
        h = np.ones(siz) / np.prod(siz)
        return h.astype(dtype)
    
    elif type == 'gaussian':
        p2 = args[0]
        p3 = args[1]
        siz = np.array([(p2[0]-1)/2 , (p2[1]-1)/2])
        std = p3
        x1 = np.arange(-siz[1], siz[1] + 1, 1)
        y1 = np.arange(-siz[0], siz[0] + 1, 1)
        x, y = np.meshgrid(x1, y1)
        arg = -(x*x + y*y) / (2*std*std)
        h = np.exp(arg)
        sumh = sum(map(sum, h))
        if sumh != 0:
            h = h/sumh
        return h.astype(dtype)
    
    elif type == 'motion':
        p2 = args[0]
        p3 = args[1]
        len = max(1, p2)
        half = (len - 1) / 2
        phi = np.mod(p3, 180) / 180 * np.pi

        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        xsign = np.sign(cosphi)
        linewdt = 1

        eps = np.finfo(float).eps
        sx = np.fix(half * cosphi + linewdt * xsign - len * eps)
        sy = np.fix(half * sinphi + linewdt - len * eps)

        x1 = np.arange(0, sx + 1, xsign)
        y1 = np.arange(0, sy + 1, 1)
        x, y = np.meshgrid(x1, y1)

        dist2line = (y * cosphi - x * sinphi)
        rad = np.sqrt(x * x + y * y)

        lastpix = np.logical_and(rad >= half, np.abs(dist2line) <= linewdt)
        lastpix.astype(int)
        x2lastpix = half * lastpix - np.abs((x * lastpix + dist2line * lastpix * sinphi) / cosphi)
        dist2line = dist2line * (-1 * lastpix + 1) + np.sqrt(dist2line ** 2 + x2lastpix ** 2) * lastpix
        dist2line = linewdt + eps - np.abs(dist2line)
        logic = dist2line < 0
        dist2line = dist2line * (-1 * logic + 1)

        h1 = np.rot90(dist2line, 2)
        h1s = np.shape(h1)
        h = np.zeros(shape=(h1s[0] * 2 - 1, h1s[1] * 2 - 1))
        h[0:h1s[0], 0:h1s[1]] = h1
        h[h1s[0] - 1:, h1s[1] - 1:] = dist2line
        h = h / sum(map(sum, h)) + eps * len * len

        if cosphi > 0:
            h = np.flipud(h)

        return h.astype(dtype)
