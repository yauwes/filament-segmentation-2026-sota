"""Input enhancement: three physically motivated planes that replace the raw grey image for the crop segmenter.

Filaments are dark, thin ridges on a bright disk whose brightness falls off towards the limb. Each plane removes one
nuisance and is expressed in robust units (median / MAD inside the solar disk), so frames from the six GONG sites
with different exposure and seeing land on a common scale:

    bth7   black top-hat, disk element radius 7 px: dark valleys narrower than ~15 px
    flat   image minus its 101-px median: limb darkening and large-scale background removed
    ridge  Sato tubeness for dark ridges (sigma 1-3 at half resolution): elongated structures
"""
from pathlib import Path

import cv2
import numpy as np

from . import data

PLANES = ("bth7", "flat", "ridge")


def disk_mask(img):
    """Solar disk as a circle fitted to the thresholded frame (GONG frames are not always centred)."""
    ys, xs = np.nonzero(img > 0.25 * np.percentile(img, 99))
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    return np.hypot(yy - ys.mean(), xx - xs.mean()) < 0.99 * np.sqrt(len(xs) / np.pi)


def enhance(img, names=PLANES):
    """uint8 grey image -> uint8 (len(names), H, W); 128 is the disk median, 16 grey levels are one MAD."""
    from skimage.filters import sato
    disk, x, out = disk_mask(img), img.astype(np.float32), []
    for n in names:
        if n == "raw":
            c = x
        elif n.startswith("bth"):
            r = int(n[3:])
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
            c = -cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, k).astype(np.float32)
        elif n == "flat":
            c = x - cv2.medianBlur(img, 101).astype(np.float32)
        elif n == "ridge":
            small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
            tube = sato(small.astype(np.float32) / 255, sigmas=(1, 2, 3), black_ridges=True)
            c = -cv2.resize(tube.astype(np.float32), img.shape[::-1], interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError(n)
        v = c[disk]
        med = float(np.median(v))
        mad = 1.4826 * float(np.median(np.abs(v - med))) + 1e-6
        z = np.where(disk, np.clip((c - med) / mad, -8, 8), 0)
        out.append(np.clip((z + 8) * (255 / 16), 0, 255).astype(np.uint8))
    return np.stack(out)


def planes(stem, cache=True):
    """Enhanced planes of a photo, memory-mapped from WORK/planes when cached (see scripts/cache_planes.py)."""
    f = data.WORK / "planes" / f"{stem}.npy"
    if f.exists():
        return np.load(f, mmap_mode="r")
    p = enhance(data.read_gray(stem))
    if cache:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f".{np.random.randint(1 << 30)}.tmp.npy")
        np.save(tmp, p)
        Path(tmp).replace(f)                                 # atomic: parallel workers never read a partial file
    return p
