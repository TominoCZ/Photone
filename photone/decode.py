"""Turn a photo file into the pixels the pipeline draws.

Every image leaves here the same way, whatever it was: scene-linear RGB in Rec.2020 primaries
(D65), packed as RGBA half floats, top row first. That is what goes to the GPU as a float
texture without touching the bytes, and what every later stage of the pipeline starts from.

`load` does the work; `Workers` runs it in another process, so that a file which crashes the
decoder costs one sentence in the window and not the window.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np

from photone import dng, edit, lens

RAW = {".arw", ".cr2", ".cr3", ".dng", ".nef", ".orf", ".pef", ".raf", ".rw2"}
RASTER = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
LUMA = (0.2627, 0.6780, 0.0593)  # how much of the lightness each of Rec.2020's colors is
# What counts as a picture's top (`light_of`): the level only its brightest twentieth is above.
# A lamp or a window smaller than that does not hold back where the exposure starts; a sky does.
TOP = 95


class Unreadable(Exception):
    """This file cannot be shown; the message is for the user."""


class Pixels:
    """A decoded photo's pixels and whoever owns the memory. `close()` when done with `buf`."""

    def __init__(self, buf, shared: SharedMemory | None = None) -> None:
        self.buf, self.shared = buf, shared

    def close(self) -> None:
        if self.shared:
            self.buf = None
            self.shared.close()
            self.shared.unlink()
            self.shared = None


def pack(rgb: np.ndarray) -> bytes:
    """(h, w, 3) linear RGB, float 0..1, as RGBA half floats (the test chart: small)."""
    out = np.ones((*rgb.shape[:2], 4), np.float16)
    out[..., :3] = rgb
    return out.tobytes()


def meta(name: str, width: int, height: int, full: tuple[int, int] | None = None) -> dict:
    """`full` is the photo's real size; width and height are what was decoded, which for a raw
    file at half size is half of it. Zoom is measured against the real size."""
    full = full or (width, height)
    return {"name": name, "width": width, "height": height, "full": list(full),
            "size": "full" if (width, height) == tuple(full) else "half",
            "space": "rec2020-linear", "format": "rgba16f",
            "raw": False,     # as the sensor measured it: it needs a profile to be a picture
            "denoised": False,  # written by Photone's neural denoiser (photone/dng.py)
            "camera": None}   # what white balance needs to know (see `camera`), if known


def camera(wb, cam_xyz) -> dict | None:
    """The camera's own white balance (as gains, green = 1) and its matrix from XYZ to its own
    colors; None when the file gives no usable ones, and white balance stays as shot."""
    wb, cam_xyz = np.array(wb[:3], float), np.array(cam_xyz, float)[:3]
    if not (wb > 0).all() or abs(np.linalg.det(cam_xyz)) < 1e-9:
        return None
    return {"wb": (wb / wb[1]).tolist(), "cam_xyz": cam_xyz.tolist()}


def sampled(raw, nominal: bool = False) -> np.ndarray | None:
    """One cell in 64 of the sensor's mosaic as colors: (rows, columns, 3) of red, green and
    blue, shares of the sensor's white with the camera's white balance on, before any
    developing: so the same whichever size is developed afterwards. None for a sensor that
    is not a mosaic of red, green and blue. `nominal`: of the white the file names, as Photone
    develops since 2026-11-26 (adjust_maximum off); else as the decoder did before (below)."""
    pattern, gains = raw.raw_pattern, np.array(raw.camera_whitebalance, float)
    gains[3] = gains[3] or gains[1]
    if pattern is None or pattern.shape != (2, 2) or raw.color_desc != b"RGBG" or gains.min() <= 0:
        return None
    mosaic, black = raw.raw_image_visible, raw.black_level_per_channel
    # (The decoder takes the highest value in the file for the sensor's white when that is
    # within a quarter of the nominal one: so does this.)
    white = float(mosaic.max())
    white = white if white > 0.75 * raw.white_level and not nominal else float(raw.white_level)
    high, wide = mosaic.shape[0] // 2 * 2, mosaic.shape[1] // 2 * 2
    seen = [[], [], []]  # red, green (two to a cell), blue
    for down, across in ((0, 0), (0, 1), (1, 0), (1, 1)):
        color = int(pattern[down, across])
        dark = float(black[color])
        share = (mosaic[down:high:16, across:wide:16] - dark) / (white - dark)
        seen[1 if color == 3 else color].append(np.clip(share * gains[color] / gains.min(), 0, 1))
    return np.stack([np.mean(planes, axis=0) for planes in seen], axis=-1)


def light_of(raw) -> list[float] | None:
    """How light the picture is as the sensor recorded it, before any developing: [the median
    lightness, the level that only the brightest twentieth of the picture is above (`TOP`), by
    each cell's highest channel], shares of the sensor's white, of `sampled`'s cells. None
    where that has none."""
    cells = sampled(raw)
    if cells is None:
        return None
    return [float(np.median(cells @ np.array(LUMA))), float(np.percentile(cells.max(axis=-1), TOP))]


def match_of(raw, about: dict | None) -> list[list[float]] | None:
    """The tone curve a raw photo starts with (edit.matched): the one that makes the sensor's
    picture, as Photone shows it at rest with the curve straight, as light as the picture the
    camera put in the file. From `sampled`'s cells and the preview at 640 pixels at most, so
    it is the same whichever size is developed. None where either is missing."""
    cells = sampled(raw, nominal=True)
    theirs = carried(raw) if cells is not None and about is not None else None
    if theirs is None:
        return None
    # (Every second cell each way: the shares come out within 0.001 of all of them, in 4 ms
    # and not 24. Reading the preview is most of what is left: a Sony's is the whole picture.)
    ours = edit.tone_at(cells[::2, ::2] @ edit.camera_matrix(about).T,
                        edit.uniforms({"curve": edit.STRAIGHT}, None, True))
    return edit.matched(np.clip(edit.lightness(ours), 0, 1),
                        np.clip(edit.lightness(theirs), 0, 1))


def carried(raw) -> np.ndarray | None:
    """The preview a raw file carries, 640 pixels at most across, as linear Rec.2020 colors
    (h, w, 3), from its own color space (sRGB where it names none). None where it has none."""
    import rawpy
    from PySide6.QtCore import QBuffer, QByteArray, QSize, Qt
    from PySide6.QtGui import QColorSpace, QImage, QImageReader

    try:
        found = raw.extract_thumb()
    except rawpy.LibRawError:
        return None
    if found.format == rawpy.ThumbFormat.JPEG:
        data = QBuffer()
        data.setData(QByteArray(found.data))
        data.open(QBuffer.OpenModeFlag.ReadOnly)
        reader = QImageReader(data, b"jpeg")
        size = reader.size()
        if size.isValid() and max(size.width(), size.height()) > 640:
            reader.setScaledSize(size.scaled(QSize(640, 640), Qt.AspectRatioMode.KeepAspectRatio))
        image = reader.read()
    else:  # plain pixels
        held = np.ascontiguousarray(found.data[..., :3], np.uint8)
        image = QImage(held.data, held.shape[1], held.shape[0], held.shape[1] * 3,
                       QImage.Format.Format_RGB888).copy()
    if image.isNull():
        return None
    if not image.colorSpace().isValid():
        image.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
    image.convertTo(QImage.Format.Format_RGBX16FPx4)
    image.convertToColorSpace(QColorSpace(QColorSpace.Primaries.Bt2020,
                                          QColorSpace.TransferFunction.Linear))
    pixels = np.frombuffer(image.constBits(), np.float16).reshape(image.height(), image.width(), 4)
    return pixels[..., :3].astype(float)


def kept(sizes, half: bool) -> tuple[slice, slice]:
    """The part of a developed raw picture that is the photo: (rows, columns), of the picture
    as it comes out, turned upright. A sensor records a few columns and rows more than the
    camera calls the picture, and the outermost can hold anything (a Canon 700D's last
    column, five times too bright: a blue line down the right edge of a dark photo; the
    owner: "that should never happen"). The file says which part is meant, and that is what
    the camera's own JPEG and every other developer show. All of it where the file names no
    such part. At half size a margin of an odd width takes the pixel it has half of with it."""
    wide, high = sizes.width, sizes.height
    left, top = sizes.crop_left_margin - sizes.left_margin, sizes.crop_top_margin - sizes.top_margin
    right, bottom = wide - left - sizes.crop_width, high - top - sizes.crop_height
    if not sizes.crop_width or not sizes.crop_height or min(left, top, right, bottom) < 0:
        left = top = right = bottom = 0
    # Turned as the picture is (LibRaw's `flip`: 3 upside down, 5 a quarter left, 6 right).
    turned = {3: (right, bottom, left, top), 5: (top, right, bottom, left),
              6: (bottom, left, top, right)}
    left, top, right, bottom = turned.get(sizes.flip, (left, top, right, bottom))
    if sizes.flip in (5, 6):
        wide, high = high, wide
    by = 2 if half else 1
    wide, high = (wide + by - 1) // by, (high + by - 1) // by
    left, top, right, bottom = (-(-margin // by) for margin in (left, top, right, bottom))
    return slice(top, high - bottom), slice(left, wide - right)


def trimmed(rgb: np.ndarray, sizes, half: bool) -> tuple[np.ndarray, tuple[int, int]]:
    """A developed raw picture cut to the photo (`kept`), and the photo's full size."""
    high, wide = sizes.height, sizes.width
    if sizes.flip in (5, 6):  # (a quarter turn)
        high, wide = wide, high
    rows, columns = kept(sizes, False)
    # Asked for half size, LibRaw may give the full one all the same (a Sony's medium and
    # small lossless raws: no mosaic to halve). Cut by half-size margins, that showed the
    # top left quarter of the photo, stretched over all of it.
    for by in (2, 1) if half else (1,):
        if rgb.shape[:2] == (-(-high // by), -(-wide // by)):
            return rgb[kept(sizes, by == 2)], (columns.stop - columns.start, rows.stop - rows.start)
    # Not the shape LibRaw's sizes say (a Fujifilm SuperCCD's, turned back 45° from a square of
    # 6559 × 6560 to 5430 × 3845): all of the picture as it came, which has no margins. (At
    # half size the full one is taken to be twice it: a pixel or two off either way, at
    # most, until the full size comes and says.) It was drawn squeezed into the square.
    by = 2 if half else 1
    return rgb, (rgb.shape[1] * by, rgb.shape[0] * by)


def developed(raw, half: bool) -> np.ndarray:
    """LibRaw's development as Photone takes it: linear Rec.2020, the camera's white balance,
    highlights clipped, at the white the file names (adjust_maximum off: LibRaw's default
    took the photo's brightest value for white when that was above 75 % of it, and so made
    one photo in six up to 1.29 times brighter, a sensor's white no longer its 1)."""
    import rawpy

    return raw.postprocess(half_size=half, gamma=(1, 1), no_auto_bright=True, output_bps=16,
                           use_camera_wb=True, output_color=rawpy.ColorSpace.Rec2020,
                           highlight_mode=rawpy.HighlightMode.Clip, adjust_maximum_thr=0.0)


# LibRaw's own figures (colorconst.cpp; double, D65 float), for `figures`.
XYZ_RGB = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])
D65 = np.array([0.95047, 1.0, 1.08883], np.float32).astype(float)
REC2020 = np.array([[0.627452, 0.329249, 0.043299], [0.069109, 0.919531, 0.011360],
                    [0.016398, 0.088030, 0.895572]])
F32 = np.float32


def _products(left: np.ndarray, rgb_cam: np.ndarray, divide=None) -> np.ndarray:
    """LibRaw's out[i][j] += float(left[i][k] * rgb_cam[k][j] (/ divide[i])), k = 0..2, in
    float32 from the left: how its matrices to Rec.2020 and to XYZ for AHD are made."""
    terms = left[:, :, None] * rgb_cam.astype(float)[None]
    terms = (terms if divide is None else terms / divide[:, None, None]).astype(F32)
    return terms[:, 0] + terms[:, 1] + terms[:, 2]


def _rgb_cam(cam_xyz) -> np.ndarray | None:
    """LibRaw's cam_xyz_coeff for three colors: its rows made to sum to 1, inverted. None
    where the file has no such matrix (all 0s, as LibRaw 0.22 gives a GFX100RF's)."""
    cam_rgb = np.asarray(cam_xyz, float)[:3] @ XYZ_RGB
    sums = cam_rgb.sum(axis=1, keepdims=True)
    cam_rgb = np.where(sums > 0.00001, cam_rgb / np.where(sums > 0.00001, sums, 1), 0)
    try:
        return np.linalg.inv(cam_rgb).astype(F32)
    except np.linalg.LinAlgError:
        return None


def visible(raw, sizes) -> np.ndarray:
    """The sensor's values LibRaw develops, by its sizes before developing (rawpy's
    raw_image_visible is cut by the sizes as they are now, and developing changes them)."""
    return raw.raw_image[sizes.top_margin:sizes.top_margin + sizes.height,
                         sizes.left_margin:sizes.left_margin + sizes.width]


def _cells(mosaic: np.ndarray, cell, black, scale) -> list:
    """LibRaw's half size of every 8th whole 2 x 2 cell of `mosaic` each way, before its
    matrix: (red, green, blue), each color's black off, scaled in floats, cut to ints, the two
    greens' mean."""
    rows, columns, values = len(mosaic) // 2 * 2, mosaic.shape[1] // 2 * 2, [None] * 4
    for down, across in ((0, 0), (0, 1), (1, 0), (1, 1)):
        color = cell[down, across]
        seen = np.maximum(mosaic[down:rows:16, across:columns:16].astype(np.int64)
                          - black[color], 0).astype(F32)
        values[color] = np.clip((seen * scale[color]).astype(np.int32), 0, 65535)
    return [values[0], (values[1] + values[3]) >> 1, values[2]]


def figures(raw, path: Path) -> tuple[dict, np.ndarray | None]:
    """What the GPU needs to develop this file's mosaic as LibRaw does (ui/develop*.frag, the
    port of Neuro's libraw.py: black, white balance, AHD, Rec.2020, highlights clipped, turned
    and cut to the photo), or {} where it can't: not a plain 2 x 2 mosaic of red, green and
    blue (X-Trans, a Sony's or Canon's small raws, Foveon, a linear DNG), a Panasonic (LibRaw
    fills its 0s in first), or figures that do not give LibRaw's own picture. That is checked:
    LibRaw's half-size picture (0.06 s for 33 MP; given back, for the half size, or None if
    none was made) has no demosaicing, so every 64th of its pixels is worked out from the
    mosaic here, with each matrix LibRaw may have taken (the file's own, or one made from the
    matrix from XYZ), and the one that gives them to within 2 (of 65535) is used. Before the
    file is developed at all: that changes its sizes and colors. ponytail: LibRaw's per-color
    black only, not the patterns of blacks some files carry; the check sends such a file to
    the CPU."""
    import rawpy

    sizes, black = raw.sizes, np.array(raw.black_level_per_channel, np.int64)
    wide, high = sizes.width, sizes.height
    pre = np.array(raw.camera_whitebalance, F32)
    pre[1] = pre[1] or 1
    pre[3] = pre[3] or pre[1]
    maximum = int(raw.white_level) - int(black.min())  # (LibRaw takes the common black off it)
    if (raw.raw_type != rawpy.RawType.Flat or raw.color_desc != b"RGBG" or raw.raw_pattern is None
            or raw.raw_pattern.shape != (2, 2) or sizes.pixel_aspect != 1 or min(wide, high) < 16
            or visible(raw, sizes).shape != (high, wide) or path.suffix.lower() == ".rw2"
            or not pre.min() > 0.00001 or maximum <= 0):
        return {}, None
    cell, mosaic = raw.raw_colors_visible[:2, :2].astype(int), visible(raw, sizes)
    pre = (pre / pre.min()).astype(F32)
    scale = ((pre * F32(65535)).astype(F32) / F32(maximum)).astype(F32)
    # LibRaw's picture, as it would be had it not been turned (its `flip`: 4 a quarter turn,
    # then 2 upside down and 1 left to right), every 8th cell each way.
    flip, half_rgb = sizes.flip, developed(raw, True)
    shown = half_rgb.transpose(1, 0, 2) if flip & 4 else half_rgb
    shown = shown[::-1 if flip & 2 else 1, ::-1 if flip & 1 else 1]
    if shown.shape[:2] != (-(-high // 2), -(-wide // 2)):
        return {}, half_rgb
    theirs = shown[:high // 2:8, :wide // 2:8].astype(int)
    rgb = _cells(mosaic, cell, black, scale)
    own = np.asarray(raw.color_matrix, F32)[:3, :3]
    made = _rgb_cam(raw.rgb_xyz_matrix)
    mats = ([own] if own[0, 0] > 0.125 else []) + ([made] if made is not None else [])
    tried = [(m, _products(REC2020, m)) for m in mats]
    # (a camera LibRaw has no matrix for: its "raw color", its rgb_cam left at 1s, the camera's
    # colors given as they are)
    plain = np.eye(3, dtype=F32)
    for rgb_cam, out_cam in tried + ([(plain, plain)] if made is None else []):
        ours = np.stack([np.clip((rgb[0] * out_cam[i, 0] + rgb[1] * out_cam[i, 1]
                                  + rgb[2] * out_cam[i, 2]).astype(np.int32), 0, 65535)
                         for i in range(3)], axis=-1)
        if np.abs(ours - theirs).max() <= 2:
            return {"scale": scale.tolist(), "black": black.tolist(),
                    "cell": cell.ravel().tolist(), "sensor": [wide, high], "flip": flip,
                    "xyz_cam": _products(XYZ_RGB, rgb_cam, D65).ravel().tolist(),
                    "out_cam": out_cam.ravel().tolist()}, half_rgb
    return {}, half_rgb


def _pixels(raw, sizes, path: Path, half: bool, gpu: bool) -> tuple[np.ndarray, dict | None]:
    """`develop_raw`'s pixels of the open raw file, and for its mosaic what tells the GPU how
    to develop it (`figures`, and where the photo lies in it), else None. The figures are
    worked out first, before developing changes the file, if `gpu`."""
    known = figures(raw, path)[0] if gpu else {}
    if half or not known:
        return developed(raw, half), None
    rows, columns = kept(sizes, False)
    place = [columns.start, rows.start, columns.stop - columns.start, rows.stop - rows.start]
    return visible(raw, sizes), {**known, "kept": place}


def _denoised(path: Path, half: bool) -> tuple[dict, np.ndarray] | None:
    """A photo Photone's neural denoiser wrote: no mosaic, and floats, which LibRaw would
    round. It comes with what the original's camera said (dng.read). None for another file."""
    try:
        ours = dng.read(path, half) if path.suffix.lower() == ".dng" else None
    except (OSError, ValueError) as e:
        raise Unreadable(str(e) if path.name in str(e) else f"{path.name}: {e}") from e
    if not ours:
        return None
    kept_, pixels = ours
    info = meta(path.name, pixels.shape[1], pixels.shape[0], kept_["full"])
    # (Read from a file: what it says is checked as a curve the user sent would be.)
    match = edit.clean({"curve": kept_["match"]}).get("curve")
    info.update(raw=True, camera=kept_["camera"], match=match, denoised=True)
    return info, pixels


def develop_raw(path: Path, half: bool, mosaic: bool = True) -> tuple[dict, np.ndarray]:
    """Developed no further than demosaicing and the camera's white balance: (h, w, 3) uint16
    (a photo Photone denoised: (h, w, 4) half floats, as it was written). The full size of a
    file the GPU can develop (`figures`) is not developed here: it is the sensor's mosaic,
    (h, w) uint16, and info["mosaic"] says how (ui/Develop.qml); 0.35 s of a Sony's 0.6 s
    saved, and 48 MB sent where 264 were. `mosaic` False: developed here all the same.

    Half size skips interpolation (one pixel per 2x2 sensor cell): 0.17 s against 0.47 s for a
    33 MP file, so it is what is shown first."""
    if ours := _denoised(path, half):
        return ours
    import rawpy  # slow to import, and the checks run without a raw file

    try:
        # (Not `with`: the mosaic is a view of the file's memory, which it keeps open.)
        raw = rawpy.imread(str(path))
        sizes = raw.sizes  # before developing, which changes them
        about = camera(raw.camera_whitebalance, raw.rgb_xyz_matrix)
        match = match_of(raw, about)
        rgb, gpu = _pixels(raw, sizes, path, half, mosaic and not half)
        del raw
    except rawpy.LibRawFileUnsupportedError as e:  # (its message is a bytes object's repr)
        raise Unreadable(f"{path.name}: Photone cannot read this kind of raw file yet") from e
    except (rawpy.LibRawError, OSError) as e:
        raise Unreadable(f"{path.name}: {e}") from e
    if gpu:  # the mosaic: the GPU turns it and cuts it to the photo
        info = meta(path.name, *gpu["kept"][2:])
        info["mosaic"] = gpu
    else:
        rgb, full = trimmed(rgb, sizes, half)
        info = meta(path.name, rgb.shape[1], rgb.shape[0], full)
    # (With the camera's figures for the lens that was on it, where the file has them:
    # they travel with what is known of the camera, to wherever the shader's numbers are made.
    # None where the file gives no usable matrix: many DNGs, whose LibRaw leaves it zero.)
    info.update(raw=True, camera=about and {**about, "lens": lens.figures(path)}, match=match)
    return info, rgb


def develop_raster(path: Path) -> tuple[dict, np.ndarray]:
    """A JPEG, PNG or TIFF, turned by Qt from its own color space (sRGB when it names none)
    into linear Rec.2020, upright: (h, w, 4) half floats."""
    from PySide6.QtGui import QColorSpace, QImage, QImageReader

    reader = QImageReader(str(path))
    reader.setAutoTransform(True)  # the camera's "which way is up"
    image = reader.read()
    if image.isNull():
        raise Unreadable(f"{path.name}: {reader.errorString()}")
    if not image.colorSpace().isValid():
        image.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
    image.convertTo(QImage.Format.Format_RGBX16FPx4)
    image.convertToColorSpace(QColorSpace(QColorSpace.Primaries.Bt2020,
                                          QColorSpace.TransferFunction.Linear))
    pixels = np.frombuffer(image.constBits(), np.float16).reshape(image.height(), image.width(), 4)
    return meta(path.name, image.width(), image.height()), pixels.copy()  # `image` ends here


def develop(path: Path, half: bool, mosaic: bool = True) -> tuple[dict, np.ndarray]:
    kind = path.suffix.lower()
    if kind in RAW:
        return develop_raw(path, half, mosaic)
    if kind in RASTER:
        return develop_raster(path)
    raise Unreadable(f"{path.name}: not a kind of photo Photone can open")


HALF = (np.arange(65536) / 65535).astype(np.float16)  # a 16-bit integer as a half float
BANDS = min(8, os.cpu_count() or 1)


def texels(info: dict) -> tuple[int, int, str]:
    """(columns, rows, QImage format) of the picture `store` writes: RGBA half floats, or a
    raw file's mosaic two sensor values to a texel, each as its low and its high byte
    (Develop.qml reads them; "premultiplied" so that Qt sends the bytes as they are: as
    plain RGBA it multiplied them by the "alpha", and as 16-bit gray it made them 8-bit)."""
    if "mosaic" in info:
        wide, high = info["mosaic"]["sensor"]
        return -(-wide // 2), high, "Format_RGBA8888_Premultiplied"
    return info["width"], info["height"], "Format_RGBA16FPx4"


def size_of(info: dict) -> int:
    """The bytes of the picture `store` writes."""
    columns, rows, _ = texels(info)
    return columns * rows * (4 if "mosaic" in info else 8)


def store(pixels: np.ndarray, buf) -> None:
    """Write developed pixels into `buf` as RGBA half floats: a raw file's three channels of
    16-bit integers (full scale = 1) looked up in a table, the fourth set to 1; a finished
    picture's half floats as they are.

    In one pass, straight to where they are wanted, in bands on threads (numpy lets other
    threads run meanwhile): 0.04 s for 33 MP. (Before 2026-11-17 it was four passes, 0.2 s:
    integers into four channels, Qt making them floats, a copy into shared memory.)"""
    if pixels.ndim == 2:  # a mosaic: its 16-bit values as they are, an even number to a row
        out = np.ndarray((len(pixels), -(-pixels.shape[1] // 2) * 2), np.uint16, buffer=buf)
        out[:, :pixels.shape[1]] = pixels
        out[:, pixels.shape[1]:] = 0
        return
    out = np.ndarray((*pixels.shape[:2], 4), np.float16, buffer=buf)
    if pixels.shape[2] == 4:
        out[...] = pixels
        return

    def band(k: int) -> None:
        rows = slice(len(out) * k // BANDS, len(out) * (k + 1) // BANDS)
        out[rows, :, :3] = HALF[pixels[rows]]
        out[rows, :, 3] = 1

    with ThreadPoolExecutor(BANDS) as bands:
        list(bands.map(band, range(BANDS)))


def load(path: Path, half: bool = True, mosaic: bool = False) -> tuple[dict, Pixels]:
    """In this process. The checks use it; the app goes through Workers."""
    info, pixels = develop(path, half, mosaic=mosaic)
    buf = bytearray(size_of(info))
    store(pixels, buf)
    return info, Pixels(buf)


def load_shared(path: Path, half: bool, mosaic: bool = True) -> tuple[dict, str]:
    """In the worker: the pixels go into shared memory, and only its name travels back (that
    is all of a SharedMemory that is sent; the program has it open as soon as the answer
    arrives). (Through a pipe, 264 MB of them took a quarter of a second.) `mosaic` False: a raw
    file's full size developed here, never left to the GPU."""
    info, pixels = develop(path, half, mosaic)
    shared = SharedMemory(create=True, size=size_of(info))
    store(pixels, shared.buf)
    if sys.platform == "win32":
        # Windows drops the memory when nobody has it open: held until the program surely has.
        later = threading.Timer(30, shared.close)
        later.daemon = True
        later.start()
    else:
        shared.close()
    return info, shared


class Workers:
    """Decoding in other processes. Two decode the photo that is asked for, its quick
    half-size picture and its full-size one side by side. Two more, at a lower priority and
    with fewer threads, decode ahead what is likely to be asked for next (`ahead`). Kept
    apart so that decoding ahead never stands in the way of the photo that is wanted: sharing
    two workers, a click on a photo further along waited 0.7 s behind the neighbors' decoding."""

    def __init__(self, first: Path | None = None) -> None:
        """`first`: a photo that will be asked for at once (the one named on the command
        line). Its decoding starts now, in both sizes, while the window is still being made."""
        self.pool, self.calm = self._start(), self._start(calm=True)
        self.early = {(first, half): self.pool.submit(load_shared, first, half)
                      for half in ((True, False) if first and first.suffix.lower() in RAW
                                   else (True,) if first else ())}
        self.before: dict = {}  # what is being or has been decoded ahead of being asked for

    @staticmethod
    def _start(calm: bool = False) -> ProcessPoolExecutor:
        # spawn, not fork: a copy of a process with Qt and a GPU driver in it is not safe.
        pool = ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_begin, initargs=(os.getpid(), calm))
        for _ in range(2):
            pool.submit(_warm)  # start them and import the decoders now, not at the first photo
        return pool

    def call(self, function, *args):
        try:
            return self.pool.submit(function, *args).result()
        except BrokenProcessPool:
            self.pool = self._start()  # the worker died: the next photo gets a new one
            raise Unreadable("the decoder crashed on this file") from None

    def ahead(self, wanted: list[tuple[Path, bool]], begin: bool = True) -> dict:
        """What is worth having before it is asked for (each a photo and whether its quick
        half-size picture or its full one): what is next to the open photo, so that stepping
        there shows it at once (0.3 s otherwise, and 0.7 s until it is sharp). What was
        decoded ahead and is not among them is given up; what is missing is begun, unless
        `begin` is false (a photo has just been asked for: first that one). Gives back what
        is being or has been decoded ahead, by the same keys (the session looks at what is
        done: Session.held)."""
        wanted = [(path, half) for path, half in wanted
                  if path.suffix.lower() in RAW or (half and path.suffix.lower() in RASTER)]
        for key in [key for key in self.before if key not in wanted]:
            self._give_up(self.before.pop(key))
        for key in wanted if begin else ():
            if key not in self.before:
                try:
                    self.before[key] = self.calm.submit(load_shared, *key)
                except BrokenProcessPool:
                    self.calm = self._start(calm=True)
                except RuntimeError:
                    pass  # shut down
        return self.before

    @staticmethod
    def _give_up(future) -> None:
        """A decoding nobody will ask for: not begun if it can still be helped, and the
        memory its pixels are in given back when it is done."""
        def free(done) -> None:
            if not done.cancelled() and not done.exception():
                Pixels(None, done.result()[1]).close()

        if not future.cancel():
            future.add_done_callback(free)

    def load(self, path: Path, half: bool = True, mosaic: bool = True) -> tuple[dict, Pixels]:
        # (May be called from two threads at once.) Decoded ahead, or being so: that one. Still
        # waiting its turn there: not that one, the workers for what is asked for are free.
        # `mosaic` False: as `load_shared`'s (not one decoded ahead, which may be a mosaic).
        early = self.early.pop((path, half), None) or self.before.pop((path, half), None)
        if early and not mosaic:
            self._give_up(early)
            early = None
        elif early and early.cancel():
            early = None
        try:
            info, shared = early.result() if early else self.call(load_shared, path, half, mosaic)
        except BrokenProcessPool:
            self.pool = self._start()
            raise Unreadable(f"{path.name}: the decoder crashed on this file") from None
        except Unreadable as e:
            raise Unreadable(str(e) if path.name in str(e) else f"{path.name}: {e}") from None
        size = size_of(info)  # (Windows hands out whole pages: cut to this)
        return info, Pixels(shared.buf if len(shared.buf) == size else shared.buf[:size], shared)

    def close(self) -> None:
        self.ahead([])
        for pool in (self.pool, self.calm):
            pool.shutdown(wait=False, cancel_futures=True)


def _begin(parent: int, calm: bool = False) -> None:
    """In the worker, first thing: end when the program that started it ends, however it ends.
    Without this a killed Photone leaves its worker behind for good (seen). `calm`: a worker
    that decodes ahead: it gives way to everything else, and its decoder gets a quarter of
    the cores (a 33 MP file then takes 0.64 s where it takes 0.48 s on all sixteen: most of
    that work is not shared out anyway, and nobody is waiting for it)."""
    import ctypes
    import signal

    if sys.platform == "win32":  # no such signal there: a thread waits for the program's end
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = kernel.GetCurrentProcess.restype = ctypes.c_void_p
        program = ctypes.c_void_p(kernel.OpenProcess(0x100000, False, parent))  # SYNCHRONIZE

        def outlived() -> None:
            kernel.WaitForSingleObject(program, -1)  # (at once, should it be gone already)
            os._exit(0)

        threading.Thread(target=outlived, daemon=True).start()
    else:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    # Ctrl+C in the terminal reaches every process of the program; the program ends its
    # workers itself (`Workers.close`). Each of them printed a traceback of it.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if os.getppid() != parent:  # it ended before we could ask
        os._exit(0)
    if calm:
        if sys.platform == "win32":
            me = ctypes.c_void_p(kernel.GetCurrentProcess())
            kernel.SetPriorityClass(me, 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(10)
        # (Before the decoder is loaded: it reads this once.)
        os.environ["OMP_NUM_THREADS"] = str(max(2, (os.cpu_count() or 2) // 4))


def _warm() -> None:
    import time

    import rawpy  # noqa: F401
    time.sleep(0.05)  # long enough that the second warm-up goes to the second worker


def synthetic(width: int = 240, height: int = 160) -> tuple[dict, Pixels]:
    """A test chart with no file behind it, for the checks and the shots: hue across, lightness
    down, and a gray step wedge along the bottom (neutral must stay neutral)."""
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    phase = np.array([0, 1 / 3, 2 / 3], np.float32)[None, None, :]
    hue = 0.5 + 0.5 * np.cos(2 * np.pi * (x - phase))
    rgb = (0.03 + 0.6 * (1 - y) ** 2.2) * (0.35 + 0.65 * hue)
    wedge = (np.minimum(np.floor(x * 8), 7) / 7) ** 2.2 * np.ones((1, 1, 3), np.float32)
    rgb = np.where(y > 0.82, wedge, rgb)
    info = meta("test chart", width, height)
    info.update(raw=True, camera=camera(  # a real camera's figures, so the chart develops
        [2.32, 1.0, 1.61], [[0.746, -0.2365, -0.0588], [-0.5687, 1.3442, 0.2474],
                            [-0.0624, 0.1156, 0.6584]]))
    return info, Pixels(pack(rgb))
