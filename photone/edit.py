"""What can be done to a photo, and what the pipeline shader needs to do it.

An edit is a plain dict of the values that differ from DEFAULTS. It is what the catalog keeps,
what undo steps through, and what `uniforms` turns into numbers for ui/pipeline.frag. Nothing
here touches pixels or Qt: it is small arithmetic, done once per change.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np

from photone import lens, lut

# None for the white balance means "as the camera shot it", which differs per photo; and None
# for the tone curve "the photo's own": for a raw file, the one that shows it as the camera's
# own picture shows it (`curve_of`).
STRAIGHT = [[0.0, 0.0], [1.0, 1.0]]  # the tone curve that changes nothing
# Black and white. A film look is a set of points of the B&W EQ and nothing else: around
# `hue` (degrees on the wheel: 30 red, 60 orange, 105 yellow, 145 green, 200 cyan, 262 blue,
# 330 magenta) colors become `gain` percent lighter (darker below 0), less and less out to
# `width` degrees (a bell, as the color EQ's points; where two reach the same hue they add
# up). The filters do what a colored glass before the lens does to film: its own color
# lighter, the opposite one darker. Few points and wide ones: a look that changes much within
# a few degrees of hue turns the noise of a photo's colors into speckle. Chosen by eye, not
# measured on film.
def _film(*points) -> list[dict]:
    return [{"hue": float(h), "gain": float(g), "width": float(w)} for h, g, w in points]


FILMS = {"none": [],
         "panchromatic": _film((40, 5, 90), (105, 6, 70), (262, -10, 100)),
         "red": _film((35, 28, 100), (160, -14, 80), (250, -34, 110)),
         "orange": _film((50, 20, 90), (120, 4, 60), (250, -24, 110)),
         "yellow": _film((95, 14, 90), (250, -14, 110)),
         "green": _film((140, 18, 90), (20, -12, 90), (275, -10, 90)),
         "orthochromatic": _film((35, -30, 90), (140, 6, 80), (250, 18, 110))}
# Film grain. How strong it is from black (left) to white, 0..1, is a curve as the tone
# curve is one; a response is a set of its points: Print, strongest in the middle tones and
# gone at black and white, as on a print; Negative, strongest in the shadows; Even. A
# character is an amount, a size and a roughness (a film's speed, as a look).
RESPONSES = {"print": [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]],
             "negative": [[0.0, 0.55], [0.3, 1.0], [1.0, 0.15]],
             "even": [[0.0, 1.0], [1.0, 1.0]]}
CHARACTERS = {"fine": {"grain": 20.0, "grainSize": 0.6, "grainRough": 30.0},
              "classic": {"grain": 35.0, "grainSize": 1.0, "grainRough": 50.0},
              "bold": {"grain": 55.0, "grainSize": 1.6, "grainRough": 65.0},
              "pushed": {"grain": 75.0, "grainSize": 2.3, "grainRough": 85.0}}
GRAINS = 4000.0  # grains along the picture's longer side at size 1: 1.75 pixels each at 33 MP
DEFAULTS = {"profile": "standard", "temperature": None, "tint": None,
            "exposure": 0.0, "contrast": 0.0, "brightness": 0.0, "highlights": 0.0,
            "midtones": 0.0, "shadows": 0.0, "whites": 0.0, "blacks": 0.0, "curve": None,
            "pins": [],
            "saturation": 100.0, "vibrance": 0.0, "chroma": STRAIGHT, "eq": [],
            "mono": False, "bw": FILMS["panchromatic"],
            "luts": [],
            # (Off until turned up. At 0 a setting is switched off: nothing of it is done, and
            # the full-size picture it is kept in is not made, PhotoView's `detailed`. A photo
            # started at 20, 10 and 10 at first, the owner's, set by eye on his photos: "these
            # as defaults would be good"; and then, 2026-11-22: "sharpening and all noise
            # reduction should be at 0 by default AND they should not take up VRAM at zero
            # either". Edits kept from before say what they had: catalog.DETAIL_ON.)
            "sharpen": 0.0, "sharpenRadius": 1.0, "sharpenMask": 0.0,
            "denoise": 0.0, "denoiseColor": 0.0,
            # (Off until switched on. They were on at first, the owner's choice then: "i think
            # all should be switched on by default"; and then, 2026-11-20: "make these off
            # by default". Edits kept from before say that they had them on: catalog.LENS_ON.)
            "lensDistortion": False, "lensColor": False, "lensVignetting": False,
            "defringe": 0.0,
            "grain": 0.0, "grainSize": 1.0, "grainRough": 50.0, "grainColor": 0.0,
            "grainCurve": RESPONSES["print"],
            "turns": 0.0, "angle": 0.0, "crop": [0.0, 0.0, 1.0, 1.0], "masks": []}
RANGES = {"temperature": (2000.0, 15000.0), "tint": (-150.0, 150.0), "exposure": (-5.0, 5.0),
          "contrast": (-100.0, 100.0), "brightness": (-100.0, 100.0),
          "highlights": (-100.0, 100.0), "midtones": (-100.0, 100.0), "shadows": (-100.0, 100.0),
          "whites": (-100.0, 100.0), "blacks": (-100.0, 100.0), "vibrance": (-100.0, 100.0),
          "saturation": (0.0, 200.0), "turns": (0.0, 3.0), "angle": (-15.0, 15.0),
          "sharpen": (0.0, 150.0), "sharpenRadius": (0.5, 3.0), "sharpenMask": (0.0, 100.0),
          "denoise": (0.0, 100.0), "denoiseColor": (0.0, 100.0),
          "defringe": (0.0, 100.0),
          "grain": (0.0, 100.0), "grainSize": (0.4, 4.0), "grainRough": (0.0, 100.0),
          "grainColor": (0.0, 100.0)}
# The photo's frame: quarter turns to the right, an angle (degrees) that turns the picture to
# the right inside its frame to straighten it, and the crop, [left, top, right, bottom] of the
# picture as it stands after the turns, 0..1. They are what stays when the original is shown.
FRAME = ("turns", "angle", "crop")
# The tools of the rail and the settings each one owns: what its dot, its reset and its
# "compare" are about.
TOOLS = {"tone": ("profile", "exposure", "contrast", "brightness", "highlights", "midtones",
                  "shadows", "whites", "blacks", "curve"),
         "balance": ("temperature", "tint"), "pins": ("pins", "masks"),
         "saturation": ("saturation", "vibrance"), "eq": ("eq",), "chroma": ("chroma",),
         "grain": ("grain", "grainSize", "grainRough", "grainColor", "grainCurve"),
         "lut": ("luts",),
         "detail": ("sharpen", "sharpenRadius", "sharpenMask", "denoise", "denoiseColor"),
         "lens": ("lensDistortion", "lensColor", "lensVignetting", "defringe"),
         "mono": ("bw",),  # (the look; black and white itself is a switch, no tool's setting)
         "crop": FRAME}
# The settings that do something at rest, and what they are when they do nothing: what a
# card's A/B shows the photo with. (Sharpening and noise reduction were here while a photo
# started with a little of each.)
# (The tone curve at rest is the photo's own, `curve_of`: held, the Exposure card's A/B shows
# the photo with it straight, as shot. It showed the exposure a photo started lighter at,
# which was the slider's start; the owner: "its a confusing design".)
IDLE = {"curve": STRAIGHT}
POINTS = 16     # of a tone curve: what the shader's three matrices hold
GAP = 0.01      # the least two neighboring curve points are apart, left to right
PINS = 12       # color pins a photo can have: what the shader's nine matrices hold
# A color pin: every color near `from` on the wheel (OKLab a, b) moves as `from` moves to `to`.
# `range` is how far from `from` it reaches, in chroma; `feather` how much of that is a soft
# edge (percent); `light` makes those colors lighter or darker; `variance` gathers their hues
# around the pin's (negative) or spreads them (positive); `low`..`high` is the span of
# lightness it works in. "offset" moves the colors together and can tint a gray; "rotate"
# turns the hue and scales the chroma, so a gray stays gray; "protect" moves nothing and
# keeps the other pins off its colors.
PIN = {"light": 0.0, "range": 0.08, "feather": 70.0, "variance": 0.0, "low": 0.0, "high": 1.0,
       "mode": "offset", "mask": 0}  # mask: which of the photo's masks it is kept to; 0: none
MASKS = 8       # masks a photo can have: what the shader's four matrices hold
# A mask: a part of the picture a pin can be kept to. Places are in the stored picture's own
# 0..1 (so a mask stays on what it covers when the photo is cropped or turned). "radial": an
# ellipse about `at`, one of its axes ending at `to`, the other `ratio` times as long; full
# inside, fading out over the outer `feather` of it. "linear": half at `at`, fading to nothing
# at `to` and to full as far the other way. `invert` swaps inside and outside.
MASK = {"ratio": 1.0, "feather": 0.5, "invert": False}
MASK_RANGES = {"ratio": (0.1, 10.0), "feather": (0.0, 1.0)}
# What a mask changes of the picture inside it, its "edit": for each of these settings an
# amount added to the whole image's (the owner: "masks that ... allow me to change exposure
# and all kinds of basic parameters like if i need to dim, saturate or color-correct a spot").
# A temperature in kelvin, like the whole image's; the rest in their own units.
LOCAL = {"exposure": 5.0, "contrast": 100.0, "brightness": 100.0, "highlights": 100.0,
         "midtones": 100.0, "shadows": 100.0, "whites": 100.0, "blacks": 100.0,
         "temperature": 4000.0, "tint": 150.0, "saturation": 100.0, "vibrance": 100.0}
# And these of a mask's edit are whole things of its own, a second pass on top of the whole
# image's inside the mask (two curves cannot be added like two numbers, so one follows the
# other): the tone curve, the chroma curve, the color EQ. A new mask has none: the straight
# line, no points. (Until 2026-11-18 a mask's took the whole image's place and showed the
# whole image's until it had one; the owner: "some cards have their settings cloned from
# Adjust to a newly created mask, thats not good", "filter cards are supposed to be complete
# separate passes stacked on top".)
LOCAL_WHOLE = ("curve", "chroma", "eq")
# "brush": painted by hand, as `strokes`, each a run of places in the stored picture (0..1 both
# ways) painted (or, `erase`, rubbed out) with a round brush: `size` its radius as a share of
# the picture's width, `feather` how much of that radius fades out. `painted` makes the
# strokes a raster, which is what the shader reads (the photo's painted masks in the red,
# green and blue of one texture, hence BRUSHES).
# `off`: the mask does nothing for now (the eye in the list); `name`: the user's own for it.
KINDS = ("radial", "linear", "brush")
BRUSHES = 3     # painted masks a photo can have: one color channel of the raster each
STROKES = 400   # strokes a painted mask keeps
REACH = 2000    # places in one stroke
PAINT = 2048    # the raster's longer side, in pixels
STROKE = {"size": (0.001, 0.5), "feather": (0.0, 1.0)}
PIN_RANGES = {"light": (-0.5, 0.5), "range": (0.01, 0.5), "feather": (0.0, 100.0),
              "variance": (-100.0, 100.0), "low": (0.0, 1.0), "high": (0.0, 1.0)}
MODES = ("offset", "rotate", "protect")
NEUTRAL = 0.02  # a source color with less chroma than this has no hue to turn: it is offset
VIVID = 0.4     # the chroma at the right end of the chroma curve
BANDS = 8       # points a color EQ can have: what the shader's two matrices hold
# A point of the color EQ: around `hue` (degrees on the wheel: 0 red, 90 yellow, 140 green, 260
# blue), colors get `gain` percent of their chroma: in full at that hue, half of it at half of
# `width` degrees to either side, not at all beyond `width` (`_bell`).
BAND_RANGES = {"hue": (0.0, 360.0), "gain": (0.0, 200.0), "width": (8.0, 120.0)}
BW_RANGES = {**BAND_RANGES, "gain": (-40.0, 40.0)}  # a point of the B&W EQ: percent lighter
TINTED = 0.04   # the chroma at which the B&W EQ does half of what it says: a gray has no hue
BAND = 0.05     # how softly a pin's span of lightness ends

# A profile is the base tone curve from what the sensor measured to a picture: its contrast,
# and how bright it shows a mid gray (which a sensor records at about 1/8 of its white).
# Standard is fitted to the camera's own JPEGs: over twelve Sony A7 IV shots, brightness at
# seven percentiles is within 4.4 levels (of 255) of the camera's; the first guess, (1.3, 0.19),
# was 16.8 off, darker from the midtones up. Other makers' cameras are untested.
PROFILES = {"standard": (1.6, 0.32), "flat": (1.2, 0.26)}
MID = 0.125
MID_FINISHED = 0.18  # the mid gray of a finished picture (JPEG, PNG, TIFF), in linear light

SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
XYZ_TO_2020 = np.array([[1.7166512, -0.3556708, -0.2533663],
                        [-0.6666844, 1.6164812, 0.0157685],
                        [0.0176399, -0.0427706, 0.9421031]])
IDENTITY = np.eye(3)
# A finished picture has no camera behind it, but its white can be moved all the same: as a
# gain on the eye's three kinds of cone (the Bradford matrix), starting from daylight. To the
# arithmetic below that is a camera like any other, without a sensor that clips.
BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367],
                     [0.0389, -0.0685, 1.0296]])
FINISHED = {"wb": (1 / (BRADFORD @ SRGB_TO_XYZ.sum(axis=1))).tolist(),
            "cam_xyz": BRADFORD.tolist(), "sensor": False}


def valid(key: str, value):
    """The value as the edit keeps it, or ValueError. Everything that enters an edit passes
    here: what the interface sends and what a catalog file holds."""
    if value is None and key in DEFAULTS and DEFAULTS[key] is None:
        return None  # (the photo's own: the white balance as shot, its tone curve)
    if key in SHAPES:  # a setting that is more than one number
        return SHAPES[key](value)
    if key not in RANGES:
        raise ValueError(f"no such setting: {key!r}")
    if not _number(value):
        raise ValueError(f"{key} must be a number")
    low, high = RANGES[key]
    return min(high, max(low, float(value)))


def _ranged(key: str, value) -> float:
    if not _number(value):
        raise ValueError(f"{key} must be a number")
    low, high = RANGES[key]
    return min(high, max(low, float(value)))


def valid_mono(value) -> bool:
    """A switch (black and white, a lens correction): on or off."""
    if not isinstance(value, bool):
        raise ValueError("a switch is on or off")
    return value


LUTS = 16       # LUTs a photo can have: what four of the shader's matrices hold


def valid_luts(value) -> list[dict]:
    """The LUTs laid on a photo, in the order they are laid on."""
    if not isinstance(value, (list, tuple)) or len(value) > LUTS:
        raise ValueError(f"at most {LUTS} LUTs")
    return [valid_lut(given) for given in value]


def valid_lut(value) -> dict:
    """One LUT of a photo: which of those installed (`id`: what its file is kept under, of
    letters and digits only: it becomes part of a path), its name, how many points a side
    it has, how much of it is laid on, in percent, and `off` while it is switched off."""
    if not (isinstance(value, dict) and isinstance(value.get("id"), str) and value["id"].isalnum()
            and len(value["id"]) <= 64 and isinstance(value.get("name"), str)
            and _number(value.get("size")) and _number(value.get("strength", 100.0))):
        raise ValueError("a LUT is an id, a name, a size and a strength")
    if not 2 <= value["size"] <= 65 or value["size"] != int(value["size"]):
        raise ValueError("a LUT has 2 to 65 points a side")
    out = {"id": value["id"], "name": value["name"][:80], "size": int(value["size"]),
           "strength": min(100.0, max(0.0, float(value.get("strength", 100.0))))}
    return {**out, "off": True} if value.get("off") is True else out


def valid_profile(value) -> str:
    if value not in PROFILES:
        raise ValueError(f"no such profile: {value!r}")
    return value


def _number(value) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def valid_curve(value) -> list[list[float]]:
    """A tone curve: 2 to POINTS points [in, out] of lightness, 0..1, running left to right.
    Lists, not tuples: it must compare equal to itself after a trip through JSON."""
    if not isinstance(value, (list, tuple)) or not 2 <= len(value) <= POINTS:
        raise ValueError("a curve has 2 to 16 points")
    points = []
    for point in value:
        if not (isinstance(point, (list, tuple)) and len(point) == 2 and all(map(_number, point))):
            raise ValueError("a curve point is two numbers")
        points.append([round(min(1.0, max(0.0, float(v))), 4) for v in point])
    if any(b[0] - a[0] < GAP - 1e-9 for a, b in pairwise(points)):
        raise ValueError("curve points must run left to right")
    return points


def _color(value) -> list[float]:
    """A place on the wheel: OKLab [a, b], each within half a unit of gray."""
    if not (isinstance(value, (list, tuple)) and len(value) == 2 and all(map(_number, value))):
        raise ValueError("a color on the wheel is two numbers")
    return [round(min(0.5, max(-0.5, float(v))), 5) for v in value]


def valid_pins(value) -> list[dict]:
    """Color pins as the edit keeps them: every pin whole, every number in its range. Plain
    lists, dicts and rounded numbers, so that an edit is equal to itself after JSON."""
    if not isinstance(value, (list, tuple)) or len(value) > PINS:
        raise ValueError(f"at most {PINS} pins")
    pins = []
    for given in value:
        if not isinstance(given, dict) or "from" not in given:
            raise ValueError("a pin is a source color and what happens to it")
        pin = {"from": _color(given["from"]), "to": _color(given.get("to", given["from"]))}
        for key, (low, high) in PIN_RANGES.items():
            number = given.get(key, PIN[key])
            if not _number(number):
                raise ValueError(f"a pin's {key} is a number")
            pin[key] = round(min(high, max(low, float(number))), 5)
        pin["mode"], pin["mask"] = given.get("mode", PIN["mode"]), given.get("mask", 0)
        if pin["mode"] not in MODES or pin["high"] - pin["low"] < 0.02:
            raise ValueError("no such kind of pin, or no span of lightness left")
        if isinstance(pin["mask"], bool) or pin["mask"] not in range(MASKS + 1):
            raise ValueError("a pin's mask is one of the photo's masks, or none")
        pin["mask"] = int(pin["mask"])
        pins.append(pin)
    return pins


def _place(point) -> list[float]:
    """A place in the picture (it may lie somewhat outside it: a mask's handle can)."""
    if not (isinstance(point, (list, tuple)) and len(point) == 2 and all(map(_number, point))):
        raise ValueError("a mask's place is two numbers")
    return [round(min(2.0, max(-1.0, float(v))), 5) for v in point]


def valid_masks(value) -> list[dict]:
    """The photo's masks as the edit keeps them."""
    if not isinstance(value, (list, tuple)) or len(value) > MASKS:
        raise ValueError(f"at most {MASKS} masks")
    masks = [_mask(given) for given in value]
    if sum(mask["kind"] == "brush" for mask in masks) > BRUSHES:
        raise ValueError(f"at most {BRUSHES} painted masks")
    return masks


def _mask(given) -> dict:
    if not isinstance(given, dict) or given.get("kind") not in KINDS:
        raise ValueError("a mask is radial, linear or painted")
    if given["kind"] == "brush":
        mask = {"kind": "brush", "strokes": _strokes(given.get("strokes", []))}
    else:
        mask = {"kind": given["kind"], "at": _place(given.get("at")), "to": _place(given.get("to"))}
        if math.dist(mask["at"], mask["to"]) < 0.005:
            raise ValueError("a mask needs some size")
        for key, (low, high) in MASK_RANGES.items():
            mask[key] = _within(given.get(key, MASK[key]), low, high, f"a mask's {key}")
    mask["invert"] = bool(given.get("invert", False))
    local = _local(given.get("edit", {}))
    if local:  # (kept only where there is one: a mask without is as it always was)
        mask["edit"] = local
    if given.get("off"):
        mask["off"] = True
    name = given.get("name", "")
    if isinstance(name, str) and name.strip():
        mask["name"] = name.strip()[:40]
    return mask


def _within(number, low: float, high: float, what: str) -> float:
    if not _number(number):
        raise ValueError(f"{what} is a number")
    return round(min(high, max(low, float(number))), 5)


class _Stroke(dict):
    """A stroke as it is kept: `_strokes` made it, and takes it as it is the next time. (A
    mask's strokes are all handed back at every move of the brush, with only the last one
    longer: looked through anew each time, a mask of ten thousand places cost 1.5 ms a move.)"""


def _strokes(given) -> list[dict]:
    """A painted mask's strokes as they are kept."""
    if not isinstance(given, (list, tuple)) or len(given) > STROKES:
        raise ValueError(f"at most {STROKES} strokes")
    out = []
    for stroke in given:
        if type(stroke) is _Stroke:
            out.append(stroke)
            continue
        if not isinstance(stroke, dict):
            raise ValueError("a stroke is a brush and the places it went")
        try:
            points = np.asarray(stroke.get("points", []), dtype=float)
        except (TypeError, ValueError):
            raise ValueError("a stroke's places are numbers") from None
        if points.ndim != 2 or points.shape[1] != 2 or not 1 <= len(points) <= REACH \
                or not np.isfinite(points).all():
            raise ValueError("a stroke is one place or more, two numbers each")
        out.append(_Stroke({**{key: _within(stroke.get(key), low, high, f"a brush's {key}")
                               for key, (low, high) in STROKE.items()},
                            "erase": bool(stroke.get("erase", False)),
                            "points": np.round(np.clip(points, -1.0, 2.0), 4).tolist()}))
    return out


def raster_size(shape: float) -> tuple[int, int]:
    """How large the raster of a picture's painted masks is: (width, height)."""
    shape = shape or 1.0
    if shape >= 1:
        return PAINT, max(1, round(PAINT / shape))
    return max(1, round(PAINT * shape)), PAINT


class Painter:
    """A painted mask's raster as it is being made: `canvas` (heights by widths, 0..1), and
    what is needed to go on with the stroke under way.

    From each place of a stroke to the next the brush leaves a straight band with round
    ends: every pixel by how far it is from that line, whole within the brush's core and
    fading over its feather. (At first the brush was pressed down every quarter of its
    radius, which showed as scallops along a soft edge; the owner: "a straight line has
    circle artifacts".)

    How bands are put together. Along a stroke, one stretch and the next take the larger of
    the two (or the band would be harder at every joint). But a stroke is cut into pieces
    some brush widths long, and a piece is laid onto all that is older than the piece before
    it as paint onto paint: what is left clear is the product of what each leaves clear. So
    where a soft edge lies over another, of another stroke or of this one coming back, they
    run into one another smoothly. (With the larger of the two everywhere, two soft edges
    met in a crease, and the gaps between passes had hard pointed ends; the owner: "there
    cannot be any sharp patches like that".) Rubbing out is the same with clear for paint.

    ponytail: each stretch costs its bounding box (a brush of 5 %: a millisecond; at its
    largest, a million pixels and some 30 ms), and a mask of hundreds of long strokes takes
    a moment to paint anew when a photo is opened. So a stroke is painted by the GPU while
    it is made (ui/brush.frag, by the same rule), and by this only once it is over."""

    def __init__(self, shape: float) -> None:
        wide, high = raster_size(shape)
        self.canvas = np.zeros((high, wide), dtype=np.float32)
        self.under = self.before = self.piece = None
        # The same as bytes, and where the canvas has changed since they were made (`bytes`).
        self.shown, self.stale, self.fresh = np.zeros((high, wide), dtype=np.uint8), None, None

    def begin(self) -> None:
        """A stroke begins: nothing of it is there yet."""
        self.under = self.canvas.copy()          # all but the piece under way
        self.before = np.zeros_like(self.canvas)  # the piece before the one under way
        self.piece = np.zeros_like(self.canvas)   # the one under way
        self.gone, self.reach, self.reached = 0.0, None, None  # (where each of the two pieces is)

    def add(self, stroke: dict, start: int = 0) -> None:
        """Go on with the stroke begun, from its place `start`."""
        high, wide = self.canvas.shape
        radius, core = brush_of(stroke, wide)
        steps, gone = stretches(stroke, wide, high, start, self.gone)
        for a, b, turn in steps:
            self._stretch(a, b, radius, core, stroke["erase"])
            if turn:  # the next stretch begins another piece
                self._turn()
        self.gone = gone

    def _turn(self) -> None:
        """The piece under way becomes the one before, and a new one begins. Only where the
        two pieces are is anything moved: whole rasters copied at every piece made a fast
        stroke with a small brush stutter (some twenty megabytes every few frames)."""
        here, there = self.reach, self.reached
        if there:
            self.before[there] = 0
        if here:
            self.under[here] = self.canvas[here]
            self.before[here] = self.piece[here]
            self.piece[here] = 0
        self.reached, self.reach, self.gone = here, None, 0.0

    def _stretch(self, a, b, radius: float, core: float, erase: bool) -> None:
        high, wide = self.canvas.shape
        low = np.floor(np.minimum(a, b) - radius).astype(int)
        top = np.ceil(np.maximum(a, b) + radius).astype(int) + 1
        box = (slice(max(0, low[1]), min(high, top[1])), slice(max(0, low[0]), min(wide, top[0])))
        if box[0].start >= box[0].stop or box[1].start >= box[1].stop:
            return
        was_at = self.reach  # (the piece under way reaches as far as its stretches' boxes)
        self.reach = box if was_at is None else _around(was_at, box)
        self.stale = box if self.stale is None else _around(self.stale, box)
        paint, far = _band(box, a, b, radius, core)
        piece = self.piece[box]
        np.maximum(piece, paint, out=piece)
        # This piece and the one before it as one (the larger), less what the one before
        # put there already: that much paint, or that much rubbed out, onto what is under.
        was = self.before[box]
        more = np.maximum(piece, was, out=paint)
        more -= was
        more /= np.maximum(1 - was, np.float32(1e-6), out=far)
        under = self.under[box]
        if erase:
            np.subtract(1, more, out=more)
            np.multiply(under, more, out=self.canvas[box])
        else:
            np.subtract(1, under, out=far)
            more *= far
            more -= far                        # -(1 - under) (1 - more)
            np.add(1, more, out=self.canvas[box])

    def bytes(self) -> np.ndarray:
        """The raster as bytes. Only where it has changed since the last time is worked out
        (`fresh`: that box, or none): the whole of it at every move of the brush was 3 ms."""
        self.fresh, self.stale = self.stale, None
        if self.fresh:
            self.shown[self.fresh] = np.round(self.canvas[self.fresh] * 255)
        return self.shown


PIECE = 6.0  # a piece of a stroke, in brush radii of the way gone


def brush_of(stroke: dict, wide: int) -> tuple:
    """A stroke's brush on a raster that wide: (its radius in pixels, the share of it whole)."""
    return max(0.5, stroke["size"] * wide), 1 - max(stroke["feather"], 1e-4)


def stretches(stroke: dict, wide: int, high: int, start: int = 0, gone: float = 0.0) -> tuple:
    """A stroke from its place `start` on, as what is painted one after another on a raster
    that large: ([from, to, whether a piece ends there], the way gone in the piece under way).
    Painter paints by this, and so does the shader that paints a stroke while it is made
    (ui/brush.frag): the two must put the same pieces together."""
    radius = brush_of(stroke, wide)[0]
    points = np.asarray(stroke["points"][max(0, start - 1):], dtype=float) * (wide, high)
    steps = []
    for a, b in list(zip(points, points[1:], strict=False)) or [(points[0], points[0])]:
        gone += float(np.hypot(*(b - a)))
        turn = gone > PIECE * radius
        steps.append((a, b, turn))
        if turn:
            gone = 0.0
    return steps, gone


def _band(box: tuple, a, b, radius: float, core: float) -> tuple:
    """What the brush leaves from a to b within that box (two slices): 1 in its core, fading
    to its edge; and an array as large to work in. (A pixel is its middle. A column and a row
    of 32-bit numbers, not two grids of 64, and in place: this runs at every move of the brush.)"""
    rows, columns = (np.arange(side.start, side.stop, dtype=np.float32) for side in box)
    ys, xs = rows[:, None] + np.float32(0.5 - a[1]), columns[None, :] + np.float32(0.5 - a[0])
    along = (b - a).astype(np.float32)
    long = np.float32(along @ along)
    t = np.clip((xs * along[0] + ys * along[1]) / long, 0, 1) if long else np.float32(0)
    far = np.hypot(xs - t * along[0], ys - t * along[1])   # how far each is from the stretch
    far *= np.float32(1 / (radius * (1 - core)))
    far -= np.float32(core / (1 - core))
    np.clip(far, 0, 1, out=far)            # smoothstep from the core to the brush's edge,
    paint = far * far
    far *= -2
    far += 3
    paint *= far
    np.subtract(1, paint, out=paint)       # and the paint is what is left of it
    return paint, far


def _around(one: tuple, other: tuple) -> tuple:
    """The box (two slices) that holds both."""
    return tuple(slice(min(old.start, new.start), max(old.stop, new.stop))
                 for old, new in zip(one, other, strict=True))


def painted(strokes: list[dict], shape: float) -> np.ndarray:
    """A painted mask as a raster: 0 where it is not, 255 where it is whole."""
    painter = Painter(shape)
    for stroke in strokes:
        painter.begin()
        painter.add(stroke)
    return painter.bytes()


def _local(given) -> dict:
    """A mask's own edit as it is kept: the amounts that are numbers, within their reach,
    and not nothing."""
    if not isinstance(given, dict):
        raise ValueError("a mask's edit is settings and their amounts")
    out = {}
    for key, reach in LOCAL.items():
        amount = given.get(key, 0.0)
        if not _number(amount):
            raise ValueError(f"a mask's {key} is a number")
        amount = round(min(reach, max(-reach, float(amount))), 5)
        if amount:
            out[key] = amount
    for key in LOCAL_WHOLE:
        if key in given:
            out[key] = valid(key, given[key])
    return out


def merged(edit: dict, local: dict, camera: dict | None) -> dict:
    """The edit of the picture inside a mask: the whole image's with the mask's amounts
    added, each kept within what the setting can be, and as `over` the mask's own curves and
    EQ, which come on top. Where the whole image's says nothing (the white balance as shot),
    that is what is added to."""
    full, out = {**DEFAULTS, **edit}, dict(edit)
    shot = (camera.get("shot") or as_shot(camera)) if camera else (6500.0, 0.0)
    rest = {"temperature": shot[0], "tint": shot[1]}
    for key, amount in local.items():
        if key in LOCAL_WHOLE:  # on top of the whole image's, which stays (over_uniforms)
            out["over"] = {**out.get("over", {}), key: amount}
            continue
        low, high = RANGES[key]
        base = rest[key] if full[key] is None else full[key]
        out[key] = min(high, max(low, base + amount))
    return out


def layers(edit: dict, camera: dict | None) -> list[tuple[int, dict]]:
    """(mask's number, the edit inside it) for every mask that changes anything: what is
    developed a second time and laid over the picture by the mask's weight."""
    return [(n + 1, merged(edit, mask["edit"], camera))
            for n, mask in enumerate(edit.get("masks", []))
            if mask.get("edit") and not mask.get("off")]


def valid_crop(value) -> list[float]:
    """A crop: [left, top, right, bottom] within the picture, with something left of it."""
    if not (isinstance(value, (list, tuple)) and len(value) == 4 and all(map(_number, value))):
        raise ValueError("a crop is four numbers")
    left, top, right, bottom = (round(min(1.0, max(0.0, float(v))), 5) for v in value)
    if right - left < 0.02 or bottom - top < 0.02:
        raise ValueError("a crop must leave something of the picture")
    return [left, top, right, bottom]


def valid_eq(value, ranges=BAND_RANGES) -> list[dict]:
    """The points of the color EQ (or, with its ranges, of the B&W EQ) as the edit keeps
    them: each a hue, a gain and a width."""
    if not isinstance(value, (list, tuple)) or len(value) > BANDS:
        raise ValueError(f"at most {BANDS} points")
    points = []
    for given in value:
        if not isinstance(given, dict) or not all(_number(given.get(key)) for key in ranges):
            raise ValueError("a point of an EQ is a hue, a gain and a width")
        point = {key: round(min(high, max(low, float(given[key]))), 3)
                 for key, (low, high) in ranges.items()}
        point["hue"] = round(float(given["hue"]) % 360, 3)  # once round the wheel is the same hue
        points.append(point)
    return points


SHAPES = {"profile": valid_profile, "curve": valid_curve, "chroma": valid_curve,
          "grainCurve": valid_curve, "luts": valid_luts,
          "pins": valid_pins, "eq": valid_eq, "crop": valid_crop, "masks": valid_masks,
          "mono": valid_mono, "bw": lambda value: valid_eq(value, BW_RANGES),
          "lensDistortion": valid_mono, "lensColor": valid_mono, "lensVignetting": valid_mono,
          "turns": lambda value: float(round(_ranged("turns", value)) % 4)}


def clean(values) -> dict:
    """An edit read from somewhere untrusted: only what is known and valid, and not default."""
    out = {}
    if isinstance(values, dict) and "lut" in values and "luts" not in values:
        values = {**values, "luts": [values["lut"]]}  # (a photo had one LUT, for a day)
    for key, value in (values.items() if isinstance(values, dict) else ()):
        try:
            value = valid(key, value)
        except ValueError:
            continue
        if value != DEFAULTS[key]:
            out[key] = value
    return out


# ---- white balance -----------------------------------------------------------------------

def planck_xy(kelvin: float) -> tuple[float, float]:
    """Where a glowing body of that temperature sits in CIE xy (Kang et al., 1667-25000 K)."""
    t = min(25000.0, max(1667.0, kelvin))
    if t <= 4000:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2222:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return x, y


def _uv(x: float, y: float) -> np.ndarray:
    """CIE 1960 uv, where a step across the temperature line is a step of tint."""
    d = -2 * x + 12 * y + 3
    return np.array([4 * x / d, 6 * y / d])


TINT = 3000.0  # tint units per unit of uv: +150 is a strong magenta, as in other raw editors


def white_xyz(kelvin: float, tint: float) -> np.ndarray:
    """The color of light of that temperature and tint, Y = 1. A positive tint is greener
    light, so that raising the tint setting makes the picture more magenta, as raising the
    temperature setting makes it warmer: both name the light to be taken out."""
    here, ahead = _uv(*planck_xy(kelvin)), _uv(*planck_xy(kelvin + 1))
    along = (ahead - here) / np.linalg.norm(ahead - here)
    u, v = here + np.array([along[1], -along[0]]) * (tint / TINT)  # toward green
    x, y = 3 * u / (2 * u - 8 * v + 4), 2 * v / (2 * u - 8 * v + 4)
    return np.array([x / y, 1.0, (1 - x - y) / y])


def light(cam_xyz, neutral) -> tuple[float, float]:
    """The temperature and tint of the light under which the camera saw `neutral` (its three
    channels' answer to something white or gray)."""
    xyz = np.linalg.inv(np.array(cam_xyz)) @ np.array(neutral)
    target = _uv(xyz[0] / xyz.sum(), xyz[1] / xyz.sum())

    def off(mired: float) -> float:
        return float(np.linalg.norm(_uv(*planck_xy(1e6 / mired)) - target))

    mired = min(range(40, 600), key=off)  # coarse, then around the best
    mired = min((mired + step / 20 for step in range(-20, 21)), key=off)
    kelvin = round(1e6 / mired / 10) * 10.0
    here, ahead = _uv(*planck_xy(kelvin)), _uv(*planck_xy(kelvin + 1))
    along = (ahead - here) / np.linalg.norm(ahead - here)
    tint = float(np.dot(target - here, [along[1], -along[0]])) * TINT
    return kelvin, float(round(tint))


def as_shot(camera: dict) -> tuple[float, float]:
    """The temperature and tint the camera's own white balance stands for."""
    return light(camera["cam_xyz"], 1 / np.array(camera["wb"]))


def neutral_at(camera: dict, rgb) -> tuple[float, float] | None:
    """The temperature and tint that make this color a gray ("pick neutral"): `rgb` is a pixel
    as decoded, linear Rec.2020. None where the pixel says nothing (a channel at or below
    zero)."""
    seen = np.linalg.inv(camera_matrix(camera)) @ np.array(rgb, dtype=float)
    if seen.min() <= 0:
        return None
    return light(camera["cam_xyz"], seen / np.array(camera["wb"]))


def camera_matrix(camera: dict) -> np.ndarray:
    """From the camera's own colors (white-balanced) to linear Rec.2020, as LibRaw builds it:
    checked against LibRaw's output to five decimals."""
    cam_rgb = np.array(camera["cam_xyz"]) @ SRGB_TO_XYZ
    cam_rgb /= cam_rgb.sum(axis=1, keepdims=True)
    return (XYZ_TO_2020 @ SRGB_TO_XYZ) @ np.linalg.inv(cam_rgb)


def balance(edit: dict, camera: dict | None) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """(to the camera's colors, back from them, [gain r, g, b, ceiling]).

    White balance is a gain on each of the camera's own channels, so the pixels go back there
    for it. The gain is relative to the camera's own white balance, which the decoder already
    applied: at the defaults it is exactly 1."""
    if not camera:
        return IDENTITY, IDENTITY, [1.0, 1.0, 1.0, 1e9]  # no sensor: nothing to cut off
    shot = camera.get("shot") or as_shot(camera)  # the caller may keep it: it is a search
    kelvin = shot[0] if edit.get("temperature") is None else edit["temperature"]
    tint = shot[1] if edit.get("tint") is None else edit["tint"]
    cam_xyz = np.array(camera["cam_xyz"])

    def gains(k: float, t: float) -> np.ndarray:
        g = 1 / (cam_xyz @ white_xyz(k, t))
        return g / g[1]

    gain = gains(kelvin, tint) / gains(*shot)
    out = camera_matrix(camera)
    # ponytail: a channel the sensor clipped stays clipped, so all three are cut at the lowest
    # gain to keep blown highlights neutral (a little gray rather than tinted). The upgrade is
    # rebuilding clipped channels from the others, in the decoder.
    ceiling = float(min(gain.min(), 1.0)) if camera.get("sensor", True) else 1e9
    return np.linalg.inv(out), out, [*map(float, gain), ceiling]


# ---- the base tone curve ------------------------------------------------------------------

def curve(profile: str) -> tuple[float, float, float]:
    """(c, s, k) of y = k * x^c / (x^c + s): mid gray lands where the profile says and the
    sensor's white lands on the display's white."""
    c, mid = PROFILES[profile]
    a = MID ** c
    s = a * (1 - mid) / (mid - a)
    return c, s, 1 + s


def _eased(t):
    """0 at 0 and 1 at 1, even between, eased in over the first three tenths: how Highlights
    and Whites grow (the shader's `eased`). A smoothstep here flattens the middle of what it
    covers and leaves the top as it was, which looks harsh."""
    t = np.clip(t, 0, 1)
    return np.where(t < 0.3, t * t / 0.6, t - 0.15) / 0.85


def tone_at(x, u: dict):
    """What is shown for neutral light `x` (a share of the sensor's white, as recorded; a
    number or an array), by the shader's numbers `u`, as the shader's `develop` does it with
    the white balance as shot: exposure; contrast and shadows as gains in stops from mid
    gray; highlights, such a gain on what those made, and midtones, a gain round mid gray, on
    what that made; the profile's curve; then Blacks and Whites, the two ends of what is
    shown, mid gray staying put (Blacks takes light away below it, most at black, or adds
    some; Whites is a gain in stops above it, most at white, growing with the stops above mid
    gray); then HDR. In HDR (`hdrLift`, the display's peak less one) what is shown
    climbs on by the square of how light it is, so that the sensor's white lands on the peak:
    all of the picture is deeper, a shadow a little, white by the whole headroom. That is the
    last thing done, after brightness and (in the shader) after the tone curve and the colors'
    tools, which so work on the same picture in HDR as in SDR.

    ponytail: the square is judged on a display with one stop above white (the owner's: 500
    nits white, 1015 peak). With two or three stops the light midtones may rise too fast; the
    upgrade is a power that grows with the peak."""
    shown, gray = np.maximum(np.asarray(x, float), 0) * u["gain"], u["mid"]
    stops = np.log2(np.maximum(shown, 1e-6) / gray)
    # (Shadows up fades out toward black, none of it eight stops under mid gray: all of it
    # there turned the blacks milky and brought their noise up. Down, it does not.)
    dark = (1 - _step(-4.5, 0.5, stops)) * (_step(-8.0, -3.0, stops) if u["shadows"] > 0 else 1.0)
    # (Contrast down counts the stops from mid gray less and less toward the two ends, none
    # eight stops under and none at white: counted in full, black was lifted three stops and
    # white taken down one and a half, a milky gray picture.)
    # (Up, it counts them less and less too, none at white and none eight stops under: both
    # ends stay where they are and what lies between is steeper, half as steep again round
    # mid gray at +100. Counted in full it showed all from a stop and a half above mid gray
    # as white.)
    lights = _step(0.5 if u["contrast"] < 0 else 0.0, -np.log2(gray), stops)
    far = 1 - np.where(stops > 0, lights, _step(1.0, 8.0, -stops))
    spread = stops * far
    shown = shown * 2.0 ** (u["contrast"] * spread + u["shadows"] * dark)
    stops = np.log2(np.maximum(shown, 1e-6) / gray)
    shown = shown * 2.0 ** (u["highlights"] * _eased((stops + 0.5) / 4.5))
    away = np.abs(np.log2(np.maximum(shown, 1e-6) / gray))
    shown = shown * 2.0 ** (u["midtones"] * (1 - _step(0.0, 3.5, away)))
    if u["curveC"] > 0:
        p, middle = shown ** u["curveC"], u["mid"] ** u["curveC"]
        shown = u["curveK"] * p / (p + u["curveS"])
        gray = u["curveK"] * middle / (middle + u["curveS"])
    shown = np.maximum(shown - u["blacks"] * (1 - _step(0.0, gray, shown)), 0)
    above = np.log2(np.maximum(shown, 1e-6) / gray) / -np.log2(gray)
    shown = shown * 2.0 ** (u["whites"] * _eased(above))
    shown = shown * np.clip(shown, 1e-6, 1.0) ** u["brightness"]
    # (HDR last: in the shader after the colors' tools too.)
    return shown + u["hdrLift"] * np.maximum(shown, 0) ** 2


# ---- the tone curve the user draws -----------------------------------------------------------

def slopes(points: list[list[float]]) -> list[float]:
    """The curve's slope at each point, for a smooth line through them that never swings past
    a point (monotone cubic: between two points it stays between their heights)."""
    rise = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in pairwise(points)]
    wide = [b[0] - a[0] for a, b in pairwise(points)]
    # At a point between two others: the slope of the parabola through the three (each
    # neighbor counting by how near it is), which bends as much on one side of the point as
    # on the other; held to three times the flatter of the two rises, which is as far as it
    # can go without the line swinging past a point, and flat where the line turns back.
    # (It was the harmonic mean of the two rises, which is always nearer the flatter one:
    # the line left a point gently on that side and with a sharp bend on the other; the
    # owner: "on one side, its properly interpolated, but on the other ... a way sharper
    # turn".)
    inner = [0.0 if a * b <= 0 else
             math.copysign(min(abs((a * h1 + b * h0) / (h0 + h1)), 3 * min(abs(a), abs(b))), a)
             for (a, b), (h0, h1) in zip(pairwise(rise), pairwise(wide), strict=True)]
    if not inner:
        return [rise[0], rise[0]]

    def end(rise_here: float, slope_next: float) -> float:  # bends on, never back
        slope = (3 * rise_here - slope_next) / 2
        if slope * rise_here <= 0:
            return 0.0
        return math.copysign(min(abs(slope), 3 * abs(rise_here)), slope)

    return [end(rise[0], inner[0]), *inner, end(rise[-1], inner[-1])]


def curve_at(points, x: float) -> float:
    """The tone curve's output for lightness x. The same arithmetic as ui/pipeline.frag:
    flat outside the end points, a cubic between two neighbors."""
    slope = slopes(points)
    if x <= points[0][0]:
        return points[0][1]
    for ((x0, y0), (x1, y1)), (m0, m1) in zip(pairwise(points), pairwise(slope), strict=True):
        if x <= x1:
            h = x1 - x0
            t = (x - x0) / h
            return ((2 * t**3 - 3 * t**2 + 1) * y0 + (t**3 - 2 * t**2 + t) * h * m0
                    + (-2 * t**3 + 3 * t**2) * y1 + (t**3 - t**2) * h * m1)
    return points[-1][1]


# OKLab's lightness of a linear Rec.2020 color: what the tone curve works on (ui/oklab.glsl).
TO_LMS = np.array([[0.6166703, 0.3602644, 0.0230653], [0.2650725, 0.6358695, 0.0990580],
                   [0.1000613, 0.2038777, 0.6960610]])
LIGHTNESS = np.array([0.2104542553, 0.7936177850, -0.0040720468])
# The shares of a picture (percent, from its darkest) at which `matched` lines up two pictures.
MATCH = (2, 10, 25, 50, 75, 90, 98)


def lightness(rgb) -> np.ndarray:
    """The lightness of linear Rec.2020 colors (..., 3), as the shader's tone curve sees it."""
    return np.cbrt(np.asarray(rgb, float) @ TO_LMS.T) @ LIGHTNESS


def curve_of(edit: dict, camera: dict | None) -> list[list[float]]:
    """The tone curve: what the edit says; or, where it says nothing, the photo's own (the
    camera's `match`, decode.match_of: the curve that shows a raw file as the camera's own
    picture shows it, at Exposure 0 and with the Standard profile, whichever profile is
    chosen); straight where there is none (a finished picture, a sensor that is no mosaic of
    red, green and blue, a file with no preview).

    How it came to this, 2026-11-20. A photo started lighter than shot as far as its picture
    had room, never lower and by two stops at most (the rule of 2026-11-17, `catalog._start_5`):
    of 200 of the owner's photos it showed half more than two thirds of a stop lighter than
    the camera's own picture, and 11 % more than a stop and a half; "i think it overshoots by
    a lot where i think ART never does". ART never moves Exposure: its default for raw files
    draws this curve ("i want it exactly like in art"). Of 150 photos, with it the middle of
    the picture is within 0.001 of the camera's, and 5 have over 2 % more of the picture at
    white than the camera's picture has (22 under the rule before).

    ponytail: an edit that names no curve follows `matched`, so a change to it changes every
    such photo, and the catalog has to write out what they had (`Catalog._keep`, four times
    for Exposure so far). The upgrade is to keep the curve with the edit when it is made."""
    given = edit.get("curve")
    return given if given is not None else (camera or {}).get("match") or STRAIGHT


def matched(ours, theirs) -> list[list[float]]:
    """The tone curve that makes a picture of `ours` lightnesses (0..1, any shape) as light
    as one of `theirs`: where a share of ours is darker than x, the same share of theirs is
    darker than y, and the curve goes through (x, y), at seven shares (`MATCH`). It is how
    ART starts a raw photo: it never touches Exposure, it draws the curve that takes its own
    plain rendering to the picture the camera put in the file (its histmatching.cc, which
    keeps five points of the mapping; the owner: "i want it exactly like in art"). As there,
    the curve never cuts off: it starts at black and ends at white, a point that would lie
    nearer than 0.04 to the one before, lower than it, or at either end of what can be shown
    is left out, and the line between the points never swings past them (`slopes`)."""
    points = [[0.0, 0.0]]
    for x, y in zip(np.percentile(ours, MATCH), np.percentile(theirs, MATCH), strict=True):
        if x - points[-1][0] >= 0.04 and x <= 0.96 and max(points[-1][1], 0.005) <= y <= 0.97:
            points.append([round(float(x), 4), round(float(y), 4)])
    return [*points, [1.0, 1.0]]


LINE = 257  # heights a curve is drawn from (65 cut the corners of a steep curve: the owner saw)


def curve_line(points) -> list[float]:
    """A curve as LINE heights, left to right, for drawing it: `curve_at` at each, with the
    slopes worked out once."""
    slope, xs = slopes(points), np.linspace(0.0, 1.0, LINE)
    out = np.where(xs <= points[0][0], points[0][1], points[-1][1]).astype(float)
    for ((x0, y0), (x1, y1)), (m0, m1) in zip(pairwise(points), pairwise(slope), strict=True):
        here = (xs > x0) & (xs <= x1)
        h, t = x1 - x0, (xs[here] - x0) / (x1 - x0)
        out[here] = ((2 * t**3 - 3 * t**2 + 1) * y0 + (t**3 - 2 * t**2 + t) * h * m0
                     + (-2 * t**3 + 3 * t**2) * y1 + (t**3 - t**2) * h * m1)
    return out.tolist()


def curve_uniforms(points: list[list[float]]) -> dict:
    """The curve for the shader: POINTS x, y and slopes (the last point repeated to fill)."""
    slope = slopes(points)
    fill = POINTS - len(points)
    return {"curveOn": 0.0 if points == STRAIGHT else 1.0,
            "curveX": [p[0] for p in points] + [points[-1][0]] * fill,
            "curveY": [p[1] for p in points] + [points[-1][1]] * fill,
            "curveM": slope + [0.0] * fill}


def luts_uniforms(luts: list[dict]) -> dict:
    """A photo's LUTs for the shader, four to a matrix, one of four numbers each: how many
    points a side, how much of it is laid on (none while it is off), and the row its slices
    begin at in the picture they are all read from (photone/lut.py: `stacked` one under the
    other, in the order of `bands`)."""
    rows, tops = [], lut.bands(luts)
    for one in luts:
        much = 0.0 if one.get("off") else one["strength"] / 100
        rows += [float(one["size"]), much, float(tops[one["id"], one["size"]]), 0.0]
    rows += [0.0] * (LUTS * 4 - len(rows))
    return {"lutCount": float(len(luts)),
            **{f"luts{letter}": rows[16 * n:16 * n + 16] for n, letter in enumerate("ABCD")}}


def grain_uniforms(full: dict, shape: float) -> dict:
    """Film grain for the shader: `grain` is how far the lightest and darkest grains are from
    the picture's own lightness (OKLab, at the response's full strength; 100 is 0.15 either
    way); `grainWide` and `grainHigh` how many grains lie across and down the stored picture
    (whose width over height is `shape`); the response curve as the tone curve is handed
    over. Not colored for a black and white photo."""
    along, shape = GRAINS / full["grainSize"], shape if shape > 0 else 1.0  # (no photo yet: 0)
    curve = curve_uniforms(full["grainCurve"])
    return {"grain": full["grain"] / 100 * 0.3,
            "grainWide": along * min(1.0, shape), "grainHigh": along * min(1.0, 1 / shape),
            "grainRough": full["grainRough"] / 100,
            "grainColor": 0.0 if full["mono"] else full["grainColor"] / 100,
            **{f"grain{name[5:]}": curve[name] for name in ("curveX", "curveY", "curveM")}}


# ---- color pins ------------------------------------------------------------------------------

def _turned(color: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * color[0] - s * color[1], s * color[0] + c * color[1]])


def _between(angle: float) -> float:
    """An angle as the short way round: within half a turn of zero."""
    return angle - 2 * math.pi * round(angle / (2 * math.pi))


def _step(low: float, high: float, x):
    """GLSL's smoothstep, for a number or an array."""
    t = np.clip((x - low) / (high - low), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def mask_weight(mask: dict, place, shape: float) -> float:
    """How much of a mask a place of the stored picture (0..1 both ways) is in. `shape`: the
    stored picture's width over its height, so that an ellipse is one in pixels."""
    if mask.get("off"):
        return 0.0
    if mask["kind"] == "brush":  # the raster, read between its pixels as the GPU reads it
        canvas = painted(mask["strokes"], shape).astype(float) / 255
        most = np.array(canvas.shape) - 1.001
        y, x = np.clip(np.array(place[::-1]) * canvas.shape - 0.5, 0, most)
        r, c, fy, fx = int(y), int(x), y - int(y), x - int(x)
        weight = float(np.array([1 - fy, fy]) @ canvas[r:r + 2, c:c + 2] @ np.array([1 - fx, fx]))
        return 1 - weight if mask["invert"] else weight
    wide = np.array([shape, 1.0])
    start, axis = np.array(mask["at"]) * wide, (np.array(mask["to"]) - np.array(mask["at"])) * wide
    here, long = np.array(place) * wide - start, float(np.hypot(*axis))
    along = float(here @ axis) / (long * long)
    if mask["kind"] == "linear":
        weight = 1 - _step(-1.0, 1.0, along)
    else:
        across = float(here @ np.array([-axis[1], axis[0]])) / (long * long * mask["ratio"])
        weight = 1 - _step(1 - max(mask["feather"], 1e-4), 1.0, math.hypot(along, across))
    return 1 - weight if mask["invert"] else weight


def pin_weight(pin: dict, lab) -> float:
    """How much of a pin a color (lightness, a, b) gets: all of it within the pin's core,
    none beyond its range or outside its span of lightness, smoothly in between. (Its mask,
    if it has one, comes on top: `pins_at`.)"""
    far = math.dist(lab[1:], pin["from"])
    core = min(pin["range"] * (1 - pin["feather"] / 100), pin["range"] - 1e-4)
    weight = 1 - _step(core, pin["range"], far)
    if pin["low"] > 0:
        weight *= _step(pin["low"] - BAND, pin["low"] + BAND, lab[0])
    if pin["high"] < 1:
        weight *= 1 - _step(pin["high"] - BAND, pin["high"] + BAND, lab[0])
    return weight


def pins_at(pins: list[dict], lab, masked=None) -> np.ndarray:
    """What the pins make of a color (lightness, a, b). The same arithmetic as
    ui/pipeline.frag: each pin moves the color as it was, the moves add up, and what a
    protecting pin covers is moved that much less. `masked`: for each of the photo's masks,
    how much of it the pixel is in (`mask_weight`); a pin with a mask works that much."""
    lab = np.array(lab, dtype=float)
    moved, shield = np.zeros(3), 0.0
    for pin in pins:
        weight = pin_weight(pin, lab) * _masked(pin, masked)
        if pin["mode"] == "protect":
            shield = max(shield, weight)
            continue
        source, target, color = np.array(pin["from"]), np.array(pin["to"]), lab[1:]
        hue = math.hypot(*source) >= NEUTRAL  # the source has one
        if hue and pin["variance"] and math.hypot(*color) > 1e-6:
            apart = _between(math.atan2(color[1], color[0]) - math.atan2(source[1], source[0]))
            moved[1:] += _turned(color, weight * pin["variance"] / 100 * apart) - color
        if pin["mode"] == "rotate" and hue:
            turn = _between(math.atan2(target[1], target[0]) - math.atan2(source[1], source[0]))
            grown = 1 + weight * (math.hypot(*target) / math.hypot(*source) - 1)
            moved[1:] += _turned(color, weight * turn) * grown - color
        else:
            moved[1:] += weight * (target - source)
        moved[0] += weight * pin["light"]
    return lab + (1 - shield) * moved


def _masked(pin: dict, masked) -> float:
    """How much of a pin its mask lets through at a pixel (all of it without one; none when
    its mask is gone)."""
    if not masked or not pin.get("mask"):
        return 1.0 if not pin.get("mask") or masked is None else 0.0
    return masked[pin["mask"] - 1] if pin["mask"] <= len(masked) else 0.0


def _shield(pins: list[dict], lab, masked=None) -> float:
    """How much of a color the protecting pins cover."""
    return max((pin_weight(pin, lab) * _masked(pin, masked)
                for pin in pins if pin["mode"] == "protect"), default=0.0)


def _bell(x: float) -> float:
    """An EQ point's weight that far from its hue, in widths: all of it at the hue itself,
    half at half the width, nothing from the width on, and no corner anywhere. (It was a
    flat top with steep sides: colors a few degrees apart, as the noise of a photo makes
    them, were then treated very differently, which showed as speckle; the owner saw it.)"""
    return 0.5 + 0.5 * math.cos(math.pi * min(abs(x), 1.0))


def eq_at(points: list[dict], hue: float) -> float:
    """What the color EQ makes of the chroma of a color of that hue (degrees): times this."""
    gain = 1.0
    for point in points:
        apart = abs((hue - point["hue"] + 180) % 360 - 180)
        gain += _bell(apart / point["width"]) * (point["gain"] / 100 - 1)
    # (A plain number: as one of numpy's the interface could not read it, and the EQ's line
    # was not drawn once it had a point.)
    return float(max(0.0, gain))


def bw_at(points: list[dict], hue: float) -> float:
    """What the B&W EQ makes of the lightness of a vivid color of that hue (degrees): times
    this. The color EQ's own sum, a point's gain being percent lighter."""
    return eq_at([{**point, "gain": 100 + point["gain"]} for point in points], hue)


def mono_at(points: list[dict], lab) -> np.ndarray:
    """A color (lightness, a, b) as black and white: its own lightness, made lighter or
    darker by the B&W EQ for its hue, the less the nearer it is to gray (half at TINTED)."""
    chroma = math.hypot(lab[1], lab[2])
    hue = math.degrees(math.atan2(lab[2], lab[1]))
    share = chroma / (chroma + TINTED)
    return np.array([lab[0] * (1 + (bw_at(points, hue) - 1) * share), 0.0, 0.0])


def chroma_at(points: list[list[float]], chroma: float) -> float:
    """What the chroma curve makes of a chroma: the curve reads it as a share of VIVID, and
    carries on at the color's own pace beyond that."""
    x = chroma / VIVID
    return (curve_at(points, min(x, 1.0)) + max(x - 1.0, 0.0)) * VIVID


def vibrance_at(vibrance: float, chroma: float, hue: float) -> float:
    """What Vibrance (-100..100) makes of a chroma: times this. The muted colors gain most
    (+100 doubles a gray-ish one) and the vivid ones nothing (from a chroma of 0.25 on), so
    that nothing is pushed past what a screen can show; and colors of skin (OKLab hue 55
    degrees, fading out 15 to 35 degrees to either side) gain only three tenths of it.
    Below 0 the muted colors go toward gray and the vivid ones stay."""
    skin = 1 - 0.7 * (1 - _step(15, 35, abs((hue - 55 + 180) % 360 - 180)))
    return 1 + vibrance / 100 * skin * (1 - _step(0.0, 0.25, chroma))


def colors_at(edit: dict, lab, place=None, shape: float = 1.0) -> np.ndarray:
    """Everything that is done to a color (lightness, a, b) after the tone curve, in the
    shader's order: the pins, then saturation and vibrance (not where a pin protects), the
    chroma curve and the color EQ. The same arithmetic as ui/pipeline.frag. `place`: where in
    the stored picture the pixel is (0..1), which the masks go by; `shape`: its width over its
    height."""
    full = {**DEFAULTS, **edit}
    if full["mono"]:  # black and white: in the color tools' place
        return mono_at(full["bw"], lab)
    masked = None if place is None else [mask_weight(m, place, shape) for m in full["masks"]]
    shield = _shield(full["pins"], lab, masked)
    out = pins_at(full["pins"], lab, masked)
    chroma = math.hypot(out[1], out[2])
    if chroma > 1e-6:
        hue = math.degrees(math.atan2(out[2], out[1]))
        after = chroma * (full["saturation"] / 100 * (1 - shield) + shield)
        after *= vibrance_at(full["vibrance"] * (1 - shield), after, hue)
        after = chroma_at(full["chroma"], after)
        after *= eq_at(full["eq"], hue)
        out[1:] *= after / chroma
    return out


def look_uniforms(full: dict) -> dict:
    """Saturation, the chroma curve and the color EQ for the shader; and `look`: whether
    anything at all is done to colors after the tone curve. Black and white (`mono`) takes
    their place: the color tools at rest, and the B&W EQ's points where the color EQ's are."""
    if full["mono"]:
        points = [{**point, "gain": 100 + point["gain"]} for point in full["bw"]]
        return {**look_uniforms({**DEFAULTS, "eq": points}), "look": 1.0, "mono": 1.0}
    slope = slopes(full["chroma"])
    fill = POINTS - len(full["chroma"])
    bands = []
    for point in full["eq"]:
        bands += [point["hue"], point["gain"] / 100 - 1, point["width"], 0.0]
    bands += [0.0] * (BANDS * 4 - len(bands))
    plain = (full["saturation"] == 100 and full["vibrance"] == 0
             and full["chroma"] == STRAIGHT and not full["eq"])
    return {"look": 0.0 if plain and not full["pins"] else 1.0, "mono": 0.0,
            "saturation": full["saturation"] / 100, "vibrance": full["vibrance"] / 100,
            "chromaOn": 0.0 if full["chroma"] == STRAIGHT else 1.0,
            "chromaX": [p[0] for p in full["chroma"]] + [full["chroma"][-1][0]] * fill,
            "chromaY": [p[1] for p in full["chroma"]] + [full["chroma"][-1][1]] * fill,
            "chromaM": slope + [0.0] * fill,
            "eqCount": float(len(full["eq"])), "eqA0": bands[:16], "eqA1": bands[16:]}


def over_uniforms(over: dict) -> dict:
    """A mask's own tone curve, chroma curve and color EQ for the shader, which applies each
    straight after the whole image's (`merged` gathers them as `over`); for the whole image
    itself, none. (Kept by what they are: they are asked for at every draw, and change only
    while one of them is being drawn.)"""
    key = repr(over)
    if key not in _OVER:
        if len(_OVER) > 32:
            _OVER.clear()
        _OVER[key] = _over_uniforms(over)
    return _OVER[key]


_OVER: dict[str, dict] = {}


def _over_uniforms(over: dict) -> dict:
    tone = curve_uniforms(over.get("curve", STRAIGHT))
    look = look_uniforms({**DEFAULTS, "chroma": over.get("chroma", STRAIGHT),
                          "eq": over.get("eq", [])})
    second = {**tone, **{k: look[k] for k in look if k.startswith(("chroma", "eq"))}}
    return {name.replace("curve", "curve2").replace("chroma", "chroma2").replace("eq", "eq2"): v
            for name, v in second.items()}


def masks_uniforms(masks: list[dict], shape: float) -> dict:
    """The masks for the shader, in four matrices of four masks each, worked out as far as
    can be: with (u, v) a place in the stored picture, a mask's A is (x, y, z, kind) and its B
    (x, y, z, feather), and u·x + v·y + z is how far along the mask's axis the place is (A)
    and how far across it (B), 1 being its edge. Kind: 0 radial, 1 linear, 2 painted (its A
    is then the channel of the raster it is in, `painted`), 4 more when inverted. A slot
    without a mask covers nothing, and neither does a mask that is off."""
    along, across = [], []
    wide = np.array([shape or 1.0, 1.0])
    brushes = 0
    for mask in masks:
        channel, brushes = brushes, brushes + (mask["kind"] == "brush")
        if mask.get("off"):
            along += [0.0, 0.0, 2.0, 0.0]
            across += [0.0, 0.0, 0.0, 1.0]
            continue
        if mask["kind"] == "brush":
            along += [float(channel), 0.0, 0.0, 2.0 + 4 * mask["invert"]]
            across += [0.0, 0.0, 0.0, 1.0]
            continue
        start, axis = np.array(mask["at"]) * wide, (np.array(mask["to"]) - mask["at"]) * wide
        long = float(axis @ axis)
        side = np.array([-axis[1], axis[0]]) / mask["ratio"]
        kind = KINDS.index(mask["kind"]) + 4 * mask["invert"]
        along += [*(axis * wide / long), -float(start @ axis) / long, float(kind)]
        across += [*(side * wide / long), -float(start @ side) / long, max(mask["feather"], 1e-4)]
    for _ in range(MASKS - len(masks)):
        along += [0.0, 0.0, 2.0, 0.0]
        across += [0.0, 0.0, 0.0, 1.0]
    return {"masksA0": along[:16], "masksA1": along[16:],
            "masksB0": across[:16], "masksB1": across[16:]}


def without_mask(edit: dict, number: int) -> dict:
    """The edit's masks and pins once mask `number` (1, 2, …) is deleted: its pins go with
    it, and the pins of later masks follow theirs down."""
    full = {**DEFAULTS, **edit}
    pins = [{**pin, "mask": pin["mask"] - (pin.get("mask", 0) > number)}
            for pin in full["pins"] if pin.get("mask", 0) != number]
    return {"masks": full["masks"][:number - 1] + full["masks"][number:], "pins": pins}


def pins_uniforms(pins: list[dict]) -> dict:
    """The pins for the shader: three numbers-of-four per pin, in nine matrices of four pins
    each. A: from, to. B: range, core, light, variance. C: low, high, kind (0 offset,
    1 rotate, 2 protect)."""
    rows = {"A": [], "B": [], "C": []}
    for pin in pins:
        core = min(pin["range"] * (1 - pin["feather"] / 100), pin["range"] - 1e-4)
        kind = MODES.index(pin["mode"])
        if kind == 1 and math.hypot(*pin["from"]) < NEUTRAL:
            kind = 0
        rows["A"] += [*pin["from"], *pin["to"]]
        rows["B"] += [pin["range"], core, pin["light"], pin["variance"] / 100]
        rows["C"] += [pin["low"], pin["high"], float(kind), float(pin.get("mask", 0))]
    out = {"pinCount": float(len(pins))}
    for letter, numbers in rows.items():
        numbers += [0.0] * (PINS * 4 - len(numbers))
        out.update({f"pins{letter}{n}": numbers[16 * n:16 * n + 16] for n in range(3)})
    return out


def detail_uniforms(full: dict, shape: float, wide: int) -> dict:
    """Sharpening and noise reduction for the shader, which reads each pixel with its
    neighbors in the stored picture (`wide` pixels across, `shape` its width over its height)
    and works in the square root of the light. `detailX`, `detailY`: one pixel of it as a
    share of the picture. `sharpen`: how much of the difference from the neighbors is added
    (100: all of it, once more); `sharpenRadius`: how far the neighbors are, in pixels;
    `sharpenMask`: differences smaller than this are not sharpened (0.05 at 100). `denoise`,
    `denoiseColor`: how far apart two pixels may be and still count as the same surface,
    for their lightness (0.07 at 100, a third of that at 50: the low end is where a photo
    taken in good light is set, and there a little is much) and for their color (0.025 at
    10, where a sensor's noise is already all within it, and slowly on to 0.06 at 100: what
    rises past 10 is `denoiseFar`, how much the color is also taken from further out, to
    all of it at 100; a wide reach across anything of another lightness would spread colors).
    Nothing of it without the picture's size."""
    color, far = full["denoiseColor"], max(full["denoiseColor"] - 10, 0) / 90
    known = wide > 0 and shape > 0
    return {"detailX": 1 / wide if known else 0.0, "detailY": shape / wide if known else 0.0,
            "sharpen": full["sharpen"] / 100 if known else 0.0,
            "sharpenRadius": full["sharpenRadius"], "sharpenMask": full["sharpenMask"] / 100 * 0.05,
            "denoise": (full["denoise"] / 100) ** 1.5 * 0.07 if known else 0.0,
            "denoiseColor": min(color, 10) * 0.0025 + far * 0.035 if known else 0.0,
            "denoiseFar": far if known else 0.0}


def uniforms(edit: dict, camera: dict | None, raw: bool, white: float = 1.0,  # noqa: PLR0913
             shape: float = 1.0, wide: int = 0) -> dict:
    """Everything ui/pipeline.frag takes, as plain numbers (matrices as 3 rows of 3, the
    curve as lists of 16). `white` is how bright the sensor's white may be shown, in multiples
    of the display's white: 1 in SDR, the display's peak in HDR. `shape`: the stored picture's
    width over its height, which the masks go by; `wide`: how many pixels it is across, which
    sharpening and noise reduction go by."""
    full = {**DEFAULTS, **edit}
    if camera and camera.get("denoised"):
        # A photo the neural denoiser wrote has had its noise taken out: none of this one.
        full.update(denoise=0.0, denoiseColor=0.0)
    to_cam, from_cam, wb = balance(full, camera)
    c, s, k = curve(full["profile"]) if raw else (0.0, 1.0, 1.0)  # 0: no curve
    out = {"gain": 2.0 ** full["exposure"], "wbGain": wb,
           "toCam": to_cam.tolist(), "fromCam": from_cam.tolist(),
           "mid": MID if raw else MID_FINISHED,
           # Contrast: stops from mid gray are stretched by this much round mid gray (-100
           # leaves 0.65 of them, as far as ART's Contrast goes: half was "a little extreme",
           # the owner; +100 makes them half as many again: more, with white and black
           # staying where they are, would leave the lights under white flat).
           "contrast": (1.5 if full["contrast"] > 0 else 1 / 0.65) ** (full["contrast"] / 100) - 1,
           # Brightness: a power on the finished picture's luminance, less one (+100: 0.5).
           "brightness": 2.0 ** (-full["brightness"] / 100) - 1,
           # Highlights: up to 2.4 stops, 1.8 at the sensor's white. The profile shows half the
           # sensor's white at level 238 of 255: one stop there moved a bright sky by little.
           "highlights": full["highlights"] / 100 * 2.4, "shadows": full["shadows"] / 100 * 1.5,
           "midtones": full["midtones"] / 100,  # a stop at mid gray
           # Whites: up to 0.8 stops at white. Blacks: light taken away at black (+100 adds
           # it: sign turned), by the square of the slider, since a display's steps near
           # black are that much finer: 0.03 of white is level 48 of 255.
           "whites": full["whites"] / 100 * 0.8,
           "blacks": -math.copysign(0.03 * (full["blacks"] / 100) ** 2, full["blacks"]),
           "curveC": c, "curveS": s, "curveK": k,
           "hdrLift": max(0.0, white - 1) if raw else 0.0,
           **curve_uniforms(curve_of(full, camera)),
           **pins_uniforms([] if full["mono"] else full["pins"]),
           **look_uniforms(full), **grain_uniforms(full, shape),
           **luts_uniforms(full["luts"]), **detail_uniforms(full, shape, wide),
           # Lens corrections, from the camera's own figures where the file had them.
           **lens.tables((camera or {}).get("lens"), full["lensDistortion"], full["lensColor"],
                         full["lensVignetting"]),
           "lensShape": shape if shape > 0 else 1.0,
           # Fringes of color taken off edges (the shader's `defringed`): how much.
           "defringe": full["defringe"] / 100 if wide > 0 else 0.0,
           **masks_uniforms(full["masks"], shape), **over_uniforms(edit.get("over", {}))}
    if full["mono"]:  # (a mask's own chroma curve and EQ have nothing to work on)
        out["chroma2On"] = out["eq2Count"] = 0.0
    if out["chroma2On"] or out["eq2Count"] or out["grain"]:
        out["look"] = 1.0  # (a mask's own chroma curve or EQ is something done to colors too,
        # and grain is laid on the color as OKLab has it)
    return out


# ---- the frame -------------------------------------------------------------------------------

# Where a point of the picture as it stands after so many quarter turns to the right lies in
# the stored picture, both 0..1: rows of a matrix for (x, y, 1).
TURNED = (((1, 0, 0), (0, 1, 0)), ((0, 1, 0), (-1, 0, 1)),
          ((-1, 0, 1), (0, -1, 1)), ((0, -1, 1), (1, 0, 0)))


def _fits(middle, half, turn, size) -> float:
    """How much of a turned rectangle (its middle, half its size, its turn, all in pixels of
    the picture) fits in the picture about its middle: all of it (1), or the share it has to
    shrink to."""
    room = [(middle if d < 0 else size - middle)[axis] / abs(d)
            for sign in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            for axis, d in enumerate(turn @ (half * sign)) if abs(d) > 1e-9]
    return min([1.0, *room]) if size.all() else 1.0


def _laid(full: dict, width: int, height: int, loose: bool) -> tuple:
    """The frame as a rectangle laid on the picture as it stands after the turns: its middle
    and half its size in pixels, the turn that takes its own directions to the picture's, and
    that picture's size. `loose`: the Crop tool's view instead, which is all of the picture
    under a level frame: the rectangle round the turned picture."""
    size = np.array([height, width] if int(full["turns"]) % 2 else [width, height], dtype=float)
    angle = math.radians(full["angle"])
    turn = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
    if loose:
        return size / 2, np.abs(turn) @ size / 2, turn, size
    left, top, right, bottom = full["crop"]
    middle = np.array([left + right, top + bottom]) * size / 2
    half = np.array([right - left, bottom - top]) * size / 2
    return middle, half * _fits(middle, half, turn, size), turn, size


def frame(edit: dict, width: int, height: int, loose: bool = False) -> dict:
    """The photo's frame, for a stored picture of that size.

    `matrix`: 16 numbers (by columns) that take a point of the frame, (0, 0) its top left and
    (1, 1) its bottom right, to its place in the stored picture; `size`: the frame in pixels;
    `shape`: its width over its height, unrounded.
    With an angle the frame is made just small enough, about its middle, to stay inside the
    picture. `loose`: what the Crop tool shows instead: the whole picture, turned by the angle
    so that the frame lies level on it, in the rectangle that just holds it (its corners are
    outside the picture)."""
    full = {**DEFAULTS, **edit}
    middle, half, turn, size = _laid(full, width, height, loose)
    # A point of the frame -> pixels of the turned picture -> 0..1 of it -> the stored picture.
    to_pixels = np.eye(3)
    to_pixels[:2, :2] = turn * (2 * half)
    to_pixels[:2, 2] = middle - turn @ half
    to_unit = np.diag([1 / max(1, size[0]), 1 / max(1, size[1]), 1.0])
    m = (np.vstack([TURNED[int(full["turns"]) % 4], (0, 0, 1)]) @ to_unit @ to_pixels).tolist()
    return {"matrix": [m[0][0], m[1][0], 0.0, 0.0, m[0][1], m[1][1], 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0, m[0][2], m[1][2], 0.0, 1.0],
            "size": (max(1, round(2 * half[0])), max(1, round(2 * half[1]))),
            "shape": float(half[0] / half[1]) if half[1] else 1.0}


def frame_in_view(edit: dict, width: int, height: int) -> list[float]:
    """Where the photo's frame lies in the Crop tool's view: left, top, right, bottom, 0..1.
    Level, as that view is the picture turned by the frame's own angle."""
    tight, loose = (frame(edit, width, height, loose=loose)["matrix"] for loose in (False, True))
    return [*unframed(loose, *framed(tight, 0, 0)), *unframed(loose, *framed(tight, 1, 1))]


def crop_for(edit: dict, size: tuple, start, wanted, together: bool = True) -> list[float]:
    """The crop that puts the frame where the Crop tool is asked to: at `wanted` (left, top,
    right, bottom in the tool's view, 0..1: `frame(loose=True)`), or as far toward it from
    `start` as it stays inside the picture. `together`: both directions at one pace (a shape
    is being kept); otherwise across first and then down, so that a frame at the picture's
    edge still slides along it."""
    middle, half, turn, pixels = _laid({**DEFAULTS, **edit}, *size, loose=True)

    def laid(rect) -> tuple:  # that rectangle on the picture: its middle, half its size
        low, high = rect[:2], rect[2:]
        return middle + turn @ ((low + high - 1) * half), (high - low) * half

    def toward(here, there):  # as far from here to there as stays inside
        low, high = 0.0, 0.0 if _fits(*laid(here), turn, pixels) < 1 - 1e-6 else 1.0
        if _fits(*laid(there), turn, pixels) >= 1 - 1e-9:
            return there
        for _ in range(24):
            mid = (low + high) / 2
            ok = _fits(*laid(here + (there - here) * mid), turn, pixels) >= 1 - 1e-9
            low, high = (mid, high) if ok else (low, mid)
        return here + (there - here) * low

    start, wanted = np.array(start, dtype=float), np.array(wanted, dtype=float)
    if start.shape != (4,) or wanted.shape != (4,) or not np.isfinite([*start, *wanted]).all():
        raise ValueError("a frame is four numbers")
    if not together:
        start = toward(start, np.array([wanted[0], start[1], wanted[2], start[3]]))
        wanted = np.array([start[0], wanted[1], start[2], wanted[3]])
    at, reach = laid(toward(start, wanted))
    return [float(v) for v in (*((at - reach) / pixels), *((at + reach) / pixels))]


def framed(matrix: list[float], u: float, v: float) -> tuple[float, float]:
    """Where a point of the frame lies in the stored picture."""
    return (matrix[0] * u + matrix[4] * v + matrix[12], matrix[1] * u + matrix[5] * v + matrix[13])


def unframed(matrix: list[float], x: float, y: float) -> tuple[float, float]:
    """Where a point of the stored picture lies in the frame: `framed` the other way."""
    x, y = x - matrix[12], y - matrix[13]
    size = matrix[0] * matrix[5] - matrix[4] * matrix[1]
    if not size:  # no frame yet (a photo with masks whose picture is still being read):
        return (x, y)  # nothing is drawn of it, and nothing may fail (it raised here)
    return ((matrix[5] * x - matrix[4] * y) / size, (matrix[0] * y - matrix[1] * x) / size)


# ---- history -------------------------------------------------------------------------------

class History:
    """Undo and redo over whole edits. A run of changes to one setting (a slider being dragged)
    is one step: it ends at `commit`, or when another setting changes."""

    KEPT = 100  # steps that are written down with a photo's edit (`kept`)

    def __init__(self, start: dict | None = None, steps: tuple | None = None) -> None:
        """`start`: the edit as it stands. `steps`: (the states, which one it stood at) as kept
        from when the photo was last open: taken up if they end where the edit stands, so that
        undo and redo carry on; not otherwise (the edit was changed elsewhere since)."""
        self.states, self.at, self.open = [dict(start or {})], 0, None
        try:
            states, at = [clean(state) for state in steps[0]], int(steps[1])
            if 0 <= at < len(states) and states[at] == self.edit:
                self.states, self.at = states, at
        except (TypeError, ValueError, IndexError, KeyError):
            pass  # none kept, or not what was written there

    def kept(self) -> tuple[list[dict], int]:
        """(The states, which one it stands at) to write down: all of them, or the KEPT round
        where it stands."""
        first = max(0, min(self.at - self.KEPT // 2, len(self.states) - self.KEPT))
        return self.states[first:first + self.KEPT], self.at - first

    @property
    def edit(self) -> dict:
        return self.states[self.at]

    def set(self, key: str, value) -> None:
        self.put({key: value})

    def put(self, values: dict) -> None:
        """Change settings. The same ones changed again and again (a slider being dragged, a
        pad that moves two at once) stay one step until `commit`."""
        now = {**self.edit, **{key: valid(key, value) for key, value in values.items()}}
        now = {k: v for k, v in now.items() if v != DEFAULTS[k]}
        if now == self.edit:
            return
        gesture = tuple(sorted(values))
        if self.open != gesture:
            del self.states[self.at + 1:]  # a new change ends what could be redone
            self.states.append(now)
            self.at += 1
            self.open = gesture
        else:
            self.states[self.at] = now

    def replace(self, whole: dict) -> None:
        """Another edit altogether (a checkpoint gone back to), as one step of its own."""
        self.commit()
        whole = clean(whole)
        if whole != self.edit:
            del self.states[self.at + 1:]
            self.states.append(whole)
            self.at += 1

    def commit(self) -> None:
        self.open = None
        if self.at and self.states[self.at] == self.states[self.at - 1]:
            del self.states[self.at]  # dragged there and back: nothing to undo
            self.at -= 1

    def step(self, by: int) -> bool:
        self.commit()
        to = self.at + by
        if not 0 <= to < len(self.states):
            return False
        self.at = to
        return True


# ---- a look ----------------------------------------------------------------------------------

# What of an edit can go from one photo to another, in parts that can be chosen: everything
# but the photo's frame and its masks (with their pins), which belong to that picture.
# (The tone curve is in the Exposure card, and stays a part of its own here: saved looks name it.)
PARTS = {"light": tuple(k for k in TOOLS["tone"] if k != "curve"), "curve": ("curve",),
         "balance": TOOLS["balance"],
         "color": ("luts", "pins", "saturation", "vibrance", "chroma", "eq"),
         "mono": ("mono", "bw"), "detail": TOOLS["detail"], "lens": TOOLS["lens"],
         "grain": TOOLS["grain"]}


def _parts(parts) -> list[str]:
    return [part for part in PARTS if parts is None or part in parts]


def look_of(edit: dict, parts=None) -> dict:
    """The look of an edit: its settings of those parts (all of them with None), cleaned. Of
    the pins, the whole image's. What is at rest is not in it."""
    full, out = clean(edit), {}
    for part in _parts(parts):
        for key in PARTS[part]:
            if key in full:
                out[key] = full[key]
    if "pins" in out:
        out["pins"] = [pin for pin in out["pins"] if not pin.get("mask")]
        if not out["pins"]:
            del out["pins"]
    return out


def with_look(edit: dict, look: dict, parts=None) -> dict:
    """An edit with a look laid on it: those parts are the look's (at rest where the look
    says nothing), the rest is as it was; the pins of the photo's own masks stay."""
    look, out = look_of(look, parts), dict(clean(edit))
    for part in _parts(parts):
        for key in PARTS[part]:
            out.pop(key, None)
    own = [pin for pin in clean(edit).get("pins", []) if pin.get("mask")]
    out.update(look)
    if "color" in _parts(parts) and own:
        out["pins"] = [*look.get("pins", []), *own][:PINS]
    return clean(out)


# ---- checkpoints ---------------------------------------------------------------------------

def clean_checkpoints(found) -> list[dict]:
    """A photo's checkpoints as read from somewhere untrusted: only those that are whole."""
    out = []
    for item in found if isinstance(found, list) else ():
        if not isinstance(item, dict):
            continue
        number, name, made = item.get("id"), item.get("name"), item.get("made")
        if (isinstance(number, int) and isinstance(name, str) and name.strip()
                and isinstance(made, (int, float)) and number not in {c["id"] for c in out}):
            out.append({"id": number, "name": name.strip()[:80], "made": float(made),
                        "edit": clean(item.get("edit"))})
    return out


def unsaved(points: list[dict], now: dict, base: int | None) -> list[str]:
    """The settings of `now` that are not as the checkpoint it came from has them: `base` (the
    one last saved or gone to) while it is there, else the newest. With no checkpoint, the
    settings not at rest; standing at a checkpoint, none. (The owner: "Unsaved modified
    parameters ... should be colored yellow".)"""
    if standing(points, now, base)[0] is not None:
        return []
    kept = {point["id"]: point["edit"] for point in points}
    was = kept.get(base, points[-1]["edit"] if points else {})
    return sorted(key for key in DEFAULTS
                  if now.get(key, DEFAULTS[key]) != was.get(key, DEFAULTS[key]))


def standing(points: list[dict], now: dict, base: int | None) -> tuple[int | None, bool]:
    """(the checkpoint the edit `now` stands at, or None; whether `now` is unsaved).

    The edit stands at a checkpoint when it is that checkpoint's edit: at `base`, the one last
    saved or gone to, if that fits, else at the first that does. Otherwise it is unsaved,
    unless nothing was ever done to the photo (no base, and every setting at rest)."""
    fitting = [c["id"] for c in points if c["edit"] == now]
    if fitting:
        return (base if base in fitting else fitting[0]), False
    return None, bool(now) or base is not None

