"""Neural denoise: a raw photo through the net Neuro trained, and the result as a DNG.

The net (models/nafnet-4m-120k.onnx: a NAFNet of 4 million weights, single precision, run by
ONNX Runtime) sees the photo after demosaicing, put back into the camera's own colors with
each made to have noise of one unit at any brightness (v = scale * sqrt(x + offset)), scaled
so that its median sits at MIDDLE. The noise is then one number all over the picture, which
is the net's second input. This is Neuro's arithmetic (neuro/tasks/denoise/net.py, test.py,
raw.py) in numpy: what the net was trained on is what it must be given.

`run` puts a photo through it and keeps the net's result (`Denoised`), so that how much of
what it took away stays away can be chosen afterwards without running it again: `color` of
the color noise (what differs between the three colors) and `grain` of the grain (what they
have in common). `save` writes that as a DNG beside the original (photone/dng.py).

ponytail: the net runs in this process, on a thread. Should a GPU driver ever take the window
down with it, the upgrade is a process of its own, as decoding has (decode.Workers).
"""

from __future__ import annotations

import contextlib
import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
from PySide6.QtCore import Property, QObject, Signal, Slot
from PySide6.QtGui import QMatrix4x4, QVector4D

from photone import __version__, decode, dng, edit, lens
from photone.catalog import identify

MODEL = Path(__file__).parent / "models" / "nafnet-4m-120k.onnx"
# The net takes pieces of one size, which its file is made for (`_Net.piece`: 512 pixels), and
# sees 98 pixels around a pixel: each piece is given MARGIN more on every side than is kept of
# it (a multiple of 8, the net's grid).
# How large the pieces are is how much of the card the net takes, measured 2026-11-20 on an
# RTX 3080 Ti, for a 33 MP photo: 1280 px 6.3 GB and 7 s of the net, 768 px 2.4 GB, 640 px
# 1.9 GB, 512 px 1.2 GB and 11 s. The owner: "i would like it to use less GPU": 512. (With
# 1280 a card that other programs were using refused pieces, and they went to the processor.)
MARGIN = 104
MIDDLE = 0.25
# The noise of the cameras measured, by ISO: a per unit of ISO (the light's grain) and b at
# each ISO (the dark's: the spread of cap-on frames, squared), in the raw file's numbers.
# From /data/projects/Denoiser (pure_noise/, calibrate_v2.py), 2026-11-19, as Neuro has them.
# ponytail: a camera is known by its sensor's size, which rawpy gives; its name it does not.
KNOWN = {
    (4688, 7028): (0.002175, {100: 4.8, 400: 7.1, 800: 15.4, 1600: 35.2, 3200: 140.7,
                              6400: 316.5, 12800: 1114.4, 25600: 4255.4, 51200: 17217.3}),
    (4660, 6984): (0.0054, {100: 19.8, 200: 19.8, 400: 35.2, 800: 55.0, 1600: 140.7,
                            3200: 430.8, 6400: 1373.8, 12800: 5495.1, 25600: 21980.4}),
}


class Stopped(Exception):
    """The user closed the window: nothing is wanted of this run any more."""


# --- The noise of a photo ------------------------------------------------------------------

@functools.cache
def _calmest(share: float = 1 / 3) -> float:
    """What the calmest third of such readings gives for pure noise of one unit: they are the
    ones that came out low by chance, so their mean is below 1 (0.61). (The same every time:
    worked out once.)"""
    runs = np.random.default_rng(7).standard_normal((200000, 8))
    each = np.sort(np.median(np.abs(runs[:, 2:] - runs[:, :-2]), axis=1) / 0.6745 / np.sqrt(2))
    return float(each[:int(len(each) * share)].mean())


# The 11 compare-and-swaps that put the third and fourth smallest of six values in places 2
# and 3 (an optimal sorting network of six, but for its last, which would only order those
# two: their sum is the same either way).
_SIX = ((1, 2), (4, 5), (0, 2), (3, 5), (0, 1), (3, 4), (1, 4), (0, 3), (2, 5), (1, 3), (2, 4))


def _median_of_six(d: np.ndarray) -> np.ndarray:
    """np.median(d, axis=-1) for a last axis of six, the same numbers: the mean of the third
    and fourth smallest, sorted by whole arrays at a time (np.median sorts six values at a
    time, a call each, and took twice as long)."""
    six = [d[..., i] for i in range(6)]
    for i, j in _SIX:
        six[i], six[j] = np.minimum(six[i], six[j]), np.maximum(six[i], six[j])
    return (six[2] + six[3]) * 0.5


def _readings(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(signal, spread) of every run of eight pixels of one color in a row of a mosaic: their
    mean, and the noise their differences four apart show."""
    v = values.astype(np.float32)
    h, w = v.shape[0] // 2 * 2, v.shape[1] // 16 * 16
    signals, spreads = [], []
    for dy, dx in product((0, 1), (0, 1)):
        runs = v[dy:h:2, dx:w:2].reshape(h // 2, w // 16, 8)  # eight of one color in a row
        signals.append(runs.mean(axis=2).ravel())
        spreads.append(_median_of_six(np.abs(runs[:, :, 2:] - runs[:, :, :-2])).ravel()
                       / 0.6745 / np.sqrt(2))
    return np.concatenate(signals), np.concatenate(spreads)


def of_mosaic(values: np.ndarray, share: float = 1 / 3) -> tuple[float, float]:
    """(a, b) of a mosaic with its black level taken off: the noise's variance is
    a * signal + b. From pixels of one color four apart along a row, by how light the place
    is; of every lightness the calmest places count, since texture is not noise.

    Each reading is put in its band of lightness once (one pass, then a sort by band number
    that keeps their order), where it was a mask of the whole photo for each of 30 bands:
    a band's readings come out the same, in the same order, and are sorted as they were, so
    (a, b) are the same to the last bit. (The masks were 0.6 s of 1.3 for a 33 MP photo.)"""
    signal, spread = _readings(values)
    edges = np.unique(np.percentile(signal, np.linspace(0.5, 99.5, 31)))
    # (Band i holds edges[i] <= signal < edges[i + 1], as the masks had it.)
    band = np.searchsorted(edges, signal, side="right") - 1
    inside = np.flatnonzero((band >= 0) & (band < len(edges) - 1))
    band = band[inside].astype(np.int16)
    order = inside[np.argsort(band, kind="stable")]
    counts = np.bincount(band, minlength=len(edges) - 1)
    xs, vs, calmest = [], [], _calmest(share)
    for start, count in zip(np.cumsum(counts) - counts, counts, strict=True):
        if count < 300:
            continue
        here = order[start:start + count]
        spread_here = spread[here]
        calm = np.argsort(spread_here)[:int(count * share)]
        xs.append(signal[here][calm].mean())  # (their own signal: the calmest are the darker)
        vs.append((spread_here[calm].mean() / calmest) ** 2)
    xs, vs = np.array(xs, float), np.array(vs, float)
    a, b = np.polyfit(xs, vs, 1, w=1 / np.maximum(vs, 1e-9)) if len(xs) > 1 else (0.0, 0.0)
    if a <= 0:  # (a picture all of one lightness shows no slope: all of it is b)
        a, b = 1e-6, float(np.median(vs)) if len(vs) else 1.0
    return float(a), float(max(b, 1e-6))


def known(shape: tuple, iso: float) -> tuple[float, float] | None:
    """(a, b) of a photo from a sensor that was measured, at its ISO: b between the ISOs
    measured goes by a straight line in the logarithms of both. None for any other sensor,
    and for a file that does not say its ISO. (Read off the photo itself, b came out as
    nothing for some photos, and the net, told the shadows were clean, left their noise.)"""
    if tuple(shape) not in KNOWN or not iso > 0:
        return None
    per_iso, dark = KNOWN[tuple(shape)]
    isos = sorted(dark)
    b = np.exp(np.interp(np.log(iso), np.log(isos), np.log([dark[i] for i in isos])))
    return iso * per_iso, float(b)


# --- A raw file, as the net wants it -------------------------------------------------------

def develop(path: Path) -> tuple[np.ndarray, dict]:
    """((h, w, 3) uint16 scene-linear Rec.2020, the very picture Photone shows at full size;
    what is known of the photo: the camera's `wb` and `cam_xyz`, the noise's `a` and `b` over
    the range `room`, how it is `turned`, the tone curve it starts with (`match`), its `lens`
    figures, `header` and `preview`).
    A ValueError says why a file cannot be denoised."""
    import rawpy

    if path.suffix.lower() not in decode.RAW:
        raise ValueError("Only raw files can be denoised: the net works on what a sensor recorded.")
    if dng.made_here(path):
        raise ValueError("This file is already denoised by Photone's neural denoiser.")
    try:
        with rawpy.imread(str(path)) as raw:
            sizes, camera = raw.sizes, decode.camera(raw.camera_whitebalance, raw.rgb_xyz_matrix)
            if camera is None:
                raise ValueError(f"{path.name} does not say how its camera sees color.")
            if raw.raw_pattern is None or raw.raw_pattern.shape != (2, 2):
                raise ValueError(f"{path.name} is not from a sensor with a mosaic of 2 x 2.")
            black = float(np.mean(raw.black_level_per_channel))
            mosaic = raw.raw_image_visible
            a, b = known(mosaic.shape, float(raw.other.iso_speed)) or of_mosaic(mosaic - black)
            about = {**camera, "a": a, "b": b, "room": float(raw.white_level - black),
                     "match": decode.match_of(raw, camera), "lens": lens.figures(path)}
            jpeg = _carried(raw)
            rgb = raw.postprocess(half_size=False, gamma=(1, 1), no_auto_bright=True,
                                  output_bps=16, use_camera_wb=True,
                                  output_color=rawpy.ColorSpace.Rec2020,
                                  highlight_mode=rawpy.HighlightMode.Clip)
    except (rawpy.LibRawError, OSError) as e:
        raise ValueError(f"{path.name} cannot be read as a raw photo ({e}).") from None
    about.update(turned={3: 3, 5: 8, 6: 6}.get(sizes.flip, 1), header=dng.carried(path),
                 preview=jpeg)
    return np.ascontiguousarray(rgb[decode.kept(sizes, False)]), about


def _carried(raw) -> tuple[bytes, tuple[int, int]] | None:
    """(the JPEG preview a raw file carries, its (wide, high)), or None where it has none."""
    import rawpy

    try:
        carried = raw.extract_thumb()
    except rawpy.LibRawError:
        return None
    size = _jpeg_size(carried.data) if carried.format == rawpy.ThumbFormat.JPEG else None
    return (carried.data, size) if size else None


def _jpeg_size(jpeg: bytes) -> tuple[int, int] | None:
    """(wide, high) of a JPEG, from the first of its blocks that says so."""
    at = 2
    while at + 9 < len(jpeg) and jpeg[at] == 0xff:
        kind, length = jpeg[at + 1], int.from_bytes(jpeg[at + 2:at + 4], "big")
        if 0xc0 <= kind <= 0xcf and kind not in (0xc4, 0xc8, 0xcc):
            return (int.from_bytes(jpeg[at + 7:at + 9], "big"),
                    int.from_bytes(jpeg[at + 5:at + 7], "big"))
        at += 2 + length
    return None


class Denoised:
    """A photo and what the net made of it. `noisy`: the picture as developed; `clean`: the
    net's result in its own terms (v, before the scaling to MIDDLE), every pixel of it."""

    def __init__(self, noisy: np.ndarray, about: dict) -> None:
        self.noisy, self.about = noisy, about
        self.clean = np.empty(noisy.shape, np.float32)
        self.where = ""  # what the net ran on: "CUDA", "DirectML" or "the processor"
        wb = np.array(about["wb"], np.float64)
        self.gain = wb / wb.min()  # what the decoder multiplied each of the camera's colors by
        slope = np.maximum(self.gain * about["a"] / max(about["room"], 1), 1e-12)
        floor = self.gain ** 2 * about["b"] / max(about["room"], 1) ** 2
        self.from_cam = edit.camera_matrix(about)
        self.to_cam = np.linalg.inv(self.from_cam).astype(np.float32)
        self.scale = (2 / np.sqrt(slope)).astype(np.float32)
        self.offset = (floor / slope).astype(np.float32)

    # (`into`, `camera` and `mixed` work in place, in one array each: the same arithmetic in
    # the same order as written plainly, but every new array of a picture's size, and every
    # sum over its three colors by numpy's reductions, cost as much as the arithmetic.)

    def into(self, part: np.ndarray) -> np.ndarray:
        """(h, w, 3) of the developed picture -> v, noise of one unit."""
        cam = part.astype(np.float32)
        cam /= np.float32(65535)
        cam = cam @ self.to_cam.T
        cam += self.offset
        np.maximum(cam, 0, out=cam)
        np.sqrt(cam, out=cam)
        cam *= self.scale
        return cam

    def camera(self, v: np.ndarray, own: bool = False) -> np.ndarray:
        """v -> the camera's own colors with its white balance on, 1 the sensor's white.
        `own`: v is an array of no use afterwards, which becomes the result."""
        out = np.maximum(v, 0, out=v if own else None)
        out /= self.scale
        np.square(out, out=out)
        out -= self.offset
        return out

    def mixed(self, rows: slice, color: float = 1.0, grain: float = 1.0) -> np.ndarray:
        """Those rows with these shares of what the net took kept away (1 and 1: the net's
        own result; 0 and 0: the photo as it was), in the camera's colors, white balance on:
        clean + (1 - grain) common + (1 - color) (removed - common), summed in that order."""
        clean = self.clean[rows]
        removed = self.into(self.noisy[rows])
        removed -= clean
        # (The mean of the three as numpy's mean has it, one after the other, then by 3.)
        common = removed[..., 0] + removed[..., 1]
        common += removed[..., 2]
        common /= np.float32(3)
        common = common[..., None]
        out = clean + common * np.float32(1 - grain)
        removed -= common
        removed *= np.float32(1 - color)
        out += removed
        return self.camera(out, own=True)

    def picture(self) -> np.ndarray:
        """The net's own result as Photone shows a photo: (h, w, 4) half floats, linear
        Rec.2020, between 0 and 1. For the preview, which mixes the shares itself."""
        out = np.ones((*self.clean.shape[:2], 4), np.float16)
        matrix = self.from_cam.astype(np.float32).T

        def band(k: int) -> None:
            rows = slice(len(out) * k // 8, len(out) * (k + 1) // 8)
            out[rows, :, :3] = np.clip(self.camera(self.clean[rows]) @ matrix, 0, 1)

        with ThreadPoolExecutor(8) as bands:
            list(bands.map(band, range(8)))
        return out


# --- The net -------------------------------------------------------------------------------

def session(processor: bool = False):
    """(the net ready to run, what it runs on). On the GPU where ONNX Runtime has a way to
    it (CUDA on Linux, DirectML on Windows), else on the processor, which takes minutes."""
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    have, options, wanted = ort.get_available_providers(), ort.SessionOptions(), []
    if "CUDAExecutionProvider" in have and not processor:
        with contextlib.suppress(AttributeError, OSError):
            ort.preload_dlls()  # (NVIDIA's libraries, from the wheels they came in)
        # (Memory as each piece needs it and no search for the fastest convolutions: 3.3 GB
        # of the card for a piece, where the defaults wanted more than 4 and failed on a
        # card that other programs were using.)
        wanted.append(("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested",
                                                 "cudnn_conv_algo_search": "HEURISTIC"}))
    if "DmlExecutionProvider" in have and not processor:
        wanted.append("DmlExecutionProvider")
        options.enable_mem_pattern = False  # (DirectML asks for both)
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    net = ort.InferenceSession(str(MODEL), options, providers=[*wanted, "CPUExecutionProvider"])
    where = {"CUDAExecutionProvider": "CUDA", "DmlExecutionProvider": "DirectML"}
    return net, where.get(net.get_providers()[0], "the processor")


PROCESSOR = "the processor"


class _Net:
    """The net, on the GPU where there is one, and on the processor for a piece the GPU
    refuses (as a rule its memory is full: another program's, for that moment). The next
    piece is tried on the GPU again; after three refusals in a row it is left alone.
    (One refusal used to put all the rest of the photo on the processor: two minutes where
    it takes eight seconds, and nothing said why. The owner: "VERY slow for some reason".)"""

    def __init__(self) -> None:
        self.fast, self.on = session()
        self.piece = int(self.fast.get_inputs()[0].shape[2])  # the side its file is made for
        self.slow, self.refused, self.pieces, self.slowly, self.why = None, 0, 0, 0, ""

    def __call__(self, seen: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        feed = {"x": seen, "sigma": sigma}
        self.pieces += 1
        if self.on != PROCESSOR and self.refused < 3:
            try:
                out = self.fast.run(None, feed)[0]
                self.refused = 0
                return out
            except Exception as e:  # noqa: BLE001 - whatever the GPU's library raises
                self.refused, self.why = self.refused + 1, str(e).strip().splitlines()[-1][:200]
                print(f"photone: the denoiser's piece {self.pieces} was refused by {self.on}: "
                      f"{self.why}", flush=True)
        self.slowly += 1
        try:
            self.slow = self.slow or (self.fast if self.on == PROCESSOR else session(True)[0])
            return self.slow.run(None, feed)[0]
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"The denoiser failed ({str(e).splitlines()[0][:160]}).") from None

    @property
    def where(self) -> str:
        """What it ran on, in words: "CUDA", "the processor", or both and how much of each."""
        if self.on == PROCESSOR or self.slowly == self.pieces > 0:
            return PROCESSOR
        return self.on if not self.slowly else (
            f"{self.on}, and {PROCESSOR} for {self.slowly} of {self.pieces} pieces")


def run(path: Path, progress=None, stopped=None) -> Denoised:
    """The photo at `path` through the net, a piece at a time. `progress(done, of, where)` is
    called after each piece (and once before the first: 0 of them), `where` saying what the
    net runs on; `stopped()` true ends it with `Stopped`. The whole photo is told one noise:
    its norm is of all of it, not of each piece. A ValueError says why it could not be done."""
    noisy, about = develop(path)
    held = Denoised(noisy, about)
    high, wide = noisy.shape[:2]
    net = _Net()
    kept_ = net.piece - 2 * MARGIN
    places = list(product(range(0, high, kept_), range(0, wide, kept_)))
    if progress:
        progress(0, len(places), net.where)
    v, norm = _terms(held)
    sigma = np.full((1, 1, 1, 1), norm, np.float32)

    def cut(at: tuple[int, int]) -> np.ndarray:
        return np.ascontiguousarray((_piece(v, *at, net.piece) * norm).transpose(2, 0, 1)[None])

    # (Closed here, on this thread, however the loop ends: left to the exception's traceback,
    # its thread would be shut down wherever that is let go, the interface's thread.)
    with contextlib.closing(_ahead(cut, places)) as pieces:
        for done, ((y, x), seen) in enumerate(zip(places, pieces, strict=True)):
            if stopped and stopped():
                raise Stopped
            kept = held.clean[y:y + kept_, x:x + kept_]
            kept[...] = net(seen, sigma)[0].transpose(1, 2, 0)[MARGIN:MARGIN + kept.shape[0],
                                                               MARGIN:MARGIN + kept.shape[1]] / norm
            held.where = net.where
            if progress:
                progress(done + 1, len(places), net.where)
    return held


def _terms(held: Denoised) -> tuple[np.ndarray, np.float32]:
    """(the whole photo in the net's terms, once (in bands, on threads): the pieces are cut
    from it; what it is multiplied by for the net, so that its median sits at MIDDLE)."""
    noisy = held.noisy
    v = np.empty(noisy.shape, np.float32)

    def band(k: int) -> None:
        rows = slice(len(noisy) * k // 8, len(noisy) * (k + 1) // 8)
        v[rows] = held.into(noisy[rows])

    with ThreadPoolExecutor(8) as bands:
        list(bands.map(band, range(8)))
    return v, np.float32(MIDDLE / max(float(np.median(v[::4, ::4].mean(axis=2))), 1e-3))


def _ahead(cut, places: list):
    """cut(place) of each place in turn, the next one cut on a thread of its own while the
    one before is worked on: ONNX Runtime lets go of Python's lock while it runs, and cutting
    a piece took 1.4 ms of the 29 a piece takes on the GPU."""
    if not places:  # (a picture of no rows or no columns: nothing to cut)
        return
    with ThreadPoolExecutor(1) as ahead:
        coming = ahead.submit(cut, places[0])
        for n in range(len(places)):
            seen = coming.result()
            if n + 1 < len(places):
                coming = ahead.submit(cut, places[n + 1])
            yield seen


def _piece(seen: np.ndarray, y: int, x: int, piece: int) -> np.ndarray:
    """The piece (`piece` a side) of the photo whose kept part begins at (y, x), with its
    margin; past the photo's edge, the photo mirrored in it. (The edge's pixels again made
    lines of the outermost row's noise, a hundred pixels long: to the net those are something
    in the picture, so it left that row half its noise and drew streaks inward from it. The
    owner: "the same artifact around the edges". Mirrored, the noise there is noise.)"""
    top, left, reach = max(y - MARGIN, 0), max(x - MARGIN, 0), piece - MARGIN
    part = seen[top:y + reach, left:x + reach]
    above, before = top - (y - MARGIN), left - (x - MARGIN)
    return np.pad(part, ((above, piece - above - part.shape[0]),
                         (before, piece - before - part.shape[1]), (0, 0)), "reflect")


def target(path: Path) -> Path:
    """Where the denoised file of that photo goes: beside it, "<name>-Denoised.dng", with a
    number after it where that is taken (a file is never written over)."""
    for n in range(1, 1000):
        out = path.with_name(f"{path.stem}-Denoised{'' if n == 1 else f'-{n}'}.dng")
        if not out.exists():
            return out
    raise ValueError("There are too many denoised copies of this photo beside it.")


def save(held: Denoised, path: Path, color: float = 1.0, grain: float = 1.0) -> Path:
    """The denoised photo as a DNG beside the original at `path` (dng.write), with these
    shares of what the net took kept away. Gives the new file's path; an OSError where it
    cannot be written (a full disk, a folder that may not be written to)."""
    about, out = held.about, target(path)
    turns = dng.TURNS.get(about["turned"], 0)
    high, wide = held.noisy.shape[:2]
    full = (wide, high)
    # The file's pixels lie as the sensor had them: the upright picture turned back.
    size = (high, wide) if turns % 2 else full
    gain = held.gain.astype(np.float32)

    def upright(top: int, bottom: int) -> np.ndarray:
        return held.mixed(slice(top, bottom), color, grain) / gain

    rows = upright
    if turns:  # (turned as a whole: the file's rows are the upright picture's columns)
        # (Mixed in bands on threads, as an unturned photo's strips are by dng.write: one
        # band after another, it took as long as all the rest of the writing.)
        upright_whole = np.empty(held.noisy.shape, np.float32)

        def band(top: int) -> None:
            upright_whole[top:top + 512] = upright(top, top + 512)

        with ThreadPoolExecutor(8) as bands:
            list(bands.map(band, range(0, high, 512)))
        whole = np.rot90(upright_whole, -turns)
        rows = lambda top, bottom: whole[top:bottom]  # noqa: E731

    matrix = (held.from_cam * held.gain).tolist()  # what is stored -> Rec.2020, balance on
    dng.write(out, rows, size, {
        "cam_xyz": about["cam_xyz"], "wb": about["wb"], "turned": about["turned"],
        "header": about["header"], "preview": about["preview"],
        "software": f"Photone {__version__} (neural denoise)",
        # (`of`: which photo it was made from, as the catalog knows a photo (catalog.identify:
        # the same after a rename or a move), so that the two can be shown as a pair.)
        "neural": {"model": MODEL.stem, "color": color, "grain": grain, "from": path.name,
                   "of": identify(path)},
        "kept": {"matrix": matrix, "match": about["match"],
                 "camera": {"wb": about["wb"], "cam_xyz": about["cam_xyz"],
                            "lens": about["lens"], "denoised": True}}})
    return out


# --- What the interface talks to -----------------------------------------------------------

class Denoiser(QObject):
    """The Neural Denoise window's side in Python (ui/Denoise.qml): the open photo through the
    net on a thread, how far that is, the net's result as a picture for the preview (which
    mixes the shares itself, on the GPU: ui/mixing.frag), and the file written when the user
    says so, with the photo's edit copied to it in the catalog."""

    changed = Signal()
    saved = Signal(str)                # the new file, written: its path
    stepped = Signal(int, float, str)  # from the thread: (which run, how far 0..1, on what)
    finished = Signal(int, object)     # from the thread: (which run, what came of it)

    def __init__(self, session) -> None:
        super().__init__()
        self.session = session
        self.held: Denoised | None = None
        self.pixels: np.ndarray | None = None  # the net's result as the preview's picture
        self._run, self._state, self._progress, self._message, self._path = 0, "", 0.0, "", None
        self._where = ""
        self.stepped.connect(self._stepped)
        self.finished.connect(self._finished)

    @Property(str, notify=changed)
    def state(self) -> str:
        """"" (nothing asked), "working" (the net runs), "ready" (its result is there),
        "saving" (the file is being written) or "failed" (`message` says why)."""
        return self._state

    @Property(float, notify=changed)
    def progress(self) -> float:
        return self._progress

    @Property(str, notify=changed)
    def message(self) -> str:
        return self._message

    @Property(str, notify=changed)
    def where(self) -> str:
        """What the net runs on, or ran on: "CUDA", "DirectML", "the processor", or the first
        with the processor for some pieces."""
        return self._where

    @Property(str, notify=changed)
    def target(self) -> str:
        """The name the denoised file will have."""
        try:
            return target(self._path).name if self._path else ""
        except ValueError:
            return ""

    @Property(str, notify=changed)
    def source(self) -> str:
        """The address of the net's result as a picture (app.PhotoProvider), once it is there."""
        return f"image://photo/denoised/{self._run}/full" if self.pixels is not None else ""

    @Property("QVariantMap", notify=changed)
    def terms(self) -> dict:
        """What ui/mixing.frag needs to go into the net's terms and back (`Denoised.into`)."""
        held = self.held
        if not held:
            return {}
        four = lambda m: QMatrix4x4(*m[0], 0, *m[1], 0, *m[2], 0, 0, 0, 0, 1)  # noqa: E731
        return {"toCam": four(held.to_cam.tolist()), "fromCam": four(held.from_cam.tolist()),
                "scale": QVector4D(*held.scale.tolist(), 0),
                "offset": QVector4D(*held.offset.tolist(), 0)}

    @Slot(str, result=bool)
    def fits(self, path: str) -> bool:
        """Whether that file can be denoised: a raw file that Photone has not denoised."""
        file = Path(path)
        return file.suffix.lower() in decode.RAW and dng.made_here(file) is None

    def _set(self, state: str, message: str = "", progress: float = 0.0) -> None:
        self._state, self._message, self._progress = state, message, progress
        self.changed.emit()

    @Slot(str)
    def start(self, path: str) -> None:
        """That photo through the net. Whatever was under way or kept is let go."""
        self._run += 1
        self.held, self.pixels, self._path, self._where = None, None, Path(path), ""
        self._set("working")
        threading.Thread(target=self._work, args=(self._run, self._path), daemon=True,
                         name="denoise").start()

    def _work(self, mine: int, path: Path) -> None:
        try:
            held = run(path, lambda done, of, on: self.stepped.emit(mine, done / max(of, 1), on),
                       lambda: mine != self._run)
            self.finished.emit(mine, (held, held.picture()))
        except Stopped:
            pass
        except Exception as e:  # noqa: BLE001 - whatever it is, the window says it and stays
            self.finished.emit(mine, e)

    def _stepped(self, mine: int, share: float, on: str) -> None:
        if mine == self._run and self._state == "working":
            self._progress, self._where = share, on
            self.changed.emit()

    def _finished(self, mine: int, result) -> None:
        if mine != self._run:
            return  # (of a window since closed, or of another photo)
        if isinstance(result, Exception):
            said = str(result) if isinstance(result, (ValueError, OSError)) else repr(result)
            return self._set("failed", said)
        if isinstance(result, Path):
            return self._written(result)
        self.held, self.pixels = result
        self._set("ready", progress=1.0)

    @Slot(float, float)
    def save(self, color: float, grain: float) -> None:
        """Write the file, with these shares (0..1) of what the net took kept away."""
        if self._state != "ready":
            return
        self.session.flush()  # (the edit that is copied is the one in the catalog)
        self._set("saving", progress=1.0)
        threading.Thread(target=self._write, args=(self._run, self.held, self._path, color, grain),
                         daemon=True, name="denoise-save").start()

    def _write(self, mine: int, held: Denoised, path: Path, color: float, grain: float) -> None:
        try:
            self.finished.emit(mine, save(held, path, color, grain))
        except Exception as e:  # noqa: BLE001
            self.finished.emit(mine, e)

    def _written(self, out: Path) -> None:
        """On the interface's thread: the new file has the original's edit (the owner: "with
        all the edit settings copied"), and the window is told."""
        catalog, original = self.session.catalog, self._path
        if catalog:
            try:
                had = catalog.load(identify(original), original)
                catalog.save(identify(out), out, had)
            except Exception as e:  # noqa: BLE001 - the file is written: say what is missing
                self.session.say(f"{out.name} was written, but its edit could not be copied: {e}")
        self.close()
        self.saved.emit(str(out))

    @Slot()
    def close(self) -> None:
        """The window is closed: a run under way ends at its next piece, and what was kept
        (a gigabyte for a 33 MP photo) is let go."""
        self._run += 1
        self.held, self.pixels, self._path = None, None, None
        self._set("")
