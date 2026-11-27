"""What the interface talks to: the session (which photos are open, what went wrong, SDR or HDR)
and the route by which a photo's pixels reach the GPU.

The pixels go: decode.load -> QImage (RGBA half floats, no conversion) -> the scene graph's
texture -> ui/pipeline.frag. Qt asks for them through an image provider on a loader thread, so
the window stays responsive while a file is being decoded.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import CancelledError, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import (
    Property,
    QCoreApplication,
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QSettings,
    QSize,
    Qt,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QColor,
    QCursor,
    QGuiApplication,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtQml import QJSValue, QQmlImageProviderBase
from PySide6.QtQuick import QQuickImageProvider, QQuickItem, QQuickView

from photone import HOMEPAGE, QUIET, __version__, decode, edit, lut, scopes, shaders
from photone.catalog import Catalog, identify
from photone.denoise import Denoiser
from photone.export import Exporter
from photone.library import Library, about
from photone.presets import Presets

UI = Path(__file__).parent / "ui"


CHART = (1200, 800)  # the test chart's size


NO_HDR = ("HDR is not available here: this desktop or display did not give the window an HDR "
          "surface. Showing SDR.")


def file_named(name: str) -> str:
    """An installed LUT's name as the file it came from, extension and all. Those installed
    before .clf files could be (2026-11-20) came from .cube files and were named without it."""
    return name if name.lower().endswith((".cube", ".clf")) else name[:75] + ".cube"


def small_copy(meta: dict, buf) -> np.ndarray:
    """A photo's pixels thinned to about 500 across, linear Rec.2020 as decoded: what "pick
    neutral" reads a color from."""
    seen = np.frombuffer(buf, np.float16).reshape(meta["height"], meta["width"], 4)
    step = max(1, meta["width"] // 500)
    return seen[::step, ::step, :3].astype(np.float32)


def unlike(seen: np.ndarray, small: np.ndarray) -> float:
    """The share of `seen`'s points (h, w, 3: the GPU's development of a raw file, linear,
    0..1, as bytes, one full-size pixel each, at about the places between small[:h + 1,
    :w + 1]'s) that match none of the four points of `small` around them (`small_copy` of
    LibRaw's half size: no demosaicing, so the four absorb the half step between the two, and
    an edge). About 0 for the same development; a channel missing, or every other pixel, is
    most of them."""
    high, wide = seen.shape[:2]
    off = np.inf
    for near in (small[a:a + high, b:b + wide] for a in (0, 1) for b in (0, 1)):
        each = np.abs(near - seen) - 0.25 * near  # (and an edge)
        off = np.minimum(off, np.maximum(np.maximum(each[..., 0], each[..., 1]), each[..., 2]))
    return float((off > 3 / 255).mean())  # (and the bytes' step)


def owned(buf, width: int, height: int, kind: str = "Format_RGBA16FPx4") -> QImage:
    """A float image that owns its pixels, copied out of `buf`. By numpy and not by Qt's own
    `copy`: that keeps Python's lock while it runs (50 ms for 33 MP), and the interface, which
    asks Python at every turn, stood still at every photo opened; numpy lets go of it."""
    image = QImage(width, height, getattr(QImage.Format, kind))
    size = image.sizeInBytes()
    to, of = np.frombuffer(image.bits(), np.uint8, size), np.frombuffer(buf, np.uint8, size)

    def part(k: int) -> None:  # a quarter of it (15 ms for 33 MP on four threads, 42 on one)
        to[size * k // 4:size * (k + 1) // 4] = of[size * k // 4:size * (k + 1) // 4]

    with ThreadPoolExecutor(4) as parts:
        list(parts.map(part, range(4)))
    return image


class Recent:
    """The decoded pictures of the photos opened last, kept so that going back to one shows it
    at once and sharp (the owner: "when im switching between two same compositions to see if
    one is better quickly, i always see the low quality render and have to wait for it to
    load every time"). Up to `room` bytes of them (a 33 MP photo is 330 MB in its two sizes),
    the one not looked at for longest let go first. A file that has changed since is not the
    photo that was kept. The half sizes of the photos in `keep` (the open one and those
    beside it, which PhotoView holds on the GPU) are not let go: a step there finds its
    description here, and nothing would ask for the picture again.

    ponytail: the pictures are kept in memory, not on the GPU: one that is gone back to is
    sent there again (some tens of milliseconds for 33 MP), unless it is beside the open
    photo (`Session.held`). The upgrade is to keep more textures."""

    def __init__(self, room: int | None = None) -> None:
        """`room`: unless told, a sixteenth of the machine's memory, 2 GB at most (six photos
        of 33 MP)."""
        try:
            memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (AttributeError, ValueError, OSError):
            memory = 16 << 30  # ponytail: Windows is not asked, 16 GB is assumed (so 1 GB here)
        self.room = min(2 << 30, memory // 16) if room is None else room
        self.kept, self.lock, self.keep = OrderedDict(), threading.Lock(), set()

    def _held(self) -> int:
        return sum(what[1].sizeInBytes() for what in self.kept.values())

    @staticmethod
    def _name(path: Path, half: bool) -> tuple:
        try:
            about = path.stat()
        except OSError:
            return path, half, None
        return path, half, (about.st_mtime_ns, about.st_size)

    def has(self, path: Path, half: bool) -> bool:
        with self.lock:
            return self._name(path, half) in self.kept

    def get(self, path: Path, half: bool):
        """(what the decoder said of it, its picture, its thinned copy), or None."""
        with self.lock:
            name = self._name(path, half)
            if name in self.kept:
                self.kept.move_to_end(name)
            return self.kept.get(name)

    def put(self, path: Path, half: bool, what: tuple) -> None:
        with self.lock:
            self.kept[name := self._name(path, half)] = what
            while self._held() > self.room:
                old = next((n for n in self.kept
                            if n != name and not (n[1] and n[0] in self.keep)), None)
                if old is None:
                    break
                del self.kept[old]


class Session(QObject):
    changed = Signal()
    scoped = Signal()  # the histogram was counted anew, or the working range changed
    brushed = Signal("QVariantMap")  # a painted stroke went on: what the GPU paints (`_brush_on`)
    scopeTaken = Signal()  # a render asked for by `grabScope` is back
    dotted = Signal()  # the wheel's dots are other dots than before
    counted = Signal(int, object)  # from the counting thread: (generation, what it found)
    # The lists of the edit, and the checkpoints, say so themselves when they change: what
    # draws them (the wheel with its dots, a curve, a list) is then not drawn anew at every
    # change of a slider.
    pinsChanged = Signal()
    curveChanged = Signal()
    chromaChanged = Signal()
    eqChanged = Signal()
    bwChanged = Signal()
    grainCurveChanged = Signal()
    lutsChanged = Signal()
    masksChanged = Signal()
    listed = Signal()  # the photos at hand are other photos (the filmstrip's row)
    saved = Signal()   # the edit has come to rest (and is in the catalog, where there is one)
    checkpointsChanged = Signal()
    LISTS = ("pins", "curve", "chroma", "eq", "bw", "grainCurve", "luts", "masks")
    photoChanged = Signal()  # another photo is shown: its view starts afresh
    described = Signal(int, object)  # from a loader thread: (generation, the photo's description)
    heldChanged = Signal()  # other photos beside the open one can be drawn at once (`held`)
    denoiser = None     # the neural denoiser (make_view gives the session one)
    _denoised = False   # the open photo is a file the neural denoiser wrote

    def __init__(self, photos: list[Path], load=decode.load, hdr: bool | None = None,
                 settings: QSettings | None = None, catalog: Catalog | None = None) -> None:
        """`settings` remembers the SDR/HDR choice and the peak, `catalog` keeps the edits;
        without them nothing is kept (the checks and the shots). `hdr` given outright (--hdr)
        wins over what is remembered."""
        super().__init__()
        # What the interface reads many times over after every change (the shader's numbers,
        # the edit, which tools are in use) is worked out once per change: `refresh` empties
        # this before it says `changed`.
        self._memo: dict = {}
        self._told: dict = {}  # the lists as the interface was last told of them
        # `exporter`: set by whoever makes one (make_view). `ahead`: by whoever can decode
        # ahead (decode.Workers.ahead). `_way`: which way the user last stepped through the
        # photos. `_seen`: the photo (its generation) whose first picture has arrived.
        self.exporter, self.ahead, self._way, self._seen, self._listing = None, None, 1, -1, 0
        # (Once the open photo's own pictures are in, or after a while if one never comes:
        # until then nothing else is decoded, the photo asked for has the machine to itself.)
        # (And `holder` looks again at what is `held`, while a photo beside the open one is
        # still being decoded ahead.)
        self.reader, self.holder = (
            QTimer(self, singleShot=True, interval=1500, timeout=self._read_ahead),
            QTimer(self, singleShot=True, interval=50, timeout=self._hold))
        self.changed.connect(self._lists)
        self.photos, self.load, self.settings, self.catalog = photos, load, settings, catalog
        self._index, self._generation = 0, 0
        self._wants_hdr = self._kept("view/hdr", False, bool) if hdr is None else hdr
        self._asked_hdr = hdr is True  # asked for outright: say so if it cannot be had
        self._surface = None  # not known before a frame
        self._linear = os.environ.get("PHOTONE_HDR_ENCODING") == "linear"
        self._headroom = 2.0  # until the desktop says otherwise (set_headroom)
        self._error, self._info, self._failed, self._size = "", "", False, (0, 0)
        self._camera, self._raw, self._original, self._unwritten = None, False, False, False
        self._bypass, self._bypass_local, self._small = "", (0, ()), None
        self._start_scopes()
        self.history = self._history()
        # (A moment after a change.)
        self.saver = QTimer(self, singleShot=True, interval=400, timeout=self.flush)
        self.described.connect(self._described)
        self.threads = ThreadPoolExecutor(2, "decode")  # a photo's two sizes at once
        # (`held`, and what `ahead` has decoded or is decoding.)
        self.fetching, self.recent, self._held, self._before = {}, Recent(), [], {}
        # The generation of the photo on show, for a thread to look at; the same and its place
        # among the photos, for Qt's image thread.
        self._open, self._on = [0], (0, 0)
        self._fetch()

    def _start_scopes(self) -> None:
        """What is counted from the picture, and what of the interface is open."""
        self._histogram, self._lab, self._after, self._cloud = {}, None, None, None
        self._inside = None  # which pixels of that count are the wheel's (scopes.inside_of)
        self._range = ("full", 0.0, 1.0)
        self._panels = {"tool": self._kept("view/tool", "tone", str),
                        "clipping": False,
                        "live": self._kept("view/live", True, bool), "chart": False,
                        # (Where the window was when it was last closed: the library, or Edit.)
                        "place": self._kept("view/place", "library", str)}
        # The painted masks' picture: how often it changed and of which strokes, and what
        # was painted last time (a stroke going on is added to that).
        self._paint, self._rasters, self._paint_lock = (0, []), {}, threading.Lock()
        # A stroke is being made, and how far the piece of it under way has gone (`_brush_on`).
        self._brushing, self._brush_gone = False, 0.0
        self._colors = (None, None, 0)  # that picture as it was last made: its shape, bytes, masks
        self.counting = ThreadPoolExecutor(1, "count")  # the scopes, one count after another
        self._wanted, self._wanting = None, threading.Lock()  # the render to count next
        self._grab = None  # the render of the scopes that is under way (`grabScope`)
        self.counted.connect(self._counted)

    def refresh(self) -> None:
        """Say that something changed: every change to the session ends here. What was worked
        out before is forgotten first, not in a slot of `changed`: Qt tells the interface's
        bindings ahead of any slot, and they would read what is no longer so (a photo that
        did not turn until the next change, after the pointer had been over it)."""
        self._memo.clear()
        self.changed.emit()

    @Slot()
    def _lists(self) -> None:
        """After a change: tell the interface of each list that is no longer what it was."""
        if not hasattr(self, "history"):
            return  # still being set up
        now = {key: self.history.edit.get(key, edit.DEFAULTS[key]) for key in self.LISTS}
        now["checkpoints"] = self._checkpoints()
        now["masks"] = self.masks  # (where they are on show changes with the frame, too)
        now["curve"] = self.curve  # (at rest, the photo's own: there once the file is decoded)
        for key, value in now.items():
            if value != self._told.get(key):
                self._told[key] = value
                getattr(self, f"{key}Changed").emit()

    @Property("QVariantList", notify=pinsChanged)
    def pins(self) -> list:
        return self.history.edit.get("pins", [])

    @Property("QVariantList", notify=masksChanged)
    def masks(self) -> list:
        """The photo's masks (edit.MASK), their places in the frame as it is on show: 0..1
        from its top left, where they are drawn and dragged."""
        def placed() -> list:
            matrix = self._frame()["matrix"]
            # (Of a painted one only how many strokes it has: the strokes themselves are
            # thousands of numbers, and the interface draws none of them.)
            return [{**mask, "strokes": len(mask["strokes"])} if mask["kind"] == "brush"
                    else {**mask, "at": list(edit.unframed(matrix, *mask["at"])),
                          "to": list(edit.unframed(matrix, *mask["to"]))}
                    for mask in self.history.edit.get("masks", [])]

        return self._once("masks", placed)

    @Slot(str, result=int)
    def addMask(self, kind: str) -> int:  # noqa: N802
        """A new mask in the middle of what is on show: its number (1, 2, …), or 0."""
        masks = list(self.history.edit.get("masks", []))
        if len(masks) >= edit.MASKS:
            self.say("A photo can have eight masks. Delete one to make another.")
            return 0
        if kind == "brush" and sum(m["kind"] == "brush" for m in masks) >= edit.BRUSHES:
            self.say("A photo can have three painted masks. Delete one to make another.")
            return 0
        matrix, to = self._frame()["matrix"], (0.5, 0.7) if kind == "linear" else (0.78, 0.5)
        masks.append({"kind": "brush", "strokes": []} if kind == "brush" else
                     {"kind": kind, "at": edit.framed(matrix, 0.5, 0.5),
                      "to": edit.framed(matrix, *to)})
        self.history.commit()
        self.setEdits({"masks": masks})
        self.history.commit()
        return len(masks) if len(self.history.edit.get("masks", [])) == len(masks) else 0

    @Slot(int, "QVariantMap")
    def setMask(self, number: int, values: dict) -> None:  # noqa: N802
        """Change mask `number` (1, 2, …). `at` and `to` are places in the frame as it is on
        show; moving `at` alone takes `to` along, that is, moves the mask."""
        masks = [dict(mask) for mask in self.history.edit.get("masks", [])]
        if not 1 <= number <= len(masks):
            return
        mask, matrix = masks[number - 1], self._frame()["matrix"]
        try:
            places = {k: list(edit.framed(matrix, *values[k])) for k in ("at", "to") if k in values}
        except TypeError:
            return  # not a place
        if "at" in places and "to" not in places:
            places["to"] = [mask["to"][i] + places["at"][i] - mask["at"][i] for i in (0, 1)]
        mask.update({**values, **places})
        self.setEdits({"masks": masks})

    @Slot(int, "QVariantList", float, float, bool, bool)
    def paint(self, number: int, places: list, size: float, feather: float, erase: bool,  # noqa: PLR0913
              begin: bool) -> None:
        """Paint on mask `number` (a painted one) with the brush, or rub out: through these
        places of the frame as it is on show. `begin`: a new stroke; else the last one goes
        on. `size`: the brush's radius as a share of the frame's width."""
        masks = [dict(mask) for mask in self.history.edit.get("masks", [])]
        if not (1 <= number <= len(masks) and masks[number - 1]["kind"] == "brush" and places):
            return
        frame = self._frame()
        try:
            points = [list(edit.framed(frame["matrix"], *place)) for place in places]
        except TypeError:
            return  # not places
        strokes = list(masks[number - 1]["strokes"])
        begin = begin or not strokes
        if begin:
            strokes.append({"size": size * frame["size"][0] / max(1, self._stored_size()[0]),
                            "feather": feather, "erase": erase, "points": points})
        else:
            strokes[-1] = {**strokes[-1], "points": strokes[-1]["points"] + points}
        if len(strokes) > edit.STROKES or len(strokes[-1]["points"]) > edit.REACH:
            return self.say("This mask holds no more strokes. Paint the rest on a second mask.")
        masks[number - 1]["strokes"] = strokes
        had = 0 if begin else len(strokes[-1]["points"]) - len(points)
        self._brushing = True   # (before the change is told: the picture is not asked for anew)
        self.setEdits({"masks": masks})
        self._brush_on(number, had)

    def _brush_on(self, number: int, had: int) -> None:
        """Tell the interface how the stroke just made longer goes on from its place `had`,
        for the GPU to paint it (PhotoView.qml, `brush`): as edit.Painter will paint it once
        it is over, from the places as they are kept."""
        masks = self.history.edit.get("masks", [])
        channel = sum(mask["kind"] == "brush" for mask in masks[:number - 1])
        strokes = masks[number - 1]["strokes"] if number <= len(masks) else []
        stroke = strokes[-1] if strokes else None
        if stroke is None or len(stroke["points"]) <= had or channel >= edit.BRUSHES:
            return  # (the change was not taken)
        stored = self._stored_size()
        wide, high = edit.raster_size(stored[0] / max(1, stored[1]))
        gone = self._brush_gone if had else 0.0
        steps, self._brush_gone = edit.stretches(stroke, wide, high, had, gone)
        radius, core = edit.brush_of(stroke, wide)
        self.brushed.emit({"channel": channel, "begin": had == 0, "radius": radius, "core": core,
                           "erase": stroke["erase"], "wide": wide, "high": high,
                           "steps": [[*map(float, a), *map(float, b), bool(turn)]
                                     for a, b, turn in steps]})

    @property
    def luts(self) -> Path:
        """Where the LUTs that were brought in are kept, each under what its file holds:
        beside the catalog, or (with none: the checks) where nothing stays."""
        home = self.catalog.path.parent if self.catalog else Path(tempfile.gettempdir()) / "photone"
        return self.luts_home or home / "luts"

    luts_home: Path | None = None  # (the checks: a folder of their own)

    # ---- LUTs: those installed in Photone, and those laid on the open photo ----------------

    installedChanged = Signal()

    def _shelf(self) -> list[dict]:
        """The LUTs that are installed, by name: `id` (what its file is kept under), `name`,
        `size`, `print` (lut.fingerprint: what tells a LUT that is installed already) and
        `also` (other ids it is found by: what its file's text is told by, which is how a
        photo edited on the first day names it). Kept in a list beside their files; a file
        that is there without being in it (put there by hand, or by the first day's Photone)
        is taken in under the name it gives itself."""
        if self._installed is None:
            self._stale = False
            found = {one["id"]: one for one in self._listed_before()}
            for file in sorted(self.luts.glob("*.cube")) if self.luts.is_dir() else []:
                read = self._read(file.stem) if file.stem not in found else None
                if read:
                    self._stale = True
                    named = file_named(lut.title(read[0]) or file.stem)
                    found[file.stem] = {"id": file.stem, "name": named,
                                        "size": len(read[1]), "print": lut.fingerprint(read[1]),
                                        "also": [lut.written(read[0])]}
            self._installed = sorted(found.values(), key=lambda one: one["name"].lower())
            if self._stale:  # (what was worked out is written down: not again the next time)
                self._keep_shelf()
        return self._installed

    def _read(self, name: str, parsed: bool = True) -> tuple[str, np.ndarray | None] | None:
        """An installed LUT's file as (its text, its table); None if it is not there, or
        no LUT. `parsed` False: the text alone."""
        try:
            if not name.isalnum():
                return None
            text = (self.luts / f"{name}.cube").read_text("utf-8", errors="replace")
            return text, lut.parse(text) if parsed else None
        except (OSError, ValueError):
            return None

    def _listed_before(self) -> list[dict]:
        """What the list beside the files says is installed, as far as it is true: each one
        valid, its file there; and what the list did not say yet (`print`, `also`) worked out
        from the file."""
        try:
            listed = json.loads((self.luts / "index.json").read_text("utf-8"))
        except (OSError, ValueError):
            return []
        found = []
        for one in listed if isinstance(listed, list) else []:
            with contextlib.suppress(ValueError, KeyError, TypeError):
                had = edit.valid_lut({**one, "strength": 100.0})
                told, also = one.get("print"), one.get("also")
                if not told or not isinstance(also, list):
                    self._stale = True
                    read = self._read(had["id"], parsed=not told)
                    told, also = told or lut.fingerprint(read[1]), [lut.written(read[0])]
                also = [str(name) for name in also if str(name).isalnum()][:8]
                named = file_named(had["name"])
                self._stale |= named != had["name"]  # (written down with it, once)
                if (self.luts / f"{had['id']}.cube").is_file():
                    found.append({"id": had["id"], "name": named, "size": had["size"],
                                  "print": str(told), "also": also})
        return found

    _installed: list | None = None
    _stale = False
    _lut_serial = 0
    twice: list = []  # (file, the installed LUT it is the same as), of the last install

    def _keep_shelf(self) -> None:
        with contextlib.suppress(OSError):
            part = self.luts / "index.part"
            part.write_text(json.dumps(self._installed, indent=1), "utf-8")
            os.replace(part, self.luts / "index.json")

    def _listed(self, now: list[dict]) -> None:
        self._installed = sorted(now, key=lambda one: one["name"].lower())
        self._keep_shelf()
        # (What is installed is another: the picture of the photo's LUTs is asked for anew,
        # under another address, and the photo's list says again which of them are gone.)
        self._lut_serial += 1
        self._tables.clear()
        self.installedChanged.emit()
        self.lutsChanged.emit()

    def _found(self, name: str) -> dict | None:
        """The installed LUT a photo means by that id: the one kept under it, or one that is
        also found by it."""
        shelf = self._shelf()
        return (next((one for one in shelf if one["id"] == name), None)
                or next((one for one in shelf if name in one["also"]), None))

    @Property("QVariantList", notify=installedChanged)
    def lutsInstalled(self) -> list:  # noqa: N802
        return self._shelf()

    def _install(self, path: Path) -> dict:
        """Take a .cube or .clf file in: a copy is kept under what its table holds
        (lut.fingerprint), so nothing hangs on where the file was. One whose table is
        installed already, under whatever name and however its file is written, is not
        installed again: `twice` then names the file and the one that is there. The LUT as
        the list has it, or ValueError saying what is wrong."""
        try:
            if path.stat().st_size > lut.BYTES:
                raise ValueError("the file is too large to be a LUT")
            text = path.read_text("utf-8", errors="replace")
            table = lut.parse(text)
            name, other = lut.fingerprint(table), lut.written(text)
            there = next((had for had in self._shelf() if had["print"] == name), None)
            if there:
                self.twice.append((path.name, there["name"]))
                if other not in (there["id"], *there["also"]):  # (found by this file's id too)
                    there = {**there, "also": [*there["also"], other][-8:]}
                    self._listed([there if one["id"] == there["id"] else one
                                  for one in self._shelf()])
                return there
            self.luts.mkdir(parents=True, exist_ok=True)
            part = self.luts / f"{name}.part"
            # (A .clf file's table is kept as a .cube: all that reads it after reads those.)
            clf = path.suffix.lower() == ".clf"
            part.write_text(lut.cube(table, path.stem + ".clf") if clf else text, "utf-8")
            os.replace(part, self.luts / f"{name}.cube")
        except OSError as trouble:
            raise ValueError(str(trouble)) from trouble
        one = {"id": name, "name": path.stem[:75] + (".clf" if clf else ".cube"),
               "size": len(table), "print": name, "also": [other]}
        self._listed([*self._shelf(), one])
        return one

    @Slot("QVariantList")
    def installLuts(self, files: list) -> None:  # noqa: N802
        """Install these .cube or .clf files (URLs or paths). Says which could not be used."""
        paths = [Path(QUrl(f).toLocalFile() if "://" in str(f) else str(f)) for f in
                 (f.toString() if isinstance(f, QUrl) else f for f in files)]
        said, self.twice = [], []
        for path in paths:
            try:
                self._install(path)
            except ValueError as trouble:
                said.append(f"{path.name} cannot be used as a LUT: {trouble}.")
        said += [f"{file} is installed already" + (f", as “{name}”." if name != file else ".")
                 for file, name in self.twice]
        self.say(" ".join(said))

    @Slot(str)
    def uninstallLut(self, which: str) -> None:  # noqa: N802
        """Take a LUT out of Photone: its file goes. Photos it is laid on keep their entry,
        which says that it is gone and changes nothing."""
        if any(one["id"] == which for one in self._shelf()):
            (self.luts / f"{which}.cube").unlink(missing_ok=True)
            self._listed([one for one in self._shelf() if one["id"] != which])

    def _luts(self) -> list[dict]:
        return list(self.history.edit.get("luts", []))

    def _put_luts(self, now: list[dict]) -> None:
        self.history.commit()
        self.history.put({"luts": now})
        self.history.commit()
        self._edited()

    @Slot(str, result=int)
    @Slot(str, int, result=int)
    def addLut(self, which: str, at: int = -1) -> int:  # noqa: N802
        """Lay an installed LUT on the open photo at that place of its list (after those it
        has, with none given), at full strength, as one step of undo. Its place in the
        photo's list, or -1 (there is room for edit.LUTS)."""
        one = next((had for had in self._shelf() if had["id"] == which), None)
        if not (one and self.photos):
            return -1
        if len(self._luts()) >= edit.LUTS:
            self.say(f"A photo can have {edit.LUTS} LUTs: take one away first.")
            return -1
        now = self._luts()
        at = len(now) if at < 0 else min(at, len(now))
        self._put_luts([*now[:at], {**one, "strength": 100.0}, *now[at:]])
        return at

    @Slot(int, "QVariantMap")
    def setLut(self, at: int, values: dict) -> None:  # noqa: N802
        """Change one of the photo's LUTs: its `strength`, or `off`. (A drag of the slider is
        one step once it is let go: commitEdit.)"""
        now = self._luts()
        if 0 <= at < len(now):
            now[at] = {**now[at], **{k: values[k] for k in ("strength", "off") if k in values}}
            self.setEdits({"luts": now})

    @Slot(int)
    def removeLut(self, at: int) -> None:  # noqa: N802
        now = self._luts()
        if 0 <= at < len(now):
            self._put_luts(now[:at] + now[at + 1:])

    @Slot(int, int)
    def moveLut(self, at: int, to: int) -> None:  # noqa: N802
        """Another place in the order they are laid on."""
        now = self._luts()
        if 0 <= at < len(now) and 0 <= to < len(now) and at != to:
            now.insert(to, now.pop(at))
            self._put_luts(now)

    @Slot(str)
    def importLut(self, file: str) -> None:  # noqa: N802
        """A .cube or .clf file dropped on the photo: installed, and laid on the open photo."""
        path = Path(QUrl(file).toLocalFile() if "://" in file else file)
        if not self.photos:
            return self.say("Open a photo first: a LUT is laid on a photo.")
        try:
            self.twice = []
            self.addLut(self._install(path)["id"])
        except ValueError as trouble:
            self.say(f"{path.name} cannot be used as a LUT: {trouble}.")

    @Property("QVariantMap", notify=changed)
    def lensHas(self) -> dict:  # noqa: N802
        """Which lens corrections the open photo's file carries the camera's figures for."""
        found = (self._camera or {}).get("lens") or {}
        return {"distortion": "distortion" in found, "color": "red" in found,
                "vignetting": "vignetting" in found}

    @Property(int, constant=True)
    def lutsMost(self) -> int:  # noqa: N802
        """How many LUTs a photo can have."""
        return edit.LUTS

    @Property("QVariantList", notify=lutsChanged)
    def lutsOn(self) -> list:  # noqa: N802
        """The LUTs laid on the open photo, in order, each under the name the installed LUT
        has now (it may have been installed again under another); `lost` on one that is not
        installed (it changes nothing then)."""
        found = [(one, self._found(one["id"])) for one in self._luts()]
        return [{**one, "name": there["name"] if there else one["name"], "lost": there is None}
                for one, there in found]

    @Property(str, notify=lutsChanged)
    def lutSource(self) -> str:  # noqa: N802 - read from QML
        """The picture of the open photo's LUTs for the shader, or nothing."""
        had = self._luts()
        return f"image://photo/lut/{lut.address(had)}/{self._lut_serial}" if had else ""

    _tables: dict = {}  # the installed LUTs' tables that were read, by id (None: no LUT there)

    def _lut_at(self, which: str) -> QImage | None:
        """Some LUTs (lut.address) as the one picture the shader reads them from; one whose
        file is gone or is no LUT any more, as a table that changes nothing. On Qt's image
        loader thread."""
        try:
            tables = []
            for name, size in lut.asked(which):
                if name not in self._tables:  # (read once: a large one takes a moment)
                    there = self._found(name)
                    read = self._read(there["id"]) if there else None
                    self._tables[name] = read[1] if read else None
                table = self._tables[name]
                there = table is not None and len(table) == size
                tables.append(table if there else lut.identity(size))
        except ValueError:
            return None
        flat = lut.stacked(tables)
        high, wide = flat.shape[:2]
        return QImage(flat.tobytes(), wide, high, wide * 8, QImage.Format.Format_RGBA16FPx4).copy()

    @Property(str, notify=changed)
    def paintedSource(self) -> str:  # noqa: N802 - read from QML
        """The picture of the photo's painted masks (`painted`), for the shader; another
        address whenever a stroke was added. Empty when the photo has none."""
        strokes = [mask["strokes"] for mask in self.history.edit.get("masks", [])
                   if mask["kind"] == "brush"]
        # (Not while a stroke is being made: the GPU paints it meanwhile, and its picture is
        # asked for when it is over. Made at every move of the brush, a large one cost
        # milliseconds a stretch and the whole picture went to the GPU each time.)
        if not self._brushing and strokes != self._paint[1]:
            self._paint = (self._paint[0] + 1, strokes)
        return f"image://photo/painted/{self._generation}/{self._paint[0]}" if strokes else ""

    def painted(self, strokes: list, shape: float) -> QImage:
        """Painted masks' strokes (one list each) as the picture the shader reads: one mask
        in each of red, green and blue (edit.painted). A stroke going on is painted onto
        what was kept from last time, not everything anew."""
        wide, high = edit.raster_size(shape)
        with self._paint_lock:
            # (Kept, and only where a stroke went on written anew: made whole at every move
            # of the brush it was some milliseconds of each frame.)
            was, colors, used = self._colors
            if was != shape:
                colors, used = np.zeros((high, wide, 4), dtype=np.uint8), 0
                colors[..., 3] = 255
            for channel, mine in enumerate(strokes[:edit.BRUSHES]):
                painter, anew = self._raster(channel, mine, shape)
                shown = painter.bytes()
                box = (slice(None), slice(None)) if anew or was != shape else painter.fresh
                if box:
                    colors[(*box, channel)] = shown[box]
            colors[..., len(strokes):max(used, len(strokes))] = 0  # (of masks that are gone)
            self._colors = (shape, colors, min(len(strokes), edit.BRUSHES))
            return QImage(colors.data, wide, high, wide * 4, QImage.Format.Format_RGBX8888).copy()

    def _raster(self, channel: int, strokes: list, shape: float) -> tuple:
        """That mask's painter with these strokes on it, and whether it was begun anew."""
        shape_was, old, painter = self._rasters.get(channel, (None, [], None))
        same = shape_was == shape and strokes and painter is not None
        grown = same and len(strokes) == len(old) and strokes[:-1] == old[:-1] \
            and {**strokes[-1], "points": 0} == {**old[-1], "points": 0} \
            and strokes[-1]["points"][:len(old[-1]["points"])] == old[-1]["points"]
        if grown:  # the last stroke went on
            if len(strokes[-1]["points"]) > len(old[-1]["points"]):
                painter.add(strokes[-1], len(old[-1]["points"]))
        elif same and strokes[:-1] == old:  # one stroke more
            painter.begin()
            painter.add(strokes[-1])
        else:
            painter = edit.Painter(shape)
            for stroke in strokes:
                painter.begin()
                painter.add(stroke)
        self._rasters[channel] = (shape, strokes, painter)
        return painter, not (grown or (same and strokes[:-1] == old))

    @Slot(int)
    def deleteMask(self, number: int) -> None:  # noqa: N802
        """Take a mask away, with the pins that were kept to it, as one step of undo."""
        if 1 <= number <= len(self.history.edit.get("masks", [])):
            self.history.commit()
            self.setEdits(edit.without_mask(self.history.edit, number))
            self.history.commit()

    @Property("QVariantList", notify=curveChanged)
    def curve(self) -> list:
        return edit.curve_of(self.history.edit, self._camera)

    @Property("QVariantList", notify=chromaChanged)
    def chroma(self) -> list:
        return self.history.edit.get("chroma", edit.STRAIGHT)

    @Property("QVariantList", notify=eqChanged)
    def eq(self) -> list:
        return self.history.edit.get("eq", [])

    @Property("QVariantList", notify=grainCurveChanged)
    def grainCurve(self) -> list:  # noqa: N802
        return self.history.edit.get("grainCurve", edit.DEFAULTS["grainCurve"])

    @Property("QVariantList", notify=grainCurveChanged)
    def grainLine(self) -> list:  # noqa: N802
        """Film grain's response as heights (edit.LINE) from black to white."""
        points = self.grainCurve
        return self._once("grainLine", lambda: edit.curve_line(points))

    @Property("QVariantMap", constant=True)
    def grains(self) -> dict:
        """What film grain can be set to at one press: `characters` (an amount, a size and a
        roughness each) and `responses` (the response curve's points)."""
        return {"characters": edit.CHARACTERS, "responses": edit.RESPONSES}

    @Property("QVariantList", notify=bwChanged)
    def bw(self) -> list:
        return self.history.edit.get("bw", edit.DEFAULTS["bw"])

    def _once(self, name: str, make):
        """`make()`, worked out once until the next change."""
        if name not in self._memo:
            self._memo[name] = make()
        return self._memo[name]

    @Slot(str, str, result=str)
    def remembered(self, key: str, otherwise: str) -> str:
        """Something the interface keeps between runs (how to export, say), as text."""
        return str(self._kept(key, otherwise, str))

    @Slot(str, str)
    def remember(self, key: str, value: str) -> None:
        if self.settings:
            self.settings.setValue(key, value)

    def _kept(self, key: str, default, kind):
        return self.settings.value(key, default, type=kind) if self.settings else default

    @Property(str, constant=True)
    def version(self) -> str:
        return __version__

    @Property(str, constant=True)
    def homepage(self) -> str:
        return HOMEPAGE

    # ---- the photo ---------------------------------------------------------------------

    def _address(self, size: str, index: int | None = None) -> str:
        """One address per photo and size while the photos at hand stay the same: PhotoView
        keeps the half size of a photo beside the open one under it (`held`), and a step there
        draws that, as it is."""
        return f"image://photo/{self._listing}/{self._index if index is None else index}/{size}"

    @Property(str, notify=changed)
    def source(self) -> str:
        """What the interface loads first; empty when nothing is open. For a raw file it is the
        half-size picture: quick to decode, and all a view of the whole photo needs. With
        the test chart on show it is the chart."""
        if self.photos and self._panels["chart"]:
            return f"image://photo/{self._listing}/chart/half"
        return self._address("half") if self.photos else ""

    @Property(str, notify=changed)
    def fullSource(self) -> str:  # noqa: N802 - read from QML
        """The same photo at its full size, fetched straight after, so that it is already
        there when the view goes closer than the half-size pixels can show. Empty where the
        first picture is the full size already (anything that is not a raw file), and until
        the first picture has arrived: Qt hands pictures over one at a time, and the large one,
        when it has been decoded ahead, must not hold up the one that is shown first."""
        raw = self.photos and self.photos[self._index].suffix.lower() in decode.RAW
        seen = self._seen == self._generation
        return self._address(self._full()) if raw and seen and not self._panels["chart"] else ""

    def _full(self) -> str:
        """The size the open photo's full size is asked for at: "cpu" if the GPU developed it
        wrong (`checkDeveloped`)."""
        return "cpu" if self.photos[self._index] in self.cpu else "full"

    @Property("QStringList", notify=heldChanged)
    def held(self) -> list:
        """The half-size pictures of the photos beside the open one that can be had at once
        (kept, or decoded ahead): PhotoView keeps them on the GPU, so that a step there shows
        the photo, through its edit, in the first frame after the key (the owner: "I would
        like switching between photos ... to be instant")."""
        return self._held

    def _hold(self) -> None:
        held = []
        for at in (self._index + 1, self._index - 1):
            if not 0 <= at < len(self.photos):
                continue
            path, address = self.photos[at], self._address("half", at)
            ahead = self._before.get((path, True))
            # (One held stays so while it is beside the open photo: on its way from the
            # decoder to Qt it is in neither.)
            if address in self._held or self.recent.has(path, True) or (
                    ahead and ahead.done() and not ahead.cancelled() and not ahead.exception()):
                held.append(address)
            elif ahead and not ahead.done():  # (one that failed is not held, nor looked at again)
                self.holder.start()
        if held != self._held:
            self._held = held
            self.heldChanged.emit()

    @Property(str, notify=changed)
    def info(self) -> str:
        return self._info

    @Property(str, notify=changed)
    def name(self) -> str:
        """The open photo's file name."""
        return self.photos[self._index].name if self.photos else ""

    @Property(int, notify=changed)
    def fullWidth(self) -> int:  # noqa: N802 - read from QML
        """The width of the photo's frame in its real pixels: what 100 % means. (Of the test
        chart's while that is on show.)"""
        return self._frame()["size"][0] if self._size[0] else 0

    @Property(int, notify=changed)
    def fullHeight(self) -> int:  # noqa: N802
        return self._frame()["size"][1] if self._size[1] else 0

    @Property(int, notify=changed)
    def wide(self) -> int:
        """The stored picture's real width, whatever the frame: what the view goes by to tell
        whether it is closer than the first picture can show."""
        return self._stored_size()[0]

    def _stored_size(self) -> tuple:
        return CHART if self._panels["chart"] and self._size[0] else self._size

    def _frame(self) -> dict:
        """The photo's frame (edit.frame): its turns, its straightening, its crop. Loose
        while the Crop tool is open: all of the picture, turned by the angle so that the frame,
        which the tool draws on it, lies level."""
        return self._once("frame", lambda: edit.frame(
            self.history.edit if self._bypass != "crop" else {}, *self._stored_size(),
            loose=self._panels["tool"] == "crop"))

    @Property("QVariantMap", notify=changed)
    def framing(self) -> dict:
        """For the Crop tool: `frame` (left, top, right, bottom of the photo's frame in the
        tool's view, 0..1), `angle`, `turns`, the view's `shape` (its width over its height)
        and the `picture`'s as it stands after the turns."""
        return self._once("framing", self._framing)

    def _framing(self) -> dict:
        full = {**edit.DEFAULTS, **self.history.edit}
        wide, high = self._stored_size()
        if not (wide and high and self._panels["tool"] == "crop"):  # (nobody is looking)
            return {"frame": [0.0, 0.0, 1.0, 1.0], "angle": 0.0, "turns": 0.0, "shape": 1.0,
                    "picture": 1.0}
        return {"frame": edit.frame_in_view(self.history.edit, wide, high),
                "angle": full["angle"], "turns": full["turns"],
                "shape": edit.frame(self.history.edit, wide, high, loose=True)["shape"],
                "picture": high / wide if int(full["turns"]) % 2 else wide / high}

    @Slot("QVariantList", "QVariantList", bool)
    def setFrame(self, start: list, wanted: list, together: bool) -> None:  # noqa: N802
        """The Crop tool's frame, dragged from `start` toward `wanted` (left, top, right,
        bottom in the tool's view): as far as it stays inside the picture (edit.crop_for)."""
        try:
            crop = edit.crop_for(self.history.edit, self._stored_size(), start, wanted, together)
        except (TypeError, ValueError):
            return  # not a frame
        self.setEdits({"crop": crop})

    @Property("QVariantMap", notify=changed)
    def inFrame(self) -> dict:  # noqa: N802
        """The shader's numbers for what the scopes count: the photo in its own frame, also
        while the Crop tool shows all of the picture."""
        if self._panels["tool"] != "crop" or self._bypass == "crop":
            return self.uniforms
        tight = edit.frame(self.history.edit, *self._stored_size())["matrix"]
        return self._once("inFrame", lambda: {**self.uniforms, "frame": shaders.by_columns(tight)})

    @Property(int, notify=changed)
    def count(self) -> int:
        return len(self.photos)

    @Property(int, notify=changed)
    def index(self) -> int:
        """Which of the photos at hand is open."""
        return self._index

    @Property(int, notify=listed)
    def listing(self) -> int:
        """Another number whenever the photos at hand are other photos."""
        return self._listing

    @Slot(int, result=str)
    def fileAt(self, index: int) -> str:  # noqa: N802
        return str(self.photos[index]) if 0 <= index < len(self.photos) else ""

    @Slot(int)
    def goTo(self, index: int) -> None:  # noqa: N802
        """Show that one of the photos at hand (a click in the filmstrip)."""
        self.step(index - self._index)

    @Property(bool, notify=changed)
    def failed(self) -> bool:
        """The open photo could not be read; `error` says why."""
        return self._failed

    def _show(self, index: int) -> None:
        self.flush()
        self._index, self._failed = index, False
        self._panels = {**self._panels, "chart": False}
        self._camera, self._raw, self._small, self._histogram = None, False, None, {}
        self._denoised = False
        self._lab, self._after, self._range = None, None, ("full", 0.0, 1.0)
        self._inside = None
        self._redot()
        self.history = self._history()
        self._generation += 1
        self._on = (self._generation, index)
        self._info, self._error, self._size = "", "", (0, 0)
        here = self.photos[index]
        self.recent.keep = set(self.photos[max(0, index - 1):index + 2])
        if self.ahead:  # this photo first: of what was decoded ahead, only what is near it stays
            self._before = self.ahead([(here, True), (here, False), *self._near()], False) or {}
        self._hold()  # (before anyone is told: PhotoView keeps what it holds that is still wanted)
        self._fetch()
        # Kept from before (beside the photo that was open, or opened lately): its size and
        # description are known at once, and a picture PhotoView holds is drawn as it is.
        kept = self.recent.get(here, True)
        if kept:
            self._small = (self._generation, kept[2])
            self._described(self._generation, kept[0])  # (and refreshes)
        else:
            self.refresh()
        self.photoChanged.emit()

    @Slot("QVariant")
    def open(self, files) -> None:
        """Files chosen in the dialog or dropped on the window (URLs or paths). A .cube or .clf
        among them is a LUT for the open photo, not a photo."""
        paths = [Path(QUrl(f).toLocalFile() if "://" in str(f) else str(f)) for f in
                 (f.toString() if isinstance(f, QUrl) else f for f in files)]
        paths = [p for p in paths if p.is_file()]
        if not paths:
            return self.say("None of these files could be found.")
        tables = [p for p in paths if p.suffix.lower() in (".cube", ".clf")]
        if tables:
            return [self.importLut(str(table)) for table in tables] and None
        self._listed_anew(paths, 0)

    @Slot("QVariantList", int)
    def openAt(self, files: list, index: int) -> None:  # noqa: N802
        """Open these photos (the library's, as it shows them) and show the one at `index`."""
        self._listed_anew([Path(f) for f in files], index)

    def _listed_anew(self, photos: list, index: int) -> None:
        """Other photos are at hand, that one of them open. The edit is written first, while
        the open photo still has its place (written after, within the moment an edit waits to
        be saved, it went to the catalog under the path of whatever stood at that place among
        the new ones, or not at all); and the place is the new one before anybody is told,
        so that nobody asks for the photo at the old place among the new photos."""
        self.flush()
        self.photos = photos
        self._index = max(0, min(len(photos) - 1, index))
        self._relisted()
        if photos:
            self._show(self._index)

    def _relisted(self) -> None:
        self._listing += 1
        self.listed.emit()

    @Property(str, notify=changed)
    def path(self) -> str:
        """The open photo's file."""
        return str(self.photos[self._index]) if self.photos else ""

    @Slot(int)
    def step(self, by: int) -> None:
        """The next or previous of the open photos; stops at the ends."""
        index = max(0, min(len(self.photos) - 1, self._index + by))
        if self.photos and index != self._index:
            self._way = 1 if by > 0 else -1
            self._show(index)

    def _fetch(self) -> None:
        """Start decoding the photo shown, in both sizes for a raw file, before anything asks:
        the half size is then on screen as soon as it can be, and the full size shortly after
        whether or not the view needs it yet."""
        # Of a photo since left: not begun at all if it has not been (gone through quickly,
        # every folder's photo was decoded all the same, and the one stopped at waited behind
        # them: 2.8 s after twenty-five folders where it takes 0.2 s); else what it brings is
        # freed. (A list: Qt's image thread takes from `fetching` meanwhile.)
        for stale in list(self.fetching.values()):
            if not stale.cancel():
                stale.add_done_callback(lambda f: f.exception() or f.result()[1].close())
        self.fetching = {}
        if not self.photos:
            return
        path, generation = self.photos[self._index], self._generation
        load, open_now = self.load, self._open
        open_now[0] = generation
        # (Not what is kept from when the photo was last open: `image` hands that over.)
        first = None if self.recent.has(path, True) else self.threads.submit(load, path, True)
        if first:
            self.fetching[self._address("half").removeprefix("image://photo/")] = first

        def then_full():
            # After the quick picture, not beside it: decoded side by side the two got in each
            # other's way, and the picture of a photo not decoded ahead took 0.37 s to come
            # where it takes 0.2 s alone. And not at all for a photo the user has left.
            # (It holds nothing of the session itself: the last to let go of that must not be
            # this thread, or the session's timers are taken down from the wrong one.)
            wait([first] if first else [])
            if generation != open_now[0]:
                raise decode.Unreadable("left before it was decoded")
            return load(path, False, mosaic=False) if cpu else load(path, False)

        cpu = path in self.cpu
        if path.suffix.lower() in decode.RAW and not self.recent.has(path, False):
            address = self._address(self._full()).removeprefix("image://photo/")
            self.fetching[address] = self.threads.submit(then_full)
        self.reader.start()

    def _read_ahead(self) -> None:
        """The photos next to the open one are decoded before they are asked for: the one the
        user is heading for in both sizes (264 MB for 33 MP: one full size, no more), and the
        quick pictures of the one after it and of the one behind."""
        if self.ahead:
            self._before = self.ahead(self._near()) or {}
            self._hold()

    def _near(self) -> list:
        on, back = self._index + self._way, self._index - self._way
        wanted = [(on, True), (on, False), (on + self._way, True), (back, True)]
        return [(self.photos[at], half) for at, half in wanted
                if 0 <= at < len(self.photos) and not self.recent.has(self.photos[at], half)]

    BLANK = (1, 1)  # what a photo that cannot be shown comes back as: see `failed`
    mosaics: dict = {}  # of the full-size pictures handed out last, by address: decode's "mosaic"
    cpu: frozenset = frozenset()  # the raw files the GPU developed wrong: their full size is a
    # picture developed by LibRaw, at the address of size "cpu" (`checkDeveloped`)
    grabbing = None  # the GPU's development being read back for `checkDeveloped`

    @Slot(QObject, str)
    def checkDeveloped(self, item, source: str) -> None:  # noqa: N802 - read from QML
        """Read back a few of the pixels of a raw file's full size as the GPU developed it
        (Develop.qml, its last pass drawn again at about 250 across: one full-size pixel each,
        none averaged) and compare them to LibRaw's half size; where they differ, that file's
        full size is developed by LibRaw from now on. Figures that agree with LibRaw
        (decode.figures) do not make a picture that does: the GPU may read them otherwise.
        (Measured: 2 to 5 ms on the interface's thread for the comparison, once a photo.)"""
        listing, index, _ = source.removeprefix("image://photo/").split("/")
        path = self.photos[int(index)] if listing == str(self._listing) else None
        kept = self.recent.get(path, True) if path else None
        if kept is None or kept[2] is None or min(kept[2].shape[:2]) < 6:
            return
        small = kept[2][::2, ::2]  # (some 250 across)
        grab = self.grabbing = item.grabToImage(QSize(small.shape[1] - 1, small.shape[0] - 1))

        def judge() -> None:
            self.grabbing = None
            seen = grab.image().convertToFormat(QImage.Format.Format_RGBX8888)
            pixels = np.frombuffer(seen.constBits(), np.uint8).reshape(seen.height(), -1)
            share = unlike(pixels[:, :seen.width() * 4].reshape(seen.height(), -1, 4)[..., :3]
                           / np.float32(255), small)
            if share > 0.2 and path not in self.cpu:
                print(f"photone: {path.name}: the GPU's development is not LibRaw's ({share:.0%}"
                      " of the points differ): developed by LibRaw", file=sys.stderr)
                self.cpu = self.cpu | {path}
                self.changed.emit()

        grab.ready.connect(judge)

    @Slot(str, result="QVariant")
    def mosaicOf(self, source: str):  # noqa: N802 - read from QML
        """How to develop the picture at that address (decode.figures, and where the photo
        lies in it), if it is a raw file's mosaic; else None (it is the picture itself)."""
        return self.mosaics.get(source.removeprefix("image://photo/"))

    def image(self, address: str) -> QImage:
        """The photo as a float image. Runs on Qt's image loader thread."""
        generation, index, size = address.split("/")
        blank = QImage(*self.BLANK, QImage.Format.Format_RGBA16FPx4)
        blank.fill(0)
        if generation in ("painted", "lut", "export", "mark", "denoised"):
            return self._other(generation, index, size) or blank
        # A photo the user has already left, or the full size of one beside the open photo: not
        # worth decoding. (The half size of one beside it is: PhotoView holds it, `held`.)
        far = index != "chart" and abs(int(index) - self._index) > (size == "half")
        if generation != str(self._listing) or far:
            return blank
        if index == "chart":  # the test chart in the photo's place: no file, nothing to describe
            meta, pixels = decode.synthetic(*CHART)
            return QImage(pixels.buf, *CHART, CHART[0] * 8, QImage.Format.Format_RGBA16FPx4).copy()
        try:
            meta, picture, small = self._picture(address, self.photos[int(index)], size == "half")
        except CancelledError:
            return blank  # left while this was being asked for
        except Exception as e:  # noqa: BLE001 - raised here, Qt drops it: a blank view, no word
            if self._on[1] == int(index):
                self._failed = True
                known = isinstance(e, (decode.Unreadable, IndexError))
                self.say(str(e) or "no such photo" if known else
                         f"{self.photos[int(index)].name}: Photone could not read this file ({e})")
            return blank
        # (Which photo is open, asked after the picture is kept: a step onto this one either
        # comes after, and is told here, or finds it kept, `_show`.)
        generation, at = self._on
        if at == int(index):
            self.described.emit(generation, meta)
            self._small = (generation, small) if size == "half" else self._small
        return picture

    def _other(self, kind: str, which: str, size: str = "") -> QImage | None:
        """A picture that is not the open photo's own: its painted masks, a LUT, the
        watermark, or the photo an export is being made of (photone/export.py)."""
        if kind == "mark":
            return self.exporter.mark_picture() if self.exporter else None
        if kind == "denoised":  # the neural denoiser's result, for its window's preview
            seen = self.denoiser.pixels if self.denoiser else None
            return None if seen is None else owned(seen, seen.shape[1], seen.shape[0])
        if kind == "painted":
            return self._painted_at(which)
        if kind == "lut":
            return self._lut_at(which)
        return self.exporter.picture if self.exporter else None

    def _painted_at(self, which: str) -> QImage | None:
        """The painted masks of the photo on show, or of the one being exported."""
        if which == "export":
            return self.exporter.painted if self.exporter else None
        wide, high = self._stored_size()
        return self.painted(self._paint[1], wide / max(1, high))

    def _picture(self, address: str, path: Path, half: bool) -> tuple:
        """(what the decoder says of the photo, its picture, its thinned copy for the quick
        size): as kept from when it was last open, or decoded now and kept."""
        kept = self.recent.get(path, half)
        cpu = not half and path in self.cpu
        if kept is None or cpu and "mosaic" in kept[0]:
            wanted = self.fetching.pop(address, None)
            meta, pixels = (wanted.result() if wanted else self.load(path, half, mosaic=False)
                            if cpu else self.load(path, half))
            try:
                kept = (meta, owned(pixels.buf, *decode.texels(meta)),
                        small_copy(meta, pixels.buf) if half else None)
            finally:
                pixels.close()
            self.recent.put(path, half, kept)
        if not half:  # (before Qt has the picture: Develop.qml asks when it is there)
            self.mosaics = {**dict(list(self.mosaics.items())[-7:]), address: kept[0].get("mosaic")}
        return kept

    def _described(self, generation: int, meta: dict) -> None:
        """On the session's own thread: what the decoder found out about the photo shown."""
        if generation != self._generation or self._seen == generation:
            return  # a photo the user has since left, or told already
        self._size, self._seen = tuple(meta["full"]), generation
        # Its first picture is there: now its neighbors, ahead of being asked for. (On workers
        # of their own that give way: the full size of this one, still to come, is not held up.)
        self.reader.start(20)
        self._info = f"{meta['name']}  ·  {self._size[0]} × {self._size[1]}"
        camera = meta["camera"] or edit.FINISHED  # no camera behind it: white moves all the same
        self._camera = {**camera, "shot": edit.as_shot(camera), "match": meta.get("match")}
        self._raw, self._denoised = meta["raw"], bool(meta.get("denoised"))
        self.refresh()

    # ---- the edit (photone/edit.py) -------------------------------------------------------

    def _history(self) -> edit.History:
        """The photo's edit as the catalog holds it, with its undo and redo as they were left."""
        stored = self._stored()
        return edit.History(stored, self.catalog.steps(self._key) if self.catalog else None)

    def _stored(self) -> dict:
        """The edit the catalog holds for the photo shown, found by the photo's fingerprint."""
        self._key, self.points, self._base = None, [], None
        if not (self.catalog and self.photos):
            return {}
        photo = self.photos[self._index]
        # ponytail: three short reads on the interface's thread, about a millisecond on a
        # local disk; on a slow network share it is a brief wait when a photo is opened. The
        # upgrade is to take the fingerprint in the decoder, which opens the file anyway.
        self._key = identify(photo)
        stored = self.catalog.load(self._key, photo)
        self.points = self.catalog.checkpoints(self._key)
        return stored

    @Property(bool, notify=changed)
    def denoised(self) -> bool:
        """The photo is a file Photone's neural denoiser wrote: it has no noise reduction."""
        return self._denoised

    @Slot(str)
    def openBeside(self, path: str) -> None:  # noqa: N802
        """Show that file as the photo after the open one in the row (a denoised copy, just
        written, beside its original)."""
        if not self.photos:
            return self.open([path])
        self.flush()
        self.photos = [*self.photos[:self._index + 1], Path(path), *self.photos[self._index + 1:]]
        self._relisted()
        self._show(self._index + 1)

    @Slot(list)
    def drop(self, paths: list) -> None:
        """These photos were taken out (their files are gone: Library.remove): they leave the
        row. Was the open one among them, the one that stood after it is shown, or the last."""
        gone = {Path(path) for path in paths}
        if not gone.intersection(self.photos):
            return
        self.flush()  # (while the open photo still has its place)
        was = self.photos[self._index] in gone
        before = sum(photo in gone for photo in self.photos[:self._index])
        self.photos = [photo for photo in self.photos if photo not in gone]
        self._index = max(0, min(self._index - before, len(self.photos) - 1))
        self._on = (self._generation, self._index)
        # ponytail: the places, and so the addresses, are new: the open photo, kept, is read
        # again from `recent` and shows its thumbnail for a moment. Rare (its neighbor's file
        # went); the upgrade is addresses by file, not by place.
        self._relisted()  # (the new place first: QML asks for the open photo when told)
        self._hold()
        if was and self.photos:
            return self._show(self._index)
        if was:  # nothing is left: no photo is open
            self._size, self._info, self._failed = (0, 0), "", False
            self.history = self._history()
            self._generation += 1
            self.photoChanged.emit()
        self.refresh()

    @Property(bool, notify=changed)
    def develops(self) -> bool:
        """The photo is a raw file: it has a profile to choose."""
        return self._raw

    @Property("QVariantMap", notify=changed)
    def shot(self) -> dict:
        """Where three settings rest for this photo: the temperature and tint the camera
        chose, and its own tone curve (edit.curve_of)."""
        kelvin, tint = self._camera["shot"] if self._camera else (6500.0, 0.0)
        return {"temperature": kelvin, "tint": tint, "curve": edit.curve_of({}, self._camera)}

    @Property("QVariantMap", notify=changed)
    def edit(self) -> dict:
        """Every setting's value now, the white balance as numbers also when it is as shot."""
        def whole() -> dict:
            # (The masks as `masks` has them, without the strokes of painted ones: thousands
            # of numbers that the interface took in at every change of anything, and reads
            # none of. With a painted mask a curve dragged ran at 30 frames a second.)
            values = {**edit.DEFAULTS, **self.history.edit, "masks": self.masks}
            return {**values, **{k: v for k, v in self.shot.items() if values[k] is None}}

        return self._once("edit", whole)

    @Property("QVariantMap", notify=changed)
    def uniforms(self) -> dict:
        """What the pipeline shader takes, for the edit as it is, or without one tool's part
        of it (`setBypass`), or for none at all (`original`)."""
        return self._once("uniforms", self._uniforms)

    def _uniforms(self) -> dict:
        # (The original is the photo without its edit, but in its frame: it must not jump.)
        return self._numbers({k: v for k, v in self._compared().items()
                              if not (self._original and k not in edit.FRAME)})

    def _without(self, tool: str) -> dict:
        """The edit without one tool's part: its settings doing nothing, which for most is
        their rest and for some is not (edit.IDLE: the tone curve at rest is the photo's own)."""
        keys = edit.TOOLS.get(tool, ())
        return {**{k: v for k, v in self.history.edit.items() if k not in keys},
                **{k: edit.IDLE[k] for k in keys if k in edit.IDLE}}

    def _compared(self) -> dict:
        """The edit without what a held A/B leaves out: a tool's settings (`setBypass`), or
        some of a mask's own (`setMaskBypass`)."""
        shown = self._without(self._bypass)
        number, keys = self._bypass_local
        if "pins" in keys and "pins" in shown:
            shown["pins"] = [pin for pin in shown["pins"] if (pin.get("mask") or 0) != number]
        if keys and 1 <= number <= len(shown.get("masks", [])):
            masks = list(shown["masks"])
            own = masks[number - 1].get("edit", {})
            masks[number - 1] = {**masks[number - 1],
                                 "edit": {k: v for k, v in own.items() if k not in keys}}
            shown["masks"] = masks
        return shown

    def _numbers(self, shown: dict) -> dict:
        """The shader's numbers for an edit of the photo on show."""
        wide, high = self._stored_size()
        camera = self._camera
        if self._panels["chart"]:  # no lens drew the test chart: nothing of a lens to undo
            shown = {k: v for k, v in shown.items() if k not in edit.TOOLS["lens"]}
            camera = camera and {**camera, "lens": None}
        numbers = edit.uniforms(shown, camera, self._raw, self.peak if self.hdr else 1.0,
                                wide / max(1, high), wide)
        return shaders.for_qml({**numbers, "frame": self._frame()["matrix"]})

    def _layers(self, frame: list | None = None) -> list:
        """The picture inside each mask that has an edit of its own (edit.layers), as the
        shader's numbers; none while the original is shown. `frame`: another than the one on
        show (the photo's own, while the Crop tool shows all of the picture)."""
        if self._original:
            return []
        shown = self._compared()
        return [{"mask": number, "numbers": {**self._numbers(local),
                                             **({"frame": frame} if frame else {})}}
                for number, local in edit.layers(shown, self._camera)]

    @Property("QVariantList", notify=changed)
    def maskLayers(self) -> list:  # noqa: N802 - read from QML
        return self._once("maskLayers", self._layers)

    @Property("QVariantList", notify=changed)
    def maskLayersInFrame(self) -> list:  # noqa: N802 - read from QML
        """The same in the photo's own frame: for its thumbnail (as `inFrame` is)."""
        if self._panels["tool"] != "crop" or self._bypass == "crop":
            return self.maskLayers
        tight = shaders.by_columns(edit.frame(self.history.edit, *self._stored_size())["matrix"])
        return self._once("maskLayersInFrame", lambda: self._layers(tight))

    @Slot(int, str, float)
    def setMaskEdit(self, number: int, key: str, amount: float) -> None:  # noqa: N802
        """One setting of what mask `number` changes inside it (edit.LOCAL): an amount added
        to the whole image's. Like any setting: a drag of it is one step of undo."""
        masks = [dict(mask) for mask in self.history.edit.get("masks", [])]
        if 1 <= number <= len(masks) and key in edit.LOCAL:
            masks[number - 1]["edit"] = {**masks[number - 1].get("edit", {}), key: amount}
            self.setEdits({"masks": masks})

    @Slot(int, str, "QVariantList")
    def setMaskWhole(self, number: int, key: str, value: list) -> None:  # noqa: N802
        """A mask's own tone curve, chroma curve or color EQ (edit.LOCAL_WHOLE): it works
        inside the mask on top of the whole image's."""
        masks = [dict(mask) for mask in self.history.edit.get("masks", [])]
        if 1 <= number <= len(masks) and key in edit.LOCAL_WHOLE:
            masks[number - 1]["edit"] = {**masks[number - 1].get("edit", {}), key: value}
            self.setEdits({"masks": masks})

    @Slot(int, str, result="QVariantList")
    def localWhole(self, number: int, key: str) -> list:  # noqa: N802
        """Mask `number`'s own of that setting, which comes on top of the whole image's; a
        new mask's is the setting doing nothing (a straight curve: the whole image's at rest
        is the photo's own)."""
        masks = self.history.edit.get("masks", [])
        own = masks[number - 1].get("edit", {}) if 1 <= number <= len(masks) else {}
        return own.get(key, edit.IDLE.get(key, edit.DEFAULTS[key]))

    @Slot(int, str, result="QVariantList")
    def localLine(self, number: int, key: str) -> list:  # noqa: N802
        """That setting inside the mask as a line to draw: a curve's heights (edit.LINE), or the
        EQ's gain at every other degree of hue."""
        points = self.localWhole(number, key)
        if key == "eq":
            return [edit.eq_at(points, hue) for hue in range(0, 361, 2)]
        return edit.curve_line(points)

    @Slot(int, "QVariantList")
    def resetMaskEdit(self, number: int, keys: list) -> None:  # noqa: N802
        """Those settings of a mask's own edit back to nothing, as one step of undo."""
        masks = [dict(mask) for mask in self.history.edit.get("masks", [])]
        if 1 <= number <= len(masks):
            kept = masks[number - 1].get("edit", {})
            masks[number - 1]["edit"] = {k: v for k, v in kept.items() if k not in keys}
            self.history.commit()
            self.setEdits({"masks": masks})
            self.history.commit()

    @Property("QVariantList", notify=curveChanged)
    def curveLine(self) -> list:  # noqa: N802 - read from QML
        """The tone curve as heights (edit.LINE) from black to white, for drawing it."""
        points = edit.curve_of(self.history.edit, self._camera)
        return self._once("curveLine", lambda: edit.curve_line(points))

    @Property("QVariantList", notify=chromaChanged)
    def chromaLine(self) -> list:  # noqa: N802
        """The chroma curve as heights (edit.LINE) from gray to vivid."""
        points = self.history.edit.get("chroma", edit.STRAIGHT)
        return self._once("chromaLine", lambda: edit.curve_line(points))

    @Property("QVariantList", notify=eqChanged)
    def eqLine(self) -> list:  # noqa: N802
        """The color EQ as the gain at every other degree of hue, 0 to 360."""
        points = self.history.edit.get("eq", [])
        return self._once("eqLine", lambda: [edit.eq_at(points, hue) for hue in range(0, 361, 2)])

    @Property("QVariantList", notify=bwChanged)
    def bwLine(self) -> list:  # noqa: N802
        """The B&W EQ as percent lighter at every other degree of hue, 0 to 360."""
        points = self.bw
        return self._once("bwLine", lambda: [edit.bw_at(points, hue) * 100 - 100
                                             for hue in range(0, 361, 2)])

    @Property("QVariantMap", constant=True)
    def films(self) -> dict:
        """The film looks to start black and white from: each a set of the B&W EQ's points."""
        return edit.FILMS

    @Property("QVariantList", constant=True)
    def hues(self) -> list:
        """The hues round the wheel as colors: the color EQ's strip."""
        return scopes.hues()

    @Property(bool, notify=changed)
    def original(self) -> bool:
        return self._original

    @Slot(bool)
    def setOriginal(self, on: bool) -> None:  # noqa: N802 - called from QML
        """Show the photo without its edit, for as long as B is held."""
        if on != self._original:
            self._original = on
            self.refresh()

    @Property(bool, notify=changed)
    def canUndo(self) -> bool:  # noqa: N802
        return self.history.at > 0

    @Property(bool, notify=changed)
    def canRedo(self) -> bool:  # noqa: N802
        return self.history.at < len(self.history.states) - 1

    def _edited(self) -> None:
        self._unwritten = True
        self.saver.start()
        self.refresh()

    @Slot(str, "QVariant")
    def setEdit(self, key: str, value) -> None:  # noqa: N802
        self.setEdits({key: value.toVariant() if isinstance(value, QJSValue) else value})

    @Slot("QVariantMap")
    def setEdits(self, values: dict) -> None:  # noqa: N802
        """Several settings as one gesture (the white-balance pad moves two). A tone curve that
        is the photo's own is no change: the edit says nothing of it, as at rest (a last point
        taken away on a photo whose own curve is straight left a change that showed none)."""
        try:
            if "curve" in values and edit.valid("curve", values["curve"]) == self.shot["curve"]:
                values = {**values, "curve": None}
            self.history.put(values)
        except ValueError:
            return  # not a setting, or not a value for it: nothing happens
        self._edited()

    @Slot()
    def commitEdit(self) -> None:  # noqa: N802
        """The gesture is over (a slider let go): what follows is another undo step."""
        self._brushing = False
        self.history.commit()
        self.refresh()

    @Slot(str)
    def resetEdit(self, key: str) -> None:  # noqa: N802
        if key in edit.DEFAULTS:
            self.history.commit()
            self.history.set(key, edit.DEFAULTS[key])
            self.history.commit()
            self._edited()

    # ---- the tools of the rail (edit.TOOLS) ---------------------------------------------------

    @Property(bool, notify=changed)
    def edited(self) -> bool:
        """Anything at all has been done to the photo."""
        return bool(self.history.edit)

    @Slot()
    def resetAll(self) -> None:  # noqa: N802
        """Every setting back to rest, as one step of undo. Checkpoints stay."""
        self.history.replace({})
        self._edited()

    @Slot("QVariantMap")
    @Slot("QVariantMap", "QVariantList")
    def pasteLook(self, look: dict, parts=None) -> None:  # noqa: N802
        """Lay a look (edit.look_of) on the open photo, all of it or those parts of it, as one
        step of undo: its frame and its masks stay."""
        if self.photos:
            self.history.replace(edit.with_look(self.history.edit, look, parts))
            self._edited()

    @Slot("QVariantMap", "QVariantList", result="QVariantMap")
    def numbersWith(self, look: dict, parts: list) -> dict:  # noqa: N802
        """The shader's numbers for the open photo with that look laid on it: what a preset
        would make of it, for its small picture."""
        laid = edit.with_look(self.history.edit, look, parts)
        if laid.get("luts") != self.history.edit.get("luts"):
            laid.pop("luts", None)  # (other LUTs than the photo's: no picture of them here)
        return self._numbers(laid)

    @Property("QVariantMap", notify=changed)
    def look(self) -> dict:
        """The open photo's look as it stands (edit.look_of): what a preset is saved from."""
        return edit.look_of(self.history.edit)

    @Property("QVariantMap", notify=changed)
    def adjusted(self) -> dict:
        """For each tool, whether any of its settings is off its rest: the dot on its icon."""
        return self._once("adjusted", lambda: {tool: any(k in self.history.edit for k in keys)
                                               for tool, keys in edit.TOOLS.items()})

    @Property("QVariantMap", notify=changed)
    def comparable(self) -> dict:
        """For each tool, whether its A/B has anything to show: a setting off its rest, or
        one that does something at rest (to this photo: lens corrections do nothing to a
        file without figures)."""
        def acts(tool: str) -> bool:
            if not any(k in edit.IDLE for k in edit.TOOLS[tool]):
                return False
            return self._numbers(self.history.edit) != self._numbers(self._without(tool))
        return self._once("comparable", lambda: {tool: on or acts(tool)
                                                 for tool, on in self.adjusted.items()})

    @Slot(str)
    def resetTool(self, tool: str) -> None:  # noqa: N802
        """Every setting of one tool back to rest, as one step of undo."""
        if tool in edit.TOOLS:
            self.history.commit()
            self.history.put({key: edit.DEFAULTS[key] for key in edit.TOOLS[tool]})
            self.history.commit()
            self._edited()

    @Slot(str)
    def setBypass(self, tool: str) -> None:  # noqa: N802
        """Show the photo without one tool's part of the edit, while its Compare is held."""
        self._bypass = tool
        self.refresh()

    @Slot(int, "QVariantList")
    def setMaskBypass(self, number: int, keys: list) -> None:  # noqa: N802
        """Show the photo without those settings of a mask's own edit, while the A/B of their
        card is held; "pins": without the pins inside it (mask 0: the whole image's)."""
        self._bypass_local = (number, tuple(keys))
        self.refresh()

    @Property("QVariantMap", notify=changed)
    def panels(self) -> dict:
        """What is open: `tool` ("crop" while the Crop tab is, or ""), `clipping`; `live`: the
        wheel shows the photo's colors after the pins, not before them; and `chart`: a test
        chart stands in the photo's place, with the photo's edit on it."""
        return self._panels

    @Slot(str, "QVariant")
    def setPanel(self, key: str, value) -> None:  # noqa: N802
        if type(value) is not type(self._panels.get(key)):
            return
        self._panels = {**self._panels, key: value}
        if key == "chart":
            return self._set_chart(value)
        if self.settings and key != "clipping":  # the paint is not something to start up with
            self.settings.setValue(f"view/{key}", value)
        self.refresh()
        if key == "live":
            self._redot()

    def _set_chart(self, on: bool) -> None:
        """Put the test chart in the photo's place, or the photo back. The edit stays the
        photo's: the chart only shows what it does. Never kept between runs, and off again
        with the next photo, so that nobody finds a chart where a photo should be."""
        self._panels = {**self._panels, "chart": on and bool(self.photos)}
        self._lab, self._after, self._histogram = None, None, {}  # counted from the other picture
        self._inside = None
        self._redot()
        self.refresh()
        self.photoChanged.emit()  # the view starts afresh: the two are not of one shape

    @Slot(float, float)
    def pickNeutral(self, u: float, v: float) -> None:  # noqa: N802
        """Set the white balance so that the spot at (u, v) of the photo (0..1) is gray. Asked
        again and again while the pointer is slid over the photo, all of it one step of undo
        until `commitEdit`, like any drag."""
        if not (self._small and self._small[0] == self._generation and self._camera):
            return
        seen = self._small[1]
        if self._panels["chart"]:  # the spot is on the test chart, then
            meta, pixels = decode.synthetic(*CHART)
            seen = small_copy(meta, pixels.buf)
        u, v = (min(1.0, max(0.0, w)) for w in edit.framed(self._frame()["matrix"], u, v))
        y, x = int(v * (seen.shape[0] - 1)), int(u * (seen.shape[1] - 1))  # in the stored picture
        spot = seen[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3).mean(axis=0)
        found = edit.neutral_at(self._camera, spot) if spot.max() < 0.98 else None
        if not found:
            return self.say("That spot is too bright or too dark to tell its color. "
                            "Pick something gray that is neither.")
        self.history.put({"temperature": found[0], "tint": found[1]})
        self._edited()

    @Property("QVariantMap", notify=scoped)
    def histogram(self) -> dict:
        """r, g, b and light: 256 heights (0..1) each; over, under: the share of pixels at
        white and at black. Empty until counted."""
        return self._histogram

    @Slot(QImage)
    def setScope(self, seen: QImage) -> None:  # noqa: N802
        """A small render of the picture, its lightness and its colors (PhotoView.qml), to be
        counted (photone/scopes.py). The counting takes some milliseconds, which a frame of a
        slider being dragged does not have: it is done on a thread of its own, and `scoped`
        says when the result is in."""
        seen = seen.convertToFormat(QImage.Format.Format_RGBA8888)
        rows = np.frombuffer(seen.constBits(), np.uint8).reshape(seen.height(), seen.bytesPerLine())
        # ponytail: counted from a 256 x 256 reduction, so a clipped spot of a few pixels is
        # averaged away. The upgrade is counting on the GPU at full size.
        rows = rows[:, :seen.width() * 4].reshape(seen.height(), -1, 4).copy()
        # Only the latest render waits to be counted. (Each used to be queued: a count takes
        # longer than the 40 ms between two renders while something is dragged, so the queue
        # grew for as long as the drag went on and the counting thread, which shares the
        # interpreter with the interface, never rested. The owner, turning a crop for some
        # seconds: "rotating over a long period of time now lags".)
        with self._wanting:
            waiting = self._wanted is not None
            self._wanted = (self._generation, rows, self._asked())
        if not waiting:
            self.counting.submit(self._count)

    @Slot(QQuickItem, result=bool)
    def grabScope(self, item) -> bool:  # noqa: N802
        """Take a render of that item (Scopes.qml) and count it (`setScope`); `scopeTaken`
        says when it is back. False if it cannot be taken now.

        Asked for here and not by the interface's script (`grabToImage` with a function to
        call): a grab asked for by script is booked by the script engine as memory it
        holds, 3 MiB each, 25 a second while something is dragged, and once in a few
        thousand Qt (6.12) takes one off its count that it never added. The count wraps
        round below zero, the engine then finds its limit passed at every allocation and
        collects garbage at each one: every change cost 20 to 90 ms until the program was
        started again. (The owner: "low FPS and lag, this effect stays even after removing
        the mask until restart". Seen in the engine's own log: `unmanaged heap size:
        18446744073708177990`.)"""
        grab = item.grabToImage()
        if grab is None:
            return False
        self._grab = grab  # (kept: it is ours until it is ready)
        grab.ready.connect(self._scope_ready)
        return True

    def _scope_ready(self) -> None:
        grab, self._grab = self._grab, None
        if grab is not None:
            self.setScope(grab.image())
        self.scopeTaken.emit()

    def _asked(self) -> tuple:
        """What decides which of the photo's colors are counted for the wheel: the working
        range, and whether before or after the pins."""
        return (*self._range, self._panels["live"])

    def _count(self) -> None:
        """On the counting thread: the render that waits (`setScope`). `asked`: see `_asked`,
        as it was when the render was taken."""
        with self._wanting:
            (generation, rows, asked), self._wanted = self._wanted, None
        before, after = scopes.lab_of(rows), scopes.lab_of(rows, pinned=True)
        inside = scopes.inside_of(rows)
        cloud = scopes.cloud_of(*(after if asked[3] else before), asked[1:3], inside)
        # (All of it by the mask whose pins are worked on, where there is one: the owner's wish.)
        counts = {**scopes.histogram_of(rows, inside),
                  **scopes.spread_of(*(tile[inside] for tile in before))}
        self.counted.emit(generation, (counts, before, after, cloud, asked, inside))

    def _counted(self, generation: int, found: tuple) -> None:
        """On the session's own thread: take the count in, unless the photo has changed."""
        if generation != self._generation:
            return
        self._histogram, self._lab, self._after, cloud, asked, self._inside = found
        self.scoped.emit()
        if asked != self._asked():
            return self._redot()  # the range, the view or the choice moved meanwhile
        was, self._cloud = self._cloud, cloud
        same = was is not None and all(np.array_equal(was[key], cloud[key])
                                       for key in ("points", "colors", "light"))
        if not same:  # (the same: a pin dragged, and the colors before the pins counted)
            self.dotted.emit()

    def _redot(self) -> None:
        """The count for the wheel is to be made anew (another range, another photo)."""
        self._cloud = None
        self.scoped.emit()
        self.dotted.emit()

    # ---- the wheel: the photo's colors (photone/scopes.py) -------------------------------------

    def _dots(self) -> dict:
        """The photo's colors as counted for the wheel, within the working range
        (scopes.cloud_of): what Fit frames, the lightness strip, and what the checks look at.
        (The dots on the wheel are not these: the GPU draws them from the photo, dots.vert.)"""
        if self._cloud is None:
            empty = np.zeros(0, np.float32)
            shown = self._after if self._panels["live"] else self._lab
            self._cloud = scopes.cloud_of(*(shown or (empty, empty, empty)), self._range[1:],
                                          self._inside if shown else None)
        return self._cloud

    @Property("QVariantMap", notify=dotted)
    def cloud(self) -> dict:
        """That count as plain lists (scopes.listed): for the checks."""
        return scopes.listed(self._dots())

    @Property("QVariantMap", notify=dotted)
    def dots(self) -> dict:
        """What the wheel takes of that count: `box` (the least and most a, then b: what Fit
        frames), `count` (whether there is anything), `light` (the strip: 64 bars as bytes)."""
        found = self._dots()
        return {"count": len(found["colors"]), "box": found["box"],
                "light": list(scopes.heights(found["light"]))}

    @Property("QVariantMap", notify=scoped)
    def bars(self) -> dict:
        """The histogram as the interface takes it: each row of bars as a list of heights
        0..255 (r, g, b, light, chroma, hue), and `over`, `under`. Empty until counted.

        Lists, not bytes. As bytes each row became an ArrayBuffer in the interface's script
        engine, and Qt (6.9) takes such a buffer's size off its count of memory held outside
        the engine when the buffer goes, without having added it: after a few thousand
        counts the count wrapped round below zero, the engine then saw its limit passed at
        every allocation and collected garbage at each one, for good: every change cost 20
        to 70 ms until the program was started again (the owner: "low FPS and lag, this
        effect stays even after removing the mask until restart")."""
        return {key: list(scopes.heights(value)) if isinstance(value, list) else value
                for key, value in self._histogram.items()}

    @Property("QVariantMap", notify=scoped)
    def range(self) -> dict:
        """The working range: the span of lightness the wheel shows. `name` is one of
        scopes.RANGES or "custom"."""
        return {"name": self._range[0], "low": self._range[1], "high": self._range[2]}

    @Slot(str)
    def setRange(self, name: str) -> None:  # noqa: N802
        if name in scopes.RANGES:
            self._range = (name, *scopes.RANGES[name])
            self._redot()

    @Slot(float, float, result="QVariantList")
    def colorAt(self, u: float, v: float) -> list:  # noqa: N802
        """[a, b, lightness] of the picture at (u, v) of the photo (0..1), before the pins
        (that is the color a pin takes hold of); empty if unknown."""
        if not self._lab or not (0 <= u <= 1 and 0 <= v <= 1):
            return []
        light, a, b = self._lab
        y, x = int(v * (light.shape[0] - 1)), int(u * (light.shape[1] - 1))
        return [float(a[y, x]), float(b[y, x]), float(light[y, x])]

    @Slot(float, float, result=str)
    def shownAt(self, u: float, v: float) -> str:  # noqa: N802
        """The color of the picture round (u, v) of the photo as a color to draw ("#rrggbb"),
        the mean of a small patch so that noise does not decide it; empty if unknown. (What
        Pick neutral would take: its preview.)"""
        if not self._lab or not (0 <= u <= 1 and 0 <= v <= 1):
            return ""
        light, a, b = self._lab
        y, x = int(v * (light.shape[0] - 1)), int(u * (light.shape[1] - 1))
        patch = (slice(max(0, y - 1), y + 2), slice(max(0, x - 1), x + 2))
        mean = (np.array([float(tile[patch].mean())]) for tile in (light, a, b))
        red, green, blue = scopes.srgb_of(*mean)[0]
        return f"#{red:02x}{green:02x}{blue:02x}"

    @Slot(float, float)
    def pickRange(self, u: float, v: float) -> None:  # noqa: N802
        """Limit the wheel to lightness like that of the spot at (u, v) of the photo."""
        found = self.colorAt(u, v)
        if found:
            self._range = ("custom", max(0.0, found[2] - 0.12), min(1.0, found[2] + 0.12))
            self._redot()

    @Property("QVariantMap", notify=changed)
    def before(self) -> dict:
        """The shader's numbers for the photo without its edit, in its frame: one side of the
        before/after split."""
        return self._once("before", lambda: self._numbers(
            {k: v for k, v in self.history.edit.items() if k in edit.FRAME}))

    @Slot(int)
    def turn(self, by: int) -> None:
        """A quarter turn to the right (1) or to the left (-1). The crop turns with the picture."""
        full = {**edit.DEFAULTS, **self.history.edit}
        left, top, right, bottom = full["crop"]
        crop = [1 - bottom, left, 1 - top, right] if by > 0 else [top, 1 - right, bottom, 1 - left]
        self.history.commit()
        self.history.put({"turns": (int(full["turns"]) + (1 if by > 0 else 3)) % 4, "crop": crop})
        self.history.commit()
        self._edited()

    @Property(int, notify=changed)
    def steps(self) -> int:
        """How many steps of undo there are: a place to come back to (`undoTo`)."""
        return self.history.at

    @Slot(int)
    def undoTo(self, steps: int) -> None:  # noqa: N802
        """Undo back to where there were that many steps (a tool that is cancelled)."""
        while self.history.at > max(0, steps) and self.history.step(-1):
            pass
        self._edited()

    @Slot()
    def undo(self) -> None:
        if self.history.step(-1):
            self._edited()

    @Slot()
    def redo(self) -> None:
        if self.history.step(1):
            self._edited()

    @Slot()
    def flush(self) -> None:
        """Write the edit to the catalog now (also called a moment after every change, when
        another photo is opened, and at exit)."""
        self.saver.stop()
        if self.exporter:
            self.exporter.keep_mark()  # (the watermark, should it still be waiting to be written)
        if not (self._unwritten and self.photos):
            return
        try:
            if self.catalog:
                self.catalog.save(self._key, self.photos[self._index], self.history.edit,
                                  self.points)
                self.catalog.save_steps(self._key, *self.history.kept())
            self._unwritten = False
            self.saved.emit()  # (the photo's thumbnail is made anew from what is on screen)
        except sqlite3.Error as e:
            self.say(f"This edit could not be saved: {e}")

    # ---- checkpoints: edits saved to come back to --------------------------------------------

    @Property("QVariantList", notify=checkpointsChanged)
    def checkpoints(self) -> list:
        """For the list: each with whether the edit stands at it now ("here"), or stood at it
        and has been changed since ("changed"), newest first."""
        return self._once("checkpoints", self._checkpoints)

    def _checkpoints(self) -> list:
        here, unsaved = edit.standing(self.points, self.history.edit, self._base)
        today = time.strftime("%Y-%m-%d")

        def when(made: float) -> str:
            at = time.localtime(made)
            same_day = time.strftime("%Y-%m-%d", at) == today
            # (Not "%-d": Windows does not know it.)
            return time.strftime("%H:%M" if same_day else f"{at.tm_mday} %b %Y", at)

        return [{"id": c["id"], "name": c["name"], "when": when(c["made"]), "made": c["made"],
                 "state": "here" if c["id"] == here
                 else "changed" if unsaved and c["id"] == self._base else ""}
                for c in reversed(self.points)]

    @Property("QVariantMap", notify=changed)
    def about(self) -> dict:
        """What the left panel tells of the open photo (library.about): read once a photo."""
        path = self.photos[self._index] if self.photos else None
        if path is None:
            return {}
        if getattr(self, "_about", (None,))[0] != path:
            try:
                self._about = (path, about(path))
            except OSError:
                self._about = (path, {"file": path.name})
        return self._about[1]

    @Property("QVariantList", notify=changed)
    def unsavedKeys(self) -> list:  # noqa: N802
        """The settings changed since the checkpoint the edit came from (edit.unsaved)."""
        return self._once("unsaved", lambda: edit.unsaved(self.points, self.history.edit,
                                                          self._base))

    @Property(bool, notify=changed)
    def unsaved(self) -> bool:
        """The edit as it is now is in no checkpoint: going to one would lose it."""
        return edit.standing(self.points, self.history.edit, self._base)[1]

    def _points_changed(self) -> None:
        self._unwritten = True
        self.flush()  # a checkpoint is kept at once, not a moment later
        self.refresh()

    @Slot()
    def saveCheckpoint(self) -> None:  # noqa: N802
        """Keep the edit as it is now, to come back to."""
        self.history.commit()
        number = max((c["id"] for c in self.points), default=0) + 1
        self.points.append({"id": number, "name": f"Checkpoint {number}", "made": time.time(),
                            "edit": dict(self.history.edit)})
        self._base = number
        self._points_changed()

    @Slot(int)
    def goToCheckpoint(self, number: int) -> None:  # noqa: N802
        """Make the photo's edit that checkpoint's. One step of undo, like any other change.
        (The interface asks first when `unsaved`.)"""
        for point in self.points:
            if point["id"] == number:
                self.history.replace(point["edit"])
                self._base = number
                self._edited()

    @Slot(int, str)
    def renameCheckpoint(self, number: int, name: str) -> None:  # noqa: N802
        """Another name for a checkpoint; an empty one leaves it as it is."""
        if name.strip():
            self.points = [{**c, "name": name.strip()[:80]} if c["id"] == number else c
                           for c in self.points]
            self._points_changed()

    @Slot(int)
    def deleteCheckpoint(self, number: int) -> None:  # noqa: N802
        self.points = [c for c in self.points if c["id"] != number]
        if self._base == number:
            self._base = None
        self._points_changed()

    # ---- what went wrong ---------------------------------------------------------------

    @Property(str, notify=changed)
    def error(self) -> str:
        return self._error

    @Slot(str)
    def say(self, text: str) -> None:
        """One sentence for the user, near the bottom of the window; empty takes it away."""
        self._error = text
        self.refresh()

    # ---- the surface, the switch, the peak (PLAN.md 2.5) --------------------------------

    @Property(bool, notify=changed)
    def ready(self) -> bool:
        """The window's surface is known, so colors can be drawn right. False for one frame."""
        return self._surface is not None

    @Property(bool, notify=changed)
    def hdrSurface(self) -> bool:  # noqa: N802 - read from QML
        """The window is on an HDR surface. A fact about the window, fixed once known."""
        return self._surface is True

    @Property(bool, notify=changed)
    def linearSurface(self) -> bool:  # noqa: N802 - read from QML
        """The HDR surface takes linear light, so everything written to it must be linear too.
        Not what was measured: on the one system looked at (NVIDIA, Plasma 6) the desktop reads
        the scRGB surface as ordinary gamma-encoded values that may exceed 1, although its
        Vulkan name says linear. So this is off unless PHOTONE_HDR_ENCODING=linear says so."""
        return self._surface is True and self._linear

    @Property(bool, notify=changed)
    def hdr(self) -> bool:
        """The switch: highlights may go above the white of the interface. Needs the surface."""
        return self._wants_hdr and self._surface is True

    @Slot(bool)
    def setHdr(self, on: bool) -> None:  # noqa: N802 - called from QML
        if on and self._surface is False:
            return self.say(NO_HDR)
        self._wants_hdr = on
        if self.settings:
            self.settings.setValue("view/hdr", on)
        self.refresh()

    @Slot(bool)
    def surface_known(self, hdr_surface: bool) -> None:
        """Called once, after the first frame, with what the window really got."""
        self._surface = hdr_surface
        if self._asked_hdr and not hdr_surface:
            self._error = NO_HDR
        self.refresh()

    @Property(float, notify=changed)
    def peak(self) -> float:
        """The brightest the photo may get in HDR, in multiples of white: the display's
        headroom. The user's own figure if there is one (view/hdr_peak in the settings), else
        what the desktop reports, else 2."""
        return max(1.0, self._kept("view/hdr_peak", self._headroom, float))

    @Slot(float)
    def set_headroom(self, times_white: float) -> None:
        self._headroom = times_white
        self.refresh()


class PhotoProvider(QQuickImageProvider):
    """Hands the session's pictures to the interface's Image elements. Qt asks on a thread of
    its own, one picture at a time; the decoding itself is already under way (Session._fetch).

    Deliberately Qt's plain provider and nothing made here: with a response and a texture
    factory written in Python, Qt and Python each deleted them when they saw fit, and it
    crashed; and Qt's ready-made factory for such a response turned the picture into 8-bit."""

    def __init__(self, session: Session) -> None:
        super().__init__(QQmlImageProviderBase.ImageType.Image,
                         QQmlImageProviderBase.Flag.ForceAsynchronousImageLoading)
        self.session = session

    def requestImage(self, name: str, size: QSize, requested: QSize) -> QImage:  # noqa: N802
        image = self.session.image(name)
        size.setWidth(image.width())
        size.setHeight(image.height())
        return image


def headroom_from_kscreen(report: str, screen: str) -> float | None:
    """The display's peak over its white, from `kscreen-doctor -o` (Plasma), for the output
    called `screen`. None when that output has no HDR or the report says nothing useful."""
    for block in re.split(r"(?m)^Output: ", re.sub(r"\x1b\[[0-9;]*m", "", report))[1:]:
        if block.split()[1:2] != [screen] or not re.search(r"HDR: enabled", block):
            continue
        white = re.search(r"SDR brightness: (\d+)", block)
        peak = (re.search(r"overridden with: (\d+)", block)
                or re.search(r"Peak brightness: (\d+)", block))
        if white and peak and int(white[1]) > 0:
            return max(1.0, int(peak[1]) / int(white[1]))
    return None


def ask_desktop_for_headroom(session: Session, screen: str) -> None:
    """In the background, so a slow or missing tool never holds the window up. Qt itself knows
    nothing about a display's brightness; Plasma tells. Elsewhere the default stands."""
    def ask() -> None:
        try:
            report = subprocess.run(["kscreen-doctor", "-o"], capture_output=True, text=True,
                                    timeout=3, **QUIET).stdout
        except (OSError, subprocess.TimeoutExpired):
            return
        if found := headroom_from_kscreen(report, screen):
            QtCore.QMetaObject.invokeMethod(session, "set_headroom",
                                            Qt.ConnectionType.QueuedConnection,
                                            QtCore.Q_ARG(float, found))

    threading.Thread(target=ask, daemon=True).start()


def surface_environment(platform: str, vulkan: bool, env: dict) -> dict:
    """What to add to the environment before the window is made. Qt picks the kind of surface
    then, from QSG_RHI_*. An HDR surface (scRGB: half floats, 1.0 is SDR white, more is
    brighter) exists only through Vulkan and only on Wayland, so it is asked for there and
    nowhere else: everywhere else Qt's own default (OpenGL on Linux) stays, which is what works
    on the most machines. Asking is only a request; watch_surface finds out what the window
    got. Setting QSG_RHI_BACKEND yourself overrides all of this."""
    if "QSG_RHI_BACKEND" in env or not platform.startswith("wayland") or not vulkan:
        return {}
    return {"QSG_RHI_BACKEND": "vulkan", "QSG_RHI_HDR": "scrgb"}


def vulkan_works() -> bool:
    """Is there a Vulkan driver for a real GPU? Asked of the Vulkan loader itself (PySide6 has
    no binding for it). A software Vulkan (llvmpipe) does not count: OpenGL may be faster there."""
    class CreateInfo(ctypes.Structure):
        _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint),
                    ("app", ctypes.c_void_p),
                    ("layers", ctypes.c_uint), ("layerNames", ctypes.c_void_p),
                    ("extensions", ctypes.c_uint), ("extensionNames", ctypes.c_void_p)]

    try:
        vk = ctypes.CDLL("libvulkan.so.1")
    except OSError:
        return False
    instance, count = ctypes.c_void_p(), ctypes.c_uint()
    if vk.vkCreateInstance(ctypes.byref(CreateInfo(sType=1)), None, ctypes.byref(instance)):
        return False
    try:
        vk.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None)
        devices = (ctypes.c_void_p * count.value)()
        vk.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), devices)
        properties = ctypes.create_string_buffer(1024)  # VkPhysicalDeviceProperties
        for device in devices:
            vk.vkGetPhysicalDeviceProperties(ctypes.c_void_p(device), properties)
            if int.from_bytes(properties.raw[16:20], sys.byteorder) != 4:  # deviceType, not CPU
                return True
        return False
    finally:
        vk.vkDestroyInstance(instance, None)


def choose_surface(platform: str) -> None:
    os.environ.update(surface_environment(platform, vulkan_works(), dict(os.environ)))


def make_view(session: Session, library: Library | None = None,
              presets: Presets | None = None) -> QQuickView:
    # Qt gathers small pictures into one shared texture, an "atlas", which is 8-bit: a photo
    # under about a thousand pixels across lost its floats there (coarse shadows, nothing
    # above white). No picture of ours may go in. scripts/shots.py checks what it guards.
    os.environ.setdefault("QSG_ATLAS_SIZE_LIMIT", "1")
    view = QQuickView()
    view.setTitle("Photone")
    view.setResizeMode(QQuickView.ResizeMode.SizeRootObjectToView)
    view.setMinimumSize(QSize(900, 560))
    view.resize(1600, 1000)
    library = library or Library(None)  # (none: an empty one, as in most checks)
    session.exporter = Exporter(session, view)
    session.denoiser = Denoiser(session)
    view.engine().addImageProvider("photo", PhotoProvider(session))
    presets = presets or Presets()  # (none: kept nowhere, as in the checks)
    context = view.rootContext()
    # What is not ready for everyone yet, by name: PHOTONE_TRIAL=presets (several: commas).
    context.setContextProperty("trial", os.environ.get("PHOTONE_TRIAL", ""))
    # (The first three are the view's own from here: they outlive the interface that reads
    # them, also at exit.)
    for name, thing in (("session", session), ("library", library), ("presets", presets),
                        ("exporter", session.exporter), ("denoiser", session.denoiser),
                        ("cursors", TurnCursors(view)),
                        ("usage", Usage(view))):
        thing.setParent(view)
        context.setContextProperty(name, thing)
    for name, file in (("pipelineShader", "pipeline.frag"), ("pipelineVertex", "pipeline.vert"),
                       ("wheelShader", "wheel.frag"), ("wheelVertex", "wheel.vert"),
                       ("ovalShader", "oval.frag"), ("ovalVertex", "oval.vert"),
                       ("dotsShader", "dots.frag"), ("dotsVertex", "dots.vert"),
                       ("tallyShader", "tally.frag"), ("tallyVertex", "tally.vert"),
                       ("barsShader", "bars.frag"), ("barsVertex", "bars.vert"),
                       ("brushShader", "brush.frag"), ("brushVertex", "brush.vert"),
                       ("brushedShader", "brushed.frag"), ("brushedVertex", "brushed.vert"),
                       ("mixingShader", "mixing.frag"), ("mixingVertex", "mixing.vert"),
                       ("developShader", "develop.frag"), ("developVertex", "develop.vert"),
                       ("developGreenShader", "develop_green.frag"),
                       ("developHomoShader", "develop_homo.frag")):
        url = QUrl.fromLocalFile(str(shaders.compiled(file)))
        context.setContextProperty(name, url.toString())
    view.setSource(QUrl.fromLocalFile(str(UI / "Main.qml")))
    if view.status() != QQuickView.Status.Ready:
        problems = "; ".join(e.toString() for e in view.errors())
        raise RuntimeError(problems or "the interface did not load")
    context.setContextProperty("perFrame", PerFrameInput(view))
    return view


def is_hdr_surface(view: QQuickView) -> bool:
    """Whether the window really got an HDR surface (asking for one is only a request)."""
    from PySide6.QtGui import QRhiSwapChain

    chain = view.swapChain()
    return bool(chain) and chain.format() in (QRhiSwapChain.Format.HDRExtendedSrgbLinear,
                                              QRhiSwapChain.Format.HDR10)


def turn_cursor(degrees: float, ratio: float) -> QCursor:
    """An arc with an arrow at each end, bulging toward `degrees` (0: up, clockwise), its
    middle the hot spot; 64 points across (the owner: "much bigger"), drawn at the screen's
    pixel ratio."""
    side, reach = 64, 22.0
    pixmap = QPixmap(round(side * ratio), round(side * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(QColor(0, 0, 0, 0))
    path = QPainterPath()
    box = (-reach, 0.0, 2 * reach, 2 * reach)  # the circle the arc is of: its top is the hot spot
    path.arcMoveTo(*box, 40)
    path.arcTo(*box, 40, 100)
    for at, way in ((40, 1), (140, -1)):  # the arrows, along the circle, away from the middle
        a = math.radians(at)
        tip = (reach * math.cos(a), reach - reach * math.sin(a))
        back = (-way * math.sin(a), -way * math.cos(a))  # opposite to travel
        for turn in (0.6, -0.6):
            c, s = math.cos(turn), math.sin(turn)
            path.moveTo(*tip)
            end = (back[0] * c - back[1] * s, back[0] * s + back[1] * c)
            path.lineTo(tip[0] + 9 * end[0], tip[1] + 9 * end[1])
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.translate(side / 2, side / 2)
    painter.rotate(degrees)
    for color, wide in ((QColor(0, 0, 0, 190), 6.0), (QColor(255, 255, 255), 2.5)):
        pen = QPen(color, wide)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.strokePath(path, pen)
    painter.end()
    return QCursor(pixmap, side // 2, side // 2)


class Usage(QObject):
    """What the program is using, for the window's foot: processor (of one core; of this
    process, its decoding workers apart), memory, and the GPU's memory where it can be asked.
    Read every two seconds.

    ponytail: Linux only (/proc), and the GPU's figure only from NVIDIA's `nvidia-smi`, asked
    on a thread; elsewhere those parts are left out. The upgrade is psutil and the vendors'
    own interfaces."""

    changed = Signal()
    asked = Signal(str)

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self._text, self._gpu, self._was = "", "", (time.monotonic(), time.process_time())
        self._tool = shutil.which("nvidia-smi")
        self.asked.connect(self._answered)
        self._timer = QTimer(self, interval=2000, timeout=self._read)
        self._timer.start()

    @Property(str, notify=changed)
    def text(self) -> str:
        return self._text

    def _read(self) -> None:
        now = (time.monotonic(), time.process_time())
        share = 100 * (now[1] - self._was[1]) / max(1e-6, now[0] - self._was[0])
        self._was = now
        parts = [f"CPU {share:.0f} %"]
        try:
            pages = int(Path("/proc/self/statm").read_text().split()[1])
            parts.append(f"RAM {pages * os.sysconf('SC_PAGE_SIZE') / 2 ** 20:,.0f} MB")
        except (OSError, ValueError, IndexError, AttributeError):  # (the last: Windows in Wine)
            pass
        if self._tool:
            threading.Thread(target=self._ask, daemon=True).start()
        self._text = "  ·  ".join([*parts, *([self._gpu] if self._gpu else [])])
        self.changed.emit()

    def _ask(self) -> None:
        try:
            said = subprocess.run(
                [self._tool, "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1.5, check=False, **QUIET).stdout
        except (OSError, subprocess.SubprocessError):
            return
        mine = [line.split(",")[1].strip() for line in said.splitlines()
                if line.split(",")[0].strip() == str(os.getpid())]
        with contextlib.suppress(RuntimeError):  # (shut down meanwhile)
            self.asked.emit(f"GPU {mine[0]} MB" if mine else "")

    def _answered(self, text: str) -> None:
        self._gpu = text


def pipette_cursor(ratio: float) -> QCursor:
    """A pipette, its tip (bottom left) the hot spot: the pointer while a neutral gray is
    picked from the photo."""
    side = 40
    pixmap = QPixmap(round(side * ratio), round(side * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    # The whole pipette (its tube from the tip, a collar across it, the bulb), and apart from
    # it, just off its tip, a small ring round the very spot the color is taken from (the
    # owner: "full pipette but with a separated circle"). Dark first, then light in it.
    parts = (((11.5, 28.5), (24.0, 16.0), 3.0), ((20.5, 12.5), (27.5, 19.5), 3.0),
             ((26.5, 13.5), (33.5, 6.5), 7.5))
    for color, more in ((QColor(0, 0, 0, 200), 3.0), (QColor(255, 255, 255), 0.0)):
        for start, end, wide in parts:
            pen = QPen(color, wide + more)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(*start), QPointF(*end))
        painter.setPen(QPen(color, 1.3 + more * 0.8))
        painter.drawEllipse(QPointF(5.0, 35.0), 3.0, 3.0)
    painter.end()
    return QCursor(pixmap, 5, 35)


class TurnCursors(QObject):
    """The pointer where a drag turns something (the crop frame's rim, a mask's): a real
    cursor, which the desktop draws. (Drawn by the window at first, every move of the mouse
    cost a frame of the whole window, and the arc trailed the hand: the owner, "it lags".)
    Made once for every ten degrees; set on the item only when the step changes."""

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self.made: dict[int, QCursor] = {}
        self.step: int | None = None  # the step on show, or none: the pointer is not ours
        self.picking = False  # the pointer is the pipette

    @Slot(bool)
    def pipette(self, on: bool) -> None:
        """The pointer is a pipette (Pick neutral, over the photo), or no longer."""
        if on == self.picking:
            return
        self.picking = on
        if on:
            screen = QGuiApplication.primaryScreen()
            QGuiApplication.setOverrideCursor(
                pipette_cursor(float(math.ceil(screen.devicePixelRatio()) if screen else 1)))
        else:
            QGuiApplication.restoreOverrideCursor()

    @Slot(QObject, float)
    def turn(self, item, degrees: float) -> None:
        """The pointer is the turning one, bulging toward `degrees`. As the application's
        override: set on the item, Qt Quick did not pass it on to the window, and the pointer
        stayed an arrow (the owner: "it doesnt ever appear anymore")."""
        step = round(degrees / 10) % 36
        if self.step == step:
            return
        if step not in self.made:
            screen = QGuiApplication.primaryScreen()
            # (A whole ratio: the picture's size is then a whole number of the screen's pixels.)
            ratio = math.ceil(screen.devicePixelRatio()) if screen else 1
            self.made[step] = turn_cursor(step * 10, float(ratio))
        if self.step is None:
            QGuiApplication.setOverrideCursor(self.made[step])
        else:
            QGuiApplication.changeOverrideCursor(self.made[step])
        self.step = step
        item.setProperty("turnStep", step)
        Pace.cursors += 1

    @Slot(QObject)
    def plain(self, item) -> None:
        """The pointer is whatever is under it again."""
        item.setProperty("turnStep", -1)
        if self.step is not None:
            QGuiApplication.restoreOverrideCursor()
            self.step = None
            Pace.cursors += 1


class PerFrameInput(QObject):
    """The one place the window's raw input is handled: whatever arrives faster than frames
    are shown is taken in here and handed on once a frame, as its result.

    A mouse may report a thousand times a second and more (the owner's: up to 8,000), Qt on
    Wayland hands every report on, and a report that moves a slider costs the interface about
    a millisecond: at 1,000 a second the picture was drawn ten times a second. So:

    - moves: the first after a pause goes on at once; those that follow are held until a frame
      has been shown (or, when nothing is drawn, for two frames' time), and then the latest is
      sent on. Every handler works from where the pointer is, so nothing is lost by that;
    - turns of the wheel: held likewise and added up (`wheel`), one turn a frame carrying all
      of them, so a fast scroll is not slowed, it arrives in larger steps (the owner: "count
      how much we've scrolled ... and use that as the delta the next frame");
    - presses, releases, keys: at once, first sending on what is held, so nothing arrives out
      of turn. They come at the pace of fingers, and held for a frame they would only lag.

    Nothing else in the program sees an event at the device's rate: the interface's handlers
    get only what this sends. Any new kind of input that can come fast (touch, a pen) belongs
    here too (docs/PLAN.md §3, rule 6)."""

    OTHERS = (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease,
              QEvent.Type.MouseButtonDblClick, QEvent.Type.KeyPress, QEvent.Type.KeyRelease)
    STEADY = (Qt.ScrollPhase.NoScrollPhase, Qt.ScrollPhase.ScrollUpdate)  # (what may be added up)

    def __init__(self, view: QQuickView) -> None:
        super().__init__(view)
        self.view, self.held, self.turned, self.sending = view, None, None, False
        self.trail, self.passed = [], []  # places of held moves: gathered, and of the last send
        self.pace = None  # (before anything else: making a child of the view sends it an event)
        self.quiet = QTimer(self, singleShot=True, timerType=Qt.TimerType.PreciseTimer)
        self.quiet.timeout.connect(self.shown)
        view.installEventFilter(self)
        # (Queued: no Python on the render thread, which is where this signal comes from.)
        view.frameSwapped.connect(self.shown, Qt.ConnectionType.QueuedConnection)
        if os.environ.get("PHOTONE_PACE"):
            self.pace = Pace(view)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt's name
        kind = event.type()
        if self.pace and watched is self.view:
            self.pace.saw(kind, self.sending)
        if self.sending or watched is not self.view:
            return False
        if kind == QEvent.Type.MouseMove:
            if not self.quiet.isActive():  # the first for a while: at once
                self.wait()
                return False
            if self.held and event.buttons() and len(self.trail) < 512:
                self.trail.append(self.held[0])  # (the way gone, for what draws along it)
            self.held = (QPointF(event.position()), QPointF(event.globalPosition()),
                         event.buttons(), event.modifiers())
            return True
        if kind == QEvent.Type.Wheel:
            return self.wheel(event)
        if kind in self.OTHERS:
            self.send()
        return False

    def wheel(self, event) -> bool:
        """Hold a turn of the wheel, added to those held, until the next frame. The first for
        a while goes at once; the beginning or end of a touchpad's scroll, or a turn with other
        keys held, first sends on what is held."""
        held = self.turned
        if held and (held[5] != event.modifiers() or event.phase() not in self.STEADY):
            self.send()
        if event.phase() not in self.STEADY:
            return False
        if not self.quiet.isActive():
            self.wait()
            return False
        if self.turned is None:
            self.turned = [None, None, QPoint(0, 0), QPoint(0, 0), event.buttons(),
                           event.modifiers(), event.phase(), event.inverted()]
        turned = self.turned
        turned[0], turned[1] = QPointF(event.position()), QPointF(event.globalPosition())
        turned[2] += event.pixelDelta()
        turned[3] += event.angleDelta()
        return True

    def wait(self) -> None:
        rate = self.view.screen().refreshRate() if self.view.screen() else 60.0
        self.quiet.start(max(4, int(2000 / max(30.0, rate))))

    @Slot()
    def shown(self) -> None:
        """A frame has been shown, or none came: what is held goes on; with nothing held, the
        next move goes through at once."""
        if self.pace and self.sender() is self.view:
            self.pace.frames += 1
        if self.held is None and self.turned is None:
            self.quiet.stop()
        else:
            self.send()

    def send(self) -> None:
        move, turned = self.held, self.turned
        if move is None and turned is None:
            return
        self.held = self.turned = None
        self.passed, self.trail = self.trail, []
        self.sending = True
        if move:
            position, on_screen, buttons, keys = move
            QCoreApplication.sendEvent(self.view, QMouseEvent(
                QEvent.Type.MouseMove, position, on_screen, Qt.MouseButton.NoButton, buttons, keys))
        if turned:
            QCoreApplication.sendEvent(self.view, QWheelEvent(*turned))
        self.sending = False
        self.passed = []
        self.wait()

    @Slot(result="QVariantList")
    def skipped(self) -> list:
        """While a move is being handled: where in the window the pointer was at the moves
        held back since the one before, in order. A brush paints through them, so that a fast
        stroke is the way the hand went and not a chord a frame long (the owner: "looks like
        its framerate limited"). Still one change a frame."""
        return self.passed


class Pace(QObject):
    """PHOTONE_PACE=1: a line a second on the terminal with what the window really gets and
    what it costs, for a lag that only the owner's own mouse brings about (scripts send events
    to the window and do not come the desktop's way). Per second: events that arrived, by
    kind; how many were handed on to the interface; frames shown; changes to the session; the
    longest the interface's thread did not come back to its loop; times a pointer was set."""

    cursors = 0  # (TurnCursors counts here)

    def __init__(self, view: QQuickView) -> None:
        super().__init__(view)
        self.kinds: dict[str, int] = {}
        self.on, self.frames, self.changes, self.worst = 0, 0, 0, 0.0
        self.last = time.perf_counter()
        session = view.findChild(Session)
        if session:
            session.changed.connect(self.changed)
        self.beat = QTimer(self, interval=5, timerType=Qt.TimerType.PreciseTimer)
        self.beat.timeout.connect(self.tick)
        self.beat.start()
        self.second = QTimer(self, interval=1000)
        self.second.timeout.connect(self.say)
        self.second.start()

    def saw(self, kind, ours: bool) -> None:
        name = kind.name if hasattr(kind, "name") else str(kind)
        if ours:
            self.on += 1
        else:
            self.kinds[name] = self.kinds.get(name, 0) + 1

    @Slot()
    def changed(self) -> None:
        self.changes += 1

    @Slot()
    def tick(self) -> None:
        now = time.perf_counter()
        self.worst, self.last = max(self.worst, now - self.last), now

    @Slot()
    def say(self) -> None:
        most = sorted(self.kinds.items(), key=lambda item: -item[1])[:5]
        kinds = ", ".join(f"{kind} {count}" for kind, count in most)
        print(f"pace  arrived {sum(self.kinds.values()):5d} ({kinds})"
              f"  handed on {self.on:3d}  frames {self.frames:3d}  changes {self.changes:3d}"
              f"  loop {self.worst * 1000:4.0f} ms  pointers set {Pace.cursors}",
              file=sys.stderr, flush=True)
        self.kinds, self.on, self.frames, self.changes, self.worst = {}, 0, 0, 0, 0.0
        Pace.cursors = 0


class FirstFrame(QObject):
    """Tells the session, after the window's first frame, what kind of surface it really got.

    A QObject of the interface's thread, reached by a queued connection, so that no Python
    runs on Qt's render thread: Python there has to wait for the interpreter, and if the
    interface's thread is at that moment waiting for the render thread (hiding the window
    does), the two wait for each other for good. (Seen in a test script, not in the program.)"""

    def __init__(self, view: QQuickView, session: Session) -> None:
        super().__init__(view)
        self.view, self.session, self.done = view, session, False

    @Slot()
    def seen(self) -> None:
        if self.done:  # (a second frame came before this ran: seen on Windows)
            return
        self.done = True
        self.view.frameSwapped.disconnect(self.seen)  # once is enough
        self.view.setTitle("Photone")
        self.session.surface_known(is_hdr_surface(self.view))


def watch_surface(view: QQuickView, session: Session) -> None:
    """After the first frame, tell the session what kind of surface the window really got.
    Until then the interface is not drawn (it would not know how to draw its colors).

    Until then the title also says why the window may stay white: on Windows the first launch
    turns every shader into the graphics card's code before the first frame (a few seconds;
    Qt keeps the result, so later launches are past it at once)."""
    view.setTitle("Photone (Compiling shaders…)")
    view.frameSwapped.connect(FirstFrame(view, session).seen, Qt.ConnectionType.QueuedConnection)


def close_view(view: QQuickView) -> None:
    """Take the window apart now, while Qt and Python are both whole. Left to the end of the
    program, the two tear the same objects down in an order of their own, and it crashed."""
    view.hide()
    view.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    QtCore.QCoreApplication.processEvents()
    # Python's own leftovers of it go now too (a session and its timers hold each other): a
    # window made next in the same run (the shots make dozens) was otherwise handed objects at
    # the addresses of those, and its interface read their properties as undefined.
    gc.collect()

