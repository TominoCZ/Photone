"""Photos leave the program here, as files. The picture is the one the screen shows: the photo
at its full size goes through the same shader (ui/Bench.qml, ui/pipeline.frag), in tiles, and
is read back from the GPU; this module says which tiles, puts them together, and writes the
file: never over one that is there, never where a check has not looked (PLAN.md, rule 2).
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from itertools import count
from pathlib import Path

import numpy as np
from PySide6.QtCore import Property, QBuffer, QObject, QSize, Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import (
    QColor,
    QColorSpace,
    QFont,
    QFontMetrics,
    QImage,
    QImageReader,
    QImageWriter,
    QPainter,
)
from PySide6.QtQuick import QQuickItem

from photone import decode, edit, lut, shaders
from photone.catalog import identify

# What can be written: the file's ending, what Qt calls the format, and how many bits a color has.
FORMATS = {"jpeg": (".jpg", b"jpeg", 8), "png8": (".png", b"png", 8),
           "png16": (".png", b"png", 16), "tiff16": (".tif", b"tiff", 16)}
TABLE = 65   # points a side of a look written out as a .cube file
RECENT = 12  # how many of the last exported files are kept to show
TILE = 2048  # a tile's side: small enough for any GPU, and for a window of the checks


def lattice() -> QImage:
    """A picture of every color of a table of TABLE points a side (lut.atlas of the table that
    changes nothing), as decode.py hands over a PNG: linear Rec.2020 half floats."""
    side = TABLE * TABLE
    colors = np.rint(lut.atlas(lut.identity(TABLE)).astype(np.float64) * 65535).astype(np.uint16)
    image = QImage(colors.tobytes(), side, TABLE, side * 8, QImage.Format.Format_RGBA64).copy()
    image.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
    image.convertTo(QImage.Format.Format_RGBA16FPx4)
    image.convertToColorSpace(QColorSpace(QColorSpace.Primaries.Bt2020,
                                          QColorSpace.TransferFunction.Linear))
    return image


class Failed(Exception):
    """A photo could not be exported; the message is for the user."""


def destination(folder: Path, name: str, ending: str) -> Path:
    """Where a photo's export goes: `name` with that ending in the folder, or with a number
    after it if that is taken. Taken at once (an empty file), so that two exports running into
    each other cannot both be given the same name."""
    stem = Path(name).stem
    for n in count(1):
        path = folder / (f"{stem}{ending}" if n == 1 else f"{stem}-{n}{ending}")
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
            return path
        except FileExistsError:
            continue
    raise AssertionError("unreachable")


def tiles(width: int, height: int, deep: bool, side: int = TILE,
          layers: int = 0) -> list[list[int]]:
    """The renders an export is made of: [x, y, wide, high, pass] each. One pass for 8 bits
    (0: the picture with what its masks change inside them, as the screen has it). For 16,
    two (1, 2), each value's high and low byte travelling apart (`deep_of`); and three more
    for each of `layers` masks that have an edit of their own (3, 4: the picture inside it,
    likewise; 5: how much the mask weighs; then 6, 7, 8 for the next): half-values cannot be
    mixed by the screen, so `assembled` mixes them."""
    turns = (1, 2, *range(3, 3 + 3 * layers)) if deep else (0,)
    return [[x, y, min(side, width - x), min(side, height - y), turn]
            for y in range(0, height, side) for x in range(0, width, side) for turn in turns]


def bytes_of(grab: QImage) -> np.ndarray:
    """A grab as rows x columns x 3 bytes: red, green, blue. A copy: the image's own memory
    is gone when the image is (and was once read after that: the first pixels of every tile
    were whatever had been put there since)."""
    grab = grab.convertToFormat(QImage.Format.Format_RGBA8888)
    rows = np.frombuffer(grab.constBits(), np.uint8).reshape(grab.height(), grab.bytesPerLine())
    return rows[:, :grab.width() * 4].reshape(grab.height(), grab.width(), 4)[..., :3].copy()


def deep_of(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Sixteen bits a color out of two renders of eight: the first carries red's high and low
    byte and green's high one, the second green's low one and blue's two (pipeline.frag,
    `scope` 6 and 7)."""
    first, second = first.astype(np.uint16), second.astype(np.uint16)
    return np.stack([first[..., 0] << 8 | first[..., 1], first[..., 2] << 8 | second[..., 0],
                     second[..., 1] << 8 | second[..., 2]], axis=-1)


def picture(pixels: np.ndarray, longest: int = 0) -> QImage:
    """Pixels (rows x columns x 3, bytes or 16-bit numbers) as an sRGB image; scaled down so
    that its longer side is `longest`, if that is given and smaller."""
    high, wide = pixels.shape[:2]
    if pixels.dtype == np.uint8:
        image = QImage(pixels.tobytes(), wide, high, wide * 3, QImage.Format.Format_RGB888).copy()
    else:
        four = np.full((high, wide, 4), 65535, np.uint16)
        four[..., :3] = pixels
        image = QImage(four.tobytes(), wide, high, wide * 8, QImage.Format.Format_RGBX64).copy()
    if 0 < longest < max(wide, high):
        image = image.scaled(QSize(longest, longest), Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
    image.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
    return image


# A watermark: nothing, some text, or a picture from a file; in one of nine places of the
# photo (0 top left … 8 bottom right, row by row), as high as `size` percent of the photo's
# shorter side, as strong as `strength` percent. The same for every export, and kept.
MARK = {"kind": "none", "text": "", "file": "", "place": 8, "size": 4.0, "strength": 60.0}
MARK_SIDE = 8000  # a picture for a watermark is no larger than this a side


def valid_mark(given) -> dict:
    """A watermark's settings read from somewhere untrusted: what is not valid is as MARK
    has it."""
    given, out = given if isinstance(given, dict) else {}, dict(MARK)
    if given.get("kind") in ("none", "text", "picture"):
        out["kind"] = given["kind"]
    for key, most in (("text", 120), ("file", 1000)):
        if isinstance(given.get(key), str):
            out[key] = given[key][:most]
    if isinstance(given.get("place"), (int, float)) and not isinstance(given["place"], bool):
        out["place"] = int(min(8, max(0, given["place"])))
    for key, low, high in (("size", 1.0, 30.0), ("strength", 5.0, 100.0)):
        value = given.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value:
            out[key] = float(min(high, max(low, value)))
    return out


MARK_TALL = 200  # how high a watermark's text is drawn, to be scaled to what is asked


def _mark_text(words: str) -> QImage:
    """Text as a watermark, MARK_TALL high: white, with a dark copy behind it, so that it
    reads on a bright sky too."""
    font = QFont()
    font.setPixelSize(MARK_TALL)
    font.setBold(True)
    metrics = QFontMetrics(font)
    drop = MARK_TALL // 16
    drawn = QImage(min(MARK_SIDE, metrics.horizontalAdvance(words) + 2 * drop),
                   metrics.height() + drop, QImage.Format.Format_ARGB32_Premultiplied)
    drawn.fill(0)
    painter = QPainter(drawn)
    painter.setFont(font)
    for color, off in ((QColor(0, 0, 0, 150), drop), (QColor(255, 255, 255), 0)):
        painter.setPen(color)
        painter.drawText(drop + off, metrics.ascent() + off, words)
    painter.end()
    return drawn


def _mark_picture(file: str) -> QImage | None:
    """A picture file as a watermark, no larger than 2048 px a side. Read as untrusted: None
    for what is no picture, or one too large to be a watermark."""
    reader = QImageReader(file)
    reader.setAutoTransform(True)
    size = reader.size()
    if not size.isValid() or max(size.width(), size.height()) > MARK_SIDE:
        return None
    drawn = reader.read()
    if drawn.isNull():
        return None
    drawn = drawn.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    if max(drawn.width(), drawn.height()) > 2048:
        drawn = drawn.scaled(QSize(2048, 2048), Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
    return drawn


def mark_base(mark: dict) -> QImage | None:
    """The watermark itself, at a size of its own and at full strength; None where there is
    none (switched off, no text, a file that is no picture). It is made once for a text or a
    file: an export scales it to the file's picture and Edit to the photo on screen, so that
    its scale, its opacity and its place cost nothing to change. (It was drawn anew at every
    step of a slider, a picture read from its file each time; the owner: "LAGS like hell".)"""
    if mark["kind"] == "text" and mark["text"].strip():
        return _mark_text(mark["text"].strip())
    if mark["kind"] == "picture" and mark["file"]:
        return _mark_picture(mark["file"])
    return None


def mark_size(mark: dict, base: tuple[int, int], wide: int, high: int) -> tuple[int, int]:
    """How large the watermark (`base`: its own width and height) is on a picture of that
    size: as high as `size` percent of the shorter side, and no wider than the picture less
    its edges (then lower, too)."""
    tall = max(4.0, mark["size"] / 100 * min(wide, high))
    across = base[0] * tall / max(1, base[1])
    fit = min(1.0, max(8, wide - 2 * mark_edge(wide, high)) / max(1.0, across))
    return max(1, round(across * fit)), max(1, round(tall * fit))


def mark_edge(wide: int, high: int) -> int:
    """How far a watermark stays from the photo's edges: 3 % of its shorter side."""
    return round(0.03 * min(wide, high))


def mark_at(place: int, wide: int, high: int, across: int, down: int) -> tuple[int, int]:
    """Where the top left of a watermark `across` by `down` goes on a picture of that size."""
    edge = mark_edge(wide, high)
    return ((edge, (wide - across) // 2, wide - across - edge)[place % 3],
            (edge, (high - down) // 2, high - down - edge)[place // 3])


def stamp(image: QImage, mark: dict, base: QImage | None = None) -> None:
    """Paint the watermark on a picture that is about to be written."""
    base = base if base is not None else mark_base(mark)
    if base is not None:
        size = mark_size(mark, (base.width(), base.height()), image.width(), image.height())
        drawn = base.scaled(QSize(*size), Qt.AspectRatioMode.IgnoreAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        painter = QPainter(image)
        painter.setOpacity(mark["strength"] / 100)
        painter.drawImage(*mark_at(mark["place"], image.width(), image.height(), *size), drawn)
        painter.end()


def write(image: QImage, path: Path, kind: str, quality: int = 90) -> None:
    """The image into the file that `destination` took. Failed if it cannot be written; the
    file is then removed again, so that nothing half-written or empty is left behind."""
    writer = QImageWriter(str(path), FORMATS[kind][1])
    if kind == "jpeg":
        writer.setQuality(max(1, min(100, int(quality))))
    # (A TIFF is written uncompressed: the LZW that Qt offers made a 33 MP photo's 16-bit file
    # larger, 260 MB against 198, and slower.)
    if not writer.write(image):
        path.unlink(missing_ok=True)
        raise Failed(f"{path.name} could not be written: {writer.errorString()}")


def small(image: QImage, side: int = 96) -> str:
    """A small picture of an exported one, as an address an Image can show (for the log of
    the last exports: the file itself may be 200 MB, or gone)."""
    held = QBuffer()
    held.open(QBuffer.OpenModeFlag.WriteOnly)
    image.scaled(QSize(side, side), Qt.AspectRatioMode.KeepAspectRatio,
                 Qt.TransformationMode.SmoothTransformation).convertToFormat(
        QImage.Format.Format_RGB888).save(held, "JPEG", 80)
    return "data:image/jpeg;base64," + base64.b64encode(bytes(held.data())).decode()


def assembled(job: dict, grabs: list[np.ndarray]) -> np.ndarray:
    """The whole picture out of its tiles' grabs, in the order `tiles` named them."""
    deep = any(tile[4] for tile in job["tiles"])
    whole = np.zeros((job["height"], job["width"], 3), np.uint16 if deep else np.uint8)
    grabs = iter(grabs)
    held = inside = None
    for (x, y, wide, high, turn), grab in zip(job["tiles"], grabs, strict=True):
        part, place = grab[:high, :wide], (slice(y, y + high), slice(x, x + wide))
        if turn == 0:
            whole[place] = part
        elif turn in (1, 2):  # the picture: its high bytes, then its low ones
            held, whole[place] = (part, whole[place]) if turn == 1 else (None, deep_of(held, part))
        elif turn % 3 == 0:  # a mask's own picture: high bytes
            held = part
        elif turn % 3 == 1:  # its low ones
            inside = deep_of(held, part).astype(np.float64)
        else:  # how much the mask weighs: mixed as the screen mixes the two
            weight = part[..., :1] / 255.0
            whole[place] = np.rint(whole[place] * (1 - weight) + inside * weight).astype(np.uint16)
    return whole


class Exporter(QObject):
    """Exports photos one after the other. For each: its pixels are decoded at full size (on a
    thread), ui/Bench.qml renders them through the pipeline tile by tile and hands every grab
    to `tile`, and the whole is put together and written (on a thread again). The interface
    reads `job` (what the bench is to render), `running`, `finished`, `total`, `failures`."""

    changed = Signal()
    decoded = Signal(object)  # from the decoding thread: the picture, or the sentence why not
    tileTaken = Signal()  # a render asked for by `grabTile` has been taken in
    written = Signal(str, str, int)  # from the writing thread: why not or "", the file, the run

    def __init__(self, session, parent: QObject | None = None, side: int = TILE) -> None:
        super().__init__(parent)
        self.session, self.side = session, side
        self.queue: list[Path] = []
        self.picture: QImage | None = None  # of the photo on the bench: the provider hands it out
        self._job, self._grabs, self._serial = {}, [], 0
        self._run = 0  # counts starts and stops: what a thread of an earlier one brings is dropped
        self._total, self._finished, self._failures, self._last = 0, 0, [], ""
        self.options = {"kind": "jpeg", "quality": 90, "longest": 0, "folder": ""}
        try:  # the last files exported, kept between runs
            self._recent = list(json.loads(session.remembered("export/recent", "[]")))[:RECENT]
        except (ValueError, TypeError):
            self._recent = []
        try:  # the watermark, kept between runs too
            self._mark = valid_mark(json.loads(session.remembered("export/mark", "{}")))
        except (ValueError, TypeError):
            self._mark = dict(MARK)
        self._mark_shown = session.remembered("export/mark_shown", "yes") == "yes"
        self._mark_serial, self._marked = 0, (None, None)
        # The watermark is written to the settings when it has come to rest, not at every
        # step of a slider: a write is 2.5 ms on the owner's disk, 6 at worst, and a frame
        # of his display is 4 ("the faster i move the sliders ... the laggier it is").
        self._mark_keeper = QTimer(self, singleShot=True, interval=400, timeout=self.keep_mark)
        self.decoded.connect(self._decoded)
        self.written.connect(self._written)

    # ---- what the interface reads -----------------------------------------------------------

    @Property("QVariantMap", notify=changed)
    def job(self) -> dict:
        """The photo on the bench: `source`, `width`, `height`, `uniforms`, `tiles`; empty
        when there is none."""
        return self._job

    @Property(bool, notify=changed)
    def running(self) -> bool:
        return bool(self.queue or self._job or self.picture is not None or self._busy)

    @Property(int, notify=changed)
    def total(self) -> int:
        return self._total

    @Property(int, notify=changed)
    def finished(self) -> int:
        """How many photos have been dealt with, written or failed."""
        return self._finished

    @Property("QVariantList", notify=changed)
    def failures(self) -> list:
        """One sentence for each photo that could not be exported."""
        return self._failures

    @Property(str, notify=changed)
    def last(self) -> str:
        """The folder the last export went to."""
        return self._last

    @Property("QVariantList", notify=changed)
    def recent(self) -> list:
        """The last files exported, the newest first: `path`, `name`, `source` (the photo it is of),
        `folder`, `when` (seconds since 1970), `thumb` (a small picture of it: `small`), `wide`,
        `high`, `bytes`."""
        return self._recent

    # ---- the watermark ----------------------------------------------------------------------

    markChanged = Signal()

    @Property("QVariantMap", notify=markChanged)
    def mark(self) -> dict:
        """The watermark every export gets (MARK)."""
        return self._mark

    @Property(int, notify=markChanged)
    def markSerial(self) -> int:  # noqa: N802
        """Another number whenever the watermark itself is another (its kind, its text, its
        file; not its scale, opacity or place): part of its picture's address."""
        return self._mark_serial

    @Property(bool, notify=markChanged)
    def markShown(self) -> bool:  # noqa: N802
        """Whether Edit shows the watermark on the photo (an export has it either way)."""
        return self._mark_shown

    @Slot("QVariantMap")
    def setMark(self, values: dict) -> None:  # noqa: N802
        values = dict(values)
        if "file" in values and "://" in str(values["file"]):
            values["file"] = QUrl(str(values["file"])).toLocalFile()
        now = valid_mark({**self._mark, **values})
        # (Its picture is made anew for another kind, text or file; and when a file is chosen,
        # also the one it had: the file itself may be another by now.)
        if "file" in values or any(now[k] != self._mark[k] for k in ("kind", "text", "file")):
            self._mark_serial += 1
        elif now == self._mark:
            return
        self._mark = now
        self._mark_keeper.start()
        self.markChanged.emit()

    def keep_mark(self) -> None:
        """Write the watermark to the settings now (also at exit, should it not be yet)."""
        self._mark_keeper.stop()
        self.session.remember("export/mark", json.dumps(self._mark))

    @Slot(bool)
    def setMarkShown(self, shown: bool) -> None:  # noqa: N802
        self._mark_shown = shown
        self.session.remember("export/mark_shown", "yes" if shown else "no")
        self.markChanged.emit()

    def mark_picture(self) -> QImage | None:
        """The watermark itself (mark_base), made once until it is another. Also asked for on
        Qt's image loader thread."""
        if self._marked[0] != self._mark_serial:
            self._marked = (self._mark_serial, mark_base(self._mark))
        return self._marked[1]

    @Slot(int, int, result="QVariantList")
    def markPlace(self, wide: int, high: int) -> list:  # noqa: N802
        """Where the watermark lies on a picture of that size, as shares of it: left, top,
        width, height. Empty: there is none."""
        base = self.mark_picture()
        if base is None or wide <= 0 or high <= 0:
            return []
        across, down = mark_size(self._mark, (base.width(), base.height()), wide, high)
        x, y = mark_at(self._mark["place"], wide, high, across, down)
        return [x / wide, y / high, across / wide, down / high]

    @Slot()
    def forget(self) -> None:
        """Empty that list (the files stay)."""
        self._recent = []
        self.session.remember("export/recent", "[]")
        self.changed.emit()

    # ---- the run -------------------------------------------------------------------------------

    _busy = False  # a thread of ours is decoding or writing

    @Slot("QVariantList", "QVariantMap")
    def start(self, photos: list, options: dict) -> None:
        """Export these photos (paths). `options`: `kind` (one of FORMATS), `quality` (JPEG),
        `longest` (pixels of the longer side, 0: as it is), `folder` (where to)."""
        folder = Path(str(options.get("folder", "")))
        if self.running or not photos:
            return
        if options.get("kind") not in FORMATS or not folder.is_dir():
            self._failures = [f"There is no folder {folder} to export to."]
            return self.changed.emit()
        self.session.flush()  # the edit of the open photo, as it is now
        self.options = {**self.options, **options, "folder": str(folder)}
        self.queue = [Path(str(p)) for p in photos]
        self._total, self._finished, self._failures = len(self.queue), 0, []
        self._last = str(folder)
        self._next()

    @Slot(str)
    def startTable(self, file: str) -> None:  # noqa: N802
        """Write the open photo's look as a .cube file of TABLE points a side: what the look
        makes of every color of a finished picture, found by sending a picture of all the
        table's colors down the same road as a photo to be exported. Without what a table
        cannot hold: the crop, the masks and their pins, the grain, and what goes by a
        pixel's neighbors (sharpening, noise reduction, the lens)."""
        path = Path(QUrl(file).toLocalFile() if "://" in file else file).with_suffix(".cube")
        if self.running or not self.session.photos:
            return
        if not path.parent.is_dir():
            self._failures = [f"There is no folder {path.parent} to write to."]
            return self.changed.emit()
        # (Nor what goes by a pixel's neighbors, which in this picture are other colors of the
        # table and no neighbors: sharpening, noise reduction and the lens, switched off.)
        parts = [part for part in edit.PARTS if part not in ("grain", "detail", "lens")]
        look = {**edit.look_of(self.session.history.edit, parts), "sharpen": 0.0, "denoise": 0.0,
                "denoiseColor": 0.0}
        image = lattice()
        self._table = look
        self._photo, self._busy = path, True
        self._total, self._finished, self._failures = 1, 0, []
        self.changed.emit()
        meta = {"camera": None, "raw": False, "width": TABLE * TABLE, "height": TABLE}
        self._decoded((self._run, (meta, image)))

    _table: dict | None = None  # the look whose table is being written, while it is

    @Slot()
    def stop(self) -> None:
        """Stop now: the photos still waiting are not exported, and the one under way is
        given up (the owner: "stopping an export doesnt work": it used to finish that one,
        which for a single large PNG is all of it). A thread that is decoding or writing
        cannot be interrupted: what it brings is dropped (`_run`), and a file it has written
        meanwhile it takes away again itself."""
        self._run += 1
        self._serial += 1
        self.queue, self._total, self._busy = [], self._finished, False
        self._job, self._grabs, self.picture, self.painted, self._table = {}, [], None, None, None
        self.changed.emit()

    def _next(self) -> None:
        self._job, self._grabs, self.picture, self.painted, self._table = {}, [], None, None, None
        if self.queue:
            self._busy, self._photo = True, self.queue.pop(0)
            threading.Thread(target=self._decode, args=(self._photo, self._run),
                             daemon=True).start()
        self.changed.emit()

    def _decode(self, photo: Path, run: int) -> None:
        """On a thread: the photo's pixels at full size, as the image the shader takes."""
        try:
            meta, pixels = self.session.load(photo, False)
        except (decode.Unreadable, OSError) as e:
            return self.decoded.emit((run, str(e) or f"{photo.name} could not be read"))
        try:
            # Its own copy of the pixels, before they are given back. (A raw file's mosaic, as
            # a rule: the bench develops it as the photo on screen is, Develop.qml.)
            wide, high, kind = decode.texels(meta)
            image = QImage(pixels.buf, wide, high, decode.size_of(meta) // high,
                           getattr(QImage.Format, kind)).copy()
        finally:
            pixels.close()
        self.decoded.emit((run, (meta, image)))

    def _edit_of(self, photo: Path) -> dict:
        """The photo's edit: the open one's as it stands, any other's from the catalog. (Or,
        for a look's table, that look.)"""
        if self._table is not None:
            return self._table
        if self.session.path == str(photo):
            return dict(self.session.history.edit)
        kept = self.session.catalog
        return kept.load(identify(photo), photo) if kept else {}

    def _decoded(self, brought) -> None:
        run, found = brought
        if run != self._run:
            return
        self._busy = False
        if isinstance(found, str):
            return self._written(found, "", run)
        meta, self.picture = found
        camera = meta["camera"] or edit.FINISHED
        camera = {**camera, "shot": edit.as_shot(camera), "match": meta.get("match")}
        stored = self._edit_of(self._photo)
        frame = edit.frame(stored, meta["width"], meta["height"])  # its turns, angle and crop
        shape = meta["width"] / max(1, meta["height"])
        sized = {"shape": shape, "wide": meta["width"]}
        numbers = {**edit.uniforms(stored, camera, meta["raw"], **sized),
                   "frame": frame["matrix"]}
        # (What its masks change inside them: the picture once more for each, edit.layers.)
        inside = [{"mask": n, "numbers": shaders.for_qml(
            {**edit.uniforms(local, camera, meta["raw"], **sized), "frame": frame["matrix"]})}
            for n, local in edit.layers(stored, camera)]
        self._serial += 1
        # (And its painted masks, as the picture the shader reads them from: not the open
        # photo's kept raster, which is another photo's or may be painted on meanwhile.)
        strokes = [m["strokes"] for m in stored.get("masks", []) if m["kind"] == "brush"]
        self.painted = self._painted(strokes, shape) if strokes else None
        deep = self._table is not None or FORMATS[self.options["kind"]][2] == 16
        tables = stored.get("luts")
        wide, high = frame["size"]
        self._job = {"source": f"image://photo/export/{self._serial}/full", "width": wide,
                     "height": high, "uniforms": shaders.for_qml(numbers), "layers": inside,
                     "painted": f"image://photo/painted/export/{self._serial}" if strokes else "",
                     "lut": f"image://photo/lut/{lut.address(tables)}/export" if tables else "",
                     # (A look's table: each color of it read by itself, not mixed with its
                     # neighbor's as a photo's pixels are, however little.)
                     "exact": self._table is not None, "mosaic": meta.get("mosaic"),
                     "tiles": tiles(wide, high, deep, self.side, len(inside))}
        self.changed.emit()

    @staticmethod
    def _painted(strokes: list, shape: float) -> QImage:
        wide, high = edit.raster_size(shape)
        colors = np.zeros((high, wide, 4), dtype=np.uint8)
        colors[..., 3] = 255
        for channel, mine in enumerate(strokes[:edit.BRUSHES]):
            colors[..., channel] = edit.painted(mine, shape)
        return QImage(colors.data, wide, high, wide * 4, QImage.Format.Format_RGBX8888).copy()

    @Slot(QQuickItem, result=bool)
    def grabTile(self, item) -> bool:  # noqa: N802
        """Take the render of the bench as it stands (one of `job.tiles`); `tileTaken` says
        when it has been taken in. (Asked for here, not by the interface's script:
        app.Session.grabScope says why.)"""
        grab = item.grabToImage()
        if grab is None:
            return False
        self._grab = (grab, self._serial)
        grab.ready.connect(self._tile_ready)
        return True

    def _tile_ready(self) -> None:
        held, self._grab = getattr(self, "_grab", None), None
        if held is not None and held[1] == self._serial and self._job:  # (not after Stop)
            self.tile(held[0].image())
            self.tileTaken.emit()

    @Slot(QImage)
    def tile(self, grab: QImage) -> None:
        """One render of the bench, in the order of `job.tiles`."""
        self._grabs.append(bytes_of(grab))

    @Slot()
    def rendered(self) -> None:
        """The bench has rendered every tile (or could not load the picture)."""
        if not self._job:
            return  # (stopped meanwhile)
        job, grabs, self._busy = self._job, self._grabs, True
        options = {**self.options, "table": self._table is not None, "mark": dict(self._mark),
                   "mark_base": self.mark_picture()}
        self._job, self._grabs, self.picture, self.painted, self._table = {}, [], None, None, None
        self.changed.emit()
        threading.Thread(target=self._write, daemon=True,
                         args=(self._photo, job, grabs, options, self._run)).start()

    def _write(self, photo: Path, job: dict, grabs: list, options: dict, run: int) -> None:  # noqa: PLR0913
        """On a thread: put the tiles together and write the file."""
        if len(grabs) != len(job.get("tiles", [None])):
            return self.written.emit(f"{photo.name} could not be rendered.", "", run)
        try:
            if options["table"]:  # a look's table: the colors that came back, as a .cube file
                flat = assembled(job, grabs).astype(np.float64) / 65535
                table = flat.reshape(TABLE, TABLE, TABLE, 3).transpose(1, 0, 2, 3)
                part = photo.with_name(photo.name + ".part")
                part.write_text(lut.cube(table, photo.stem), "utf-8")
                os.replace(part, photo)
                return self.written.emit("", "", run)
            image = picture(assembled(job, grabs), int(options["longest"]))
            stamp(image, options["mark"], options["mark_base"])
            if run != self._run:
                return None
            path = destination(Path(options["folder"]), photo.name, FORMATS[options["kind"]][0])
            write(image, path, options["kind"], options["quality"])
            if run != self._run:  # stopped while it was written: it is not left behind
                return path.unlink(missing_ok=True)
            self.written.emit("", json.dumps({
                "path": str(path), "name": path.name, "source": str(photo),
                "folder": str(path.parent), "when": int(time.time()), "thumb": small(image),
                "wide": image.width(), "high": image.height(), "bytes": path.stat().st_size}), run)
        except Failed as e:
            self.written.emit(str(e), "", run)
        except OSError as e:
            self.written.emit(f"{photo.name} could not be written: {e}", "", run)

    def _written(self, trouble: str, file: str, run: int) -> None:
        if run != self._run:
            return
        self._busy = False
        self._finished += 1
        if trouble:
            self._failures = [*self._failures, trouble]
        elif file:
            self._recent = [json.loads(file), *self._recent][:RECENT]
            self.session.remember("export/recent", json.dumps(self._recent))
        self._next()

    @Slot("QVariant", result=str)
    def folder_of(self, chosen) -> str:
        """A folder as the dialog gives it, as a path."""
        return chosen.toLocalFile() if isinstance(chosen, QUrl) else str(chosen)
