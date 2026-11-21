"""The checks: plain asserts over the logic under the window, no framework and no display.

    ./run.sh check
    PHOTONE_SAMPLE=/path/photo.ARW ./run.sh check    also decodes a real raw file
"""

import faulthandler
import gc
import json
import math
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from itertools import pairwise, product  # noqa: E402
from multiprocessing.shared_memory import SharedMemory  # noqa: E402

from PySide6.QtCore import QCoreApplication, QSettings, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtQml import QQmlComponent, QQmlEngine  # noqa: E402

from photone import app, catalog, decode, denoise, dng, edit, export, library, scopes, shaders  # noqa: E402


def check_store() -> None:
    """Developed pixels into a buffer of four half floats a pixel: a raw file's integers with
    full scale as 1, in every band of a picture of any height; a finished picture's untouched."""
    raw = (np.arange(12, dtype=np.uint16).reshape(2, 2, 3) + 1) * 5000  # as LibRaw gives them
    buf = bytearray(2 * 2 * 8)
    decode.store(raw, buf)
    out = np.frombuffer(buf, np.float16).reshape(2, 2, 4)
    assert np.allclose(out[..., :3], raw / 65535, atol=3e-4) and (out[..., 3] == 1).all(), "same numbers, opaque"
    assert out[0, 1, 0] < out[1, 0, 0], "row-major, the first row first"
    ends = np.array([[[0, 32768, 65535]]], np.uint16)
    decode.store(ends, buf)
    assert np.frombuffer(buf, np.float16)[:4].tolist() == [0.0, 0.5, 1.0, 1.0], "full scale is 1"
    tall = np.random.default_rng(3).integers(0, 65536, (37, 5, 3), dtype=np.uint16)  # more rows than bands, unevenly
    buf = bytearray(37 * 5 * 8)
    decode.store(tall, buf)
    out = np.frombuffer(buf, np.float16).reshape(37, 5, 4)
    assert np.abs(out[..., :3].astype(float) - tall / 65535).max() < 3e-4 and (out[..., 3] == 1).all(), "every row, once"
    buf = bytearray(2 * 2 * 8)
    floats = np.full((1, 3, 4), 0.25, np.float16)  # as a JPEG arrives from Qt
    buf = bytearray(3 * 8)
    decode.store(floats, buf)
    assert np.frombuffer(buf, np.float16).tolist() == [0.25] * 12
    half = np.frombuffer(decode.pack(np.full((1, 1, 3), 0.5, np.float32)), np.float16)
    assert half.tolist() == [0.5, 0.5, 0.5, 1.0], "the test chart's own packing"


def check_synthetic() -> None:
    meta, pixels = decode.synthetic(40, 20)
    assert (meta["width"], meta["height"]) == (40, 20)
    assert meta["space"] == "rec2020-linear" and meta["format"] == "rgba16f"
    assert meta["full"] == [40, 20] and meta["size"] == "full"
    assert len(pixels.buf) == 40 * 20 * 8, "8 bytes a pixel"
    px = np.frombuffer(pixels.buf, np.float16).reshape(20, 40, 4).astype(np.float32)
    assert px.min() >= 0 and px[..., :3].max() <= 1.0001
    wedge = px[-1, :, :3]
    assert np.allclose(wedge[:, 0], wedge[:, 1]) and np.allclose(wedge[:, 1], wedge[:, 2]), "the wedge is neutral"
    assert wedge[0, 0] == 0 and wedge[-1, 0] == 1, "from black to white"
    assert decode.meta("x", 10, 6, (20, 12))["size"] == "half"


def chart(path: Path, half: bool):
    meta, pixels = decode.synthetic(8, 4)
    return {**meta, "name": path.name, "full": [16, 8], "size": "half" if half else "full"}, pixels


def damaged(path: Path, half: bool):
    raise decode.Unreadable(f"{path.name}: damaged")


def check_session() -> None:
    assert app.Session([]).source == "", "nothing open, nothing to load"
    session = app.Session([Path("/photos/a.arw"), Path("/photos/b.arw")], load=chart)
    assert session.source == "image://photo/0/0/half" and session.count == 2
    image = session.image("0/0/half")
    assert (image.width(), image.height()) == (8, 4)
    assert image.format() == QImage.Format.Format_RGBA16FPx4, "half floats, not converted"
    assert (session.fullWidth, session.fullHeight) == (16, 8), "the real size, not the decoded one"
    assert session.info == "a.arw  ·  16 × 8" and not session.failed and session.error == ""
    assert "/photos" not in session.info, "the interface shows names, not paths"
    # The bytes mean the same to Qt as to numpy: same pixel, same channels, same row order.
    px = np.frombuffer(decode.synthetic(8, 4)[1].buf, np.float16).reshape(4, 8, 4).astype(float)
    for x, y in ((0, 0), (5, 1), (7, 3)):
        got = image.pixelColor(x, y)
        assert np.allclose([got.redF(), got.greenF(), got.blueF()], px[y, x, :3], atol=2e-3), (x, y)

    # A raw file has a sharper picture to fetch after the first; anything else does not.
    assert session.fullSource == "image://photo/0/0/full"
    changes = []
    session.photoChanged.connect(lambda: changes.append(session.source))
    session.step(1)
    assert changes == ["image://photo/1/1/half"] and session.info == ""
    session.step(1)
    session.step(-5)
    assert changes[-1] == "image://photo/2/0/half" and len(changes) == 2, "stops at the ends"
    # A late answer for a photo the user has left changes nothing about the one shown.
    assert session.image("0/1/half").width() == 1, "nor is it decoded at all"
    assert session.info == "", "the old photo's size must not land on the new one"

    def integers(path: Path, half: bool):  # as the raw decoder hands them over
        buf = bytearray(2 * 1 * 8)
        decode.store(np.array([[[32768, 16384, 0], [65535, 65535, 65535]]], np.uint16), buf)
        return decode.meta(path.name, 2, 1), decode.Pixels(buf)

    ints = app.Session([Path("i.arw")], load=integers).image("0/0/half")
    assert ints.format() == QImage.Format.Format_RGBA16FPx4, "the program's own copy, as half floats"
    got = ints.pixelColor(0, 0)
    assert np.allclose([got.redF(), got.greenF(), got.blueF()], [0.5, 0.25, 0], atol=1e-3)
    assert abs(ints.pixelColor(1, 0).greenF() - 1.0) < 1e-3, "full scale is 1"

    here = Path(__file__)
    session.open([str(here), "file://" + str(here.parent / "nothing.arw")])
    assert session.photos == [here] and session.count == 1, "only files that exist"
    assert session.fullSource == "", "not a raw file: the first picture is all there is"
    session.open(["/nonexistent/x.arw"])
    assert session.photos == [here] and "could be found" in session.error

    broken = app.Session([Path("b.arw")], load=damaged)
    assert broken.image("0/0/half").width() == 1 and broken.failed, "unreadable is a state"
    assert broken.error == "b.arw: damaged", "and a sentence, never an exception"
    assert app.Session([]).image("0/3/half").width() == 1, "a wrong index does not raise either"

    # SDR | HDR: the surface is a fact about the window, the switch a choice on top of it.
    fresh = app.Session([])
    assert not fresh.ready and not fresh.hdr and not fresh.hdrSurface, "unknown before a frame"
    fresh.surface_known(True)
    assert fresh.ready and fresh.hdrSurface and not fresh.hdr, "an HDR surface starts on SDR"
    fresh.setHdr(True)
    assert fresh.hdr and fresh.hdrSurface
    fresh.setHdr(False)
    assert not fresh.hdr and fresh.hdrSurface, "switching off leaves the surface as it is"
    assert not fresh.linearSurface, "an HDR surface is written gamma-encoded unless told otherwise"
    os.environ["PHOTONE_HDR_ENCODING"] = "linear"
    told = app.Session([])
    del os.environ["PHOTONE_HDR_ENCODING"]
    assert not told.linearSurface, "there is no linear surface before there is an HDR one"
    told.surface_known(True)
    assert told.linearSurface

    plain = app.Session([])
    plain.surface_known(False)
    plain.setHdr(True)
    assert not plain.hdr and plain.error == app.NO_HDR, "no HDR surface: a sentence, and SDR"
    asked = app.Session([], hdr=True)
    assert not asked.hdr, "wanting is not having"
    asked.surface_known(False)
    assert not asked.hdr and asked.error == app.NO_HDR, "asked for outright: say it cannot be"


def check_settings() -> None:
    """What is remembered between runs: the switch, and the user's own peak."""
    with tempfile.TemporaryDirectory() as folder:
        def kept() -> QSettings:
            return QSettings(str(Path(folder) / "photone.ini"), QSettings.Format.IniFormat)

        first = app.Session([], settings=kept())
        first.surface_known(True)
        assert not first.hdr and first.peak == 2.0, "SDR and a peak of 2 until told otherwise"
        first.set_headroom(2.03)
        assert first.peak == 2.03, "what the desktop reports"
        first.setHdr(True)
        again = app.Session([], settings=kept())
        again.surface_known(True)
        assert again.hdr, "the switch is remembered"
        quiet = app.Session([], settings=kept())
        quiet.surface_known(False)
        assert not quiet.hdr and quiet.error == "", "remembered HDR on a plain surface: no nagging"
        assert not app.Session([], hdr=False, settings=kept()).hdr
        kept().setValue("view/hdr_peak", 1.6)
        mine = app.Session([], settings=kept())
        mine.set_headroom(4.0)
        assert mine.peak == 1.6, "the user's own figure wins over the desktop's"


def check_headroom() -> None:
    report = """\x1b[01;32mOutput: \x1b[0;0m1 DP-1 546c
\tenabled
\tHDR: enabled
\t\tSDR brightness: 500 nits
\t\tPeak brightness: 1015 nits, overridden with: 1015 nits
Output: 2 HDMI-A-1 77aa
\tHDR: disabled
Output: 3 DP-2 99bb
\tHDR: enabled
\t\tSDR brightness: 200 nits
\t\tPeak brightness: 600 nits, overridden with: 1000 nits
"""
    assert abs(app.headroom_from_kscreen(report, "DP-1") - 2.03) < 0.001
    assert app.headroom_from_kscreen(report, "DP-2") == 5.0, "the user's override of the peak wins"
    assert app.headroom_from_kscreen(report, "HDMI-A-1") is None, "no HDR on that output"
    assert app.headroom_from_kscreen(report, "eDP-1") is None, "no such output"
    assert app.headroom_from_kscreen("", "DP-1") is None and app.headroom_from_kscreen("junk", "DP-1") is None


def check_raster() -> None:
    """A PNG comes out as linear Rec.2020: mid gray stays neutral, red lands where it should."""
    with tempfile.TemporaryDirectory() as folder:
        picture = QImage(4, 2, QImage.Format.Format_RGB32)
        picture.fill(QColor(128, 128, 128))
        picture.setPixelColor(3, 1, QColor(255, 0, 0))
        path = Path(folder) / "p.png"
        picture.save(str(path))
        meta, pixels = decode.load(path)
        assert (meta["width"], meta["height"], meta["full"], meta["size"]) == (4, 2, [4, 2], "full")
        assert meta["format"] == "rgba16f"
        px = np.frombuffer(pixels.buf, np.float16).reshape(2, 4, 4).astype(float)
        assert np.allclose(px[0, 0, :3], 0.2159, atol=2e-3), px[0, 0]
        assert np.allclose(px[1, 3, :3], [0.6274, 0.0691, 0.0164], atol=3e-3), px[1, 3]
        assert np.allclose(px[..., 3], 1.0), "opaque"
        (Path(folder) / "broken.jpg").write_bytes(b"not a picture")
        for name in ("broken.jpg", "missing.png", "notes.txt"):
            try:
                decode.load(Path(folder) / name)
            except decode.Unreadable as e:
                assert name in str(e)
            else:
                raise AssertionError(f"{name} should not open")


def check_workers() -> None:
    """Decoding in another process: the same answer, and a dead worker is one sentence."""
    workers = decode.Workers()
    try:
        with tempfile.TemporaryDirectory() as folder:
            picture = QImage(4, 2, QImage.Format.Format_RGB32)
            picture.fill(QColor(128, 128, 128))
            path = Path(folder) / "p.png"
            picture.save(str(path))
            (there, shared), (here, own) = workers.load(path), decode.load(path)
            assert there == here and bytes(shared.buf) == bytes(own.buf), "the same as in this process"
            name = shared.shared.name
            shared.close()
            try:
                SharedMemory(name=name)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("the shared memory must be gone once the pixels are taken")
            # Decoded ahead: there when asked for; given up: its memory is given back.
            other = Path(folder) / "q.png"
            picture.save(str(other))
            # (On workers of its own, which give way to those that decode what is asked for.)
            assert workers.pool.submit(os.nice, 0).result() == 0 and workers.calm.submit(os.nice, 0).result() == 10
            workers.ahead([(path, True)], False)
            assert workers.before == {}, "told only what is worth keeping, it begins nothing"
            workers.ahead([(path, True), (other, True), (other, False)])
            assert set(workers.before) == {(path, True), (other, True)}, "no full size of what has none"
            waiting = workers.before[(other, True)]
            info, pixels = workers.load(path)
            assert info["name"] == "p.png" and bytes(pixels.buf) == bytes(own.buf) and (path, True) not in workers.before
            pixels.close()
            left = waiting.result()[1].name
            workers.ahead([])
            assert workers.before == {}
            for _ in range(200):  # (given back from the thread that finished the decoding)
                try:
                    SharedMemory(name=left).close()
                except FileNotFoundError:
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("what was decoded ahead and never asked for must be given back")
            try:
                workers.call(os._exit, 1)
            except decode.Unreadable as e:
                assert "crashed" in str(e)
            else:
                raise AssertionError("a dead worker must be reported")
            info, pixels = workers.load(path)
            pixels.close()
            assert info["name"] == "p.png", "and the next photo gets a new one"
            try:
                workers.load(Path(folder) / "none.png")
            except decode.Unreadable as e:
                assert "none.png" in str(e), "errors cross the process with their message"
    finally:
        workers.close()


def check_no_orphan() -> None:
    """Kill the program outright: its worker must end with it (it once stayed for good)."""
    project = str(Path(__file__).resolve().parent.parent)
    script = (f"import os, sys, time; sys.path.insert(0, {project!r})\n"
              "from photone import decode\n"
              "if __name__ == '__main__':\n"
              "    print(decode.Workers().call(os.getpid), flush=True); time.sleep(60)\n")
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "app.py").write_text(script)
        program = subprocess.Popen([sys.executable, str(Path(folder) / "app.py")],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        worker = int(program.stdout.readline())
        program.kill()
        program.wait()
        for _ in range(40):
            if not Path(f"/proc/{worker}").exists():
                return
            time.sleep(0.05)
        os.kill(worker, 9)
        raise AssertionError("the worker outlived the program that started it")


CAMERA = decode.synthetic(8, 4)[0]["camera"]  # a real camera's figures
ON = catalog.LENS_ON  # what an edit kept before the lens corrections were off at rest says
THEN = {**ON, "curve": edit.STRAIGHT}  # and one of a raw file, kept before its curve was its own


def check_edit_values() -> None:
    """What may enter an edit (photone/edit.py)."""
    assert edit.valid("exposure", 9) == 5.0 and edit.valid("tint", -999) == -150.0, "kept in range"
    assert edit.valid("temperature", None) is None, "None is 'as shot'"
    assert edit.valid("curve", None) is None, "None is 'the photo's own'"
    for key, bad in (("exposure", None), ("exposure", "1"), ("exposure", float("nan")), ("contrast", None),
                     ("exposure", True), ("profile", "vivid"), ("sharpness", 1)):
        try:
            edit.valid(key, bad)
        except ValueError:
            continue
        raise AssertionError(f"{key}={bad!r} must be refused")
    assert edit.clean({"exposure": 0.5, "shadows": 0, "profile": "flat", "junk": 1,
                       "tint": "x", "highlights": 1e9}) == {"exposure": 0.5, "profile": "flat",
                                                           "highlights": 100.0}
    assert edit.clean("not a dict") == {} and edit.clean(None) == {}
    # A tone curve: points left to right, kept in 0..1, the same after a trip through JSON.
    curve = edit.valid("curve", ((0, 0), (0.5, 1.4), (1, 1)))
    assert curve == [[0.0, 0.0], [0.5, 1.0], [1.0, 1.0]] and json.loads(json.dumps(curve)) == curve
    assert edit.clean({"curve": edit.STRAIGHT}) == {"curve": edit.STRAIGHT}, "straight is a choice: a raw file's own curve is not"
    assert edit.clean({"curve": None, "exposure": 0}) == {}, "at rest: the photo's own, as shot"
    for bad in ([[0, 0]], [[0, 0], [0.5, 0.5], [0.5, 0.6], [1, 1]], [[0.6, 0], [0.4, 1]], "x",
                [[0, 0], [1, "1"]], [[0, 0, 0], [1, 1, 1]], [[i / 20, 0] for i in range(17)]):
        assert edit.clean({"curve": bad}) == {}, f"curve {bad!r} must be refused"


def check_user_curve() -> None:
    """The tone curve the user draws: through its points, never swinging past one, flat
    outside its end points; and packed for the shader."""
    # The curve a raw photo starts with (edit.matched): a darker picture of the same thing is
    # shown as light as the other, and neither end is cut off.
    theirs = np.random.default_rng(3).beta(2, 3, 20000)
    ours = theirs ** 1.8
    made = edit.valid_curve(edit.matched(ours, theirs))
    shown = np.array([edit.curve_at(made, x) for x in ours])
    assert abs(np.median(shown) - np.median(theirs)) < 0.01 and np.abs(np.percentile(shown, [25, 90]) - np.percentile(theirs, [25, 90])).max() < 0.02
    assert made[0] == [0, 0] and made[-1] == [1, 1] and all(b[1] >= a[1] for a, b in pairwise(made))
    assert edit.matched(ours, np.ones(9)) == edit.STRAIGHT, "a picture all white: nothing to go by"
    assert abs(edit.lightness([1, 1, 1]) - 1) < 1e-4 and abs(edit.lightness([0.18] * 3) - 0.18 ** (1 / 3)) < 1e-4
    at = edit.curve_at
    assert all(abs(at(edit.STRAIGHT, x / 50) - x / 50) < 1e-12 for x in range(51)), "straight is no change"
    points = [[0.0, 0.0], [0.25, 0.15], [0.6, 0.75], [1.0, 1.0]]
    assert all(abs(at(points, x) - y) < 1e-12 for x, y in points), "it passes through its points"
    line = [at(points, x / 400) for x in range(401)]
    assert all(a <= b + 1e-12 for a, b in zip(line, line[1:], strict=False)), "rising points, rising line"
    bump = [[0.0, 0.0], [0.3, 0.5], [0.5, 0.5], [1.0, 1.0]]
    assert all(abs(at(bump, x / 100) - 0.5) < 1e-12 for x in range(30, 51)), "level between two level points"
    hat = [at([[0.0, 0.2], [0.5, 0.9], [1.0, 0.4]], x / 200) for x in range(201)]
    assert max(hat) <= 0.9 + 1e-12 and min(hat) >= 0.2 - 1e-12, "no swing past a point"
    moved = [[0.1, 0.05], [0.9, 0.8]]
    assert at(moved, 0.0) == 0.05 and at(moved, 1.0) == 0.8, "flat outside the black and white points"
    u = edit.uniforms({"curve": points}, None, True)
    assert u["curveOn"] == 1 and edit.uniforms({}, None, True)["curveOn"] == 0
    assert len(u["curveX"]) == len(u["curveY"]) == len(u["curveM"]) == 16
    assert u["curveX"][:4] == [0.0, 0.25, 0.6, 1.0] and set(u["curveX"][4:]) == {1.0} and set(u["curveM"][4:]) == {0.0}


def check_pins() -> None:
    """Color pins (photone/edit.py): what may be one, how much of one a color gets, and what
    the pins make of a color."""
    pin = edit.valid("pins", [{"from": (0.1, 0.05), "range": 9, "junk": 1}])[0]
    assert pin == {"from": [0.1, 0.05], "to": [0.1, 0.05], "light": 0.0, "range": 0.5, "feather": 70.0,
                   "variance": 0.0, "low": 0.0, "high": 1.0, "mode": "offset", "mask": 0}, pin
    assert json.loads(json.dumps([pin])) == [pin] and edit.clean({"pins": []}) == {}
    for bad in ("x", [{"to": [0, 0]}], [{"from": [0, 0, 0]}], [{"from": [0, 0], "mode": "twist"}],
                [{"from": [0, 0], "range": "wide"}], [{"from": [0, 0], "low": 0.6, "high": 0.6}],
                [{"from": [0, 0]}] * 13):
        assert edit.clean({"pins": bad}) == {}, f"pins {bad!r} must be refused"

    def pinned(color, **settings) -> np.ndarray:
        pins = edit.valid("pins", [{"from": [0.1, 0.0], "to": [0.1, 0.06], "range": 0.1, "feather": 50, **settings}])
        return edit.pins_at(pins, color)

    here, edge, far = [0.6, 0.1, 0.0], [0.6, 0.1, 0.075], [0.6, -0.1, 0.0]
    assert np.allclose(pinned(here), [0.6, 0.1, 0.06]), "the pinned color goes where the pin goes"
    assert np.allclose(pinned([0.6, 0.13, 0.03]), [0.6, 0.13, 0.09]), "and its near neighbors with it"
    assert 0 < pinned(edge)[2] - edge[2] < 0.06, "less toward the edge of the range"
    assert np.allclose(pinned(far), far), "and not at all beyond it"
    assert np.allclose(pinned(here, feather=0)[1:], [0.1, 0.06]) and np.allclose(pinned(edge, feather=0), [0.6, 0.1, 0.135])
    assert np.allclose(pinned(here, light=0.1), [0.7, 0.1, 0.06]), "lighter too, if asked"
    assert np.allclose(pinned(here, low=0.7), here) and np.allclose(pinned([0.9, 0.1, 0], low=0.7)[1:], [0.1, 0.06])
    assert np.allclose(pinned(here, high=0.3), here), "only within its span of lightness"
    gray = [0.5, 0.0, 0.0]
    assert pinned(gray, range=0.2)[2] > 0.005, "an offset tints a gray within its range"
    assert np.allclose(pinned(gray, range=0.2, mode="rotate"), gray), "a rotation leaves a gray gray"
    turned = pinned(here, mode="rotate", to=[0.0, 0.2])
    assert np.allclose(turned, [0.6, 0.0, 0.2]), f"rotate: the hue turned, the chroma doubled: {turned}"
    assert np.allclose(pinned([0.6, 0.01, 0.0], mode="rotate", **{"from": [0.01, 0.0]}), [0.6, 0.1, 0.06]), "no hue to turn: offset"
    off = [0.6, 0.1, 0.02]  # a little off the pin's hue
    gathered, spread = (pinned(off, to=[0.1, 0.0], variance=v) for v in (-100, 100))
    assert abs(gathered[2]) < 0.001 and spread[2] > 0.035, "variance gathers the hues, or spreads them"
    assert abs(np.hypot(*gathered[1:]) - np.hypot(*off[1:])) < 1e-9, "and leaves the chroma alone"
    shield = edit.valid("pins", [{"from": [0.1, 0.0], "to": [0.1, 0.06], "range": 0.1},
                                 {"from": [0.1, 0.0], "range": 0.03, "feather": 0, "mode": "protect"}])
    assert np.allclose(edit.pins_at(shield, here), here), "a protected color is not moved"
    assert edit.pins_at(shield, [0.6, 0.1, 0.05])[2] > 0.06, "what lies outside the protection is"
    two = edit.valid("pins", [{"from": [0.1, 0.0], "to": [0.12, 0.0]}, {"from": [0.1, 0.0], "to": [0.1, 0.03]}])
    assert np.allclose(edit.pins_at(two, here), [0.6, 0.12, 0.03]), "two pins add up"
    u = edit.uniforms({"pins": shield}, None, True)
    assert u["pinCount"] == 2 and len(u["pinsA0"]) == 16 and u["pinsA0"][:8] == [0.1, 0.0, 0.1, 0.06, 0.1, 0.0, 0.1, 0.0]
    assert np.allclose(u["pinsB0"][:4], [0.1, 0.03, 0.0, 0.0]) and u["pinsC0"][2] == 0 and u["pinsC0"][6] == 2
    assert u["pinsB0"][5] < u["pinsB0"][4], "the core is always inside the range: the shader divides by the gap"
    assert edit.uniforms({}, None, True)["pinCount"] == 0 and not any(edit.uniforms({}, None, True)["pinsA2"])


def check_mono() -> None:
    """Black and white (photone/edit.py): a gray of the color's own lightness, which the B&W
    EQ makes lighter or darker by hue, the less the nearer the color is to gray; the color
    grade is kept beside it and untouched."""
    green, blue = [0.6, -0.1, 0.1], [0.5, -0.02, -0.14]  # hues 135 and 262 degrees
    assert np.allclose(edit.colors_at({"mono": True, "bw": []}, green), [0.6, 0, 0]), "no look: its lightness"
    points = edit.valid("bw", [{"hue": 135, "gain": 90, "width": 40}, {"hue": 262, "gain": -20, "width": 500}])
    assert points == [{"hue": 135.0, "gain": 40.0, "width": 40.0}, {"hue": 262.0, "gain": -20.0, "width": 120.0}], "kept in range"
    assert abs(edit.bw_at(points, 135) - 1.4) < 1e-12 and abs(edit.bw_at(points, 262) - 0.8) < 1e-12 and edit.bw_at(points, 60) == 1
    share = math.hypot(-0.1, 0.1) / (math.hypot(-0.1, 0.1) + edit.TINTED)
    assert np.allclose(edit.colors_at({"mono": True, "bw": points}, green), [0.6 * (1 + 0.4 * share), 0, 0])
    assert edit.colors_at({"mono": True, "bw": points}, blue)[0] < 0.5, "the blues darker"
    assert np.allclose(edit.colors_at({"mono": True, "bw": points}, [0.5, 0, 0]), [0.5, 0, 0]), "a gray has no hue: as it is"
    assert np.allclose(edit.colors_at({"mono": True, "bw": points}, [0.0, -0.1, 0.1]), [0, 0, 0]), "black stays black"
    grade = {"saturation": 150.0, "pins": edit.valid("pins", [{"from": [0.1, 0.0], "to": [0.1, 0.06]}]), "eq": edit.valid("eq", [{"hue": 135, "gain": 150, "width": 40}])}
    assert np.allclose(edit.colors_at({**grade, "mono": True, "bw": points}, green), edit.colors_at({"mono": True, "bw": points}, green)), "the color tools do nothing to it"
    assert edit.clean({**grade, "mono": True}) == {**grade, "mono": True}, "and are kept beside it"
    assert edit.DEFAULTS["bw"] == edit.FILMS["panchromatic"] and edit.clean({"mono": False, "bw": edit.FILMS["panchromatic"]}) == {}
    for name, film in edit.FILMS.items():
        assert edit.valid("bw", film) == film, f"the film look {name} is not as an edit keeps it"
        assert all(0.59 < edit.bw_at(film, hue) < 1.41 for hue in range(360)), f"{name}: its points add up to more than one point can say"
    assert edit.bw_at(edit.FILMS["red"], 30) > 1.2 > 0.8 > edit.bw_at(edit.FILMS["red"], 262), "a red filter: reds lighter, blues darker"
    assert edit.bw_at(edit.FILMS["orthochromatic"], 30) < 0.8 and edit.bw_at(edit.FILMS["orthochromatic"], 262) > 1.1
    for bad in (1, "yes", None):
        assert edit.clean({"mono": bad}) == {}, f"mono {bad!r} must be refused"
    # For the shader: one switch, the color tools at rest, the B&W EQ in the color EQ's place.
    u = edit.uniforms({**grade, "mono": True, "bw": points}, None, True)
    assert u["mono"] == 1 and u["look"] == 1 and u["pinCount"] == 0 and u["saturation"] == 1 and u["chromaOn"] == 0
    assert u["eqCount"] == 2 and np.allclose(u["eqA0"][:8], [135, 0.4, 40, 0, 262, -0.2, 120, 0])
    assert edit.uniforms(grade, None, True)["mono"] == 0 and edit.uniforms(grade, None, True)["pinCount"] == 1
    inside = edit.uniforms(edit.merged({"mono": True}, {"eq": grade["eq"], "exposure": 1.0}, None), None, True)
    assert inside["mono"] == 1 and inside["eq2Count"] == 0 and inside["gain"] == 2, "inside a mask: black and white too, its own color EQ idle"


def check_grain() -> None:
    """Film grain (photone/edit.py): its settings, the characters and responses as sets of
    them, and the numbers the shader is handed."""
    rest = edit.uniforms({}, None, True)
    assert rest["grain"] == 0 and rest["look"] == 0, "no grain: nothing done after the tone curve"
    for name, character in edit.CHARACTERS.items():
        assert edit.clean(character) == {k: v for k, v in character.items() if v != edit.DEFAULTS[k]}, name
    for name, response in edit.RESPONSES.items():
        assert edit.valid("grainCurve", response) == response and max(p[1] for p in response) == 1, name
    assert edit.DEFAULTS["grainCurve"] == edit.RESPONSES["print"] and edit.clean({"grainCurve": edit.RESPONSES["print"]}) == {}
    assert edit.clean({"grain": 250, "grainSize": 0.1, "grainColor": "much", "grainCurve": [[0, 0]]}) == {"grain": 100.0, "grainSize": 0.4}
    u = edit.uniforms({"grain": 50.0, "grainSize": 2.0, "grainRough": 80.0, "grainColor": 40.0}, None, True, shape=1.5)
    assert u["look"] == 1 and abs(u["grain"] - 0.15) < 1e-12 and u["grainRough"] == 0.8 and u["grainColor"] == 0.4
    assert u["grainWide"] == 2000 and abs(u["grainHigh"] - 2000 / 1.5) < 1e-9, "sized by the longer side: half as many at twice the size"
    upright = edit.uniforms({"grain": 50.0}, None, True, shape=2 / 3)
    assert abs(upright["grainWide"] - 4000 * 2 / 3) < 1e-9 and upright["grainHigh"] == 4000
    assert u["grainX"][:3] == [0.0, 0.5, 1.0] and u["grainY"][:3] == [0.0, 1.0, 0.0] and len(u["grainM"]) == 16
    assert edit.uniforms({"grain": 50.0, "grainColor": 40.0, "mono": True}, None, True)["grainColor"] == 0, "black and white: no colored grain"
    assert edit.uniforms({"grain": 50.0}, None, True, shape=0.0)["grainWide"] == 4000, "before a photo is there its shape is 0"
    assert set(edit.TOOLS["grain"]) == {k for k in edit.DEFAULTS if k.startswith("grain")}


def check_look() -> None:
    """A look (photone/edit.py): what of an edit goes to another photo, whole or in parts,
    and what it leaves of the edit it is laid on."""
    oval = {"kind": "radial", "at": [0.5, 0.5], "to": [0.7, 0.5]}
    pins = edit.valid("pins", [{"from": [0.1, 0.0], "to": [0.1, 0.06]}, {"from": [0.0, 0.1], "to": [0.0, 0.05], "mask": 1}])
    source = edit.clean({"exposure": 0.7, "contrast": 20.0, "temperature": 5200.0, "curve": [[0, 0], [0.5, 0.6], [1, 1]],
                         "saturation": 130.0, "pins": pins, "masks": [oval], "crop": [0.1, 0.1, 0.9, 0.9], "angle": 3.0,
                         "grain": 35.0, "mono": True, "bw": edit.FILMS["red"]})
    look = edit.look_of(source)
    assert not set(look) & {"masks", *edit.FRAME}, "the frame and the masks are the photo's own"
    assert look["pins"] == pins[:1] and look["exposure"] == 0.7 and look["bw"] == edit.FILMS["red"], "of the pins, the whole image's"
    assert set(edit.look_of(source, ["light"])) == {"exposure", "contrast"} and edit.look_of(source, ["balance"]) == {"temperature": 5200.0}
    assert edit.look_of({"nonsense": 1, "exposure": "much"}) == {}, "cleaned: a preset file is not trusted"
    assert {key for keys in edit.PARTS.values() for key in keys} | {"masks", *edit.FRAME} == set(edit.DEFAULTS), "every setting is in a part, or is the photo's own"
    target = edit.clean({"exposure": -1.0, "highlights": -30.0, "tint": 12.0, "vibrance": 40.0, "turns": 1.0, "masks": [oval],
                         "pins": edit.valid("pins", [{"from": [-0.1, 0.0], "to": [-0.1, 0.03]}, {"from": [0.05, 0.05], "to": [0.0, 0.0], "mask": 1}])})
    laid = edit.with_look(target, look)
    assert laid["turns"] == 1.0 and laid["masks"] == target["masks"], "the frame and the masks stay"
    assert laid["exposure"] == 0.7 and "highlights" not in laid and "tint" not in laid and "vibrance" not in laid, "a part is the look's, at rest where it says nothing"
    assert laid["pins"] == [pins[0], target["pins"][1]], "the look's pins, and the photo's own masks' pins"
    part = edit.with_look(target, look, ["balance", "grain"])
    assert part == {**{k: v for k, v in target.items() if k != "tint"}, "temperature": 5200.0, "grain": 35.0}, "only the parts asked for"
    assert edit.with_look(target, {}) == {k: target[k] for k in ("turns", "masks")} | {"pins": target["pins"][1:]}, "an empty look: everything of a look at rest"
    assert edit.with_look(source, edit.look_of(source)) == source, "its own look changes nothing"


def check_presets() -> None:
    """Presets (photone/presets.py): kept by name in a file, cleaned when read, one of the
    same name replaced; a file that is not what it should be costs nothing but its presets."""
    from photone import presets
    with tempfile.TemporaryDirectory() as folder:
        file = Path(folder) / "deep" / "presets.json"
        kept = presets.Presets(file)
        assert kept.all == [] and not kept.error
        source = {"exposure": 0.7, "temperature": 5200.0, "saturation": 130.0, "crop": [0.1, 0.1, 0.9, 0.9]}
        assert kept.save("  Warm   day ", source, ["light", "color"]) and kept.all == [
            {"name": "Warm day", "parts": ["light", "color"], "look": {"exposure": 0.7, "saturation": 130.0}}]
        assert not kept.save("", source, ["light"]) and not kept.save("x", source, []) and not kept.save("x", source, ["nonsense"])
        assert kept.save("Cool", {"temperature": 4800.0}, ["balance"]) and not kept.save(" Warm  day", {"exposure": 1.0}, ["light"])
        assert kept.taken("Warm day ") and not kept.taken("Warm")
        assert [r["name"] for r in kept.all] == ["Warm day", "Cool"] and kept.all[0]["look"] == {"exposure": 0.7, "saturation": 130.0}, "the same name: refused, the first stays"
        kept.rename(1, "Warm day")
        assert kept.all[1]["name"] == "Cool", "a name that is taken is not given twice"
        kept.rename(1, "Blue hour")
        again = presets.Presets(file)
        assert again.all == kept.all and again.all[1]["name"] == "Blue hour", "what was saved is there the next time"
        again.remove(0)
        again.remove(7)
        assert [r["name"] for r in presets.Presets(file).all] == ["Blue hour"]
        file.write_text(json.dumps([{"name": "Odd", "parts": ["light", "x"], "look": {"exposure": "much", "contrast": 500, "masks": 3}},
                                    {"name": 5, "parts": ["light"]}, "nonsense", {"name": "No parts", "look": {}}]), "utf-8")
        assert presets.Presets(file).all == [{"name": "Odd", "parts": ["light"], "look": {"contrast": 100.0}}], "read as untrusted"
        file.write_text("{ not json", "utf-8")
        broken = presets.Presets(file)
        assert broken.all == [] and "could not be read" in broken.error
    assert presets.Presets().save("In memory", {"exposure": 1.0}, ["light"]), "with no file: kept while the program runs"
    with tempfile.TemporaryDirectory() as folder:  # (they were recipes for a day: that file is still read)
        (Path(folder) / "recipes.json").write_text(json.dumps([{"name": "Old", "parts": ["light"], "look": {"exposure": 1.0}}]), "utf-8")
        moved = presets.Presets(Path(folder) / "presets.json")
        assert [r["name"] for r in moved.all] == ["Old"] and moved.save("New", {"contrast": 5.0}, ["light"])
        assert [r["name"] for r in presets.Presets(Path(folder) / "presets.json").all] == ["Old", "New"]


def check_lut() -> None:
    """LUTs (photone/lut.py): a .cube file read as untrusted, its table laid out for the
    shader, and written again; and the edit's setting for one."""
    from photone import lut
    same = lut.identity(5)
    assert same.shape == (5, 5, 5, 3) and same[4][0][2].tolist() == [0.5, 0.0, 1.0], "indexed [blue][green][red]"
    text = lut.cube(same, 'My "look"')
    assert text.startswith('TITLE "My look"\nLUT_3D_SIZE 5\n') and text.count("\n") == 3 + 125
    assert text.splitlines()[4] == "0.250000 0.000000 0.000000", "red runs fastest"
    assert np.allclose(lut.parse(text), same, atol=1e-6)
    spoken = "# a comment\nTITLE \"x\"\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1.0 1.0 1.0\nLUT_3D_SIZE 2\n" + "\n".join(["0 0 0  # black"] * 7 + ["1 1 1"]) + "\n"
    assert lut.parse(spoken)[1][1][1].tolist() == [1.0, 1.0, 1.0]
    flat = lut.atlas(same)
    assert flat.shape == (5, 25, 4) and flat.dtype == np.float16 and flat[1, 3 * 5 + 2].tolist() == [0.5, 0.25, 0.75, 1.0], "slices of one blue side by side, red across, green down"
    rows = "\n".join(["0 0 0"] * 8)
    for bad, why in ((f"LUT_1D_SIZE 8\n{rows}", "1D"), (rows, "size"), (f"LUT_3D_SIZE 3\n{rows}", "8 colors where 27"),
                     (f"LUT_3D_SIZE 99\n{rows}", "99 points"), (f"LUT_3D_SIZE 2\n{rows[:-5]}0 nan 0", "no colors"),
                     (f"LUT_3D_SIZE 2\n{rows[:-5]}0 b c", "no number"), (f"LUT_3D_SIZE 2\n{rows}\n0 0", "9 colors"),
                     (f"LUT_3D_SIZE 2\nDOMAIN_MAX 2 2 2\n{rows}", "range"), (f"LUT_3D_SIZE two\n{rows}", "not a number"),
                     ("x" * (lut.BYTES + 1), "too large")):
        try:
            lut.parse(bad)
            raise AssertionError(f"a file that is no LUT was read: {why}")
        except ValueError as said:
            assert why in str(said), f"{why!r} is not in {str(said)!r}"
    # A photo's LUTs: a list, each an installed one's id, name and size, a strength, on or off.
    had = {"id": "0a1b2c", "name": "Kodak " * 30, "size": 33, "strength": 250}
    kept = {"id": "0a1b2c", "name": ("Kodak " * 30)[:80], "size": 33, "strength": 100.0}
    assert edit.valid("luts", [had, {**had, "id": "ff", "size": 17, "strength": 40, "off": True, "more": 1}]) == [
        kept, {"id": "ff", "name": kept["name"], "size": 17, "strength": 40.0, "off": True}]
    assert edit.clean({"luts": []}) == {} and edit.clean({"lut": had}) == {"luts": [kept]}, "one LUT, as a photo had it for a day"
    for bad in ({**had, "id": "../x"}, {**had, "size": 1}, {**had, "size": 2.5}, {"id": "a"}, "a.cube", {**had, "name": 5}, None):
        assert edit.clean({"luts": [bad]}) == {}, f"a LUT {bad!r} must be refused"
    assert edit.clean({"luts": [had] * 17}) == {} and len(edit.clean({"luts": [had] * 16})["luts"]) == 16 == edit.LUTS
    two = [{**kept, "strength": 40.0}, {"id": "ff", "name": "x", "size": 17, "strength": 80.0, "off": True}, {**kept, "size": 9}]
    u = edit.uniforms({"luts": two}, None, True)
    assert u["lutCount"] == 3 and u["lutsA"] == [33, 0.4, 9, 0, 17, 0, 42, 0, 9, 1, 0, 0, 0, 0, 0, 0], "size, strength (none while off), the row its band begins at"
    many = edit.uniforms({"luts": [kept] * 4 + [{**kept, "strength": 30.0}] + [kept] * 11}, None, True)
    assert many["lutCount"] == 16 and many["lutsB"][:4] == [33, 0.3, 0, 0] and many["lutsD"][12:] == [33, 1, 0, 0] and not any(u["lutsD"]), "four to a matrix, sixteen in all"
    assert lut.bands(two) == {("0a1b2c", 9): 0, ("0a1b2c", 33): 9, ("ff", 17): 42} == lut.bands(two[::-1] + two), "by id: whatever their order, and one band for one that is there twice"
    assert edit.uniforms({}, None, True)["lutCount"] == 0
    assert edit.look_of({"luts": two})["luts"] == two and "luts" not in edit.look_of({"luts": two}, ["light"]), "LUTs are of a look's colors"
    # The picture they are read from: each one's slices in a band, one under the other.
    whole = lut.stacked([lut.identity(3), 1 - lut.identity(5)])
    assert whole.shape == (8, 25, 4) and whole[1, 2 * 3 + 1].tolist() == [0.5, 0.5, 1.0, 1.0] and whole[3 + 1, 3 * 5 + 2].tolist() == [0.5, 0.75, 0.25, 1.0]
    assert lut.address(two) == "0a1b2c.9-0a1b2c.33-ff.17" and lut.asked("0a1b2c.33-ff.17") == [("0a1b2c", 33), ("ff", 17)]
    for bad in ("", "a", "a.1", "../x.9", "a.9-", "a.b", "a.99"):
        try:
            lut.asked(bad)
            raise AssertionError(f"an address {bad!r} must be refused")
        except ValueError:
            pass
    assert lut.title('# x\nTITLE "Warm print"\nLUT_3D_SIZE 2') == "Warm print" and lut.title("LUT_3D_SIZE 2") == ""
    # Told by its colors: the same table in a file written otherwise is the same LUT.
    table = 0.2 + 0.6 * lut.identity(3)
    other = "# from another program\r\n" + lut.cube(table, "Another name").replace("\n", "\r\n").replace("0.500000", "0.5  ")
    assert other != lut.cube(table, "x") and lut.fingerprint(lut.parse(other)) == lut.fingerprint(lut.parse(lut.cube(table, "x")))
    assert lut.fingerprint(table) != lut.fingerprint(table + np.float32(0.001)) and lut.fingerprint(lut.identity(3)) != lut.fingerprint(lut.identity(4))
    assert len(lut.fingerprint(table)) == 20 and lut.fingerprint(table).isalnum(), "it is what the file is kept under"


def check_clf() -> None:
    """A .clf file (photone/lut.py, `_clf`): one 3D table read as a .cube's is, anything more
    refused with what it holds."""
    from photone import lut
    odd = np.random.default_rng(3).random((3, 3, 3, 3)).astype(np.float32)  # (no symmetry to hide a turned axis)

    def clf(table, depth="32f", scale=1.0, more="", head='<?xml version="1.0" encoding="UTF-8"?>\n'):
        rows = "\n".join(" ".join(f"{v:.9g}" for v in row) for row in table.transpose(2, 1, 0, 3).reshape(-1, 3) * scale)
        return (f'{head}<ProcessList xmlns="urn:AMPAS:CLF:v3.0" id="x" compCLFversion="3.0"><Description>a look</Description>{more}'
                f'<LUT3D inBitDepth="32f" outBitDepth="{depth}"><Array dim="3 3 3 3">\n{rows}\n</Array></LUT3D></ProcessList>')
    assert np.array_equal(lut.parse(clf(odd)), odd), "blue runs fastest in a CLF"
    tenth = np.round(odd * 1023) / 1023
    assert np.allclose(lut.parse(chr(0xFEFF) + "\n" + clf(tenth, "10i", 1023)), tenth, atol=1e-6), "10 bits run up to 1023 (after a byte order mark)"
    matrix = '<Matrix inBitDepth="32f" outBitDepth="32f"><Array dim="3 3">1 0 0 0 1 0 0 0 1</Array></Matrix>'
    for bad, why in ((clf(odd, more=matrix), "does Matrix, LUT3D"), (clf(odd).replace('dim="3 3 3 3"', 'dim="3 3 4 3"'), "N N N 3"),
                     (clf(odd, "9i"), "outBitDepth '9i'"), (clf(odd).replace("</ProcessList>", ""), "not XML"),
                     ('<!DOCTYPE x [<!ENTITY a "aaaa">]>' + clf(odd, head=""), "DOCTYPE"), ("<cube/>", "no CLF"),
                     (clf(odd).replace("\n</Array>", " 1 2 3\n</Array>"), "28 colors where 27"), (clf(odd).replace("0.", "x.", 1), "no number")):
        try:
            lut.parse(bad)
            raise AssertionError(f"a CLF that cannot be used was read: {why}")
        except ValueError as said:
            assert why in str(said), f"{why!r} is not in {str(said)!r}"


def check_detail() -> None:
    """Sharpening and noise reduction (photone/edit.py): their settings, and the numbers the
    shader is handed, which go by the size of the stored picture."""
    rest = edit.uniforms({}, None, True, shape=1.5, wide=6000)
    assert rest["sharpen"] == 0.2 and abs(rest["denoise"] - 0.1 ** 1.5 * 0.07) < 1e-12 and rest["denoiseColor"] == 0.025, "what a photo starts with: the owner's"
    none = edit.uniforms({"sharpen": 0.0, "denoise": 0.0, "denoiseColor": 0.0}, None, True, shape=1.5, wide=6000)
    assert none["sharpen"] == 0 and none["denoise"] == 0 and none["denoiseColor"] == 0 and edit.clean({"sharpen": 0, "denoise": 10}) == {"sharpen": 0.0}, "0 is off, and is kept"
    assert rest["detailX"] == 1 / 6000 and abs(rest["detailY"] - 1 / 4000) < 1e-12, "a pixel of the picture, as a share of it"
    u = edit.uniforms({"sharpen": 80.0, "sharpenRadius": 2.0, "sharpenMask": 50.0, "denoise": 100.0, "denoiseColor": 40.0}, None, True, shape=1.5, wide=6000)
    assert u["sharpen"] == 0.8 and u["sharpenRadius"] == 2 and u["sharpenMask"] == 0.025 and abs(u["denoise"] - 0.07) < 1e-12 and abs(u["denoiseColor"] - (0.025 + 30 / 90 * 0.035)) < 1e-12 and abs(u["denoiseFar"] - 1 / 3) < 1e-12
    assert 0.02 < edit.uniforms({"denoise": 50.0}, None, True, wide=6000)["denoise"] < 0.03, "half way is a third of all of it"
    low = edit.uniforms({"denoiseColor": 10.0}, None, True, wide=6000)
    assert abs(low["denoiseColor"] - 0.025) < 1e-12 and low["denoiseFar"] == 0, "at 10, what a photo starts with, the color is taken from no further than it was"
    assert edit.uniforms({"denoiseColor": 100.0}, None, True, wide=6000)["denoiseFar"] == 1
    blind = edit.uniforms({"sharpen": 80.0, "denoise": 100.0, "denoiseColor": 40.0}, None, True)
    assert blind["sharpen"] == 0 and blind["denoise"] == 0 and blind["denoiseColor"] == 0 and blind["detailX"] == 0, "without the picture's size: none of it"
    assert edit.clean({"sharpen": 500, "sharpenRadius": 0.1, "denoise": -3, "denoiseColor": "much"}) == {"sharpen": 150.0, "sharpenRadius": 0.5, "denoise": 0.0}
    assert edit.look_of({"sharpen": 40.0, "denoise": 20.0}, ["detail"]) == {"sharpen": 40.0, "denoise": 20.0} and "sharpen" not in edit.look_of({"sharpen": 40.0}, ["light"])


def check_lens() -> None:
    """Lens corrections (photone/lens.py): the camera's figures read out of a raw file's
    header as untrusted, and the tables the shader takes."""
    from photone import lens

    def tiff(tags: dict) -> bytes:  # a header whose picture's directory (0x14a) holds those tags of 16-bit numbers
        entries, data, at = b"", b"", 8 + 2 + 12 + 4 + 2 + 12 * len(tags) + 4
        for tag, values in tags.items():
            packed = struct.pack(f"<{len(values)}h", *values)
            entries += struct.pack("<HHI", tag, 8, len(values)) + (packed.ljust(4, b"\0") if len(packed) <= 4 else struct.pack("<I", at + len(data)))
            data += packed if len(packed) > 4 else b""
        first = struct.pack("<H", 1) + struct.pack("<HHII", 0x14a, 4, 1, 26) + struct.pack("<I", 0)
        return b"II*\0" + struct.pack("<I", 8) + first + struct.pack("<H", len(tags)) + entries + struct.pack("<I", 0) + data

    far, dim = list(range(0, 800, 50)), [0, 0, *range(512, 7680, 512)]
    with tempfile.TemporaryDirectory() as folder:
        file = Path(folder) / "a.arw"
        file.write_bytes(tiff({0x7037: [16, *far], 0x7032: [16, *dim], 0x7035: [32, *[256] * 16, *[-256] * 16]}))
        found = lens.figures(file)
        assert found == {"distortion": far, "vignetting": dim, "red": [256] * 16, "blue": [-256] * 16}, found
        file.write_bytes(tiff({0x7037: [5, 1, 2, 3, 4, 5]}))
        assert lens.figures(file) == {"distortion": [1, 2, 3, 4, 5]}, "fewer figures, and only one of the three"
        for bad in (tiff({0x7037: [40, *far]}), tiff({0x7037: [16, 1, 2]}), tiff({0x7035: [7, 1, 2, 3, 4, 5, 6, 7]}),
                    tiff({0x7037: [16, *far]})[:60], b"II*\0", b"not a tiff at all", b""):
            file.write_bytes(bad)
            assert lens.figures(file) is None, "figures that are not what they say are none"
        assert lens.figures(Path(folder) / "none.arw") is None
    none = lens.tables(found, False, False, False)
    assert none["lensOn"] == 0 and none["lensGreen"] == [1.0] * 16 == none["lensLight"] and lens.tables(None, True, True, True)["lensOn"] == 0
    bent = lens.tables(found, True, False, False)
    assert bent["lensOn"] == 1 and bent["lensGreen"][-1] == 1 and bent["lensGreen"][0] < 0.96 and bent["lensRed"] == bent["lensGreen"], "the corner shows the corner; the middle is drawn larger"
    assert all(a <= b for a, b in pairwise(bent["lensGreen"])) and bent["lensLight"] == [1.0] * 16
    assert abs(bent["lensGreen"][8] - (1 + lens._at(far, 8 / 15) * 2 ** -14) / (1 + lens._at(far, 1.0) * 2 ** -14)) < 1e-12
    assert lens._at(far, 0.0) == far[0] and lens._at(far, 0.5 / 15) == far[0] and lens._at(far, 1.5 / 15) == far[1] and lens._at(far, 1.0) == (far[14] + far[15]) / 2
    fringed = lens.tables(found, False, True, False)
    assert fringed["lensGreen"] == [1.0] * 16 and abs(fringed["lensRed"][5] - (1 + 256 * 2 ** -21)) < 1e-12 and fringed["lensBlue"][5] < 1
    lit = lens.tables(found, False, False, True)
    assert lit["lensLight"][0] == 1 and abs(lit["lensLight"][-1] - 2 ** (2 ** (lens._at(dim, 1.0) * 2 ** -13 - 1) - 0.5)) < 1e-12 and lit["lensLight"][-1] > 1.3
    u = edit.uniforms({**catalog.LENS_ON, "lensColor": False}, {**edit.FINISHED, "lens": found}, False, shape=1.5)
    assert u["lensOn"] == 1 and u["lensShape"] == 1.5 and u["lensGreen"] == bent["lensGreen"] == u["lensRed"] and u["lensLight"] == lit["lensLight"]
    whole = edit.uniforms({}, {**edit.FINISHED, "lens": found}, False, shape=1.5)
    assert whole["lensOn"] == 0, "all three are off until switched on"
    whole = edit.uniforms(catalog.LENS_ON, {**edit.FINISHED, "lens": found}, False, shape=1.5)
    assert whole["lensOn"] == 1 and whole["lensRed"] != whole["lensGreen"] == bent["lensGreen"]
    assert edit.uniforms({}, None, True)["lensOn"] == 0, "no figures: nothing is done, whatever is switched on"
    # Defringe, for any file: a share of 1, and nothing of it without the picture's size.
    assert edit.uniforms({"defringe": 40.0}, None, True, shape=1.5, wide=6000)["defringe"] == 0.4
    assert edit.uniforms({"defringe": 40.0}, None, True)["defringe"] == 0, "without the picture's size: no defringe"
    assert set(edit.TOOLS["lens"]) == {"lensDistortion", "lensColor", "lensVignetting", "defringe"}
    assert edit.clean({"defringe": 30, "caRed": 50.0, "defringePurple": 40.0}) == {"defringe": 30.0}, "the sliders there were before are not known any more"
    assert edit.clean({"lensDistortion": True, "lensColor": "yes", "lensVignetting": False}) == {"lensDistortion": True}


def check_mark() -> None:
    """A watermark (photone/export.py): its settings read as untrusted, and where it goes."""
    assert export.valid_mark(None) == export.MARK and export.valid_mark({"kind": "sticker", "place": True, "size": "big"}) == export.MARK
    got = export.valid_mark({"kind": "text", "text": "x" * 500, "place": 11, "size": 0.2, "strength": 400, "more": 1})
    assert got == {"kind": "text", "text": "x" * 120, "file": "", "place": 8, "size": 1.0, "strength": 100.0}
    assert export.mark_edge(6000, 4000) == 120 == export.mark_edge(4000, 6000), "3 % of the shorter side"
    at = [export.mark_at(place, 6000, 4000, 600, 200) for place in range(9)]
    assert at[0] == (120, 120) and at[4] == (2700, 1900) and at[8] == (5280, 3680) and at[2] == (5280, 120) and at[6] == (120, 3680)
    assert export.mark_base({**export.MARK, "kind": "none", "text": "x"}) is None
    assert export.mark_base({**export.MARK, "kind": "text", "text": "   "}) is None
    assert export.mark_base({**export.MARK, "kind": "picture", "file": "/nowhere/at/all.png"}) is None
    assert export.mark_size({**export.MARK, "size": 10.0}, (800, 200), 6000, 4000) == (1600, 400), "as high as 10 % of the shorter side, its shape kept"
    assert export.mark_size({**export.MARK, "size": 30.0}, (4000, 200), 600, 400) == (576, 29), "no wider than the picture less its edges: then lower too"


def check_colors() -> None:
    """Saturation, the chroma curve and the color EQ (photone/edit.py): what each makes of a
    color, and all of them with the pins, in the shader's order."""
    green = [0.6, -0.1, 0.1]  # hue 135 degrees, chroma 0.141
    assert np.allclose(edit.colors_at({}, green), green), "nothing set: nothing done"
    assert np.allclose(edit.colors_at({"saturation": 150.0}, green), [0.6, -0.15, 0.15])
    assert np.allclose(edit.colors_at({"saturation": 0.0}, green), [0.6, 0, 0]), "no saturation: gray"
    assert np.allclose(edit.colors_at({"saturation": 150.0}, [0.5, 0, 0]), [0.5, 0, 0]), "gray stays gray"
    lift = edit.valid("chroma", [[0, 0], [0.25, 0.5], [1, 1]])  # the muted made more vivid
    assert abs(edit.chroma_at(lift, 0.1) - 0.2) < 1e-9 and abs(edit.chroma_at(lift, 0.4) - 0.4) < 1e-9
    assert abs(edit.chroma_at(lift, 0.5) - 0.5) < 1e-9, "beyond the curve's end: on at the color's own pace"
    assert abs(edit.chroma_at(edit.STRAIGHT, 0.23) - 0.23) < 1e-12
    vivid = edit.colors_at({"chroma": lift}, [0.6, 0.1, 0.0])
    assert np.allclose(vivid, [0.6, 0.2, 0.0]), "the chroma curve keeps the hue"
    eq = edit.valid("eq", [{"hue": 135, "gain": 150, "width": 40}, {"hue": 495, "gain": 80, "width": 500}])
    assert eq[1] == {"hue": 135.0, "gain": 80.0, "width": 120.0}, "kept in range, the hue once round"
    eq = eq[:1]
    assert edit.eq_at(eq, 135) == 1.5 and abs(edit.eq_at(eq, 155) - 1.25) < 1e-9, "in full at its hue, half at half the width"
    assert 1 < edit.eq_at(eq, 165) < 1.25 and edit.eq_at(eq, 175) == 1 and edit.eq_at(eq, 315) == 1
    steps = [edit.eq_at(eq, 135 + n / 10) for n in range(-450, 451)]
    assert max(abs(b - a) for a, b in zip(steps, steps[1:], strict=False)) < 0.003, "a bell: no step anywhere along it"
    assert edit.eq_at([{"hue": 350.0, "gain": 0.0, "width": 40.0}], 350) == 0
    assert abs(edit.eq_at([{"hue": 350.0, "gain": 0.0, "width": 40.0}], 10) - 0.5) < 1e-9, "round the end of the circle"
    assert edit.eq_at(eq * 2, 135) == 2.0 and edit.eq_at([{"hue": 0.0, "gain": 0.0, "width": 40.0}] * 2, 0) == 0
    assert np.allclose(edit.colors_at({"eq": eq}, green), [0.6, -0.15, 0.15])
    for bad in ("x", [{"hue": 1}], [{"hue": 1, "gain": "much", "width": 30}], [eq[0]] * 9):
        assert edit.clean({"eq": bad}) == {}, f"an EQ {bad!r} must be refused"
    # All of it: the pin first, then saturation (not on what a pin protects), curve, EQ.
    pins = [{"from": [-0.1, 0.1], "to": [0.0, 0.1], "range": 0.05, "feather": 0}]
    both = edit.colors_at({"pins": edit.valid("pins", pins), "saturation": 150.0, "eq": eq}, green)
    assert np.allclose(both, [0.6, 0.0, 0.15]), f"moved to hue 90, where the EQ no longer reaches: {both}"
    guard = edit.valid("pins", [{"from": [-0.1, 0.1], "range": 0.05, "feather": 0, "mode": "protect"}])
    assert np.allclose(edit.colors_at({"pins": guard, "saturation": 150.0}, green), green), "protected from saturation"
    u = edit.uniforms({"saturation": 150.0, "chroma": lift, "eq": eq}, None, True)
    assert u["look"] == 1 and u["saturation"] == 1.5 and u["chromaOn"] == 1 and u["eqCount"] == 1
    assert u["chromaX"][:3] == [0.0, 0.25, 1.0] and u["eqA0"][:4] == [135.0, 0.5, 40.0, 0.0] and len(u["eqA1"]) == 16
    rest = edit.uniforms({}, None, True)
    assert rest["look"] == 0 and rest["saturation"] == 1 and rest["chromaOn"] == 0 and rest["eqCount"] == 0
    assert edit.uniforms({"pins": edit.valid("pins", pins)}, None, True)["look"] == 1


def check_masks() -> None:
    """Masks (photone/edit.py): how much of one a place of the picture is in, a pin kept to
    one, what the shader is given, and what deleting one does to the pins."""
    shape = 1.5  # a picture 3 wide and 2 high
    oval = edit.valid("masks", [{"kind": "radial", "at": [0.5, 0.5], "to": [0.7, 0.5], "ratio": 0.5,
                                 "feather": 0.5}])[0]
    assert oval["invert"] is False and oval["ratio"] == 0.5
    at = lambda mask, x, y: edit.mask_weight(mask, [x, y], shape)  # noqa: E731
    assert at(oval, 0.5, 0.5) == 1 and at(oval, 0.59, 0.5) == 1, "full within the inner half"
    assert abs(at(oval, 0.65, 0.5) - 0.5) < 1e-9 and at(oval, 0.7, 0.5) == 0 and at(oval, 0.9, 0.5) == 0
    # Its other axis is half as long, in pixels: 0.1 of the width is 0.15 of the height.
    assert abs(at(oval, 0.5, 0.5 + 0.1125) - 0.5) < 1e-9 and at(oval, 0.5, 0.66) == 0
    assert at({**oval, "invert": True}, 0.5, 0.5) == 0 and at({**oval, "invert": True}, 0.9, 0.5) == 1
    sky = edit.valid("masks", [{"kind": "linear", "at": [0.5, 0.4], "to": [0.5, 0.6]}])[0]
    assert at(sky, 0.1, 0.2) == 1 and at(sky, 0.9, 0.1) == 1 and at(sky, 0.3, 0.4) == 0.5
    assert at(sky, 0.5, 0.6) == 0 and at(sky, 0.5, 0.9) == 0 and 0 < at(sky, 0.5, 0.5) < 0.5
    # What the shader is given is the same thing, worked out as far as can be.
    tilted = {**oval, "to": [0.62, 0.71], "ratio": 2.2, "feather": 0.3, "invert": True}
    numbers = edit.masks_uniforms([oval, sky, tilted], shape)
    for index, mask in enumerate((oval, sky, tilted)):
        a, b = numbers["masksA0"][index * 4:][:4], numbers["masksB0"][index * 4:][:4]
        for x, y in np.random.default_rng(3).random((40, 2)):
            along, across = x * a[0] + y * a[1] + a[2], x * b[0] + y * b[1] + b[2]
            far = (along + 1) / 2 if a[3] % 2 else (math.hypot(along, across) - 1 + b[3]) / b[3]
            far = min(1.0, max(0.0, far))
            weight = 1 - far * far * (3 - 2 * far)
            weight = 1 - weight if a[3] > 1.5 else weight
            assert abs(weight - at(mask, x, y)) < 1e-9, f"mask {index} at {x:.2f}, {y:.2f}"
    spare = numbers["masksA0"][12:], numbers["masksB1"][:4]
    assert spare == ([0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]), "a slot without a mask covers nothing"
    # A pin kept to a mask works only there.
    green = [0.6, -0.1, 0.1]
    pins = edit.valid("pins", [{"from": [-0.1, 0.1], "to": [0.0, 0.1], "feather": 0, "mask": 1},
                               {"from": [-0.1, 0.1], "light": 0.1, "feather": 0}])
    assert pins[0]["mask"] == 1 and pins[1]["mask"] == 0
    look = {"pins": pins, "masks": [oval]}
    assert np.allclose(edit.colors_at(look, green, [0.5, 0.5], shape), [0.7, 0.0, 0.1]), "inside: both pins"
    assert np.allclose(edit.colors_at(look, green, [0.9, 0.5], shape), [0.7, -0.1, 0.1]), "outside: one"
    assert np.allclose(edit.colors_at(look, green, [0.65, 0.5], shape), [0.7, -0.05, 0.1]), "half way"
    assert np.allclose(edit.colors_at({"pins": pins}, green, [0.5, 0.5], shape), [0.7, -0.1, 0.1]), \
        "a pin whose mask is gone does nothing"
    u = edit.uniforms(look, None, True, shape=shape)
    assert u["pinsC0"][3] == 1.0 and u["pinsC0"][7] == 0.0 and u["look"] == 1
    # Deleting a mask takes its pins along; the pins of later masks follow theirs down.
    three = {"masks": [oval, sky, tilted], "pins": edit.valid("pins", [
        {"from": [0.1, 0], "mask": 1}, {"from": [0.2, 0], "mask": 2}, {"from": [0.3, 0], "mask": 3},
        {"from": [0.4, 0]}])}
    left = edit.without_mask(three, 2)
    assert left["masks"] == [oval, tilted]
    assert [(pin["from"][0], pin["mask"]) for pin in left["pins"]] == [(0.1, 1), (0.3, 2), (0.4, 0)]
    for bad in ("x", [{"kind": "round", "at": [0, 0], "to": [1, 1]}], [{"kind": "radial", "at": [0, 0]}],
                [{"kind": "radial", "at": [0.5, 0.5], "to": [0.5, 0.5]}], [oval] * 9,
                [{"kind": "linear", "at": [0, "x"], "to": [1, 1]}]):
        assert edit.clean({"masks": bad}) == {}, f"masks {bad!r} must be refused"
    for bad in (9, -1, 1.5, True, "1"):
        assert edit.clean({"pins": [{"from": [0.1, 0], "mask": bad}]}) == {}, f"a pin's mask {bad!r}"
    # Where a mask is on show, and back: the frame's own 0..1 (edit.unframed).
    matrix = edit.frame({"turns": 1, "angle": 4.0, "crop": [0.1, 0.2, 0.8, 0.9]}, 3000, 2000)["matrix"]
    back = edit.unframed(matrix, *edit.framed(matrix, 0.3, 0.8))
    # (A photo whose picture has no size yet has no frame: asked all the same, it must not fail.)
    assert edit.unframed(edit.frame({}, 0, 0)["matrix"], 0.3, 0.8) == (0.3, 0.8)
    assert np.allclose(back, [0.3, 0.8]), back


def check_frame() -> None:
    """The photo's frame (edit.frame): turns, straightening, crop, as the place in the stored
    picture of every point of the frame."""
    def places(values: dict, size=(600, 400), **more) -> tuple:
        made = edit.frame(values, *size, **more)
        return made, [tuple(round(v, 4) for v in edit.framed(made["matrix"], *p)) for p in ((0, 0), (1, 0), (1, 1), (0, 1))]

    made, corners = places({})
    assert made["size"] == (600, 400) and corners == [(0, 0), (1, 0), (1, 1), (0, 1)], "nothing set: the whole picture"
    made, corners = places({"turns": 1.0})
    assert made["size"] == (400, 600) and corners == [(0, 1), (0, 0), (1, 0), (1, 1)], "a turn to the right: the top left was the bottom left"
    assert places({"turns": 2.0})[1] == [(1, 1), (0, 1), (0, 0), (1, 0)] and places({"turns": 3.0})[0]["size"] == (400, 600)
    made, corners = places({"crop": [0.25, 0.5, 0.75, 1.0]})
    assert made["size"] == (300, 200) and corners == [(0.25, 0.5), (0.75, 0.5), (0.75, 1.0), (0.25, 1.0)]
    made, corners = places({"crop": [0.0, 0.0, 0.5, 1.0], "turns": 1.0})  # the left half of the turned picture
    assert made["size"] == (200, 600) and corners == [(0, 1), (0, 0.5), (1, 0.5), (1, 1)]
    for angle in (-15.0, -3.0, 4.0, 15.0):
        for crop in ([0.0, 0.0, 1.0, 1.0], [0.6, 0.1, 1.0, 0.5], [0.0, 0.7, 0.3, 1.0]):
            made, corners = places({"angle": angle, "crop": crop})
            assert all(-1e-6 <= v <= 1 + 1e-6 for corner in corners for v in corner), f"the frame leaves the picture: {angle}, {crop}: {corners}"
            wide, high = (crop[2] - crop[0]) * 600, (crop[3] - crop[1]) * 400
            assert 0.3 < made["size"][0] / wide < 1 and abs(made["size"][0] / made["size"][1] - wide / high) < 0.02, "smaller, and of the same shape"
            across = np.subtract(corners[1], corners[0]) * (600, 400)  # the frame's top edge in the picture
            assert abs(np.degrees(np.arctan2(across[1], across[0])) + angle) < 0.1, "turned by the angle, the other way"
    middle = edit.framed(edit.frame({"angle": 9.0}, 600, 400)["matrix"], 0.5, 0.5)
    assert np.allclose(middle, (0.5, 0.5)), "straightened about its middle"
    assert edit.frame({}, 0, 0)["size"] == (1, 1)
    # What the Crop tool shows: all of the picture, turned by the angle under a level frame.
    loose = places({"turns": 1.0, "crop": [0.2, 0.2, 0.8, 0.8]}, loose=True)
    assert loose[1] == places({"turns": 1.0})[1] and loose[0]["size"] == (400, 600), "no angle: the turns only"
    tilted = {"angle": 9.0, "crop": [0.3, 0.2, 0.9, 0.7]}
    made = edit.frame(tilted, 600, 400, loose=True)
    lean = np.radians(9.0)
    want = (round(600 * np.cos(lean) + 400 * np.sin(lean)), round(600 * np.sin(lean) + 400 * np.cos(lean)))
    assert made["size"] == want, f"the view just holds the turned picture: {made['size']}, not {want}"
    seen = np.array([edit.unframed(made["matrix"], *corner) for corner in ((0, 0), (1, 0), (1, 1), (0, 1))])
    assert np.allclose(seen.min(axis=0), 0) and np.allclose(seen.max(axis=0), 1), "the picture's corners touch the view's sides"
    assert np.allclose(edit.framed(made["matrix"], 0.5, 0.5), (0.5, 0.5)), "turned about the picture's middle"
    # The picture turns to the right under the frame: its top edge runs downhill to the right.
    assert seen[1][1] > seen[0][1] + 0.05, f"an angle to the right turns the picture to the left: {seen}"
    left, top, right, bottom = edit.frame_in_view(tilted, 600, 400)
    tight = edit.frame(tilted, 600, 400)
    ends = [edit.unframed(made["matrix"], *edit.framed(tight["matrix"], *p)) for p in ((1, 0), (0, 1))]
    assert np.allclose(ends, [(right, top), (left, bottom)]), "the frame lies level in that view"
    assert abs((right - left) * want[0] - tight["size"][0]) < 1 and abs((bottom - top) * want[1] - tight["size"][1]) < 1

    def moved(start, wanted, **how) -> list:  # the frame in the view once it is dragged there
        crop = edit.valid("crop", edit.crop_for(tilted, (600, 400), start, wanted, **how))
        return edit.frame_in_view({**tilted, "crop": crop}, 600, 400)

    start = [left, top, right, bottom]
    inner = [left + 0.05, top + 0.03, right - 0.02, bottom - 0.06]
    assert np.allclose(moved(start, inner), inner, atol=1e-4), f"a frame dragged to a place inside the picture is there: {moved(start, inner)}"
    far = moved(start, [left, top, 1.4, 1.3])  # its bottom right corner dragged out of the picture
    held = {**tilted, "crop": edit.valid("crop", edit.crop_for(tilted, (600, 400), start, [left, top, 1.4, 1.3]))}
    assert np.allclose(far[:2], start[:2], atol=1e-4) and far[2] > right and far[3] > bottom, f"it goes as far as it can, the other corner held: {far}"
    assert abs((far[2] - right) / (1.4 - right) - (far[3] - bottom) / (1.3 - bottom)) < 1e-3, "along the way it was dragged"
    at = [edit.framed(edit.frame(held, 600, 400)["matrix"], *p) for p in ((0, 0), (1, 0), (1, 1), (0, 1))]
    assert all(-1e-4 <= v <= 1 + 1e-4 for corner in at for v in corner), f"and stays inside the picture: {at}"
    assert min(min(v, 1 - v) for corner in at for v in corner) < 2e-3, "right up to the picture's edge"
    wide, high = right - left, bottom - top
    slid = moved(start, [left + 0.8, top - 0.05, right + 0.8, bottom - 0.05], together=False)
    assert abs(slid[1] - (top - 0.05)) < 1e-3 and slid[0] > left and abs(slid[2] - slid[0] - wide) < 1e-4 and abs(slid[3] - slid[1] - high) < 1e-4, \
        f"moved against the picture's side, it still goes the other way: {slid}"
    stuck = moved(start, [left + 0.8, top - 0.05, right + 0.8, bottom - 0.05])
    assert stuck[1] > slid[1] + 0.01, "in one go, it stops where it first meets the side"
    for bad in ([0, 0, 1], "frame", [0, 0, 1, float("nan")]):
        try:
            edit.crop_for(tilted, (600, 400), start, bad)
            raise AssertionError(f"a frame {bad!r} must be refused")
        except (ValueError, TypeError):
            pass
    assert edit.valid("turns", 2.4) == 2.0 and edit.valid("angle", 40) == 15.0 and edit.clean({"turns": 0, "angle": 0, "crop": [0, 0, 1, 1]}) == {}
    for bad in ([0, 0, 1], [0.5, 0, 0.51, 1], "x", [0, 0, 1, "1"]):
        assert edit.clean({"crop": bad}) == {}, f"a crop {bad!r} must be refused"
    assert edit.TOOLS["crop"] == edit.FRAME


def check_tone_settings() -> None:
    """Contrast and Brightness as the shader's numbers: nothing at rest, the named way otherwise."""
    rest = edit.uniforms({}, None, True)
    assert rest["contrast"] == 0 and rest["brightness"] == 0 and rest["mid"] == edit.MID
    assert edit.uniforms({}, None, False)["mid"] == edit.MID_FINISHED, "a finished picture's mid gray"
    more = edit.uniforms({"contrast": 100, "brightness": 100}, None, True)
    less = edit.uniforms({"contrast": -100, "brightness": -100}, None, True)
    assert more["contrast"] == 0.5 and abs(less["contrast"] + 0.35) < 1e-12, "stops from mid gray half as many again, 0.65 of them"
    assert more["brightness"] == -0.5 and less["brightness"] == 1, "the power on luminance, less one"
    assert rest["whites"] == 0 and rest["blacks"] == 0 and rest["vibrance"] == 0 and rest["look"] == 0 and rest["midtones"] == 0
    assert {"whites", "blacks", "midtones"} <= set(edit.TOOLS["tone"]) and "vibrance" in edit.TOOLS["saturation"]
    assert edit.uniforms({"midtones": 100}, None, True)["midtones"] == 1 and edit.clean({"midtones": -300}) == {"midtones": -100.0}
    assert edit.clean({"whites": 30, "blacks": -250, "vibrance": 1e9}) == {"whites": 30.0, "blacks": -100.0, "vibrance": 100.0}


def check_tone_even() -> None:
    """Highlights, Whites and Midtones at their ends squeeze evenly: no band of tones is left
    flat next to one left as it was, which looks harsh (the owner). How much of the contrast
    is kept, at the least and at the most."""
    def kept(wedge, values: dict, curve: bool = True) -> np.ndarray:
        plain, moved = (edit.uniforms({"exposure": 0.0, **v}, None, True) for v in ({}, values))
        was, now = (np.log(edit.tone_at(wedge, u if curve else {**u, "curveC": 0.0})) for u in (plain, moved))
        return np.diff(now) / np.diff(was)

    lights = np.geomspace(edit.MID * 1.05, 1.0, 400)
    assert kept(lights, {"whites": -100.0}).min() > 0.4, "Whites -100 flattens a band of the lights"
    assert kept(lights, {"highlights": -100.0}).min() > 0.35, "Highlights -100 flattens a band of the lights"
    assert kept(lights, {"highlights": -100.0, "whites": -100.0}).min() > 0.14
    # Shadows up lifts the shadows and leaves black: a stop and more four stops under mid
    # gray, nothing eight under; down, all of it there.
    def lift(stops: float, shadows: float) -> float:
        plain, moved = (edit.uniforms({"exposure": 0.0, **v}, None, True) for v in ({}, {"shadows": shadows}))
        return float(np.log2(edit.tone_at(edit.MID * 2 ** stops, {**moved, "curveC": 0.0}) / edit.tone_at(edit.MID * 2 ** stops, {**plain, "curveC": 0.0})))
    assert lift(-4, 100.0) > 1.2 and abs(lift(-8, 100.0)) < 1e-9 and abs(lift(-8, -100.0) + 1.5) < 1e-9
    # Contrast down flattens the middle and leaves the two ends: 0.65 of the contrast round
    # mid gray at -100, black and the sensor's white where they were.
    def flat(stops: float) -> float:
        plain, moved = (edit.uniforms({"exposure": 0.0, **v}, None, True) for v in ({}, {"contrast": -100.0}))
        return float(np.log2(edit.tone_at(edit.MID * 2 ** stops, {**moved, "curveC": 0.0}) / edit.tone_at(edit.MID * 2 ** stops, {**plain, "curveC": 0.0})))
    assert abs(flat(-1) - 0.35) < 1e-9 and abs(flat(0.5) + 0.175) < 1e-9 and abs(flat(-8)) < 1e-9 and abs(flat(3)) < 1e-9
    # Contrast up steepens the middle by half and leaves the two ends: black and the sensor's
    # white where they were, nothing between them flat (over half of the contrast at the least, deep in the shadows).
    def steep(stops: float) -> float:
        plain, moved = (edit.uniforms({"exposure": 0.0, **v}, None, True) for v in ({}, {"contrast": 100.0}))
        return float(np.log2(edit.tone_at(edit.MID * 2 ** stops, {**moved, "curveC": 0.0}) / edit.tone_at(edit.MID * 2 ** stops, {**plain, "curveC": 0.0})))
    assert abs(steep(3)) < 1e-9 and abs(steep(-8)) < 1e-9 and abs(steep(-1) + 0.5) < 1e-9 and 0.3 < steep(1) < 0.5
    k = kept(np.geomspace(edit.MID / 256, 1.0, 800), {"contrast": 100.0}, False)
    assert 0.55 < k.min() and abs(k.max() - 1.5) < 0.01, f"Contrast +100 keeps {k.min():.2f} to {k.max():.2f} of the contrast"
    wedge = np.geomspace(edit.MID / 16, edit.MID * 16, 800)  # (on scene light: no curve)
    for m in (100.0, -100.0):
        k = kept(wedge, {"midtones": m}, False)
        assert k.min() > 0.55 and k.max() < 1.45, f"Midtones {m} flattens or steepens a band by more than 0.45"


def check_tone_ends() -> None:
    """Whites and Blacks (edit.tone_at): the two ends of what is shown move, mid gray stays;
    Blacks cuts off at black or lifts it; nothing ever makes a lighter gray darker."""
    def shown(x: float, raw: bool = True, white: float = 1.0, **values) -> float:
        return float(edit.tone_at(x, edit.uniforms(values, None, raw, white)))

    for raw in (True, False):
        mid = edit.MID if raw else edit.MID_FINISHED
        rest = shown(mid, raw)
        for values in ({"whites": 100.0}, {"whites": -100.0}, {"blacks": 100.0}, {"blacks": -100.0}):
            assert abs(shown(mid, raw, **values) - rest) < 1e-9, f"{values} moves mid gray"
        assert abs(shown(1.0, raw, whites=-100.0) - 2 ** -0.8) < 1e-9, "Whites -100: white is 0.8 stops under white"
        assert shown(0.5, raw, whites=100.0) > 1.15 * shown(0.5, raw) and shown(0.5, raw, whites=-100.0) < 0.87 * shown(0.5, raw)
        assert shown(0.0, raw, blacks=100.0) == 0.03 and shown(0.0, raw, blacks=50.0) == 0.0075, "Blacks +: black is lifted"
        assert shown(0.004, raw, blacks=-100.0) == 0 and 0 < shown(0.004, raw, blacks=-10.0) < shown(0.004, raw), "Blacks -: cut off at black"
        assert abs(shown(0.6, raw, blacks=-100.0) - shown(0.6, raw)) < 1e-9, "Blacks leaves the lights alone"
        assert abs(shown(0.02, raw, whites=100.0) - shown(0.02, raw)) < 1e-9, "Whites leaves the darks alone"
    # Highlights: 1.8 stops at the sensor's white, which is what brings a bright sky down
    # (half the sensor's white is shown at level 238 of 255), and little at mid gray.
    low = edit.uniforms({"exposure": 0.0, "highlights": -100.0}, None, True)
    plain = edit.uniforms({"exposure": 0.0}, None, True)
    assert float(edit.tone_at(0.5, low)) < 0.7 * float(edit.tone_at(0.5, plain)), "Highlights -100 does not bring a bright sky down"
    assert float(edit.tone_at(edit.MID, low)) > 0.93 * float(edit.tone_at(edit.MID, plain)), "Highlights moves mid gray"
    # Midtones: a stop at mid gray, nearly that a third of a stop from it, nothing three and a half
    # stops away; on scene light, before the curve.
    for raw in (True, False):
        mid = edit.MID if raw else edit.MID_FINISHED
        plain = edit.uniforms({"exposure": 0.0}, None, raw)
        more, less = (edit.uniforms({"exposure": 0.0, "midtones": m}, None, raw) for m in (100.0, -100.0))
        assert abs(float(edit.tone_at(mid, more)) - float(edit.tone_at(2 * mid, plain))) < 1e-9, "Midtones +100: mid gray a stop up"
        assert abs(float(edit.tone_at(mid, less)) - float(edit.tone_at(mid / 2, plain))) < 1e-9, "Midtones -100: a stop down"
        assert float(edit.tone_at(mid * 1.3, more)) > float(edit.tone_at(2.5 * mid, plain)), "nearly the same a third of a stop from it"
        for far in (mid * 2 ** 3.5, mid * 2 ** -3.5, mid * 2 ** -5, 0.0):
            assert abs(float(edit.tone_at(far, more)) - float(edit.tone_at(far, plain))) < 1e-9, f"Midtones reaches {far}"
    ends = [(w, b) for w in (-100.0, 0.0, 100.0) for b in (-100.0, 0.0, 100.0)]
    others = ({}, {"contrast": -100.0, "shadows": 100.0, "highlights": -100.0}, {"contrast": 100.0, "shadows": -100.0, "highlights": 100.0},
              {"midtones": 100.0, "highlights": -100.0, "contrast": -100.0}, {"midtones": -100.0, "shadows": 100.0, "contrast": -100.0},
              {"midtones": 100.0}, {"midtones": -100.0, "highlights": 100.0, "shadows": -100.0})
    for (whites, blacks), more, white in ((e, o, w) for e in ends for o in others for w in (1.0, 2.03)):
        steps = [shown(x / 400, True, white, whites=whites, blacks=blacks, **more) for x in range(801)]
        assert all(a <= b + 1e-12 for a, b in zip(steps, steps[1:], strict=False)), f"a lighter gray shown darker: whites {whites}, blacks {blacks}, {more}, peak {white}"
    held, opened = (shown(1.0, True, white, whites=-100.0) for white in (1.0, 2.03))
    assert abs(shown(1.0, True, 2.03) - 2.03) < 1e-9 and held < opened < 2.03, "HDR opens up what Whites leaves"


def check_vibrance() -> None:
    """Vibrance (edit.vibrance_at, edit.colors_at): the muted colors gain, the vivid ones do
    not, skin gains least; below 0 the muted go gray; a more vivid color never ends up less so."""
    assert edit.vibrance_at(100.0, 0.0, 200.0) == 2 and edit.vibrance_at(100.0, 0.25, 200.0) == 1 and edit.vibrance_at(100.0, 0.4, 200.0) == 1
    assert 1.4 < edit.vibrance_at(100.0, 0.125, 200.0) < 1.6 and edit.vibrance_at(0.0, 0.05, 200.0) == 1
    assert abs(edit.vibrance_at(100.0, 0.0, 55.0) - 1.3) < 1e-9 and abs(edit.vibrance_at(100.0, 0.0, 70.0) - 1.3) < 1e-9, "skin gains three tenths"
    assert edit.vibrance_at(100.0, 0.0, 90.0) == 2 and edit.vibrance_at(100.0, 0.0, 20.0) == 2 and 1.3 < edit.vibrance_at(100.0, 0.0, 80.0) < 2
    assert edit.vibrance_at(-100.0, 0.0, 200.0) == 0 and edit.vibrance_at(-100.0, 0.3, 200.0) == 1
    for amount, hue in ((100.0, 200.0), (-100.0, 200.0), (100.0, 55.0), (37.0, 80.0)):
        after = [c / 1000 * edit.vibrance_at(amount, c / 1000, hue) for c in range(401)]
        assert all(a <= b + 1e-12 for a, b in zip(after, after[1:], strict=False)), f"vibrance {amount} at hue {hue} turns the order of colors round"
    muted, vivid = np.array([0.6, -0.03, -0.04]), np.array([0.6, -0.18, -0.24])  # two blues
    more = edit.colors_at({"vibrance": 100.0}, muted.copy())
    assert more[0] == 0.6 and 1.85 < more[1] / muted[1] < 1.95 and abs(more[1] / muted[1] - more[2] / muted[2]) < 1e-9, "more chroma, the same hue and lightness"
    assert np.allclose(edit.colors_at({"vibrance": 100.0}, vivid.copy()), vivid), "a vivid color is left as it is"
    both = edit.colors_at({"vibrance": 100.0, "saturation": 150.0}, muted.copy())
    assert both[1] / muted[1] > 1.5 * 1.5, "after Saturation, on what that left"
    guard = {"from": [-0.03, -0.04], "to": [-0.03, -0.04], "range": 0.03, "feather": 0, "mode": "protect"}
    assert np.allclose(edit.colors_at(edit.clean({"vibrance": 100.0, "pins": [guard]}), muted.copy()), muted), "a protecting pin keeps Vibrance off"
    assert edit.uniforms({"vibrance": 40.0}, None, True)["look"] == 1 and edit.uniforms({"vibrance": 40.0}, None, True)["vibrance"] == 0.4


def check_tone_curve() -> None:
    """Mid gray where the profile says, the sensor's white on the display's; in HDR the
    sensor's white on the display's peak, mid gray and below as in SDR, and all between rising."""
    def tone(x: float, u: dict) -> float:
        return float(edit.tone_at(x, u))

    for profile, (_, mid) in edit.PROFILES.items():
        u = edit.uniforms({"profile": profile}, None, True)
        assert abs(tone(edit.MID, u) - mid) < 1e-9 and abs(tone(1.0, u) - 1.0) < 1e-9, profile
        steps = [tone(x / 200, u) for x in range(201)]
        assert steps[0] == 0 and all(a < b for a, b in zip(steps, steps[1:], strict=False)), "rises all the way"
    standard, flat = (edit.uniforms({"profile": p}, None, True) for p in ("standard", "flat"))
    assert tone(0.03, flat) > tone(0.03, standard) and tone(0.4, flat) < tone(0.4, standard), "Flat is flatter"
    assert edit.uniforms({}, None, True)["hdrLift"] == 0, "SDR: the sensor's white is white"
    assert abs(edit.uniforms({}, None, True, 2.03)["hdrLift"] - 1.03) < 1e-9, "HDR: up to the peak"
    assert edit.uniforms({}, None, False, 2.03)["hdrLift"] == 0, "a finished picture is left alone"
    for profile in edit.PROFILES:
        sdr, hdr = (edit.uniforms({"profile": profile}, None, True, white) for white in (1.0, 2.03))
        assert abs(tone(1.0, hdr) - 2.03) < 1e-9, "HDR: the sensor's white is at the display's peak"
        # The whole picture climbs (the owner's choice of three, 2026-11-18): black stays, a
        # shadow gains a little, the middle more, white the whole headroom.
        assert tone(0.0, hdr) == tone(0.0, sdr), "HDR: black is black"
        lifts = [tone(x / 200, hdr) - tone(x / 200, sdr) for x in range(1, 201)]
        assert all(0 < a < b for a, b in zip(lifts, lifts[1:], strict=False)), "from black on, more the brighter"
        assert 1.0 < tone(0.01, hdr) / tone(0.01, sdr) < 1.2, "a deep shadow, a little"
        assert tone(edit.MID, hdr) > 1.25 * tone(edit.MID, sdr), "mid gray is lifted too: all of the picture is deeper"
        steps = [tone(x / 200, hdr) for x in range(401)]
        assert all(a < b for a, b in zip(steps, steps[1:], strict=False)), "rises all the way, past the sensor's white too"
    assert edit.uniforms({}, None, False)["curveC"] == 0, "and gets no curve"
    assert edit.uniforms({"exposure": 1}, None, False)["gain"] == 2.0


def check_white_balance() -> None:
    """Nothing at the defaults; warmer, cooler, greener, more magenta as named."""
    shot = edit.as_shot(CAMERA)
    assert 4000 < shot[0] < 7000 and abs(shot[1]) < 40, shot

    def gray(values: dict) -> np.ndarray:
        u = edit.uniforms(values, CAMERA, True)
        cam = np.minimum((np.array(u["toCam"]) @ np.full(3, 0.2)) * u["wbGain"][:3], u["wbGain"][3])
        return np.array(u["fromCam"]) @ cam

    assert np.allclose(gray({}), 0.2, atol=1e-9), "as shot: a gray stays exactly as decoded"
    assert np.allclose(gray({"temperature": shot[0], "tint": shot[1]}), 0.2, atol=1e-9)
    warm, cool = gray({"temperature": shot[0] + 1500}), gray({"temperature": shot[0] - 1500})
    assert warm[0] > warm[2] + 0.02 and cool[2] > cool[0] + 0.02, (warm, cool)
    pink, green = gray({"tint": shot[1] + 40}), gray({"tint": shot[1] - 40})
    assert pink[1] < min(pink[0], pink[2]) and green[1] > max(green[0], green[2]), (pink, green)
    blown = edit.uniforms({"temperature": shot[0] + 1500}, CAMERA, True)
    top = np.minimum(np.ones(3) * blown["wbGain"][:3], blown["wbGain"][3])
    assert np.allclose(top, top[0]), "what the sensor clipped stays neutral"
    for kelvin, tint in ((3200, 0), (5600, 20), (8000, -30)):  # the search finds what was put in
        gains = 1 / (np.array(CAMERA["cam_xyz"]) @ edit.white_xyz(kelvin, tint))
        found = edit.as_shot({"wb": (gains / gains[1]).tolist(), "cam_xyz": CAMERA["cam_xyz"]})
        assert abs(found[0] - kelvin) <= 20 and abs(found[1] - tint) <= 1, (kelvin, tint, found)
    # A finished picture has no camera, and its white moves all the same, from daylight.
    daylight = edit.as_shot(edit.FINISHED)
    assert abs(daylight[0] - 6500) <= 50 and abs(daylight[1]) < 20, daylight
    still = edit.uniforms({}, {**edit.FINISHED, "shot": daylight}, False)
    assert np.allclose(still["wbGain"][:3], 1, atol=1e-12) and still["wbGain"][3] > 1e6, "nothing at rest, no ceiling"
    # Pick neutral: the temperature and tint it finds make that color a gray.
    for camera in (CAMERA, edit.FINISHED):
        for color in ([0.8, 1.0, 1.3], [1.4, 1.0, 0.7], [1.0, 1.2, 1.0], [1.0, 0.85, 1.05]):
            rgb = edit.camera_matrix(camera) @ (np.array(color) * 0.2)
            kelvin, tint = edit.neutral_at(camera, rgb)
            to_cam, _, gain = edit.balance({"temperature": kelvin, "tint": tint}, camera)
            seen = (to_cam @ rgb) * gain[:3]
            assert seen.max() / seen.min() < 1.01, (color, kelvin, tint, seen)
    assert edit.neutral_at(CAMERA, [0.0, 0.0, 0.0]) is None, "black says nothing about the light"


def check_history() -> None:
    """A drag is one step; another setting or a commit starts the next."""
    h = edit.History({"shadows": 10.0})
    for value in (0.1, 0.2, 0.3):
        h.set("exposure", value)
    h.set("tint", 5)
    h.commit()
    h.set("tint", 9)
    assert h.edit == {"shadows": 10.0, "exposure": 0.3, "tint": 9.0} and h.at == 3
    assert h.step(-1) and h.edit["tint"] == 5.0 and h.step(-1) and "tint" not in h.edit
    assert h.step(-1) and h.edit == {"shadows": 10.0} and not h.step(-1), "back to the start"
    assert h.step(1) and h.edit["exposure"] == 0.3
    h.set("exposure", 0.0)
    assert h.edit == {"shadows": 10.0} and len(h.states) == 3, "a new change ends the redo"
    h.commit()
    h.set("shadows", 50)
    h.set("shadows", 10)
    h.commit()
    assert len(h.states) == 3, "there and back within one drag is no step"
    try:
        h.set("exposure", "much")
    except ValueError:
        pass
    else:
        raise AssertionError("a bad value must not enter the history")
    pad = edit.History()  # a pad moves two settings at once: still one step for the gesture
    for kelvin, tint in ((5200, 2), (5400, 5), (5600, 9)):
        pad.put({"temperature": kelvin, "tint": tint})
    assert pad.at == 1 and pad.edit == {"temperature": 5600.0, "tint": 9.0}
    pad.set("tint", 12)
    assert pad.at == 2, "one of the two alone is another gesture"
    try:
        pad.put({"tint": 1, "exposure": "x"})
    except ValueError:
        assert pad.edit["tint"] == 12.0, "one bad value: none of them is taken"


def photo_file(folder: str, name: str, seed: int) -> Path:
    """A file to stand for a photo: its content is its identity, the loaders here ignore it."""
    path = Path(folder) / name
    path.write_bytes(np.random.default_rng(seed).bytes(300_000))
    return path


def check_identify() -> None:
    """A photo is known by what it is: the same after a rename or a move, another as a copy."""
    with tempfile.TemporaryDirectory() as folder:
        one, two = photo_file(folder, "IMG_0001.ARW", 1), photo_file(folder, "IMG_0002.ARW", 2)
        key = catalog.identify(one)
        assert key and key.startswith("c1:") and key != catalog.identify(two)
        moved = one.rename(Path(folder) / "renamed.arw")
        (Path(folder) / "sub").mkdir()
        moved = moved.rename(Path(folder) / "sub" / "IMG_0002.ARW")
        os.utime(moved, (0, 0))
        assert catalog.identify(moved) == key, "renamed, moved, its dates changed: the same photo"
        copy = Path(folder) / "copy.arw"
        shutil.copy2(moved, copy)  # as a file manager copies: the dates are kept
        assert catalog.content(copy) == catalog.content(moved), "the same content"
        if catalog.born(moved):  # where the file system says when a file was made
            assert catalog.identify(copy) != key, "but a copy is a photo of its own"
        data = bytearray(moved.read_bytes())
        for at in (10, 150_000, 299_990):  # a change at the start, in the middle, at the end
            changed = bytearray(data)
            changed[at] ^= 1
            copy.write_bytes(changed)
            assert catalog.content(copy) != catalog.content(moved), at
        copy.write_bytes(data + b"x")
        assert catalog.content(copy) != catalog.content(moved), "one byte longer is another file"
        tiny = Path(folder) / "tiny.png"
        tiny.write_bytes(b"abc")
        assert catalog.identify(tiny), "a file shorter than a piece still has one"
        assert catalog.identify(Path(folder) / "none.arw") is None
        assert catalog.identify(Path(folder)) is None, "a folder is not a photo"


def check_catalog() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path, photo = Path(folder) / "deep" / "catalog.db", Path("/photos/a.arw")
        kept = catalog.Catalog(path)
        assert kept.load("c1:aa:1", photo) == {}
        kept.save("c1:aa:1", photo, {"exposure": 0.5, "profile": "flat"})
        kept.save("c1:bb:1", Path("/photos/b.arw"), {"tint": 4.0})
        kept.save(None, Path("/photos/gone.arw"), {"tint": 9.0})  # unreadable: nothing kept
        kept.close()
        again = catalog.Catalog(path)
        assert again.load("c1:aa:1", Path("/elsewhere/renamed.arw")) == {"exposure": 0.5, "profile": "flat"}, "found by what it is, not where"
        assert again.load(None, photo) == {}
        again.save("c1:aa:1", photo, {})
        assert again.load("c1:aa:1", photo) == {} and again.load("c1:bb:1", photo) == {"tint": 4.0}
        rows = again.db.execute("SELECT count(*) FROM edits").fetchone()[0]
        assert rows == 1, "an untouched photo keeps no row"
        # What is in the file is not trusted: nonsense is no edit, never an exception.
        with again.db:
            again.db.execute("INSERT INTO edits VALUES ('c1:cc:1', 'cc', '/c.arw', 'not json', 0, '{')")
            again.db.execute("INSERT INTO edits VALUES ('c1:dd:1', 'dd', '/d.arw', ?, 0, ?)",
                             ('{"exposure": 99, "evil": [1], "profile": 7}',
                              '[{"id": 1, "name": "ok", "made": 5, "edit": {"tint": 999, "x": 1}}, 7, '
                              '{"id": 1, "name": "same id", "made": 5}, {"id": 2, "name": " ", "made": 5}, '
                              '{"id": "3", "name": "n", "made": 5}]'))
        assert again.load("c1:cc:1", photo) == {} and again.checkpoints("c1:cc:1") == []
        assert again.load("c1:dd:1", photo) == {"exposure": 5.0}
        assert again.checkpoints("c1:dd:1") == [{"id": 1, "name": "ok", "made": 5.0, "edit": {"tint": 150.0}}], "only whole checkpoints, their edits cleaned"
        assert again.checkpoints("c1:none:1") == [] and again.checkpoints(None) == []
        again.save("c1:ee:1", photo, {}, [{"id": 1, "name": "start", "made": 1.0, "edit": {}}])
        assert again.checkpoints("c1:ee:1")[0]["name"] == "start", "a photo with only checkpoints keeps its row"
        again.save("c1:ee:1", photo, {}, [])
        assert again.db.execute("SELECT count(*) FROM edits WHERE key = 'c1:ee:1'").fetchone()[0] == 0
        with again.db:
            again.db.execute(f"PRAGMA user_version={catalog.VERSION + 1}")
        again.close()
        for bad in (path, Path(folder) / "deep"):  # from a newer Photone; not a file at all
            try:
                catalog.Catalog(bad)
            except catalog.Unusable as e:
                assert "will not be saved" in str(e)
            else:
                raise AssertionError(f"{bad} must not be used as a catalog")


def check_catalog_before_checkpoints() -> None:
    """A catalog made in the few hours before checkpoints gains the column, and keeps its edits."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "catalog.db"
        old = sqlite3.connect(path)
        with old:
            old.execute("CREATE TABLE edits (key TEXT PRIMARY KEY, content TEXT NOT NULL, path TEXT, "
                        "edit TEXT NOT NULL, saved REAL NOT NULL)")
            old.execute("INSERT INTO edits VALUES ('c1:aa:1', 'aa', '/a.arw', '{\"exposure\": 1.5}', 1)")
            old.execute("PRAGMA user_version=1")
        old.close()
        for _ in range(2):  # opening it again changes nothing more
            kept = catalog.Catalog(path)
            assert kept.load("c1:aa:1", Path("/a.arw")) == {"exposure": 1.5, **THEN}
            assert kept.checkpoints("c1:aa:1") == []
            kept.save("c1:aa:1", Path("/a.arw"), {"exposure": 1.5, **THEN}, [])
            kept.close()


def check_checkpoints() -> None:
    """Edits saved to come back to: which one the edit stands at, and when it is unsaved."""
    a, b = {"id": 1, "edit": {"exposure": 1.0}}, {"id": 2, "edit": {"exposure": 1.0}}
    c = {"id": 3, "edit": {}}
    assert edit.standing([], {}, None) == (None, False), "an untouched photo has nothing to lose"
    assert edit.standing([], {"tint": 3.0}, None) == (None, True), "an edit in no checkpoint has"
    assert edit.standing([a], {"exposure": 1.0}, None) == (1, False)
    assert edit.standing([a, b], {"exposure": 1.0}, 2) == (2, False), "of two alike, the one last used"
    assert edit.standing([a, b], {"exposure": 1.0}, 7) == (1, False)
    assert edit.standing([a], {"exposure": 1.2}, 1) == (None, True)
    assert edit.standing([a], {}, 1) == (None, True), "back to the defaults from a checkpoint is a change too"
    assert edit.standing([a, c], {}, 1) == (3, False), "unless a checkpoint holds exactly that"
    h = edit.History({"tint": 2.0})
    h.replace({"exposure": 1.0, "junk": 1})
    assert h.edit == {"exposure": 1.0} and h.at == 1 and h.step(-1) and h.edit == {"tint": 2.0}
    h.replace({"tint": 2.0})
    assert len(h.states) == 2, "going to the edit already there is no step"

    with tempfile.TemporaryDirectory() as folder:
        kept = catalog.Catalog(Path(folder) / "catalog.db")
        photo = photo_file(folder, "a.arw", 1)
        session = app.Session([photo], load=chart, catalog=kept)
        assert session.checkpoints == [] and not session.unsaved
        session.setEdit("exposure", 0.5)
        assert session.unsaved
        session.saveCheckpoint()
        session.setEdit("exposure", 1.5)
        session.saveCheckpoint()
        assert [(p["name"], p["state"]) for p in session.checkpoints] == [("Checkpoint 2", "here"), ("Checkpoint 1", "")], "newest first"
        session.setEdit("shadows", 20)
        assert session.unsaved and [p["state"] for p in session.checkpoints] == ["changed", ""]
        session.goToCheckpoint(1)
        assert session.history.edit == {"exposure": 0.5} and not session.unsaved
        assert [p["state"] for p in session.checkpoints] == ["", "here"]
        session.undo()
        assert session.history.edit == {"exposure": 1.5, "shadows": 20.0}, "going to one is one undo step"
        session.goToCheckpoint(99)
        assert session.history.edit["shadows"] == 20.0, "no such checkpoint: nothing happens"
        session.setEdit("exposure", 1.5)
        session.resetEdit("shadows")
        assert [p["state"] for p in session.checkpoints] == ["here", ""], "changed back by hand, it stands there again"

        again = app.Session([photo], load=chart, catalog=kept)  # another run
        assert [p["name"] for p in again.checkpoints] == ["Checkpoint 2", "Checkpoint 1"]
        assert again.checkpoints[0]["state"] == "here" and not again.unsaved
        again.deleteCheckpoint(2)
        again.saveCheckpoint()
        assert [p["name"] for p in again.checkpoints] == ["Checkpoint 2", "Checkpoint 1"], "numbers go on from the highest left"
        again.deleteCheckpoint(2)
        again.deleteCheckpoint(1)
        assert again.checkpoints == [] and again.unsaved, "the edit stays; it is in no checkpoint now"
        again.saveCheckpoint()
        moved = photo.rename(Path(folder) / "elsewhere.arw")
        assert len(app.Session([moved], load=chart, catalog=kept).checkpoints) == 1, "they follow the photo"
        if catalog.born(moved):
            copy = Path(folder) / "copy.arw"
            shutil.copy2(moved, copy)
            assert app.Session([copy], load=chart, catalog=kept).checkpoints == [], "a copy has none"
            moved.unlink()  # and now the copy is all there is: the photo, on another disk
            assert len(app.Session([copy], load=chart, catalog=kept).checkpoints) == 1
        kept.close()


def check_copies_and_moves() -> None:
    """The owner's rule: a renamed or moved photo keeps its edit, a copy is a photo of its own.
    And a move to another disk, which makes a new file, is told from a copy by the old file
    being gone."""
    with tempfile.TemporaryDirectory() as folder:
        if not catalog.born(Path(folder)):
            return  # this file system does not say when a file was made
        kept = catalog.Catalog(Path(folder) / "catalog.db")
        look = {"exposure": 1.0}
        original = photo_file(folder, "a.arw", 1)
        kept.save(catalog.identify(original), original, look)

        copy = Path(folder) / "a copy.arw"
        shutil.copy2(original, copy)
        assert kept.load(catalog.identify(copy), copy) == {}, "a copy starts without the edit"
        kept.save(catalog.identify(copy), copy, {"tint": 5.0})
        assert kept.load(catalog.identify(original), original) == look, "and has its own"
        assert kept.load(catalog.identify(copy), copy) == {"tint": 5.0}

        # To another disk: a new file there, the old one gone. Stood in for by copy and delete.
        (Path(folder) / "nas").mkdir()
        arrived = Path(folder) / "nas" / "a.arw"
        shutil.copy2(original, arrived)
        original.unlink()
        assert kept.load(catalog.identify(arrived), arrived) == look, "moved across: the edit came along"
        kept.save(catalog.identify(arrived), arrived, {"exposure": 2.0})
        renamed = arrived.rename(Path(folder) / "nas" / "renamed.arw")  # then renamed, then copied:
        assert kept.load(catalog.identify(renamed), renamed) == {"exposure": 2.0}
        arrived = renamed
        third = Path(folder) / "nas" / "another copy.arw"
        shutil.copy2(arrived, third)
        assert kept.load(catalog.identify(third), third) == {}, "a later copy gets no edit: not the new one, not the old one again"
        assert kept.load(catalog.identify(copy), copy) == {"tint": 5.0}, "the first copy is untouched by all this"
        kept.close()


def check_start() -> None:
    """Where a photo starts (edit.curve_of): Exposure at 0, as shot, for every photo; the tone
    curve at a raw file's own (`match`, made at decode), straight where there is none. In the
    session (the curve editor shows it once the file is decoded, reset is back there, straight
    is a choice that is kept, the Exposure card's A/B shows it straight), and in a thumbnail
    (the camera's preview as it is)."""
    own = [[0.0, 0.0], [0.25, 0.32], [0.6, 0.66], [1.0, 1.0]]
    camera = {**CAMERA, "match": own}
    assert edit.curve_of({}, camera) == own and edit.curve_of({"curve": edit.STRAIGHT}, camera) == edit.STRAIGHT
    assert edit.curve_of({}, CAMERA) == edit.curve_of({}, None) == edit.STRAIGHT, "no curve of its own: straight"
    at_rest = edit.uniforms({}, camera, True)
    assert at_rest["gain"] == 1 and at_rest["curveOn"] == 1 and at_rest == edit.uniforms({"curve": own}, camera, True)
    assert edit.uniforms({}, CAMERA, True)["curveOn"] == 0
    with tempfile.TemporaryDirectory() as folder:
        dark = Path(folder) / "dark.arw"
        dark.write_bytes(exif("SONY", "ILCE-7M4", "2026:07:04 13:35:44", 1, -2.0) + np.random.default_rng(5).bytes(200_000))
        plain = photo_file(folder, "plain.arw", 6)
        told = picture_file(Path(folder) / "told.jpg", exif("SONY", "ILCE-7M4", "2026:07:04 13:35:44", 1, 0.7))
        assert library.describe(dark)["bias"] == -2.0 and library.describe(told)["bias"] == 0.7
        assert library.describe(plain)["bias"] == 0
        wild = Path(folder) / "wild.arw"
        wild.write_bytes(exif("X", "Y", "2026:07:04 13:35:44", 1, 300.0))
        assert library.describe(wild)["bias"] == 0, "a compensation no camera has is not believed"

        gray = QImage(64, 48, QImage.Format.Format_RGB32)
        gray.fill(QColor(64, 64, 64))
        carried, library._preview = library._preview, lambda photo: gray
        try:
            made = QImage(str(library.thumbnail(dark, "c1:dark:1", Path(folder) / "cache")))
        finally:
            library._preview = carried
        assert 62 <= made.pixelColor(30, 20).green() <= 66, "an unedited raw file's thumbnail is its preview as it is"

        def some(path: Path, half: bool = True):  # the chart, as a raw file or as a finished picture
            meta, pixels = chart(path, half)
            return {**meta, "raw": path.suffix == ".arw", "match": {"dark.arw": own}.get(path.name)}, pixels

        session = app.Session([dark, plain, told], load=some)
        shown = []
        session.curveChanged.connect(lambda: shown.append(session.curve))
        session.image(session.source.removeprefix("image://photo/"))
        QCoreApplication.processEvents()
        assert shown and shown[-1] == own, "the curve editor is told when the photo's own curve is there"
        assert session.shot["curve"] == session.edit["curve"] == session.curve == own and session.history.edit == {}
        assert session.edit["exposure"] == 0 and session.uniforms["gain"] == 1 and session.unsavedKeys == []
        assert session.curveLine == edit.curve_line(own)
        assert session.localWhole(1, "curve") == edit.STRAIGHT and session.localLine(1, "curve") == edit.curve_line(edit.STRAIGHT), \
            "a mask's own curve starts straight, on top of the photo's own"
        assert session.comparable["tone"] and not session.adjusted["tone"], "at rest, a curve to compare and no change"
        session.setBypass("tone")
        assert session.uniforms["curveOn"] == 0, "the Exposure card's A/B: the curve straight, as shot"
        session.setBypass("")
        session.setEdit("curve", edit.STRAIGHT)
        assert session.history.edit == {"curve": edit.STRAIGHT} and session.curve == edit.STRAIGHT and "curve" in session.unsavedKeys
        session.setEdit("curve", own)
        assert session.history.edit == {}, "set to the photo's own curve: at rest again"
        session.resetTool("tone")
        assert session.history.edit == {} and session.curve == own, "reset is back to the photo's own curve"
        session.setEdit("exposure", 0.5)
        assert session.unsavedKeys == ["exposure"]
        session.setEdit("exposure", 0.0)
        assert session.history.edit == {} and session.unsavedKeys == [], "at 0 Exposure is at rest"
        for _ in range(2):  # a raw file with no curve of its own, then a finished picture: straight
            session.step(1)
            session.image(session.source.removeprefix("image://photo/"))
            QCoreApplication.processEvents()
            assert session.curve == edit.STRAIGHT and shown[-1] == edit.STRAIGHT and session.uniforms["curveOn"] == 0
            session.setEdit("curve", [[0.0, 0.0], [1.0, 1.0]])
            assert session.history.edit == {}, "straight, where that is the photo's own: no change"
            assert not session.comparable["tone"]


def check_light() -> None:
    """decode.light_of: how light the sensor's picture is, from its mosaic with the camera's
    white balance on; and library.forget_starts: thumbnails made under the old rule go, once."""
    class Sensor:  # what the raw decoder shows of a file, as far as light_of asks
        raw_pattern, color_desc = np.array([[0, 1], [3, 2]]), b"RGBG"
        camera_whitebalance, black_level_per_channel, white_level = [2048.0, 1024.0, 1536.0, 0.0], [512, 512, 512, 512], 16380

        def __init__(self, red: int, green: int, blue: int, size: tuple = (64, 96)) -> None:
            self.raw_image_visible = np.zeros(size, np.uint16) + 512
            self.raw_image_visible[0::2, 0::2] += red
            self.raw_image_visible[0::2, 1::2] += green
            self.raw_image_visible[1::2, 0::2] += green
            self.raw_image_visible[1::2, 1::2] += blue

    full = 16380 - 512
    middle, top = decode.light_of(Sensor(1000, 2000, 500))
    want = [2 * 1000 / full, 2000 / full, 1.5 * 500 / full]  # white balance on: red twice, blue one and a half times
    assert abs(top - max(want)) < 1e-9 and abs(middle - float(np.dot(decode.LUMA, want))) < 1e-9, (middle, top)
    middle, top = decode.light_of(Sensor(9000, 14000, 3000))
    assert top == 1.0 and middle > 0.9, "the highest value in the file is the sensor's white, and red, doubled, is cut off there"
    spot = Sensor(100, 200, 50, (1600, 1600))
    spot.raw_image_visible[0:80, 0:160] = 15000  # a lamp: half a hundredth of the picture
    assert decode.light_of(spot)[1] < 0.05, "a small light does not count as the picture's top"
    spot.raw_image_visible[0:320, 0:160] = 15000  # two hundredths: nor is that (a window)
    assert decode.light_of(spot)[1] < 0.05
    spot.raw_image_visible[0:1280, 0:160] = 15000  # eight hundredths: it is (a sky)
    middle, top = decode.light_of(spot)
    assert top == 1.0 and abs(middle - 0.0133) < 0.001, f"the middle is the picture's median, which a lamp does not move: {middle:.4f}"
    for odd in ({"raw_pattern": None}, {"color_desc": b"CMYG"}, {"camera_whitebalance": [0.0, 1024.0, 0.0, 0.0]}, {"raw_pattern": np.zeros((6, 6), int)}):
        other = Sensor(1000, 2000, 500)
        for name, value in odd.items():
            setattr(other, name, value)
        assert decode.light_of(other) is None, f"not a mosaic of red, green and blue ({odd}): no figures"
    with tempfile.TemporaryDirectory() as folder:
        cache = Path(folder) / "cache"
        photos = [{"key": "c1:a:1", "kind": "raw", "edited": False}, {"key": "c1:b:1", "kind": "raw", "edited": True},
                  {"key": "c1:c:1", "kind": "jpeg", "edited": False}]
        library.forget_starts(cache, [])
        assert cache.is_dir() and (cache / ".starts-5").exists(), "a new cache folder is marked: nothing in it is old"
        (cache / ".starts-5").unlink()
        files = [cache / f"{library.thumb_name(p['key'])}.jpg" for p in photos]
        for file in files:
            file.write_bytes(b"old")
        library.forget_starts(cache, photos)
        assert [f.exists() for f in files] == [False, True, True], "only the thumbnails of raw photos without an edit go"
        files[0].write_bytes(b"new")
        library.forget_starts(cache, photos)
        assert files[0].read_bytes() == b"new", "and only once"


def check_catalog_before_compensation() -> None:
    """A catalog from before version 3: every edit and checkpoint kept there has its Exposure
    written out as the 0 it was, so that no photo changes under the owner; and the file as it
    was is kept beside it."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "catalog.db"
        catalog.Catalog(path).close()
        old = sqlite3.connect(path)
        points = json.dumps([{"id": 1, "name": "One", "made": 5.0, "edit": {"shadows": 20.0}},
                             {"id": 2, "name": "Two", "made": 6.0, "edit": {"exposure": -1.0}}])
        with old:
            for key, stored, marks in (("c1:aa:1", '{"contrast": 30.0}', points), ("c1:bb:1", '{"exposure": 1.5}', "[]"),
                                       ("c1:cc:1", "{}", "[]"), ("c1:dd:1", "not json", "[]")):
                old.execute("INSERT INTO edits VALUES (?, ?, ?, ?, 1, ?)", (key, key.split(":")[1], f"/{key}.arw", stored, marks))
            old.execute("PRAGMA user_version=2")
        old.close()
        for _ in range(2):
            kept = catalog.Catalog(path)
            assert kept.load("c1:aa:1", Path("/c1:aa:1.arw")) == {"contrast": 30.0, **THEN}, "an edit of before keeps its 0 (at rest now), its straight curve and the lens corrections it had"
            assert kept.load("c1:bb:1", Path("/c1:bb:1.arw")) == {"exposure": 1.5, **THEN}
            assert kept.load("c1:cc:1", Path("/c1:cc:1.arw")) == {}, "a photo with no edit takes the new start"
            assert kept.load("c1:dd:1", Path("/c1:dd:1.arw")) == {}
            assert [p["edit"] for p in kept.checkpoints("c1:aa:1")] == [{"shadows": 20.0, **THEN}, {"exposure": -1.0, **THEN}]
            kept.close()
        copy = sqlite3.connect(path.with_name("catalog.db.v2"))
        assert copy.execute("SELECT edit FROM edits WHERE key = 'c1:aa:1'").fetchone()[0] == '{"contrast": 30.0}', "as it was before"
        copy.close()
    # A catalog of version 3, where an edit that named no Exposure had the camera's whole
    # compensation taken back: such an edit says so now, if its file is there to ask.
    with tempfile.TemporaryDirectory() as folder:
        path, under = Path(folder) / "catalog.db", Path(folder) / "under.arw"
        under.write_bytes(exif("SONY", "ILCE-7M4", "2026:07:04 13:35:44", 1, -1.3))
        picture_file(Path(folder) / "done.jpg", exif("SONY", "ILCE-7M4", "2026:07:04 13:35:44", 1, -1.3))
        catalog.Catalog(path).close()
        old = sqlite3.connect(path)
        points = json.dumps([{"id": 1, "name": "One", "made": 5.0, "edit": {"shadows": 20.0}}])
        rows = (("c1:aa:1", str(under), '{"contrast": 30.0}', points), ("c1:bb:1", str(under), '{"exposure": 0.2}', "[]"),
                ("c1:cc:1", str(under), "{}", "[]"), ("c1:dd:1", str(Path(folder) / "gone.arw"), '{"contrast": 30.0}', "[]"),
                ("c1:ee:1", str(Path(folder) / "done.jpg"), '{"contrast": 30.0}', "[]"), ("c1:ff:1", None, '{"contrast": 30.0}', "[]"))
        with old:
            for key, where, stored, marks in rows:
                old.execute("INSERT INTO edits VALUES (?, ?, ?, ?, 1, ?)", (key, key.split(":")[1], where, stored, marks))
            old.execute("PRAGMA user_version=3")
        old.close()
        for _ in range(2):
            kept = catalog.Catalog(path)
            edits = {key: json.loads(kept.db.execute("SELECT edit FROM edits WHERE key = ?", (key,)).fetchone()[0]) for key, *_ in rows}
            assert edits["c1:aa:1"] == {"contrast": 30.0, "exposure": 1.3, **THEN}, "an edit of version 3 keeps the start it had"
            assert edits["c1:bb:1"] == {"exposure": 0.2, **THEN} and edits["c1:cc:1"] == {}, "a named Exposure stays; no edit takes the new start"
            assert edits["c1:dd:1"] == {"contrast": 30.0, **THEN} and edits["c1:ee:1"] == edits["c1:ff:1"] == {"contrast": 30.0}, "a file that is gone, a finished picture, no path: left alone"
            assert [p["edit"] for p in kept.checkpoints("c1:aa:1")] == [{"shadows": 20.0, "exposure": 1.3, **THEN}]
            kept.close()
        assert path.with_name("catalog.db.v3").exists(), "the file as it was is kept beside it"
    # And one of version 4, where the compensation was taken back as far as the picture had
    # room: what that was is asked of the file (here, of a stand-in for the asking).
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "catalog.db"
        catalog.Catalog(path).close()
        old = sqlite3.connect(path)
        with old:
            for key, where, stored in (("c1:aa:1", "/there.arw", '{"contrast": 30.0}'), ("c1:bb:1", "/there.arw", '{"exposure": 0.2}'),
                                       ("c1:cc:1", "/gone.arw", '{"contrast": 30.0}'), ("c1:dd:1", "/there.arw", "{}")):
                old.execute("INSERT INTO edits VALUES (?, ?, ?, ?, 1, '[]')", (key, key.split(":")[1], where, stored))
            old.execute("PRAGMA user_version=4")
        old.close()
        asked, catalog._held_back = catalog._held_back, lambda where: 0.45 if where == "/there.arw" else None
        try:
            kept = catalog.Catalog(path)
        finally:
            catalog._held_back = asked
        edits = {key: json.loads(stored) for key, stored in kept.db.execute("SELECT key, edit FROM edits")}
        assert edits == {"c1:aa:1": {"contrast": 30.0, "exposure": 0.45, **THEN}, "c1:bb:1": {"exposure": 0.2, **THEN},
                         "c1:cc:1": {"contrast": 30.0, **THEN}, "c1:dd:1": {}}, f"edits of version 4 keep the start they had: {edits}"
        assert kept.db.execute("PRAGMA user_version").fetchone()[0] == catalog.VERSION == 7 and path.with_name("catalog.db.v4").exists()
        kept.close()
    # And one of version 5, where Exposure started by the picture alone (`_start_5`), the
    # curve was straight and the lens corrections on unless an edit said otherwise: edits,
    # checkpoints and kept undo steps of raw files say so now; a finished picture's say
    # nothing new (they started at 0, straight and uncorrected then as now).
    assert catalog._start_5([0.02, 0.1]) == 1.58 and catalog._start_5([0.01, 0.4]) == 0.32, "its middle at 0.06, its top at half of white"
    assert catalog._start_5([0.2, 1.0]) == 0 and catalog._start_5([0.0001, 0.001]) == 2.0 and catalog._start_5(None) == 0
    assert catalog._by_picture(None) is None and catalog._by_picture("/no/such/file.arw") is None and catalog._by_picture("/a/picture.jpg") is None
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "catalog.db"
        catalog.Catalog(path).close()
        old = sqlite3.connect(path)
        points = json.dumps([{"id": 1, "name": "One", "made": 5.0, "edit": {}}])
        with old:
            for key, where, stored, marks in (("c1:aa:1", "/there.arw", '{"contrast": 30.0}', points), ("c1:bb:1", "/there.arw", '{"lensColor": false, "exposure": 0.0}', "[]"),
                                              ("c1:cc:1", "/there.arw", "{}", "[]"), ("c1:dd:1", "/done.jpg", '{"contrast": 30.0}', "[]")):
                old.execute("INSERT INTO edits VALUES (?, ?, ?, ?, 1, ?)", (key, key.split(":")[1], where, stored, marks))
            old.execute("INSERT INTO steps VALUES ('c1:aa:1', ?, 1)", (json.dumps([{}, {"contrast": 30.0}]),))
            old.execute("PRAGMA user_version=5")
        old.close()
        asked, catalog._by_picture = catalog._by_picture, lambda where: 0.45 if where == "/there.arw" else None
        try:
            kept = catalog.Catalog(path)
        finally:
            catalog._by_picture = asked
        edits = {key: json.loads(stored) for key, stored in kept.db.execute("SELECT key, edit FROM edits")}
        assert edits == {"c1:aa:1": {"contrast": 30.0, "exposure": 0.45, **THEN}, "c1:bb:1": {**THEN, "lensColor": False, "exposure": 0.0},
                         "c1:cc:1": {}, "c1:dd:1": {"contrast": 30.0}}, edits
        assert [p["edit"] for p in kept.checkpoints("c1:aa:1")] == [{"exposure": 0.45, **THEN}], "a checkpoint at rest then is written out too"
        history = edit.History(kept.load("c1:aa:1", Path("/there.arw")), kept.steps("c1:aa:1"))
        assert history.at == 1 and history.states[0] == {"exposure": 0.45, **THEN}, "undo carries on, to the photo as it was"
        assert kept.db.execute("PRAGMA user_version").fetchone()[0] == catalog.VERSION and path.with_name("catalog.db.v5").exists()
        kept.close()
    # And one of version 6, the hour the lens corrections were off at rest and the rest as in
    # version 5: Exposure and the curve are written out, the corrections are not (they were
    # off at rest then too, and edits from before had them written out already).
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "catalog.db"
        catalog.Catalog(path).close()
        old = sqlite3.connect(path)
        with old:
            for key, where, stored in (("c1:aa:1", "/there.arw", '{"contrast": 30.0}'), ("c1:bb:1", "/there.arw", '{"lensColor": true, "exposure": 1.0}'),
                                       ("c1:cc:1", "/done.jpg", json.dumps({**ON, "contrast": 30.0}))):
                old.execute("INSERT INTO edits VALUES (?, ?, ?, ?, 1, '[]')", (key, key.split(":")[1], where, stored))
            # (Version 6's round wrote the corrections into the edit, not into its undo steps.)
            old.execute("INSERT INTO steps VALUES ('c1:cc:1', ?, 1)", (json.dumps([{}, {"contrast": 30.0}]),))
            old.execute("PRAGMA user_version=6")
        old.close()
        asked, catalog._by_picture = catalog._by_picture, lambda where: 0.45 if where.endswith(".arw") else None
        try:
            kept = catalog.Catalog(path)
        finally:
            catalog._by_picture = asked
        edits = {key: json.loads(stored) for key, stored in kept.db.execute("SELECT key, edit FROM edits")}
        assert edits == {"c1:aa:1": {"contrast": 30.0, "exposure": 0.45, "curve": edit.STRAIGHT},
                         "c1:bb:1": {"lensColor": True, "exposure": 1.0, "curve": edit.STRAIGHT}, "c1:cc:1": {**ON, "contrast": 30.0}}, edits
        history = edit.History(kept.load("c1:cc:1", Path("/done.jpg")), kept.steps("c1:cc:1"))
        assert history.at == 1 and history.states[0] == ON, "its undo steps end where its edit stands again"
        kept.close()
    assert catalog._held_back(None) is None and catalog._held_back("/no/such/file.arw") is None and catalog._held_back("/a/picture.jpg") is None
    assert catalog._start_4(0.7, [0.01, 0.02]) == 0.7 and catalog._start_4(0.0, [0.01, 0.02]) == 0, "version 4: never more than the camera was set under"
    assert catalog._start_4(1.3, [0.04, 0.1]) == 0.58 and catalog._start_4(5.0, [0.0001, 0.001]) == 3.0 and catalog._start_4(1.0, None) == 0


def check_saved() -> None:
    """An edit that has come to rest says so, with or without a catalog: the photo's thumbnail
    is made anew from it."""
    session = app.Session([Path("a.arw")], load=chart)
    said = []
    session.saved.connect(lambda: said.append(dict(session.history.edit)))
    session.flush()
    assert said == [], "nothing was changed, and yet it says so"
    session.setEdit("exposure", 1.0)
    session.setEdit("exposure", 1.5)
    assert said == [] and session.saver.isActive(), "a moment after the last change, not at each"
    session.flush()
    session.flush()
    assert said == [{"exposure": 1.5}], said


def check_ahead() -> None:
    """Stepping through photos: the neighbors are decoded ahead, the one the user is heading
    for first and in both sizes; the full-size picture is asked for only once the first one has
    arrived; and a picture handed to Qt is a copy that is the picture."""
    photos = [Path(f"{n}.arw") for n in range(4)]
    session = app.Session(photos, load=chart)
    asked, given_up = [], []
    session.ahead = lambda wanted, begin=True: (asked if begin else given_up).append(wanted)
    session._read_ahead()
    assert asked[-1] == [(photos[1], True), (photos[1], False), (photos[2], True)], f"at the first photo: {asked[-1]}"
    session.step(1)
    assert given_up[-1][:2] == [(photos[1], True), (photos[1], False)] and (photos[2], True) in given_up[-1] and len(asked) == 1, \
        "a photo asked for: what is near it stays, and nothing is begun ahead of it"
    session.step(1)
    session._read_ahead()
    assert asked[-1] == [(photos[3], True), (photos[3], False), (photos[1], True)], f"going on: {asked[-1]}"
    session.step(-1)
    session._read_ahead()
    assert asked[-1] == [(photos[0], True), (photos[0], False), (photos[2], True)], f"going back: {asked[-1]}"
    assert session.fullSource == "", "the full-size picture is asked for before the first one has arrived"
    session.image(session.source.removeprefix("image://photo/"))
    QCoreApplication.processEvents()  # the description comes as an event
    assert session.fullSource.endswith("/full"), session.fullSource
    session.step(1)
    assert session.fullSource == "", "and again for the next photo"
    # The quick picture is decoded before the full size, not beside it.
    order = []

    def slow(path: Path, half: bool = True):
        order.append(("begun", half))
        time.sleep(0.05)
        order.append(("done", half))
        return chart(path, half)

    slow_session = app.Session(photos, load=slow)
    for wanted in list(slow_session.fetching.values()):
        wanted.result()[1].close()
    assert order == [("begun", True), ("done", True), ("begun", False), ("done", False)], order
    pixels = np.arange(4 * 3 * 4, dtype=np.float16).reshape(3, 4, 4)
    copy = app.owned(pixels.tobytes(), 4, 3)
    assert (copy.width(), copy.height(), copy.format()) == (4, 3, QImage.Format.Format_RGBA16FPx4)
    assert bytes(copy.constBits())[:96] == pixels.tobytes(), "the copy is not the picture"


def check_steps_kept() -> None:
    """A photo's undo and redo are kept with its edit (the owner: "Undo/Redo queue seems to
    clear when photos are switched"): there again after another photo was open and after the
    program was closed; not when the edit was changed elsewhere meanwhile; a hundred steps at
    most, round where it stands; and what was written there is not trusted."""
    with tempfile.TemporaryDirectory() as folder:
        kept = catalog.Catalog(Path(folder) / "catalog.db")
        photos = [photo_file(folder, "a.arw", 1), photo_file(folder, "b.arw", 2)]
        session = app.Session(photos, load=chart, catalog=kept)
        for key, value in (("exposure", 0.5), ("contrast", 20), ("shadows", 30)):
            session.setEdit(key, value)
            session.commitEdit()
        session.undo()
        assert session.history.edit == {"exposure": 0.5, "contrast": 20.0} and session.canUndo and session.canRedo
        session.step(1)
        assert session.history.edit == {} and not session.canUndo and not session.canRedo, "another photo has its own steps"
        session.setEdit("tint", 5)
        session.commitEdit()
        session.step(-1)
        assert session.history.edit == {"exposure": 0.5, "contrast": 20.0} and session.canUndo and session.canRedo, "gone and come back: no steps"
        session.redo()
        assert session.history.edit == {"exposure": 0.5, "contrast": 20.0, "shadows": 30.0}
        for _ in range(3):
            session.undo()
        assert session.history.edit == {} and not session.canUndo, "all the way back to the untouched photo"
        session.redo()
        session.step(1)
        assert session.history.edit == {"tint": 5.0} and session.canUndo, "and the other photo's are its own still"
        session.flush()
        again = app.Session(photos, load=chart, catalog=kept)  # the program, started anew
        assert again.history.edit == {"exposure": 0.5} and again.canUndo and again.canRedo, "closed and opened: no steps"
        again.redo()
        again.redo()
        assert again.history.edit == {"exposure": 0.5, "contrast": 20.0, "shadows": 30.0} and not again.canRedo
        again.flush()
        key = catalog.identify(photos[0])
        kept.save(key, photos[0], {"exposure": -1.0})  # the edit changed elsewhere (a reset from a menu, an older program)
        other = app.Session(photos, load=chart, catalog=kept)
        assert other.history.edit == {"exposure": -1.0} and not other.canUndo and not other.canRedo, "steps that do not end at the edit are not taken up"
        with kept.db:
            kept.db.execute("UPDATE steps SET states = 'not json' WHERE key = ?", (key,))
        assert kept.steps(key) is None and kept.steps(None) is None and kept.steps("c1:none:1") is None
        assert app.Session(photos, load=chart, catalog=kept).history.edit == {"exposure": -1.0}
        other.resetAll()
        other.flush()
        assert kept.load(key, photos[0]) == {} and kept.steps(key) is not None, "a photo reset keeps the steps back to what it was"
        kept.close()
    long = edit.History({})
    for n in range(250):
        long.set("exposure", (n % 50) / 10 + 0.01)
        long.commit()
    states, at = long.kept()
    assert len(states) == 100 and states[at] == long.edit and at == 99, "the last hundred steps"
    for _ in range(200):
        long.step(-1)
    states, at = long.kept()
    assert len(states) == 100 and states[at] == long.edit and at == 50, "a hundred round where it stands"
    assert edit.History({"exposure": 1.0}, ([{"junk": 1, "contrast": 1e9}, {"exposure": 1.0}], 1)).states == [{"contrast": 100.0}, {"exposure": 1.0}]
    for bad in (("x", 0), ([{"exposure": 1.0}], 5), ([{"exposure": 1.0}], "one"), None, ([3], 0), ([{"exposure": 0.2}], 0)):
        assert edit.History({"exposure": 1.0}, bad).states == [{"exposure": 1.0}], f"steps that are not steps, or not this edit's: {bad}"


def check_recent() -> None:
    """The pictures of the photos opened last are kept (app.Recent): going back to one decodes
    nothing, a file that changed is decoded anew, and what is not looked at for longest goes
    when there is no more room."""
    asked = []

    def counting(path: Path, half: bool = True):
        asked.append((path.name, half))
        return chart(path, half)

    def look(session) -> None:  # as Qt asks for the photo shown: both of its sizes
        for size in ("half", "full"):
            assert session.image(session._address(size).removeprefix("image://photo/")).width() > 1
        QCoreApplication.processEvents()

    with tempfile.TemporaryDirectory() as folder:
        a, b = (Path(folder) / name for name in ("a.arw", "b.arw"))
        a.write_bytes(b"one")
        b.write_bytes(b"two")
        session = app.Session([a, b], load=counting)
        look(session)
        session.step(1)
        look(session)
        assert sorted(asked) == [("a.arw", False), ("a.arw", True), ("b.arw", False), ("b.arw", True)], asked
        session.step(-1)
        look(session)
        session.step(1)
        look(session)
        assert len(asked) == 4, f"going back to a photo decoded it again: {asked[4:]}"
        assert session.fullWidth == 16 and session.info.startswith("b.arw"), "what is kept says what the photo is, too"
        assert session._small[0] == session._generation and session._small[1].shape[2] == 3, "and its thinned copy is there"
        a.write_bytes(b"one, written again and longer")
        session.step(-1)
        look(session)
        assert asked[4:] == [("a.arw", True), ("a.arw", False)], f"a file that changed is decoded anew: {asked[4:]}"
        session.threads.shutdown()
    picture = QImage(10, 10, QImage.Format.Format_RGBA16FPx4)  # 800 bytes
    few = app.Recent(2000)
    for n in range(4):
        few.put(Path(f"/none/{n}.arw"), True, ({}, picture, None))
    assert [few.has(Path(f"/none/{n}.arw"), True) for n in range(4)] == [False, False, True, True], "room for two: the last two"
    assert few.get(Path("/none/2.arw"), True) is not None and few.get(Path("/none/2.arw"), False) is None
    few.put(Path("/none/4.arw"), True, ({}, picture, None))
    assert few.has(Path("/none/2.arw"), True) and not few.has(Path("/none/3.arw"), True), "what was looked at last stays"
    tight = app.Recent(10)
    tight.put(Path("/none/big.arw"), True, ({}, picture, None))
    assert tight.has(Path("/none/big.arw"), True), "the one just opened is kept whatever its size"
    assert 2 ** 26 < app.Recent().room <= 2 << 30, "by the machine's memory, 2 GB at most"


def check_fresh() -> None:
    """What the session works out once per change is never read stale by the interface. Qt
    tells QML bindings of a change before any slot runs, so forgetting must come before the
    telling: a photo that had been looked at (the pointer over it) did not turn until the
    next change."""
    session = app.Session([Path("a.arw")], load=chart)
    session.image("0/0/half")
    engine = QQmlEngine()
    engine.rootContext().setContextProperty("session", session)
    watcher = QQmlComponent(engine)
    watcher.setData(b"import QtQml\nQtObject { property int wide: session.fullWidth\n"
                    b"property real gain: session.uniforms.gain }", QUrl())
    seen = watcher.create()
    assert seen is not None, watcher.errorString()
    wide, high = session.fullWidth, session.fullHeight
    assert seen.property("wide") == wide and wide != high
    session.colorAt(0.5, 0.5)  # between two changes: the frame is worked out, and kept
    assert session.uniforms["gain"] == 1.0
    session.setEdits({"turns": 1.0, "exposure": 1.0})
    assert seen.property("wide") == high, f"the interface saw the old frame: {seen.property('wide')} wide"
    assert seen.property("gain") == 2.0, "the interface saw the old numbers"
    assert session.masks == [] and session.fullWidth == high
    # B and Compare: the scopes read the numbers a moment after every change, and the next
    # change must not be served those. (It was: B showed the edit while held and the original
    # once let go, until two changes came close together.)
    for hold, let_go, what in ((lambda: session.setOriginal(True), lambda: session.setOriginal(False), "B"),
                               (lambda: session.setBypass("tone"), lambda: session.setBypass(""), "Compare")):
        for _ in range(3):
            assert session.uniforms["gain"] == 2.0  # read between two changes, as the scopes do
            hold()
            assert seen.property("gain") == 1.0, f"{what} held, and the interface is shown the edit"
            assert session.uniforms["gain"] == 1.0
            let_go()
            assert seen.property("gain") == 2.0, f"{what} let go, and the interface is shown the original"
    del seen, watcher, engine  # before the session goes: a binding left without it complains


def check_session_edit() -> None:
    """The edit as the interface sees it, kept in the catalog across photos and runs."""
    with tempfile.TemporaryDirectory() as folder:
        kept = catalog.Catalog(Path(folder) / "catalog.db")
        photos = [photo_file(folder, "a.arw", 1), photo_file(folder, "b.arw", 2)]
        keys = [catalog.identify(p) for p in photos]
        session = app.Session(photos, load=chart, catalog=kept)
        assert session.edit["exposure"] == 0 and not session.canUndo and not session.develops
        session.image("0/0/half")  # the decoder's description arrives
        assert session.develops
        shot = session.shot
        assert session.edit["temperature"] == shot["temperature"], "as shot, as a number"
        assert session.uniforms["gain"] == 1.0 and session.uniforms["wbGain"].x() == 1.0

        for value in (0.2, 0.6, 1.0):  # a drag
            session.setEdit("exposure", value)
        session.commitEdit()
        session.setEdit("temperature", 7000)
        session.setEdit("nonsense", 1)
        session.setEdit("exposure", "lots")
        assert session.history.edit == {"exposure": 1.0, "temperature": 7000.0}
        assert session.uniforms["gain"] == 2.0 and session.uniforms["wbGain"].x() > 1.05
        session.setOriginal(True)
        assert session.uniforms["gain"] == 1.0 and session.edit["exposure"] == 1.0, "only the picture"
        session.setOriginal(False)
        session.undo()
        assert session.edit["temperature"] == shot["temperature"] and session.canRedo
        session.undo()
        assert session.edit["exposure"] == 0 and not session.canUndo
        session.redo()
        session.redo()
        session.resetEdit("temperature")
        assert session.history.edit == {"exposure": 1.0}
        assert kept.load(keys[0], photos[0]) == {}, "not written at once"
        session.flush()
        assert kept.load(keys[0], photos[0]) == {"exposure": 1.0}

        session.setEdit("shadows", 40)
        session.step(1)  # leaving a photo writes its edit
        assert kept.load(keys[0], photos[0]) == {"exposure": 1.0, "shadows": 40.0}
        assert session.history.edit == {} and not session.canUndo, "the next photo starts clean"
        session.step(-1)
        assert session.history.edit == {"exposure": 1.0, "shadows": 40.0}, "and the first comes back"

        # The edit follows the photo: moved to another folder under another name, it is found.
        (Path(folder) / "archive").mkdir()
        moved = photos[0].rename(Path(folder) / "archive" / "holiday-001.arw")
        later = app.Session([moved, photos[1]], load=chart, catalog=kept)
        assert later.edit["shadows"] == 40.0 and later.edit["exposure"] == 1.0
        assert not list(Path(folder).rglob("*.xmp")) and len(list(Path(folder).rglob("*.arw"))) == 2, "nothing is written beside a photo"
        hdr = app.Session(photos[1:], load=chart, hdr=True)
        hdr.image("0/0/half")
        assert hdr.uniforms["hdrLift"] == 0, "no HDR surface yet"
        hdr.surface_known(True)
        assert hdr.uniforms["hdrLift"] == hdr.peak - 1, "in HDR the top reaches the peak"

        kept.close()
        later.setEdit("exposure", 2)
        later.flush()
        assert "could not be saved" in later.error, "a failed write is said, not swallowed"


def check_session_tools() -> None:
    """The rail's tools as the interface drives them: the dot, Reset, Compare, Reset all, Pick
    neutral, what is open; and the histogram's counting."""
    session = app.Session([Path("chart.arw")], load=lambda path, half: decode.synthetic(240, 160))
    session.image("0/0/half")
    assert not any(session.adjusted.values()) and set(session.adjusted) == set(edit.TOOLS) and not session.edited
    session.setEdits({"contrast": 30, "shadows": 10})
    session.setEdits({"curve": [[0, 0], [0.5, 0.6], [1, 1]], "saturation": 120})
    session.commitEdit()
    assert [tool for tool, on in session.adjusted.items() if on] == ["tone", "saturation"] and session.edited
    session.setBypass("tone")
    seen = session.uniforms
    assert seen["contrast"] == 0 and seen["shadows"] == 0 and seen["curveOn"] == 0, "without Tone (its curve too), and only Tone"
    session.setBypass("")
    assert session.uniforms["contrast"] > 0
    # A card whose settings do something at rest: its A/B shows the photo without them, and
    # is there to be held though nothing of the card was changed.
    assert session.uniforms["sharpen"] > 0 and not session.adjusted["detail"] and session.comparable["detail"] and not session.comparable["eq"]
    session.setBypass("detail")
    assert session.uniforms["sharpen"] == 0 and session.uniforms["denoise"] == 0 and session.uniforms["denoiseColor"] == 0
    session.setBypass("")
    assert not session.comparable["lens"], "nothing of the Lens card is on: nothing to compare"
    steps = session.history.at
    session.resetTool("tone")
    assert session.history.edit == {"saturation": 120.0} and session.history.at == steps + 1
    session.resetTool("nonsense")
    session.resetAll()
    assert session.history.edit == {} and session.history.at == steps + 2
    session.undo()
    session.undo()
    assert session.history.edit["contrast"] == 30.0, "each reset is one step"
    # A/B on a mask's card, and on the Tone Wheel: without those settings of that mask, or its pins.
    whole = dict(session.history.edit)
    session.setEdits({"masks": [{"kind": "radial", "at": [0.5, 0.5], "to": [0.7, 0.5], "edit": {"exposure": 1.0, "contrast": 20.0}}],
                      "pins": [{"from": [0.1, 0.0], "to": [0.1, 0.06]}, {"from": [0.0, 0.1], "to": [0.0, 0.05], "mask": 1}]})
    assert len(session.maskLayers) == 1 and session.uniforms["look"] == 1
    session.setMaskBypass(1, ["exposure"])
    assert session._compared()["masks"][0]["edit"] == {"contrast": 20.0} and len(session.maskLayers) == 1
    session.setMaskBypass(1, ["exposure", "contrast"])
    assert session.maskLayers == [] and len(session.history.edit["masks"][0]["edit"]) == 2, "held, and the edit is as it was"
    session.setMaskBypass(0, ["pins"])
    assert [pin.get("mask", 0) for pin in session._compared()["pins"]] == [1], "the whole image's pins, not the mask's"
    session.setMaskBypass(1, ["pins"])
    assert [pin.get("mask", 0) for pin in session._compared()["pins"]] == [0] and len(session.maskLayers) == 1
    session.setMaskBypass(0, [])
    assert session._compared() == session.history.edit
    session.history.replace(whole)
    session.refresh()
    assert len(session.curveLine) == edit.LINE and abs(session.curveLine[128] - 0.6) < 1e-9
    wild = [[0.0, 0.0], [0.2, 0.9], [0.25, 0.1], [0.6, 0.7], [1.0, 1.0]]
    assert max(abs(a - edit.curve_at(wild, i / 256)) for i, a in enumerate(edit.curve_line(wild))) < 1e-12, "the line drawn is the curve"

    # Pick neutral on the chart: a colored spot becomes gray, the black corner says nothing.
    session.resetAll()
    session.pickNeutral(0.07, 0.35)
    found = session.history.edit
    assert set(found) == {"temperature", "tint"} and session.error == "", found
    pixels = np.frombuffer(decode.synthetic(240, 160)[1].buf, np.float16).reshape(160, 240, 4)
    u = edit.uniforms(found, session._camera, True)
    spot = (np.array(u["toCam"]) @ pixels[int(0.35 * 159), int(0.07 * 239), :3].astype(float)) * u["wbGain"][:3]
    assert spot.max() / spot.min() < 1.03, spot
    session.pickNeutral(0.3, 0.5)  # slid on to another spot: the same gesture
    assert session.history.edit != found
    session.commitEdit()
    session.undo()
    assert session.history.edit == {}, "a pick, however far it is slid, is one step"
    session.pickNeutral(0.01, 0.99)  # the wedge's black step
    assert session.history.edit == {} and "too bright or too dark" in session.error

    assert session.panels == {"tool": "tone", "clipping": False, "live": True, "chart": False, "place": "library"}
    session.setPanel("tool", "curve")
    session.setPanel("clipping", True)
    session.setPanel("clipping", "yes")
    session.setPanel("nonsense", True)
    assert session.panels == {"tool": "curve", "clipping": True, "live": True, "chart": False, "place": "library"}

    # The test chart in the photo's place: the chart's picture and size, the photo's edit.
    session.setEdit("exposure", 0.5)
    was = (session.source, session.fullWidth, session.fullHeight, session.history.edit)
    turned = []
    session.photoChanged.connect(lambda: turned.append(1))
    session.setPanel("chart", True)
    assert session.panels["chart"] and "/chart/" in session.source and session.fullSource == ""
    assert (session.fullWidth, session.fullHeight) == app.CHART and turned == [1], "the view starts afresh"
    shown = session.image(session.source.removeprefix("image://photo/"))
    assert (shown.width(), shown.height()) == app.CHART and shown.format() == QImage.Format.Format_RGBA16FPx4
    assert session.history.edit == was[3] and session.uniforms["gain"] == 2 ** 0.5, "the edit is still the photo's"
    session.pickNeutral(0.07, 0.35)
    assert "temperature" in session.history.edit, "pick neutral reads the chart while it is on show"
    session.undo()
    session.setPanel("chart", False)
    assert (session.source, session.fullWidth, session.fullHeight, session.history.edit) == was
    session.setPanel("chart", True)
    session.step(0)
    session.open([str(Path(__file__))])
    assert not session.panels["chart"], "never left on for another photo"
    assert not app.Session([]).panels["chart"]

    # The renders arrive as four tiles; the session keeps what was counted from them.
    image = QImage(12, 8, QImage.Format.Format_RGBA8888)
    image.fill(QColor(10, 20, 200))  # (blue: of the tile beside b, the pixel counts)
    session.setScope(image)  # counted on a thread of its own; the result comes as an event
    limit = time.monotonic() + 5
    while not session.histogram and time.monotonic() < limit:
        QCoreApplication.processEvents()
    assert session.histogram["g"][20] == 1 and session.histogram["over"] == 0
    assert len(session.colorAt(0.5, 0.5)) == 3 and session.colorAt(2, 0) == [], "only on the photo"
    assert session.bars["g"][20] == 255 and session.bars["over"] == 0, "the same, packed for the interface"
    assert session.dots["count"] == len(session.cloud["colors"]) > 0 and len(session.dots["box"]) == 4


def check_scopes() -> None:
    """What is counted from the pipeline's renders (photone/scopes.py): the histogram, and the
    photo's colors as dots for the wheel."""
    def packed(value: float) -> list[int]:  # -0.5..0.5 in two bytes, as the shader writes it
        wide = round((value + 0.5) * 65535)
        return [wide // 256, wide % 256]

    render = np.zeros((8, 12, 4), np.uint8)  # six tiles of 4 x 4, three by two
    render[:4, :4, :3] = [[10, 20, 30], [10, 20, 30], [255, 0, 0], [0, 0, 0]]
    render[:4, 4:8, 0] = 128
    render[:4, 8:, :2], render[:4, 8:, 2] = packed(0.1234), 204   # a, lightness 0.8
    render[4:, :4, :2] = packed(-0.2)                              # b
    render[0, 8, :3], render[4, 0, :2] = [*packed(-0.05), 51], packed(0.05)  # one dark pixel
    render[4:, 4:8, :2], render[4:, 4:8, 2] = packed(0.3), 204     # after the pins: a,
    render[4:, 8:, :2] = packed(-0.2)                              # and b
    counts = scopes.histogram_of(render)
    assert counts["r"][10] == 1 and counts["g"][20] == 1 and counts["b"][30] == 1 and counts["g"][10] == 0
    assert counts["over"] == 0.25 and counts["under"] == 0.25, "a quarter at white, a quarter at black"
    assert max(range(256), key=lambda i: counts["light"][i]) == 128
    light, a, b = scopes.lab_of(render)
    assert abs(a[1, 1] - 0.1234) < 2e-5 and abs(b[1, 1] + 0.2) < 2e-5 and abs(light[1, 1] - 0.8) < 1e-6
    moved = scopes.lab_of(render, pinned=True)
    assert abs(moved[1][1, 1] - 0.3) < 2e-5 and abs(moved[2][1, 1] + 0.2) < 2e-5, "the colors after the pins"
    def cloud_of(*given, **named) -> dict:  # as plain lists
        return scopes.listed(scopes.cloud_of(*given, **named))

    cloud = cloud_of(light, a, b)
    assert len(cloud["colors"]) == 2 and len(cloud["points"]) == 4, "two colors, two dots"
    assert np.allclose(cloud["points"][2:], [0.1234, -0.2], atol=1e-4), "the commonest last"
    assert np.allclose(cloud["box"], [-0.05, 0.1234, -0.2, 0.05], atol=1e-4)
    assert max(range(64), key=lambda i: cloud["light"][i]) == 51 and cloud["light"][12] > 0
    dark = cloud_of(light, a, b, scopes.RANGES["shadows"])
    assert np.allclose(dark["points"], [-0.05, 0.05], atol=1e-4), "only the range's pixels"
    assert dark["light"] == cloud["light"], "the strip shows the whole photo whatever the range"
    assert cloud_of(light, a, b, (0.4, 0.6))["points"] == [], "nothing that light: no dots"
    # Colors: gray is gray, and the wheel's directions are the colors they should be.
    gray, red, yellow, blue = scopes.srgb_of(np.array([0.6, 0.6, 0.8, 0.5]), np.array([0, 0.2, 0, 0]),
                                             np.array([0, 0.08, 0.15, -0.15])).tolist()
    assert gray[0] == gray[1] == gray[2] and 110 < gray[0] < 140, gray
    assert red[0] > red[1] + 60 and yellow[2] < min(yellow[:2]) - 60 and blue[2] > blue[0] + 60
    spread = scopes.spread_of(light, a, b)
    assert len(spread["chroma"]) == 64 and len(spread["hue"]) == 90
    assert max(range(64), key=lambda i: spread["chroma"][i]) == int(np.hypot(0.1234, 0.2) / 0.4 * 64)
    assert max(range(90), key=lambda i: spread["hue"][i]) == int(np.degrees(np.arctan2(-0.2, 0.1234)) % 360 / 4)
    assert scopes.heights([0, 0.5, 1, 7]) == bytes([0, 128, 255, 255]), "bars as bytes for the interface"
    strip = scopes.hues()
    assert len(strip) == 37 and strip[0] == strip[-1] and len(set(strip)) > 30, "once round the wheel"
    crowd = np.random.default_rng(1).uniform(-0.3, 0.3, (2, 60000)).astype(np.float32)
    many = cloud_of(np.full(60000, 0.5, np.float32), *crowd)
    assert 500 < len(many["colors"]) <= scopes.DOTS, "never more dots than the wheel should draw"


def exif(maker: str, model: str, taken: str, turned: int, bias: float | None = None) -> bytes:
    """A TIFF header as a camera writes it into a raw file or a JPEG's EXIF block: maker, model
    and orientation in the first directory, the date in the EXIF directory it points to, and
    there, if one is given, the exposure compensation as a signed fraction."""
    texts = [t.encode() + b"\0" for t in (maker, model, taken)]
    first = 8 + 2 + 4 * 12 + 4            # where the texts begin: after the first directory
    second = first + sum(map(len, texts))  # and the EXIF directory after them
    out = b"II*\0" + struct.pack("<I", 8) + struct.pack("<H", 4)
    out += struct.pack("<HHII", 0x10f, 2, len(texts[0]), first)
    out += struct.pack("<HHII", 0x110, 2, len(texts[1]), first + len(texts[0]))
    out += struct.pack("<HHIHH", 0x112, 3, 1, turned, 0)
    out += struct.pack("<HHII", 0x8769, 4, 1, second) + struct.pack("<I", 0)
    out += b"".join(texts[:2])
    out += texts[2]
    dated = struct.pack("<HHII", 0x9003, 2, len(texts[2]), first + len(texts[0]) + len(texts[1]))
    if bias is None:
        return out + struct.pack("<H", 1) + dated + struct.pack("<I", 0)
    compensated = struct.pack("<HHII", 0x9204, 10, 1, second + 2 + 2 * 12 + 4)
    return out + struct.pack("<H", 2) + dated + compensated + struct.pack("<I", 0) + struct.pack("<ii", round(bias * 10), 10)


def picture_file(path: Path, header: bytes = b"", size: tuple[int, int] = (900, 600)) -> Path:
    """A real JPEG, with an EXIF block if a header is given."""
    image = QImage(*size, QImage.Format.Format_RGB32)
    image.fill(QColor(90, 120, 60))
    image.save(str(path), "JPEG", 90)
    if header:
        body = path.read_bytes()
        block = b"Exif\0\0" + header
        path.write_bytes(body[:2] + b"\xff\xe1" + struct.pack(">H", len(block) + 2) + block + body[2:])
    return path


def check_describe() -> None:
    """What a file's header says: when, with what camera, turned how (photone/library.py)."""
    with tempfile.TemporaryDirectory() as folder:
        header = exif("SONY", "ILCE-7M4", "2026:07:04 13:35:44", 6)
        raw = Path(folder) / "a.arw"
        raw.write_bytes(header + np.random.default_rng(1).bytes(5000))
        said = library.describe(raw)
        assert said["camera"] == "SONY ILCE-7M4", said
        assert time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(said["taken"])) == "2026-07-04 13:35:44"
        same = Path(folder) / "b.dng"
        same.write_bytes(exif("Canon", "Canon EOS R6", "2025:01:02 03:04:05", 1) + b"x" * 100)
        assert library.describe(same)["camera"] == "Canon EOS R6", "the maker is not said twice"
        photo = picture_file(Path(folder) / "c.jpg", header)
        assert library.describe(photo)["camera"] == "SONY ILCE-7M4"
        for name, content in (("none.jpg", None), ("junk.arw", b"II*\0" + b"\xff" * 300), ("short.nef", b"MM"),
                              ("other.raf", b"FUJIFILMCCD-RAW 0201" + b"\0" * 200)):
            odd = Path(folder) / name
            picture_file(odd) if content is None else odd.write_bytes(content)
            said = library.describe(odd)  # nothing to read: the file's own date, no camera, upright
            assert said["camera"] == "" and abs(said["taken"] - odd.stat().st_mtime) < 1, (name, said)
        loop = Path(folder) / "loop.dng"  # a directory whose EXIF pointer points back at itself
        loop.write_bytes(b"II*\0" + struct.pack("<IH", 8, 1) + struct.pack("<HHII", 0x8769, 4, 1, 8) + b"\0" * 8)
        assert library.describe(loop)["camera"] == "", "a file that points in circles is not followed for ever"
        assert library.thumb_name("c1:aa:1") != library.thumb_name("c1:aa:2") and len(library.thumb_name("x")) == 24


def check_library() -> None:
    """Scanning a folder, what the catalog keeps of it, the marks, and thumbnails."""
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder) / "photos"
        (root / "trip").mkdir(parents=True)
        files = [photo_file(root, "a.arw", 1), photo_file(root / "trip", "c.dng", 3)]
        picture_file(root / "a.jpg")  # beside a.arw: one photo with it
        alone = picture_file(root / "b.jpg", exif("SONY", "ILCE-7M4", "2020:01:01 10:00:00", 6))
        (root / "notes.txt").write_text("not a photo")
        (root / ".hidden.jpg").write_bytes(b"x")
        before = sorted(str(p.relative_to(folder)) for p in Path(folder).rglob("*"))

        rows, lost = library.scan(root, {})
        assert [r["name"] for r in rows] == ["a.arw", "b.jpg", "c.dng"] and lost == [], [r["name"] for r in rows]
        assert rows[0]["pair"] == str(root / "a.jpg") and rows[1]["pair"] is None and rows[0]["kind"] == "raw"
        assert rows[1]["kind"] == "picture" and rows[1]["camera"] == "SONY ILCE-7M4" and rows[2]["folder"] == str(root / "trip")
        assert rows[0]["key"] == catalog.identify(files[0])

        kept = catalog.Catalog(Path(folder) / "catalog.db")
        kept.add_folder(root)
        kept.add_folder(root)
        kept.found(rows)
        listed = kept.photos()
        assert kept.folders() == [str(root)] and [p["name"] for p in listed][-1] == "b.jpg", "newest first: the one from 2020 last"
        assert all(p["flag"] == p["stars"] == p["label"] == 0 and not p["edited"] for p in listed)
        again, lost = library.scan(root, kept.known(root))
        assert again == [] and lost == [], "nothing new the second time"

        keys = {p["name"]: p["key"] for p in listed}
        kept.mark([keys["a.arw"], keys["c.dng"]], "stars", 4)
        kept.mark([keys["a.arw"]], "flag", 1)
        kept.mark([keys["b.jpg"]], "label", 99)
        kept.save(keys["c.dng"], files[1], {"exposure": 1.0})
        marked = {p["name"]: p for p in kept.photos()}
        assert marked["a.arw"]["stars"] == 4 and marked["a.arw"]["flag"] == 1 and marked["b.jpg"]["label"] == 5
        assert marked["c.dng"]["edited"] and not marked["a.arw"]["edited"]

        # Renamed: the same photo, its marks with it. Deleted: gone from the list.
        moved = files[0].rename(root / "trip" / "best.arw")
        alone.unlink()
        rows, lost = library.scan(root, kept.known(root))
        kept.found(rows, lost)
        now = {p["name"]: p for p in kept.photos()}
        assert set(now) == {"best.arw", "a.jpg", "c.dng"}, set(now)  # the JPEG stands alone now
        assert now["best.arw"]["key"] == keys["a.arw"] and now["best.arw"]["stars"] == 4 and now["best.arw"]["flag"] == 1
        assert now["best.arw"]["path"] == str(moved) and now["best.arw"]["pair"] is None

        kept.remove_folder(root)
        assert kept.folders() == [] and kept.photos() == []
        kept.add_folder(root)
        kept.found(library.scan(root, kept.known(root))[0])
        assert {p["name"]: p["stars"] for p in kept.photos()}["best.arw"] == 4, "added back: its marks are there"
        # A folder taken out while a folder inside it is in the library too: that one keeps
        # its photos (they went with the outer one, and it stood at 0 until the next scan).
        kept.add_folder(root / "trip")
        kept.remove_folder(root)
        assert {p["name"] for p in kept.photos()} == {p.name for p in (root / "trip").iterdir() if p.suffix != ".jpg" or not p.with_suffix(".arw").exists()} != set(), \
            {p["name"] for p in kept.photos()}
        kept.add_folder(root)
        kept.found(library.scan(root, kept.known(root))[0])
        kept.remove_folder(root / "trip")
        assert len(kept.photos()) == 3, "the inner folder taken out: the outer one still holds its photos"
        kept.close()

        # Thumbnails: in the cache, the right size, upright; and nothing was written beside a photo.
        cache = Path(folder) / "cache"
        turned = picture_file(root / "portrait.jpg", exif("X", "Y", "2020:01:01 10:00:00", 6))
        thumb = QImage(str(library.thumbnail(turned, "c1:t:1", cache)))
        assert (thumb.width(), thumb.height()) == (341, 512), (thumb.width(), thumb.height())
        wide = QImage(str(library.thumbnail(root / "a.jpg", "c1:w:1", cache)))
        assert (wide.width(), wide.height()) == (512, 341) and abs(wide.pixelColor(100, 100).green() - 120) < 6
        made = library.thumbnail(root / "a.jpg", "c1:w:1", cache)
        assert made.parent == cache and len(list(cache.iterdir())) == 2, "made once, and only in the cache"
        try:
            library.thumbnail(root / "notes.txt", "c1:n:1", cache)
        except decode.Unreadable:
            pass
        else:
            raise AssertionError("a file that is no picture must be refused")
        turned.unlink()
        after = sorted(str(p.relative_to(folder)) for p in root.rglob("*"))
        assert after == sorted(a.replace("photos/a.arw", "photos/trip/best.arw") for a in before
                               if a.startswith("photos/") and not a.endswith("b.jpg")), after


def check_raw_only() -> None:
    """The library starts with raw files only, on show and in the side list's counts; the
    switch shows finished pictures too, and is remembered."""
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder) / "photos"
        root.mkdir()
        photo_file(root, "a.arw", 1), photo_file(root, "c.dng", 3)
        picture_file(root / "a.jpg")  # beside a.arw: one photo with it, a raw one
        picture_file(root / "b.jpg", exif("SONY", "ILCE-7M4", "2020:01:01 10:00:00", 1))
        kept = catalog.Catalog(Path(folder) / "catalog.db")
        kept.add_folder(root)
        kept.found(*library.scan(root, {}))
        settings = QSettings(str(Path(folder) / "settings.ini"), QSettings.Format.IniFormat)
        shown = library.Library(kept, Path(folder) / "cache", settings)
        shown.rest(True)  # (no thumbnails: these files are not pictures)
        names = lambda: sorted(p["name"] for p in shown.model.rows)  # noqa: E731
        assert shown.filter["raw"] and names() == ["a.arw", "c.dng"] and shown.total == 3, names()
        assert shown.places["all"] == 2 and shown.places["folders"][0]["count"] == 2
        assert sum(c["count"] for c in shown.places["cameras"]) == 2 and sum(m["count"] for m in shown.places["months"]) == 2
        shown.setFilter("raw", False)
        assert names() == ["a.arw", "b.jpg", "c.dng"] and shown.places["all"] == 3 and shown.places["folders"][0]["count"] == 3
        shown.close()
        again = library.Library(kept, Path(folder) / "cache", settings)
        again.rest(True)
        assert not again.filter["raw"] and again.count == 3, "the switch is not remembered"
        again.setFilter("raw", True)
        again.close()
        fresh = library.Library(kept, Path(folder) / "cache")
        fresh.rest(True)
        assert fresh.filter["raw"] and fresh.count == 2, "with nothing remembered: raw files only"
        fresh.close()
        kept.close()


def check_about() -> None:
    """What the left panel tells of a photo (library.about): the file's own facts always; with
    a real raw file (PHOTONE_SAMPLE), what its header says of the camera and the exposure."""
    with tempfile.TemporaryDirectory() as folder:
        made = photo_file(Path(folder), "a.arw", 1)
        told = library.about(made)
        assert told["file"] == "a.arw" and told["folder"] == folder and told["size"].endswith(("MB", "kB")) and told["taken"], told
    sample = os.environ.get("PHOTONE_SAMPLE")
    if sample:
        told = library.about(Path(sample))
        assert told.get("camera") and told.get("iso", "").startswith("ISO ") and told.get("shutter", "").endswith(" s"), told


def check_camera_names() -> None:
    """Cameras by the names they are sold under (library.camera_name)."""
    for written, sold in (("SONY ILCE-7M4", "Sony Alpha 7 IV"), ("SONY ILCE-7RM5", "Sony Alpha 7R V"),
                          ("SONY ILCE-7SM3", "Sony Alpha 7S III"), ("SONY ILCE-7CR", "Sony Alpha 7CR"),
                          ("SONY ILCE-7C", "Sony Alpha 7C"), ("SONY ILCE-6700", "Sony Alpha 6700"),
                          ("SONY ILCE-1", "Sony Alpha 1"), ("SONY ILCA-99M2", "Sony Alpha 99 II"),
                          ("SONY DSC-RX100M7", "Sony RX100 VII"), ("SONY ZV-E10", "Sony ZV-E10"),
                          ("NIKON CORPORATION NIKON Z 6_2", "Nikon Z 6II"), ("NIKON D850", "Nikon D850"),
                          ("Canon EOS R6", "Canon EOS R6"), ("FUJIFILM X-T4", "Fujifilm X-T4"), ("", "")):
        assert library.camera_name(written) == sold, (written, library.camera_name(written))


def check_lens_names() -> None:
    """Lenses by the names they are sold under (library.lens_name), the owner's four among them."""
    for written, camera, sold in (("E 35-150mm F2.0-F2.8 A058", "Sony Alpha 7 IV", "Tamron 35-150mm F/2-2.8 Di III VXD"),
                                  ("TAMRON 25-200mm F2.8-5.6 A075 E", "Sony Alpha 7 IV", "Tamron 25-200mm F/2.8-5.6 Di III VXD G2"),
                                  ("FE 28-70mm F3.5-5.6 OSS II", "Sony Alpha 7 IV", "Sony FE 28-70mm F3.5-5.6 OSS II"),
                                  ("Canon EF-S 55-250mm f/4-5.6 IS STM", "Canon EOS 250D", "Canon EF-S 55-250mm f/4-5.6 IS STM"),
                                  ("E 18-50mm F2.8 A999", "Sony Alpha 6700", "E 18-50mm F2.8 A999"), ("", "", "")):
        assert library.lens_name(written, camera) == sold, (written, library.lens_name(written, camera))


def check_painted() -> None:
    """A painted mask (edit.painted): whole under the brush, fading over its feather, gone
    where rubbed out; a stroke going on painted onto what was there is the stroke painted
    whole; a mask that is off weighs nothing; a name is kept short."""
    line = {"size": 0.05, "feather": 0.5, "erase": False, "points": [[0.2, 0.5], [0.5, 0.5], [0.8, 0.5]]}
    mask = edit.valid_masks([{"kind": "brush", "strokes": [line], "name": "  Sky " + "x" * 60, "off": 1}])[0]
    assert mask["name"].startswith("Sky x") and len(mask["name"]) == 40 and mask["off"] is True, mask
    assert "off" not in edit.valid_masks([{"kind": "brush"}])[0] and "name" not in edit.valid_masks([{"kind": "brush", "name": " "}])[0]
    assert edit.mask_weight(mask, [0.5, 0.5], 1.5) == 0, "a mask that is off weighs nothing"
    on = {**mask, "off": False}
    weigh = lambda x, y, m=on: edit.mask_weight(m, [x, y], 1.5)  # noqa: E731
    # (The brush's radius is 0.05 of the width: 0.075 of the height of a 3:2 picture.)
    assert abs(weigh(0.5, 0.5) - 1) < 0.01 and abs(weigh(0.35, 0.5 + 0.02) - 1) < 0.01, "under the brush's core"
    assert weigh(0.5, 0.5 + 0.09) == 0 and weigh(0.9, 0.5) == 0, "beyond the brush"
    assert abs(weigh(0.5, 0.5 + 0.075 * 0.75) - 0.5) < 0.04, f"half way through the feather: {weigh(0.5, 0.5 + 0.075 * 0.75)}"
    assert abs(weigh(0.5, 0.5, {**on, "invert": True})) < 0.01
    rubbed = {**on, "strokes": [line, {**line, "erase": True, "feather": 0.0, "points": [[0.5, 0.5]]}]}
    assert weigh(0.5, 0.5, rubbed) < 0.01 and abs(weigh(0.25, 0.5, rubbed) - 1) < 0.01, "rubbed out in one spot only"
    going = edit.Painter(1.5)
    going.begin()
    going.add({**line, "points": line["points"][:2]})
    going.bytes()  # (as the picture is made at every move: then only what the rest adds is made anew)
    going.add(line, 2)
    assert (edit.painted([line], 1.5) == going.bytes()).all(), "a stroke going on is the stroke painted whole"
    # Two soft edges lying over each other run into one another: more there than either has
    # (with the larger of the two they met in a crease), and never more than whole.
    side = {**line, "points": [[0.2, 0.5 + 0.11], [0.8, 0.5 + 0.11]]}
    one, two = (edit.painted(strokes, 1.5).astype(int) for strokes in ([line], [line, side]))
    other = edit.painted([side], 1.5).astype(int)
    between = (slice(None), 1024)
    assert (two[between] >= np.maximum(one[between], other[between])).all() and two.max() == 255
    assert (two[between] > np.maximum(one[between], other[between])).any(), "two soft edges do not run into one another"
    # ... but not along one stroke: its band is as soft at a joint as anywhere.
    bent = edit.painted([{**line, "points": [[0.2, 0.5], [0.35, 0.5], [0.5, 0.5], [0.8, 0.5]]}], 1.5)
    assert np.abs(bent.astype(int) - one).max() <= 1, "a stroke is harder where two of its stretches meet"
    numbers = edit.masks_uniforms(edit.valid_masks([{"kind": "radial", "at": [0.5, 0.5], "to": [0.7, 0.5]}, {"kind": "brush"}, {"kind": "brush", "invert": True}, {"kind": "brush", "off": True}]), 1.5)
    assert numbers["masksA0"][4:8] == [0.0, 0.0, 0.0, 2.0] and numbers["masksA0"][8:12] == [1.0, 0.0, 0.0, 6.0], numbers["masksA0"]
    assert numbers["masksA0"][12:16] == [0.0, 0.0, 2.0, 0.0], "a mask that is off is an empty slot"
    for bad in ([{"kind": "brush"}] * (edit.BRUSHES + 1), [{"kind": "brush", "strokes": [{**line, "points": []}]}],
                [{"kind": "brush", "strokes": [{**line, "points": [[0, "a"]]}]}], [{"kind": "brush", "strokes": [{**line, "size": None}]}]):
        assert edit.clean({"masks": bad}) == {}, f"masks {bad!r} must be refused"
    assert not edit.layers({"masks": [{**on, "off": True, "edit": {"exposure": 1.0}}]}, None), "an off mask changes nothing"


def check_local() -> None:
    """A mask's own edit (edit.LOCAL): cleaned, added to the whole image's, kept in range."""
    mask = {"kind": "radial", "at": [0.5, 0.5], "to": [0.7, 0.5]}
    kept = edit.valid_masks([{**mask, "edit": {"exposure": 0.5, "contrast": 0, "saturation": 500, "nonsense": 3}}])[0]
    assert kept["edit"] == {"exposure": 0.5, "saturation": 100.0}, kept
    assert "edit" not in edit.valid_masks([{**mask, "edit": {"exposure": 0}}])[0], "an empty edit is not kept"
    camera = {"shot": (5200.0, 10.0), "match": [[0.0, 0.0], [0.4, 0.5], [1.0, 1.0]]}
    whole = {"exposure": 1.0, "saturation": 150.0, "contrast": 10.0}
    inside = edit.merged(whole, {"exposure": 0.5, "saturation": 100.0, "temperature": -700.0, "tint": 5.0}, camera)
    assert inside == {"exposure": 1.5, "saturation": 200.0, "contrast": 10.0, "temperature": 4500.0, "tint": 15.0}, inside
    # (Where the whole image says nothing of the exposure, its 0 is added to; and inside, the
    # photo's own curve holds as it does outside.)
    assert edit.merged({}, {"exposure": -1.0}, camera)["exposure"] == -1.0
    assert edit.uniforms(edit.merged({}, {"exposure": -1.0}, camera), {**CAMERA, **camera}, True)["curveY"][1] == 0.5
    # A curve or the EQ of a mask's own holds in place of the whole image's; without one, that holds.
    bent = [[0.0, 0.0], [0.5, 0.7], [1.0, 1.0]]
    own = edit.valid_masks([{**mask, "edit": {"curve": bent, "eq": [{"hue": 120, "gain": 150, "width": 40}]}}])[0]["edit"]
    assert own["curve"] == bent and own["eq"][0]["gain"] == 150.0, own
    both = edit.merged({"curve": [[0.0, 0.1], [1.0, 1.0]], "chroma": bent}, own, camera)
    assert both["curve"] == [[0.0, 0.1], [1.0, 1.0]] and both["chroma"] == bent and "eq" not in both, "the whole image's stay"
    assert both["over"] == {"curve": bent, "eq": own["eq"]}, "and the mask's own come on top"
    over, none = edit.uniforms(both, None, True), edit.uniforms({}, None, True)
    assert over["curve2On"] == 1 and over["curve2Y"][:3] == [0.0, 0.7, 1.0] and over["chroma2On"] == 0 and over["eq2Count"] == 1 and over["eq2A0"][:3] == [120.0, 0.5, 40.0]
    assert over["curveY"][:2] == [0.1, 1.0] and over["chromaOn"] == 1 and over["eqCount"] == 0 and over["look"] == 1
    assert edit.uniforms({"over": {"eq": own["eq"]}}, None, True)["look"] == 1, "a mask's EQ alone is a look too"
    assert none["curve2On"] == none["chroma2On"] == none["eq2Count"] == 0 and len(none["curve2X"]) == len(none["eq2A1"]) == 16
    photo = {"masks": [kept, edit.valid_masks([mask])[0]], "contrast": 5.0}
    found = edit.layers(photo, camera)
    assert [n for n, _ in found] == [1] and found[0][1]["contrast"] == 5.0 and found[0][1]["saturation"] == 200.0, found


def check_unsaved() -> None:
    """Which settings show as changed since the last checkpoint (edit.unsaved)."""
    assert edit.unsaved([], {}, None) == []
    assert edit.unsaved([], {"exposure": 0.5}, None) == ["exposure"]  # no checkpoint: all that moved
    one = {"id": 1, "edit": {"exposure": 0.5}}
    two = {"id": 2, "edit": {"exposure": 0.5, "contrast": 20.0}}
    assert edit.unsaved([one], {"exposure": 0.5}, 1) == []  # standing at it
    assert edit.unsaved([one, two], {"exposure": 0.5, "contrast": 30.0}, 2) == ["contrast"]
    assert edit.unsaved([one, two], {"exposure": 0.7, "contrast": 20.0}, 1) == ["contrast", "exposure"]  # from 1
    assert edit.unsaved([one, two], {"exposure": 0.5, "contrast": 25.0}, None) == ["contrast"]  # newest


def check_folders_fast() -> None:
    """Gone through quickly, the folders leave nothing behind: the photo of one that was left
    before its decoding began is not decoded (each was, and the one stopped at waited behind
    them all), an edit not yet written goes to its own photo, and the open photo is one of
    the new ones before anybody is told of them."""
    gate, begun = threading.Event(), []

    def slow(path: Path, half: bool):
        begun.append(path.name)
        gate.wait(10)
        return chart(path, half)

    saved = []
    keeper = type("Keeper", (), {"save": lambda self, key, photo, values, points=(): saved.append(photo.name),
                                 "save_steps": lambda self, key, *kept: None, "load": lambda self, key, photo: {},
                                 "checkpoints": lambda self, key: [], "steps": lambda self, key: None})()
    session = app.Session([Path(f"/a/{n}.jpg") for n in range(9)], load=slow)
    session.goTo(8)
    session._unwritten, session.catalog, paths = True, keeper, []
    session.listed.connect(lambda: paths.append(session.path))  # (raised, out of the open photo's place)
    for folder in "bcdefg":
        session.openAt([f"/{folder}/0.jpg", f"/{folder}/1.jpg"], 1)
    session.catalog = None
    assert saved == ["8.jpg"], f"the edit of /a/8.jpg went to {saved}"
    assert paths == [f"/{folder}/1.jpg" for folder in "bcdefg"], f"told of the new photos, the open one was {paths}"
    gate.set()
    session.threads.shutdown(wait=True)
    assert begun.count("1.jpg") <= 2 and len(begun) <= 3, f"six folders gone through at once: {len(begun)} photos decoded"


def check_drop_last() -> None:
    """The open photo taken out when it was the last: the one before it is open before
    anybody is told (QML asked for the old place among fewer photos: IndexError)."""
    session = app.Session([Path(f"/a/{n}.jpg") for n in range(3)], load=chart)
    session.goTo(2)
    paths = []
    session.listed.connect(lambda: paths.append(session.path))
    session.drop(["/a/2.jpg"])
    session.threads.shutdown(wait=True)
    assert paths == ["/a/1.jpg"], f"told of the photos left, the open one was {paths}"

def check_one_count_waits() -> None:
    """Renders to be counted do not queue: while one count runs, only the latest waits (they
    used to pile up for as long as something was dragged, and the program lagged)."""
    session = app.Session([])
    gate, runs, inner = threading.Event(), [], session._count
    session._count = lambda: (runs.append(1), inner())
    session.counting.submit(gate.wait)  # the counting thread is busy
    for _ in range(6):
        session.setScope(QImage(768, 512, QImage.Format.Format_RGBA8888))
    gate.set()
    session.counting.submit(lambda: None).result(10)
    assert len(runs) == 1 and session._wanted is None, f"{len(runs)} counts for six renders in a row"
    # (The session must not be let go of on the counting thread: its timers belong here.)
    session.counting.shutdown(wait=True)
    del session._count


def check_where_i_was() -> None:
    """The library remembers the place on show and the photo last looked at, and starts there
    (the owner: "When i reopen the app, i want the same folder and photo selected"); a folder
    since taken out of the library is not gone back to; and the session whether the window
    was in Edit."""
    with tempfile.TemporaryDirectory() as folder:
        one, two = Path(folder) / "one", Path(folder) / "two"
        for where in (one, two):
            where.mkdir()
            photo_file(where, "a.arw", 1), photo_file(where, "b.arw", 2)
        kept = catalog.Catalog(Path(folder) / "catalog.db")
        for where in (one, two):
            kept.add_folder(where)
            kept.found(*library.scan(where, {}))
        settings = QSettings(str(Path(folder) / "settings.ini"), QSettings.Format.IniFormat)

        def opened() -> library.Library:
            made = library.Library(kept, Path(folder) / "cache", settings)
            made.rest(True)
            return made

        first = opened()
        assert first.where == "all" and first.lastPhoto == "" and first.rowOf("/nowhere.arw") == -1
        first.show(f"folder:{two}")
        first.lookAt(1)
        looked = first.model.rows[1]["path"]
        assert looked.startswith(str(two)) and first.rowOf(looked) == 1
        first.show("no such place")
        first.lookAt(99)
        # Each place keeps the photo last looked at there, also over a restart.
        first.show(f"folder:{one}")
        assert first.rowLast() == -1
        first.lookAt(0)
        first.show(f"folder:{two}")
        assert first.rowLast() == 1, first.rowLast()
        first.close()
        between = opened()
        between.show(f"folder:{one}")
        assert between.rowLast() == 0
        between.show(f"folder:{two}")
        assert between.rowLast() == 1
        between.lookAt(1)
        between.close()
        again = opened()
        assert again.where == f"folder:{two}" and again.count == 2, f"the library does not start where it was: {again.where}"
        assert again.lastPhoto == looked and again.rowOf(again.lastPhoto) == 1, "nor at the photo last looked at"
        again.lookingAt(str(one / "a.arw"))
        again.show("month:2020-01")
        again.removeFolder(str(two))
        again.close()
        third = opened()
        assert third.where == "month:2020-01" and third.lastPhoto == str(one / "a.arw")
        third.show(f"folder:{one}")
        third.close()
        kept.remove_folder(one)
        assert opened().where == "all", "a folder that is no longer in the library is not gone back to"
        session = app.Session([], settings=settings)
        assert session.panels["place"] == "library"
        session.setPanel("place", "edit")
        assert app.Session([], settings=settings).panels["place"] == "edit", "whether the window was in Edit is remembered"
        kept.close()


def check_library_rests() -> None:
    """While a photo is being edited the library does nothing: it begins no scan and makes no
    thumbnail, and takes both up when it is on show again. And what it is asked about one
    photo it looks up, it does not search for."""
    def wait(done, seconds: float = 8.0) -> bool:
        until = time.time() + seconds
        while not done() and time.time() < until:
            QCoreApplication.processEvents()
            time.sleep(0.005)
        return done()

    with tempfile.TemporaryDirectory() as folder:
        root, cache = Path(folder) / "photos", Path(folder) / "cache"
        root.mkdir()
        for n in range(6):
            picture_file(root / f"{n}.jpg", exif("X", "Y", f"2020:01:0{n + 1} 10:00:00", 1))
        kept = catalog.Catalog(Path(folder) / "catalog.db")
        kept.add_folder(root)
        shown = library.Library(kept, cache)
        shown.setFilter("raw", False)  # (these are JPEGs: the library starts with raw files only)
        thumbs = lambda: len(list(cache.glob("*.jpg"))) if cache.is_dir() else 0  # noqa: E731
        shown.rest(True)
        shown.rescan()
        assert not shown.busy and not wait(lambda: shown.total > 0, 0.5), "resting, the library scanned its folders"
        shown.take(str(Path(folder) / "elsewhere.jpg")), shown.take(str(root / "none.jpg"))
        assert shown.total == 0, "a file outside the library's folders, or one that is not there, was taken in"
        kept.save(library.identify(root / "3.jpg"), root / "3.jpg", {"exposure": 1.0})
        shown.take(str(root / "3.jpg"))  # (a denoised copy just written: known at once, edit and all)
        assert shown.total == 1 and shown.marksOf(str(root / "3.jpg")).get("edited") is True, "resting, a file just written is not one of the library's photos"
        handed, library.BATCH = [], 2  # a folder fills while it is being read, not all at the end
        rest, _ = library.scan(root, {}, lambda rows: handed.append(len(rows)))
        library.BATCH = 100
        assert handed == [2, 2, 2] and rest == [], f"a scan hands on what it finds as it goes: {handed}, {len(rest)} left"
        shown.rest(False)
        assert wait(lambda: shown.total == 6 and not shown.busy), "on show again, it did not take up the scan"
        other = Path(folder) / "more"
        other.mkdir()
        picture_file(other / "x.jpg", exif("X", "Y", "2021:01:01 10:00:00", 1))
        shown.addFolder(str(other))
        assert shown.where == f"folder:{other}", f"a folder just added is not the one on show: {shown.where}"
        assert wait(lambda: shown.total == 7 and not shown.busy) and shown.count == 1
        shown.removeFolder(str(other))
        shown.show("all")
        assert shown.total == 6 and shown.count == 6
        assert wait(lambda: thumbs() == 7), f"on show, the thumbnails are made without being asked for: {thumbs()} of 7"
        for made in cache.glob("*.jpg"):
            made.unlink()
        shown.rest(True)
        shown.asked.clear()
        shown._news()  # (the grid is filled anew: no thumbnail is there)
        for row in range(6):
            shown.want(row)
        assert not wait(lambda: thumbs() > 0, 0.5) and len(shown.waiting) == 6 and shown.making == 0, \
            f"resting, the library made {thumbs()} thumbnails"
        shown.want(4)
        told = []
        shown.model.dataChanged.connect(lambda first, last: told.append(first.row()))
        shown.rest(False)
        assert wait(lambda: thumbs() == 6 and shown.making == 0) and not shown.waiting, "on show again, they are made"
        assert wait(lambda: sorted(told) == [0, 1, 2, 3, 4, 5]), f"the grid is told of each thumbnail, at its own row: {told}"
        # The filmstrip's thumbnails are made while the library rests, but one at a time; also
        # for a photo the library does not have; and nothing else is made meanwhile.
        for made in cache.glob("*.jpg"):
            made.unlink()
        shown.rest(True)
        shown.asked.clear()
        shown._news()
        outside = picture_file(Path(folder) / "elsewhere.jpg", exif("X", "Y", "2022:01:01 10:00:00", 1))
        most = [0]
        shown.made.connect(lambda name, done: most.__setitem__(0, max(most[0], shown.making)))
        shown.want(0)  # (the grid's wish: it waits)
        assert shown.thumbFor(shown.model.rows[1]["path"]) == "" and shown.thumbFor(shown.model.rows[2]["path"]) == ""
        assert shown.thumbFor(str(outside)) == "" and shown.thumbFor("/nowhere.jpg") == "" and shown.making == 1
        assert wait(lambda: thumbs() == 3 and shown.making == 0), f"the strip's three thumbnails, and only those: {thumbs()}"
        assert most[0] <= 1 and len(shown.waiting) == 1, f"resting, {most[0]} thumbnails were being made at a time"
        assert shown.thumbFor(str(outside)).endswith(".jpg") and shown.thumbs == 3
        shown.rest(False)
        for row in range(6):
            shown.want(row)
        assert wait(lambda: thumbs() == 7 and shown.making == 0), f"on show again, the rest is made: {thumbs()} of 7"
        rows = shown.model.rows
        # The open photo as it is edited is kept as its thumbnail: written anew, under another
        # address so that Qt reads it again, and the grid told at that photo's row.
        was = shown.thumbFor(rows[1]["path"])
        dark = QImage(60, 40, QImage.Format.Format_RGBA8888)
        dark.fill(QColor(12, 12, 12))
        told.clear()
        shown.keep(rows[1]["path"], dark)
        shown.keep(str(outside), dark)
        shown.keep("/nowhere.jpg", dark)
        assert wait(lambda: shown.thumbFor(rows[1]["path"]) == was + "?1" and told == [1]), (shown.thumbFor(rows[1]["path"]), told)
        kept_ = QImage(str(cache / f"{rows[1]['thumb']}.jpg"))
        assert (kept_.width(), kept_.height()) == (60, 40) and kept_.pixelColor(30, 20).green() < 20, "the thumbnail kept is not the picture handed over"
        assert wait(lambda: shown.thumbFor(str(outside)).endswith("?1")), "a photo the library does not have keeps its thumbnail too"
        assert shown.model.data(shown.model.index(1), Qt.ItemDataRole.UserRole + 3).endswith("?1") and thumbs() == 7
        assert shown.model.named[rows[4]["thumb"]] == [4] and shown.model.pathed[rows[2]["path"]] == 2
        shown.markPhoto(rows[2]["path"], "stars", 3)
        assert shown.marksOf(shown.model.rows[2]["path"])["stars"] == 3 and shown.marksOf("/nowhere.jpg") == {}
        assert all(shown.model.data(shown.model.index(n), Qt.ItemDataRole.UserRole + 3) for n in range(6)), "a thumbnail made is one the grid is told of"
        shown.close()
        shown.makers.shutdown(wait=True)  # (one may still be writing into the folder that is about to go)
        kept.close()


def check_missing() -> None:
    """A photo whose file is deleted stays, known as missing, until it is taken out: then it
    leaves the library and the row of photos at hand, and nothing read from it is left."""
    def wait(done, seconds: float = 8.0) -> bool:
        until = time.time() + seconds
        while not done() and time.time() < until:
            QCoreApplication.processEvents()
            time.sleep(0.005)
        return done()

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder) / "photos"
        root.mkdir()
        for n in range(4):
            picture_file(root / f"{n}.jpg", exif("X", "Y", f"2020:01:0{n + 1} 10:00:00", 1))
        loose = picture_file(Path(folder) / "elsewhere.jpg", exif("X", "Y", "2022:01:01 10:00:00", 1))
        kept = catalog.Catalog(Path(folder) / "catalog.db")
        kept.add_folder(root)
        kept.found(*library.scan(root, {}))
        shown = library.Library(kept, Path(folder) / "cache")
        shown.setFilter("raw", False)
        row = [Path(path) for path in shown.paths()] + [loose]  # (newest first: 3, 2, 1, 0, and one the library does not have)
        session = app.Session(row, load=chart, catalog=kept)
        shown.attend(session)
        session.goTo(1)
        gone, there = str(root / "1.jpg"), str(root / "0.jpg")
        assert not shown.missing(gone) and not shown.missing(str(loose)) and not shown.missing("")
        key = library.identify(Path(gone))
        kept.save(key, Path(gone), {"exposure": 1.0})
        kept.mark([key], "stars", 4)
        shown.made_here[gone] = {"of": "x"}  # (as if it were a denoised copy: what its file said)
        Path(gone).unlink()
        loose.unlink()
        assert wait(lambda: shown.missing(gone) and shown.missing(str(loose))), "a deleted file is not known as missing"
        assert session.count == 5 and shown.total == 4, "a missing photo left by itself"
        shown.remove([there])
        assert session.count == 5 and shown.total == 4, "a photo whose file is there was taken out"
        shown.remove([gone, there, str(loose)])
        assert shown.total == 3 and shown.rowOf(gone) == -1 and not shown.missing(gone) and gone not in shown.made_here
        assert [photo.name for photo in session.photos] == ["3.jpg", "2.jpg", "0.jpg"] and session.path == str(root / "2.jpg"), \
            f"the row after two were taken out, the open photo as it was: {session.photos}, {session.path}"
        assert kept.load(key, Path(gone)) == {"exposure": 1.0}, "its edit went with it"
        # The open photo taken out: the one after it is shown. Back on disk, it is the photo it was.
        Path(gone).write_bytes((root / "0.jpg").read_bytes())  # (another file under that name)
        (root / "2.jpg").unlink()
        assert wait(lambda: shown.missing(str(root / "2.jpg")))
        shown.remove([str(root / "2.jpg")])
        assert session.path == there and session.index == 1 and session.count == 2
        for file in (root / "3.jpg", root / "0.jpg"):
            file.unlink()
        assert wait(lambda: shown.missing(there) and shown.missing(str(root / "3.jpg")))
        shown.remove([there, str(root / "3.jpg")])
        assert session.count == 0 and session.path == "" and session.source == "" and shown.total == 0
        shown.close()
        kept.close()


def check_catalog_before_library() -> None:
    """A catalog from before the library gains its tables; what it was is kept beside it."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "catalog.db"
        old = sqlite3.connect(path)
        with old:
            old.execute("CREATE TABLE edits (key TEXT PRIMARY KEY, content TEXT NOT NULL, path TEXT, "
                        "edit TEXT NOT NULL, saved REAL NOT NULL, checkpoints TEXT NOT NULL DEFAULT '[]')")
            old.execute("INSERT INTO edits VALUES ('c1:aa:1', 'aa', '/a.arw', '{\"exposure\": 1.5}', 1, '[]')")
            old.execute("PRAGMA user_version=1")
        old.close()
        for _ in range(2):
            kept = catalog.Catalog(path)
            assert kept.load("c1:aa:1", Path("/a.arw")) == {"exposure": 1.5, **THEN} and kept.photos() == []
            kept.close()
        copy = sqlite3.connect(path.with_name("catalog.db.v1"))
        assert copy.execute("PRAGMA user_version").fetchone()[0] == 1, "as it was before"
        assert [r[0] for r in copy.execute("SELECT name FROM sqlite_master WHERE type='table'")] == ["edits"]
        copy.close()


def check_export_files() -> None:
    """Exported files (photone/export.py): names that never take another file's place, the
    tiles a render is made of, sixteen bits out of two passes, and what gets written."""
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder)
        first, second, third = (export.destination(out, "IMG_0001.ARW", ".jpg") for _ in range(3))
        assert [p.name for p in (first, second, third)] == ["IMG_0001.jpg", "IMG_0001-2.jpg", "IMG_0001-3.jpg"]
        assert all(p.exists() and p.stat().st_size == 0 for p in (first, second, third)), "each name is taken at once"
        (out / "holiday.tif").write_text("somebody else's")
        assert export.destination(out, "holiday.dng", ".tif").name == "holiday-2.tif"
        assert (out / "holiday.tif").read_text() == "somebody else's", "what is there stays as it is"

        assert export.tiles(100, 60, False, 64) == [[0, 0, 64, 60, 0], [64, 0, 36, 60, 0]]
        deep = export.tiles(100, 130, True, 64)
        assert len(deep) == 12 and deep[:2] == [[0, 0, 64, 64, 1], [0, 0, 64, 64, 2]] and deep[-1] == [64, 128, 36, 2, 2]
        assert sum(t[2] * t[3] for t in export.tiles(7028, 4688, False)) == 7028 * 4688, "every pixel once"

        value = np.array([[[0, 300, 65535], [258, 1, 40000]]], np.uint16)  # as the shader splits it
        one = np.stack([value[..., 0] >> 8, value[..., 0] & 255, value[..., 1] >> 8], axis=-1).astype(np.uint8)
        two = np.stack([value[..., 1] & 255, value[..., 2] >> 8, value[..., 2] & 255], axis=-1).astype(np.uint8)
        assert (export.deep_of(one, two) == value).all(), "two passes of bytes are the sixteen bits again"
        def filled() -> QImage:
            grab = QImage(64, 48, QImage.Format.Format_RGB32)
            grab.fill(QColor(10, 20, 30))
            return grab

        taken = [export.bytes_of(filled()) for _ in range(40)]  # each image gone when it is read
        assert taken[0].shape == (48, 64, 3) and all((t == [10, 20, 30]).all() for t in taken), "its own copy of the pixels"

        ramp = np.zeros((40, 300, 3), np.uint16)
        ramp[..., 1] = np.linspace(1000, 60000, 300).astype(np.uint16)[None, :]
        ramp[..., 0] = 65535
        for kind, bits in (("png16", 16), ("tiff16", 16), ("png8", 8), ("jpeg", 8)):
            path = export.destination(out, "ramp.arw", export.FORMATS[kind][0])
            pixels = ramp if bits == 16 else (ramp >> 8).astype(np.uint8)
            export.write(export.picture(pixels), path, kind, 95)
            back = QImage(str(path))
            assert (back.width(), back.height()) == (300, 40) and back.colorSpace().isValid(), kind
            read = back.convertToFormat(QImage.Format.Format_RGBA64)
            greens = np.frombuffer(read.constBits(), np.uint16).reshape(40, -1)[20, 1:300 * 4:4]
            assert abs(int(greens[150]) - int(ramp[20, 150, 1])) < (40 if bits == 16 else 900), (kind, greens[150])
            assert (len(set(greens.tolist())) > 256) == (bits == 16), f"{kind}: {len(set(greens.tolist()))} levels of green"
        small = export.picture((ramp >> 8).astype(np.uint8), 100)
        assert (small.width(), small.height()) == (100, 13) and export.picture(ramp, 9999).width() == 300
        try:
            export.write(export.picture(ramp), out / "no such folder" / "x.png", "png16")
        except export.Failed as said:
            assert "could not be written" in str(said) and not (out / "no such folder").exists()
        else:
            raise AssertionError("a file that cannot be written must be said")


def check_surface_choice() -> None:
    hdr = {"QSG_RHI_BACKEND": "vulkan", "QSG_RHI_HDR": "scrgb"}
    assert app.surface_environment("wayland", True, {}) == hdr
    assert app.surface_environment("wayland-egl", True, {}) == hdr
    assert app.surface_environment("xcb", True, {}) == {}, "X11 has no HDR: Qt's default stays"
    assert app.surface_environment("wayland", False, {}) == {}, "no Vulkan: Qt's default stays"
    assert app.surface_environment("offscreen", True, {}) == {}
    mine = {"QSG_RHI_BACKEND": "opengl"}
    assert app.surface_environment("wayland", True, mine) == {}, "the user's own choice wins"


def check_shaders() -> None:
    def block(file: str) -> list[str]:
        text = re.sub(r"//[^\n]*", "", (shaders.UI / file).read_text())
        body = re.search(r"uniform buf \{(.*?)\};", text, re.S).group(1)
        return [" ".join(line.split()) for line in body.split(";") if line.strip()]

    assert block("pipeline.vert") == block("pipeline.frag"), "both stages must declare the same uniforms, in order"
    assert "frame *" in (shaders.UI / "pipeline.vert").read_text(), "the photo's frame is applied where the item's corners are"
    # Every uniform has a property of exactly its name on the item that draws, and everything
    # edit.uniforms works out is a uniform: a name that differs is silently zero in the shader.
    names = {line.split()[-1] for line in block("pipeline.frag")} - {"qt_Matrix", "qt_Opacity"}
    item = (shaders.UI / "Pipeline.qml").read_text()
    missing = [n for n in sorted(names) if not re.search(rf"property \w+ {n}\b", item)]
    assert not missing, f"Pipeline.qml has no property for {missing}"
    assert set(edit.uniforms({}, None, True)) | {"frame"} <= names, set(edit.uniforms({}, None, True)) - names
    assert block("wheel.vert") == block("wheel.frag"), "the wheel's two stages likewise"
    assert '#include "' not in shaders.whole("pipeline.frag") and "rootsOf" in shaders.whole("wheel.frag")
    for file in ("pipeline.frag", "pipeline.vert", "wheel.frag", "wheel.vert"):
        built = shaders.compiled(file)
        assert built.stat().st_size > 500, f"{file} compiled to something"
        assert shaders.compiled(file) == built, "unchanged source, same package"


def check_kept() -> None:
    """Only the part of a raw picture that the file calls the photo is kept, whichever way up."""
    from types import SimpleNamespace as Sizes

    canon = dict(width=5208, height=3476, left_margin=72, top_margin=52, crop_left_margin=84,
                 crop_top_margin=64, crop_width=5184, crop_height=3456)  # 12 left, top, right; 8 below
    span = lambda cut: (cut.start, cut.stop)  # noqa: E731
    rows, columns = decode.kept(Sizes(**canon, flip=0), False)
    assert (span(rows), span(columns)) == ((12, 3468), (12, 5196)), (rows, columns)
    rows, columns = decode.kept(Sizes(**canon, flip=0), True)
    assert (span(rows), span(columns)) == ((6, 1734), (6, 2598)), (rows, columns)
    # Turned, the margins turn with it: a picture of the margins themselves, turned by numpy.
    sensor = np.zeros((3476, 5208), bool)
    sensor[12:3468, 12:5196] = True
    for flip, turns in ((3, 2), (5, 1), (6, 3)):
        assert np.rot90(sensor, turns)[decode.kept(Sizes(**canon, flip=flip), False)].all(), flip
        assert np.rot90(sensor, turns)[decode.kept(Sizes(**canon, flip=flip), False)].size == sensor.sum(), flip
    none = dict(canon, crop_left_margin=65535, crop_top_margin=65535, crop_width=0, crop_height=0)
    rows, columns = decode.kept(Sizes(**none, flip=5), False)
    assert (span(rows), span(columns)) == ((0, 5208), (0, 3476)), "no part named: all of it"


def check_denoise() -> None:
    """The neural denoiser around its net: the shares mixed back, the DNG written and read as
    Photone shows a raw file, known as ours (never denoised twice), and no arithmetic noise
    reduction on it. (The net itself runs on a real file: PHOTONE_SAMPLE.)"""
    cam_xyz = [[0.746, -0.2365, -0.0588], [-0.5687, 1.3442, 0.2474], [-0.0624, 0.1156, 0.6584]]
    about = {"wb": [2.4, 1.0, 1.6], "cam_xyz": cam_xyz, "a": 2.0, "b": 30.0, "room": 15868.0,
             "match": [[0.0, 0.0], [0.3, 0.36], [1.0, 1.0]], "lens": {"distortion": [1, 2, 3]}, "turned": 8, "preview": None}
    rng = np.random.default_rng(3)
    # (Grays with a little color: a color no camera records has no place in the net's terms.)
    noisy = (rng.random((30, 44, 1)) * 30000 + 2000 + rng.random((30, 44, 3)) * 1500).astype(np.uint16)
    held = denoise.Denoised(noisy, about)
    # Past the photo's edge a piece is the photo mirrored, never one row of it again and again.
    ramp = np.arange(40 * 30 * 3, dtype=np.float32).reshape(40, 30, 3)
    piece = denoise._piece(ramp, 0, 0, 256)
    at = denoise.MARGIN
    assert piece.shape == (256, 256, 3) and (piece[at:at + 40, at:at + 30] == ramp).all()
    assert (piece[at - 3, at:at + 30] == ramp[3]).all() and (piece[at + 42, at:at + 30] == ramp[36]).all(), "mirrored in the edge"
    assert (piece[at:at + 40, at - 3] == ramp[:, 3]).all() and (piece[1:] != piece[:-1]).any(axis=(1, 2)).all(), "no row twice running"
    whole = slice(0, 30)
    v = held.into(noisy)
    assert np.allclose(held.camera(v) @ held.from_cam.T, noisy / 65535, atol=2e-5), "into the net's terms and back"
    held.clean[...] = v + rng.standard_normal(v.shape).astype(np.float32)  # (what a net might take: anything)
    assert np.allclose(held.mixed(whole, 0, 0), held.camera(v), atol=1e-5), "nothing kept away: the photo as it was"
    assert np.allclose(held.mixed(whole, 1, 1), held.camera(held.clean), atol=1e-5), "all kept away: the net's result"
    took = v - held.clean
    grain_back = held.clean + took.mean(2, keepdims=True)  # color noise away, the grain back
    assert np.allclose(held.mixed(whole, 1, 0), held.camera(grain_back), atol=1e-5), "Grain is what the colors have in common"
    assert np.allclose(held.picture()[..., :3], np.clip(held.camera(held.clean) @ held.from_cam.T, 0, 1), atol=2e-3)
    with tempfile.TemporaryDirectory() as folder:
        raw = Path(folder) / "shot.ARW"
        # The original's header, as a camera writes one: a first directory, EXIF with the
        # maker's notes, GPS. All of it is in the denoised file's.
        # (Sony's notes: a directory of two tags, one pointing at 8 bytes within the notes
        # by their place in the file, which is another in the new file.)
        def sony(at: int) -> bytes:
            return (b"SONY DSC \0\0\0" + struct.pack("<HHHIIHHII", 2, 0x102, 3, 1, 7, 0xb027, 4, 2, at + 38)
                    + b"lenstype")

        notes = sony(0)
        first = {0x10f: (2, "SONY"), 0x110: (2, "ILCE-7M4"), 0x8769: (4, 0), 0x8825: (4, 0)}
        exif = {0x8827: (3, 6400), 0x829a: (5, [0.004]), 0x9204: (10, [-1.3]), 0x927c: (7, notes),
                0x9003: (2, "2026:07:10 21:53:15"), 0xa434: (2, "A lens"), 0xa005: (4, 8)}
        gps = {0: (1, bytes([2, 3, 0, 0])), 2: (5, [50, 5, 12.5])}
        exif_at = 8 + len(dng._directory(first, 8))
        first[0x8769], first[0x8825] = (4, exif_at), (4, exif_at + len(dng._directory(exif, 0)))
        notes_at = exif_at + dng._directory(exif, exif_at).find(notes)
        notes = exif[0x927c] = (7, sony(notes_at))
        raw.write_bytes(b"II*\0" + struct.pack("<I", 8) + dng._directory(first, 8)
                        + dng._directory(exif, exif_at) + dng._directory(gps, first[0x8825][1]))
        about["header"] = dng.carried(raw)
        assert about["header"]["exif"][0x927c] == (7, 46, notes[1]) and about["header"]["notes"] == (notes_at, "<")
        assert 0xa005 not in about["header"]["exif"]
        other = Path(folder) / "other.NEF"  # (the other byte order: turned as it is read)
        other.write_bytes(b"MM\0*" + struct.pack(">IHHHIHHI", 8, 1, 0x8827, 3, 1, 6400, 0, 0))
        with other.open("rb") as file:
            assert dng._entries(file, 8, ">")[0x8827] == (3, 1, struct.pack("<H", 6400)), "big-endian"
        other.unlink()
        assert denoise.target(raw).name == "shot-Denoised.dng"
        out = denoise.save(held, raw, 0.7, 0.4)
        assert out.name == "shot-Denoised.dng" and denoise.target(raw).name == "shot-Denoised-2.dng", "never written over"
        assert not list(Path(folder).glob("*.part"))
        made = dng.made_here(out)
        assert made == {"model": denoise.MODEL.stem, "color": 0.7, "grain": 0.4, "from": "shot.ARW",
                        "of": catalog.identify(raw)}, made
        assert made["of"].startswith("c1:") and made["of"] != catalog.identify(out), "the original's fingerprint, not its own"
        assert dng.made_here(raw) is None and dng.made_here(Path(folder) / "none.dng") is None
        info, pixels = decode.load(out, half=False)
        # (Lying on its side in the file, as the sensor had it: upright again when read.)
        assert (info["width"], info["height"], info["full"]) == (44, 30, [44, 30]), info
        assert info["raw"] and info["denoised"] and info["match"] == about["match"], info
        assert info["camera"] == {"wb": about["wb"], "cam_xyz": cam_xyz, "lens": about["lens"], "denoised": True}
        seen = np.frombuffer(pixels.buf, np.float16).reshape(30, 44, 4)
        want = np.clip(held.mixed(whole, 0.7, 0.4) @ held.from_cam.T, 0, 1)
        assert np.allclose(seen[..., :3], want, atol=2e-3) and (seen[..., 3] == 1).all(), "what was written is what is shown"
        half, _ = decode.load(out, half=True)
        assert (half["width"], half["height"], half["full"], half["size"]) == (22, 15, [44, 30], "half"), half
        again, whole = dng.carried(out), out.read_bytes()
        assert again["gps"] == about["header"]["gps"] and {**again["exif"], 0x927c: 0} == {**about["header"]["exif"], 0x927c: 0}, "the whole header"
        now = again["notes"][0]
        assert now != notes_at and again["exif"][0x927c][2] == sony(now) and whole[now + 38:now + 46] == b"lenstype", "the maker's notes point at themselves where they now lie"
        said = library.describe(out)
        assert said["camera"] == "SONY ILCE-7M4" and said["bias"] == -1.3, said
        assert library.about(out)["iso"] == "ISO 6400" and library.about(out)["shutter"] == "1/250 s", library.about(out)
        for wrong in (lambda: denoise.develop(out), lambda: denoise.develop(Path(folder) / "a.jpg")):
            try:
                wrong()
                raise AssertionError("denoised what must not be")
            except ValueError as e:
                assert "already denoised" in str(e) or "Only raw files" in str(e), e
        # What the window does once the file is written: the original's edit is the new file's
        # too, and the new file stands after its original among the open photos.
        kept = catalog.Catalog(Path(folder) / "catalog.db")
        kept.save(catalog.identify(raw), raw, {"contrast": 20.0, "denoise": 40.0})
        session = app.Session([raw, Path(folder) / "next.ARW"], load=lambda path, half: decode.synthetic(), catalog=kept)
        window, told = denoise.Denoiser(session), []
        window.saved.connect(told.append)
        assert window.fits(str(raw)) and not window.fits(str(out)) and not window.fits("a.jpg"), "only a raw file not yet denoised"
        window._path = raw
        window._written(out)
        assert kept.load(catalog.identify(out), out) == kept.load(catalog.identify(raw), raw) != {}, "the edit is copied"
        assert told == [str(out)] and window.state == "" and window.held is None
        session.openBeside(str(out))
        assert [p.name for p in session.photos] == ["shot.ARW", "shot-Denoised.dng", "next.ARW"] and session.path == str(out)
        session.flush()
        # In the library's order a copy stands straight after its original, by what it carries.
        rows = [{"key": "k-copy", "path": str(out)}, {"key": "k-other", "path": "b.ARW"},
                {"key": catalog.identify(raw), "path": str(raw)}, {"key": "k-last", "path": "z.dng"}]
        order = [r["key"] for r in library.paired(rows, lambda path: dng.made_here(Path(path)))]
        assert order == ["k-other", catalog.identify(raw), "k-copy", "k-last"], order
        assert library.paired(rows[:2], lambda path: dng.made_here(Path(path))) == rows[:2], "no original among them: as it was"
        # Packed and unpacked, a strip is what it was, to a half float (also past one strip's rows).
        tall = rng.random((dng.STRIP + 9, 7, 3)).astype(np.float32) * 1.2 - 0.1
        assert (dng._unpack(dng._pack(tall), len(tall), 7) == tall.astype(np.float16)).all()
        cut = Path(folder) / "cut.dng"
        cut.write_bytes(out.read_bytes()[:-100])
        try:
            decode.load(cut)
            raise AssertionError("a file cut short was shown")
        except decode.Unreadable as e:
            assert "unpack" in str(e) or "cut short" in str(e), e
    numbers = edit.uniforms({"denoise": 50.0, "denoiseColor": 50.0, "sharpen": 40.0}, info["camera"], True, wide=44)
    assert numbers["denoise"] == numbers["denoiseColor"] == numbers["denoiseFar"] == 0 and numbers["sharpen"] > 0, "no noise reduction on a denoised file"
    assert edit.uniforms({"denoise": 50.0}, {**info["camera"], "denoised": False}, True, wide=44)["denoise"] > 0
    assert denoise.known((4688, 7028), 12800) == (12800 * 0.002175, 1114.4) and denoise.known((10, 10), 100) is None
    a, b = denoise.of_mosaic(rng.normal(500, 10, (640, 960)))  # (flat, noise of 10: all of it is b)
    assert 60 < a * 500 + b < 160, (a, b)
    assert denoise.MODEL.exists() and denoise.MODEL.stat().st_size > 1 << 20, "the net comes with the program"


def check_denoise_refused() -> None:
    """The net on a card that refuses a piece now and then (its memory full for a moment)."""
    # A piece the GPU refuses is done on the processor, and the next is tried on the GPU again;
    # three refusals in a row and the GPU is left alone.
    class Card:
        def __init__(self, refuses): self.refuses, self.asked = set(refuses), 0
        def run(self, _, feed):
            self.asked += 1
            if self.asked in self.refuses:
                raise RuntimeError("Failed to allocate memory")
            return [feed["x"] + 1]
    net = denoise._Net.__new__(denoise._Net)
    net.fast, net.on, net.slow = Card({2}), "CUDA", Card(())
    net.refused, net.pieces, net.slowly, net.why = 0, 0, 0, ""
    for _ in range(4):
        assert (net(np.zeros(2), None) == 1).all()
    assert (net.fast.asked, net.slow.asked) == (4, 1) and net.where == "CUDA, and the processor for 1 of 4 pieces", net.where
    net.fast, net.refused, net.pieces, net.slowly = Card({1, 2, 3, 4, 5}), 0, 0, 0
    for _ in range(6):
        net(np.zeros(2), None)
    assert net.fast.asked == 3 and net.where == "the processor", (net.fast.asked, net.where)


def check_denoise_exact() -> None:
    """The denoiser's arithmetic, made quicker, is the same to the last bit as written plainly
    beside it here: the median of six, the noise read off a mosaic, the photo in the net's
    terms and back, the shares mixed back, and the pieces cut ahead."""
    rng = np.random.default_rng(11)
    six = np.concatenate([np.array(list(product((0.0, 1.0), repeat=6))), rng.random((3000, 6))])
    six = six.astype(np.float32)
    six[::7, 3] = six[::7, 1]  # (ties too)
    assert (denoise._median_of_six(six) == np.median(six, axis=-1)).all(), "the median of six"

    def plain(values, share=1 / 3):  # of_mosaic as first written: a mask of all of it per band
        v = values.astype(np.float32)
        h, w = v.shape[0] // 2 * 2, v.shape[1] // 16 * 16
        runs = [v[dy:h:2, dx:w:2].reshape(h // 2, w // 16, 8) for dy, dx in product((0, 1), (0, 1))]
        signal = np.concatenate([r.mean(axis=2).ravel() for r in runs])
        spread = np.concatenate([np.median(np.abs(r[:, :, 2:] - r[:, :, :-2]), axis=2).ravel()
                                 / 0.6745 / np.sqrt(2) for r in runs])
        edges = np.unique(np.percentile(signal, np.linspace(0.5, 99.5, 31)))
        xs, vs = [], []
        for low, high in pairwise(edges):
            here = (signal >= low) & (signal < high)
            if here.sum() >= 300:
                calm = np.argsort(spread[here])[:int(here.sum() * share)]
                xs.append(signal[here][calm].mean())
                vs.append((spread[here][calm].mean() / denoise._calmest(share)) ** 2)
        xs, vs = np.array(xs, float), np.array(vs, float)
        a, b = np.polyfit(xs, vs, 1, w=1 / np.maximum(vs, 1e-9)) if len(xs) > 1 else (0.0, 0.0)
        if a <= 0:
            a, b = 1e-6, float(np.median(vs)) if len(vs) else 1.0
        return float(a), float(max(b, 1e-6))

    y, x = np.mgrid[0:480, 0:960]
    scene = 800 + 3000 * x / 960 + 400 * np.sin(y / 9)
    for mosaic in (scene + rng.standard_normal(scene.shape) * np.sqrt(scene + 40),  # (lit unevenly)
                   (scene + rng.standard_normal(scene.shape) * 30).astype(np.uint16),  # (whole numbers: ties)
                   rng.normal(500, 10, (640, 960)), np.zeros((40, 30))):  # (flat; too small for a band)
        assert denoise.of_mosaic(mosaic) == plain(mosaic), (denoise.of_mosaic(mosaic), plain(mosaic))

    cam_xyz = [[0.746, -0.2365, -0.0588], [-0.5687, 1.3442, 0.2474], [-0.0624, 0.1156, 0.6584]]
    about = {"wb": [2.4, 1.0, 1.6], "cam_xyz": cam_xyz, "a": 2.0, "b": 30.0, "room": 15868.0}
    noisy = rng.integers(0, 65536, (40, 50, 3), dtype=np.uint16)
    noisy[:3], noisy[3:6] = 0, 65535
    held = denoise.Denoised(noisy, about)
    v = held.scale * np.sqrt(np.maximum((noisy.astype(np.float32) / 65535) @ held.to_cam.T + held.offset, 0))
    assert (held.into(noisy) == v).all(), "into the net's terms"
    held.clean[...] = v + rng.standard_normal(v.shape).astype(np.float32)
    clean = held.clean.copy()
    back = lambda v: (np.maximum(v, 0) / held.scale) ** 2 - held.offset  # noqa: E731
    assert (held.camera(clean) == back(clean)).all() and (held.clean == clean).all(), "and back, the given left alone"
    removed = v - clean
    common = removed.mean(axis=2, keepdims=True)
    for color, grain in ((1, 1), (0, 0), (0.5, 0.5), (0.3, 0.8), (0.77, 0.13)):
        mixed = back(clean + (1 - grain) * common + (1 - color) * (removed - common))
        assert (held.mixed(slice(0, 40), color, grain) == mixed).all(), (color, grain)
    assert (held.clean == clean).all(), "mixing leaves the net's result as it was"

    assert list(denoise._ahead(lambda at: at * 2, [3, 1, 2])) == [6, 2, 4], "in their order"
    assert list(denoise._ahead(lambda at: at, [])) == [], "no pieces, nothing cut"


def check_denoise_sample(path: Path) -> None:
    """A real raw file through the net (on the GPU where there is one), written and read back."""
    with tempfile.TemporaryDirectory() as folder:
        copy = Path(shutil.copy(path, folder))
        steps = []
        held = denoise.run(copy, lambda done, of, on: steps.append((done, of)))
        assert steps[0][0] == 0 and steps[-1][0] == steps[-1][1] == len(steps) - 1, steps
        out = denoise.save(held, copy)
        was, _ = decode.load(copy, half=False)
        now, pixels = decode.load(out, half=False)
        assert (now["width"], now["height"], now["match"]) == (was["width"], was["height"], was["match"]), "the original's curve goes with it"
        seen = np.frombuffer(pixels.buf, np.float16).reshape(now["height"], now["width"], 4)[::4, ::4, :3].astype(np.float32)
        before = held.noisy[::4, ::4] / 65535
        rough = lambda p: float(np.abs(p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]).mean())  # noqa: E731
        assert rough(seen) < rough(before), "calmer than it was"
        # As light where there is light. (Not near black: there the sensor's noise, cut off at
        # black by the decoder, made the photo lighter than the scene, and that goes with it:
        # a photo at ISO 12800 came out at 0.55 of its darkest tenth and 0.99 of its lightest.)
        blocks = lambda p: p[:p.shape[0] // 8 * 8, :p.shape[1] // 8 * 8, 1].reshape(p.shape[0] // 8, 8, p.shape[1] // 8, 8).mean((1, 3)).ravel()  # noqa: E731
        light = blocks(before) >= np.percentile(blocks(before), 90)
        assert abs(blocks(seen)[light].mean() / blocks(before)[light].mean() - 1) < 0.05, "and as light"
        print(f"  denoised on {held.where}: {len(steps) - 1} pieces, {out.stat().st_size / 2**20:.0f} MB")


def check_sample(path: Path) -> None:
    meta, pixels = decode.load(path)
    px = np.frombuffer(pixels.buf, np.float16).reshape(meta["height"], meta["width"], 4)
    assert meta["width"] > 1000 and meta["height"] > 500 and meta["format"] == "rgba16f", meta
    assert 0.001 < px[..., :3].astype(np.float32).mean() < 0.9, "neither black nor blown"
    full, _ = decode.load(path, half=False)
    assert full["width"] > meta["width"] * 1.9, "full size is twice the half size"
    assert meta["full"] == [full["width"], full["height"]] == full["full"], "half knows the real size"
    assert meta["match"] == full["match"] and edit.valid_curve(meta["match"]) == meta["match"], \
        f"the photo's own curve, the same whichever size is developed: {meta['match']}, {full['match']}"
    import rawpy

    with rawpy.imread(str(path)) as raw:
        middle, top = decode.light_of(raw)
    bias = library.describe(path)["bias"]
    was = round(max(0.0, min(-bias, math.log2(0.5 / top), math.log2(0.06 / middle), 3.0)), 2)
    assert catalog._held_back(str(path)) == was, f"where this photo started under version 4: {catalog._held_back(str(path))}, not {was}"
    assert catalog._by_picture(str(path)) == catalog._start_5([middle, top]), "and under version 5"
    assert (meta["size"], full["size"]) == ("half", "full")
    print(f"  sample {path.name}: {meta['width']}×{meta['height']} half, "
          f"{full['width']}×{full['height']} full")
    said = library.describe(path)
    assert said["camera"] and said["taken"] < path.stat().st_mtime + 86400, said
    with tempfile.TemporaryDirectory() as cache:
        thumb = QImage(str(library.thumbnail(path, "c1:sample:1", Path(cache))))
        assert max(thumb.width(), thumb.height()) == library.THUMB and thumb.pixelColor(200, 200).value() > 5
    print(f"  {said['camera']}, {time.strftime('%Y-%m-%d %H:%M', time.localtime(said['taken']))}, "
          f"thumbnail {thumb.width()}×{thumb.height()}")


def main() -> None:
    # Should it ever hang: say where, and end, rather than sit there (it did once, unexplained).
    faulthandler.dump_traceback_later(180, exit=True)
    application = QCoreApplication(sys.argv[:1])  # noqa: F841 - the session's timer wants one
    # The sessions the checks leave behind are let go when this ends, and not whenever Python
    # tidies up: that may be on a thumbnail thread, and Qt says so of every timer taken down
    # from there. (The program itself has one session, for as long as it runs.)
    gc.disable()
    check_store()
    check_synthetic()
    check_session()
    check_edit_values()
    check_tone_curve()
    check_user_curve()
    check_tone_settings()
    check_tone_ends()
    check_tone_even()
    check_vibrance()
    check_pins()
    check_frame()
    check_colors()
    check_mono()
    check_grain()
    check_look()
    check_presets()
    check_lut()
    check_clf()
    check_detail()
    check_lens()
    check_mark()
    check_masks()
    check_white_balance()
    check_history()
    check_identify()
    check_catalog()
    check_catalog_before_checkpoints()
    check_catalog_before_library()
    check_describe()
    check_library()
    check_raw_only()
    check_where_i_was()
    check_library_rests()
    check_missing()
    check_export_files()
    check_checkpoints()
    check_copies_and_moves()
    check_session_edit()
    check_steps_kept()
    check_recent()
    check_one_count_waits()
    check_folders_fast()
    check_drop_last()
    check_unsaved()
    check_painted()
    check_local()
    check_about()
    check_camera_names()
    check_lens_names()
    check_fresh()
    check_start()
    check_light()
    check_catalog_before_compensation()
    check_saved()
    check_ahead()
    check_session_tools()
    check_scopes()
    check_settings()
    check_headroom()
    check_raster()
    check_kept()
    check_denoise()
    check_denoise_refused()
    check_denoise_exact()
    check_workers()
    check_no_orphan()
    check_surface_choice()
    check_shaders()
    if sample := os.environ.get("PHOTONE_SAMPLE"):
        check_sample(Path(sample))
        check_denoise_sample(Path(sample))
    print("ok")


if __name__ == "__main__":
    main()
