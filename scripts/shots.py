"""Render the interface off screen, state by state, and audit the layout.

    ./run.sh shots                 every shot -> .dev/shots/<name>.png and .dev/shots/sheet.png
    ./run.sh shots photo notice    only these
    ./run.sh shots photo           a group: photo and every photo-… shot, and photo-sheet.png
    ./run.sh shots =photo          just the shot named photo, not its group
    ./run.sh shots "(curve)"       one of the audits that drive a window with real events
    ./run.sh shots --list          what there is

Each shot is the real window (app.make_view) on a session built for that state, never shown:
Qt renders a window that is not on screen into a texture on the GPU, and that is what is saved.
The photo in a shot is decode.py's test chart, sent down the same route as a real file. After
the render the item tree is audited (boxes overlapping, things outside the window, clipped text,
warnings from the interface) and so is the picture itself (the photo is drawn, Exposure
brightens it, each slider's fill sits between its tick and its thumb). The faults are printed;
the exit code is the number of faulty shots.

Look at the PNGs after every visual change: a passing audit is not a good-looking screen.
"""

import faulthandler
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Q_ARG, QCoreApplication, QElapsedTimer, QEvent, QEventLoop, QMetaObject, QObject, QPoint, QPointF, QRectF, QSettings, Qt, QUrl, qInstallMessageHandler  # noqa: E402
from PySide6.QtGui import QColor, QGuiApplication, QImage, QKeyEvent, QMouseEvent, QPainter, QSurfaceFormat, QWheelEvent  # noqa: E402
from PySide6.QtQuick import QSGRendererInterface  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
import check as fixtures  # noqa: E402 - the files it makes for its own checks serve here too

from photone import app as ui  # noqa: E402
from photone import catalog, export, library  # noqa: E402
from photone import decode, edit, scopes  # noqa: E402

OUT = ROOT / ".dev" / "shots"
os.environ["PHOTONE_TRIAL"] = "presets"  # the Presets tab is a trial for now: shot and audited as one
BIG, SMALL = (1600, 1000), (1100, 700)
BACKGROUND = QColor("#0e0e0f")  # Theme.bg


def chart(path: Path, half: bool):
    meta, pixels = decode.synthetic(480, 320)
    return {**meta, "name": path.name}, pixels


def picture(path: Path, half: bool):
    """The chart as a JPEG would arrive: a finished picture, no camera behind it."""
    meta, pixels = decode.synthetic(480, 320)
    return {**meta, "name": path.name, "raw": False, "camera": None}, pixels


def flat(path: Path, half: bool):
    """A photo that is nothing like the test chart: one warm gray all over."""
    rgb = np.full((200, 300, 3), (0.22, 0.2, 0.17), np.float32)
    return {**decode.meta(path.name, 300, 200), "raw": True, "camera": decode.synthetic(8, 4)[0]["camera"]}, decode.Pixels(decode.pack(rgb))


SHELF = tempfile.TemporaryDirectory(prefix="photone-shots-")  # a library of its own: gone at exit
ON_SHELF: list = [None]  # the library of the shot being taken, for what the shot does to it
ui.Session.luts_home = Path(SHELF.name) / "luts"  # LUTs installed in a shot: never the owner's


def shelf(name: str, marked: bool = True) -> "library.Library":
    """A library of pictures made here (never the owner's): two folders, three months, two
    cameras, a portrait among them; scanned as the program scans, thumbnails and all."""
    home = Path(SHELF.name) / name
    colors = [(190, 120, 70), (70, 110, 160), (90, 140, 80), (200, 180, 90), (140, 90, 150), (60, 60, 70),
              (210, 210, 200), (150, 60, 60), (80, 150, 150), (120, 120, 60), (180, 140, 170), (40, 80, 50)]
    for n, color in enumerate(colors):
        folder = home / "photos" / ("2026 Iceland" if n < 7 else "2025 Garden")
        folder.mkdir(parents=True, exist_ok=True)
        when = f"2026:0{6 + n % 2}:1{n % 9} 10:0{n % 9}:00" if n < 7 else f"2025:11:0{n - 6} 08:00:00"
        camera = ("SONY", "ILCE-7M4") if n % 4 else ("Canon", "Canon EOS R6")
        size = (600, 900) if n == 3 else (900, 600)
        photo = fixtures.picture_file(folder / f"IMG_{4100 + n}.jpg", fixtures.exif(*camera, when, 1), size)
        image = QImage(str(photo))
        image.fill(QColor(*color))
        body = photo.read_bytes()
        image.save(str(photo), "JPEG", 90)
        fresh = photo.read_bytes()
        photo.write_bytes(fresh[:2] + body[2:body.index(b"\xff\xdb")] + fresh[fresh.index(b"\xff\xdb"):])
    kept = catalog.Catalog(home / "catalog.db")
    for folder in ("2026 Iceland", "2025 Garden"):
        kept.add_folder(home / "photos" / folder)
        kept.found(*library.scan(home / "photos" / folder, kept.known(home / "photos" / folder)))
    found = kept.photos()
    if marked:
        kept.mark([found[1]["key"], found[4]["key"], found[8]["key"]], "stars", 4)
        kept.mark([found[2]["key"]], "stars", 2)
        kept.mark([found[1]["key"], found[9]["key"]], "flag", 1)
        kept.mark([found[5]["key"]], "flag", -1)
        kept.mark([found[4]["key"]], "label", 3)
        kept.mark([found[10]["key"]], "label", 1)
        kept.save(found[8]["key"], Path(found[8]["path"]), {"exposure": 0.5})
    for photo in found:
        library.thumbnail(Path(photo["path"]), photo["key"], home / "thumbs")
    shown = library.Library(kept, home / "thumbs")
    if name != "library-raw-only":  # (the shelf's photos are JPEGs: hidden, as the program starts)
        shown.setFilter("raw", False)
    return shown


SPECKS = (np.arange(60, 270, 15), np.arange(30, 128, 7))  # (rows, columns) of `noisy`'s dark pixels


def noisy(path: Path, half: bool):
    """A photo to try sharpening and noise reduction on, 480 x 320: a gray with a sensor's
    noise (left third), the same noise over an edge from dark to light (middle third), and a
    clean edge (right third). The noise is as strong in the square root of the light
    everywhere, and another in each of red, green and blue."""
    spread = np.random.default_rng(7).normal(0, 0.03, (320, 480, 3))
    x = np.arange(480)[None, :, None]
    roots = np.where(x < 160, 0.45, np.where(x < 240, 0.3, np.where(x < 320, 0.6, np.where(x < 400, 0.3, 0.6))))
    roots = np.broadcast_to(roots, (320, 480, 3)) + np.where(x < 320, spread, 0)
    roots[SPECKS] = 0.15  # and a few lone dark pixels in the gray, as a sensor has them
    rgb = (np.clip(roots, 0, None) ** 2).astype(np.float32)
    return {**decode.meta(path.name, 480, 320), "raw": False, "camera": None}, decode.Pixels(decode.pack(rgb))


LENS = {"distortion": [round(900 * (n / 15) ** 2) for n in range(16)], "vignetting": [round(9000 * (n / 15) ** 2) for n in range(16)],
        "red": [12000] * 16, "blue": [-12000] * 16}  # figures as a camera writes them, stronger than a lens has


def through_sensor(rgb: np.ndarray, red: float = 1.0, blue: float = 1.0, soft: bool = False) -> np.ndarray:
    """A picture (rows x columns x 3) as a raw file has it: drawn by a lens that draws red and
    blue that many times larger than green, recorded by a sensor that has one color at each
    pixel (red, green / green, blue), and made whole again the way raw developers do: green
    from its neighbors, red and blue as green plus what they differ from it by at their own
    pixels. So red's and blue's finest detail is green's, which is what the chromatic
    aberration correction has to reckon with (pipeline.frag, `corrected`).
    `soft`: by a lens less sharp than the chart, as any is: an edge from one pixel to the next
    can fall between the pixels that see red, and then nothing tells where red has it, which
    matters where red is less than a pixel off."""
    high, wide = rgb.shape[:2]
    big = np.repeat(np.repeat(rgb, 4, 0), 4, 1)
    ys, xs = np.indices((4 * high, 4 * wide))
    drawn = np.empty_like(rgb)
    for color, larger in enumerate((red, 1.0, blue)):
        y = np.clip(np.rint((ys + 0.5 - 2 * high) / larger + 2 * high - 0.5), 0, 4 * high - 1).astype(int)
        x = np.clip(np.rint((xs + 0.5 - 2 * wide) / larger + 2 * wide - 0.5), 0, 4 * wide - 1).astype(int)
        drawn[..., color] = big[y, x, color].reshape(high, 4, wide, 4).mean(axis=(1, 3))
    drawn = np.pad(drawn, ((2, 2), (2, 2), (0, 0)), mode="edge")  # (the frame's edge has neighbors too, not the far edge's)
    ys, xs = np.indices(drawn.shape[:2])
    reds, blues = (ys % 2 == 0) & (xs % 2 == 0), (ys % 2 == 1) & (xs % 2 == 1)

    def near(values: np.ndarray) -> np.ndarray:  # each pixel and the eight around it, the nearer counting more
        return sum((1 - abs(dy) / 2) * (1 - abs(dx) / 2) * np.roll(values, (dy, dx), (0, 1)) for dy in (-1, 0, 1) for dx in (-1, 0, 1))

    if soft:
        drawn = near(drawn) / 4

    def filled(values: np.ndarray, there: np.ndarray) -> np.ndarray:  # known `there` only: everywhere, from the nearest that are
        return near(values * there) / near(there.astype(np.float32))

    green = np.where(reds | blues, filled(drawn[..., 1], ~(reds | blues)), drawn[..., 1])
    return np.stack([green + filled(drawn[..., 0] - green, reds), green, green + filled(drawn[..., 2] - green, blues)], axis=-1)[2:-2, 2:-2].astype(np.float32)


def sensed_chart(red: float, blue: float, soft: bool = False):
    """The chart `through_sensor`, as decode.synthetic gives it: (what it is, its pixels)."""
    meta, pixels = decode.synthetic(480, 320)
    rgb = np.frombuffer(pixels.buf, np.float16).reshape(320, 480, 4)[..., :3].astype(np.float32)
    return meta, decode.Pixels(decode.pack(through_sensor(rgb, red, blue, soft)))


def lensed(path: Path, half: bool):
    """The chart as a raw file of a camera that wrote down its lens's faults (LENS), with the
    colored fringes those figures are for in it (`through_sensor`)."""
    meta, seen = sensed_chart(1 + LENS["red"][0] * 2 ** -21, 1 + LENS["blue"][0] * 2 ** -21)
    return {**meta, "name": path.name, "camera": {**meta["camera"], "lens": LENS}}, seen


def fringed(path: Path, half: bool):
    """A photo to try defringe on, 480 x 320: dark on the left, light on the right, and along
    the edge between them a seam four pixels wide, purple in the upper half and green in the
    lower; and far from any edge a purple square and a green one, which are things and no
    fringes."""
    rgb = np.zeros((320, 480, 3), np.float32)
    rgb[:, :240], rgb[:, 240:] = 0.02, 0.7
    rgb[:160, 238:242], rgb[160:, 238:242] = (0.5, 0.12, 0.6), (0.12, 0.5, 0.12)
    rgb[40:100, 60:120], rgb[220:280, 60:120] = (0.5, 0.12, 0.6), (0.12, 0.5, 0.12)
    return {**decode.meta(path.name, 480, 320), "raw": False, "camera": None}, decode.Pixels(decode.pack(rgb))


def damaged(path: Path, half: bool):
    raise decode.Unreadable(f"{path.name}: the file is damaged and cannot be read")


def empty() -> ui.Session:
    return ui.Session([])


def photo() -> ui.Session:
    return ui.Session([Path("test chart.arw")], load=chart)


def pair(n: int) -> None:
    """The shelf's photo after the n-th passes for the n-th's denoised copy (what its file
    would say of itself, put where the library keeps that)."""
    shown = ON_SHELF[0]
    shown.made_here[shown.paths()[n + 1]] = {"of": shown.model.rows[n]["key"]}


def lose(*rows: int) -> None:
    """The shelf's photos at these rows pass for ones whose files are gone."""
    shown = ON_SHELF[0]
    shown.absent = {shown.paths()[row] for row in rows}
    shown.changed.emit()


def stage(root):
    return root.findChild(QObject, "box:stage")


def neural(root, session, state: str, share: float = 1.0, step: str = "Denoising") -> None:
    """The Neural Denoise window in one of its states, without the net (which wants a real raw
    file and seconds of a GPU): "working" at `share` of the way, "ready" with the chart itself
    as the net's result, or "failed"."""
    from photone import denoise

    denoiser = session.denoiser
    denoiser._path = session.photos[0]
    if state == "ready":
        meta, pixels = chart(session.photos[0], False)
        seen = np.frombuffer(pixels.buf, np.float16).reshape(meta["height"], meta["width"], 4)
        noisy = np.clip(seen[..., :3].astype(np.float32) * 65535, 0, 65535).astype(np.uint16)
        held = denoise.Denoised(noisy, {**meta["camera"], "a": 1.0, "b": 10.0, "room": 15000.0})
        held.clean[...], held.where = held.into(noisy), "CUDA"
        denoiser.held, denoiser.pixels = held, held.picture()
    denoiser._run += 1  # (another address for its picture)
    denoiser._step = step
    denoiser._set(state, "IMG_0001.ARW cannot be read as a raw photo (data error)." if state == "failed" else "", share)
    find(root, "denoise-window").setProperty("visible", True)
    find(root, "denoise-cancel").forceActiveFocus()  # (as its `open` does: the keys are its own)


# A real raw photo for the Neural Denoise shots of which net ran (PHOTONE_SAMPLE, else the
# owner's sample where it is; the test chart where there is none: a test chart is not the app).
SAMPLE = Path(os.environ.get("PHOTONE_SAMPLE", "/data/projects/Neuro/samples/IMG01184.ARW"))


def told(kind: str, why: str = ""):
    """A shot of the Neural Denoise window, ready, on SAMPLE, saying which net ran (Denoise.qml's
    `which`): `kind` "sensor" or "developed", and `why`."""
    def make() -> ui.Session:
        return ui.Session([SAMPLE]) if SAMPLE.is_file() else photo()

    def after(root, session) -> None:
        from photone import denoise

        denoiser = session.denoiser
        denoiser._path = session.photos[0]
        meta, pixels = decode.load(SAMPLE, half=False) if SAMPLE.is_file() else chart(session.photos[0], False)
        seen = np.frombuffer(pixels.buf, np.float16).reshape(meta["height"], meta["width"], 4)
        noisy = np.clip(seen[..., :3].astype(np.float32) * 65535, 0, 65535).astype(np.uint16)
        held = denoise.Denoised(noisy, {**meta["camera"], "a": 1.0, "b": 10.0, "room": 15000.0})
        held.clean[...], held.where, held.kind, held.why = held.into(noisy), "CUDA", kind, why
        denoiser.held, denoiser.pixels = held, held.picture()
        denoiser._run += 1
        denoiser._set("ready", "", 1.0)
        find(root, "denoise-window").setProperty("visible", True)
        find(root, "denoise-cancel").forceActiveFocus()
    return make, after


def call(item, function: str, *args):
    """A function of a QML item, as a shot's way of reaching a state that needs a gesture."""
    QMetaObject.invokeMethod(item, function, *(Q_ARG("QVariant", a) for a in args))


def checkpoints(session, count: int = 3) -> None:
    """Looks saved one after the other; the edit then stands at the second."""
    for n in range(count):
        session.setEdit(("exposure", "temperature", "shadows")[n % 3], (0.4, 6200, 45)[n % 3] + n)
        session.commitEdit()
        session.saveCheckpoint()
    session.goToCheckpoint(2)


def find(root, name: str):
    """An item by its objectName, through the items as they are drawn (a list's rows are not
    among the QObject children that findChild looks through)."""
    return next((i for i in items(root) if i.objectName() == name), None)


def click(window, item) -> None:
    """A real click in the middle of an item."""
    QTest.mouseClick(window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     box(item).center().toPoint())
    settle()


def CHART_BYTES():  # noqa: N802 - a small picture to stand for an exported one
    ramp = np.linspace(40, 230, 96, dtype=np.uint8)
    return np.stack([np.tile(ramp, (64, 1)), np.tile(ramp[::-1], (64, 1)), np.full((64, 96), 120, np.uint8)], axis=-1)


def find_presets(root):
    """The window's presets (kept nowhere in a shot)."""
    return root.window().rootContext().contextProperty("presets")


def some_luts() -> list[Path]:
    """A few .cube files made here to stand for ones brought in."""
    from photone import lut
    same = lut.identity(17)
    made = {"Warm print": 0.06 + 0.94 * same ** np.array([0.85, 1.0, 1.25], np.float32),
            "Cool shadows": same ** np.array([1.15, 1.0, 0.85], np.float32),
            "Bleach bypass": np.clip((same - 0.5) * 1.3 + 0.5, 0, 1) * 0.6 + same.mean(axis=-1, keepdims=True) * 0.4,
            "Faded film": 0.1 + 0.8 * same, "Teal and orange": same * np.array([1.08, 0.98, 0.9], np.float32)}
    files = []
    for name, table in made.items():
        files.append(Path(SHELF.name) / f"{name}.cube")
        files[-1].write_text(lut.cube(np.clip(table, 0, 1), name), "utf-8")
    return files


def exposure(value: float):
    return lambda root, session: session.setEdit("exposure", value)


def edits(values: dict, **panels):
    """A shot's state: these settings, and these panels open."""
    def do(root, session) -> None:
        for key, value in panels.items():
            show(root, value) if key == "tool" else session.setPanel(key, value)
        session.setEdits(values)
    return do


OVAL = {"kind": "radial", "at": [0.6, 0.42], "to": [0.8, 0.5], "ratio": 0.6, "feather": 0.4}
SLOPE = {"kind": "linear", "at": [0.5, 0.4], "to": [0.55, 0.62]}


def masked(mask: dict, shown: bool = False, chosen: int = 1):
    """A shot's state: that mask, chosen (or not: 0), and a pin kept to it that darkens what
    it covers; `shown`: with H held, which shows the mask instead of the picture."""
    def do(root, session) -> None:
        show(root, "masks")
        darker = {"from": [0, 0], "range": 0.5, "feather": 0, "light": -0.25, "mask": 1}
        session.setEdits({"masks": [mask], "pins": [darker]})
        settle()
        find(root, "wheel").setProperty("mask", chosen)
        stage(root).setProperty("showMask", int(shown))
    return do


def view_of(root):
    """The window an item of the interface is in."""
    return root.window()


def show(root, tool: str, *more: str) -> None:
    """The right column with that card open and at its top, and no other card but `more`:
    "" is the Tone Wheel, "pins" the pin's settings under it, "curve" the tone curve at the foot
    of Exposure, "crop" and "masks" their tabs."""
    call(root, "go", tool if tool in ("crop", "masks") else "edit")
    cards = {"": "wheel", "pins": "wheel", "crop": "wheel", "masks": "wheel", "curve": "tone"}.get(tool, tool)
    root.setProperty("unfolded", {**dict.fromkeys(("about", "folders", cards, *more), True)})
    settle()
    root.window().grabWindow()  # a frame: the column has its new height once one is drawn
    if tool not in ("crop", "masks"):
        bring(root, {"": "wheel-block", "pins": "panel:pins", "curve": "curve-plot"}.get(tool, "panel:" + tool))


def bring(root, name: str) -> None:
    """Scroll the right column until that item is at its top, or as near as the column goes."""
    scroll = find(root, "tools-scroll")
    for _ in range(2):  # (twice: what is scrolled into sight may be laid out only then)
        to = scroll.property("contentY") + box(find(root, name)).top() - box(scroll).top() - 8
        scroll.setProperty("contentY", max(0.0, min(scroll.property("contentHeight") - scroll.height(), to)))
        settle()
        root.window().grabWindow()
    settle()


def opened(session, tool: str | None = None):
    """The window on that session, never shown, with its photo drawn: (view, root)."""
    view = ui.make_view(session)
    session.surface_known(False)
    root = view.rootObject()
    root.setSize(view.size())
    wait(lambda: root.property("photoReady"))
    settle()
    view.grabWindow()  # a frame: columns and rows only place their items when one is drawn
    if tool is not None:
        show(root, tool)
    return view, root


def drag(view, at: QPoint, by: QPoint, held=Qt.KeyboardModifier.NoModifier, down=None) -> None:
    """A real drag with the left button; `down` is called once it is pressed, before it moves.
    The press comes half a second after whatever was before, so as not to make a double click
    of it."""
    to = QPointF(at + by)  # QTest's own move cannot hold a key
    QTest.mousePress(view, Qt.MouseButton.LeftButton, held, at, 500)
    settle()
    if down:
        down()
    QCoreApplication.sendEvent(view, QMouseEvent(
        QEvent.Type.MouseMove, to, to, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, held))
    settle()
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, held, at + by)
    settle()


def double_click(view, at: QPoint) -> None:
    """A real double click, clear in time of whatever press came before it."""
    QTest.mouseMove(view, at, 500)
    QTest.mouseDClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at)
    settle()


def tap(view, at: QPoint) -> None:
    """A real click, clear in time of whatever press came before it."""
    QTest.mouseMove(view, at, 500)
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at)
    settle()


def plain(value):
    """A property of a QML item as Python has it (lists and objects may come as JS values)."""
    return value.toVariant() if hasattr(value, "toVariant") else value


def counted(view, session) -> bool:
    """Count the scopes anew. The program has Qt grab their small render, which Qt refuses for
    a window that is not on screen; so here that render is put in the corner of the window for
    one grab, and the session is handed the same picture it would get."""
    renders = view.rootObject().findChild(QObject, "scopes")
    call(renders, "refresh")  # the edit as it is now, as the program does before it counts
    renders.setVisible(True)
    seen = view.grabWindow().copy(int(box(renders).left()), int(box(renders).top()), 768, 512)
    renders.setVisible(False)
    session._histogram = {}
    session.setScope(seen)
    counted_now = wait(lambda: bool(session.histogram), 4000)  # it is counted on a thread
    settle()
    return counted_now


LIFT = [[0.0, 0.0], [0.4, 0.6], [1.0, 1.0]]  # a tone curve that lightens the middle
SWING = [[0.0, 0.0], [0.22, 0.14], [0.55, 0.62], [0.82, 0.9], [1.0, 1.0]]


def curved(level: int, points) -> int:
    """Where a gray of that level on screen lands under a tone curve: its lightness is the
    cube root of its light."""
    v = level / 255
    light = v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    out = edit.curve_at(points, light ** (1 / 3)) ** 3
    return round(255 * (12.92 * out if out <= 0.0031308 else 1.055 * out ** (1 / 2.4) - 0.055))


# name: (window size, the session for that state, what to do once it is drawn)
# Name shots by group: "photo", "photo-small"… (`shots photo` renders the group).
SHOTS = {
    "empty": (BIG, empty, None),
    "photo": (BIG, photo, None),
    "photo-small": (SMALL, photo, None),
    "photo-exposure": (BIG, photo, exposure(1.5)),
    "photo-dark": (BIG, photo, exposure(-2.25)),
    "develop": (BIG, photo, edits({"temperature": 6400, "tint": 18, "exposure": 0.35, "contrast": 20,
                                   "brightness": -15, "highlights": -40, "shadows": 55})),
    "develop-flat": (SMALL, photo, lambda root, session: session.setEdit("profile", "flat")),
    "curve": (BIG, photo, edits({"curve": SWING}, tool="curve")),
    "curve-small": (SMALL, photo, edits({"curve": LIFT, "exposure": -0.5}, tool="curve")),
    "balance": (BIG, photo, edits({"temperature": 6900, "tint": -30}, tool="balance")),
    "balance-pick": (SMALL, photo, lambda root, session: (
        show(root, "balance"), stage(root).setProperty("picking", "neutral"))),
    "histogram": (BIG, photo, edits({"exposure": 0.8}, clipping=True)),
    "tools-closed": (SMALL, photo, edits({"contrast": 30}, tool="")),
    "wheel": (BIG, photo, edits({}, tool="")),
    "wheel-all": (BIG, photo, lambda root, session: (
        show(root, ""), settle(), counted(root.window(), session), call(find(root, "wheel"), "showAll"))),
    "wheel-close": (BIG, photo, lambda root, session: (
        show(root, ""), settle(), counted(root.window(), session),
        call(find(root, "wheel"), "zoomAt", 5.0, 150.0, 200.0))),
    "wheel-range": (BIG, photo, lambda root, session: (show(root, ""), session.setRange("shadows"))),
    "wheel-hover": (SMALL, photo, lambda root, session: (
        show(root, ""), stage(root).setProperty("hover", QPointF(0.3, 0.4)))),
    "wheel-tone": (SMALL, photo, None),
    "pins": (BIG, photo, lambda root, session: (
        show(root, ""), settle(), counted(root.window(), session),
        session.setEdit("pins", [{"from": [-0.09, 0.08], "to": [-0.03, 0.12], "range": 0.07},
                                 {"from": [0.1, -0.04], "to": [0.13, -0.1], "mode": "rotate"},
                                 {"from": [0.0, -0.1], "to": [0.0, -0.1], "light": 0.1}]),
        find(root, "wheel").setProperty("selected", 0))),
    "pins-panel": (BIG, photo, lambda root, session: (
        show(root, "pins"), settle(), counted(root.window(), session),
        session.setEdit("pins", [{"from": [-0.09, 0.08], "to": [-0.03, 0.12], "range": 0.07, "light": 0.08, "low": 0.3}]),
        find(root, "wheel").setProperty("selected", 0))),
    "pins-none": (SMALL, photo, lambda root, session: show(root, "pins")),
    "pins-reach": (BIG, photo, lambda root, session: (
        show(root, ""), settle(), counted(root.window(), session),
        session.setEdit("pins", [{"from": [-0.09, 0.08], "to": [-0.03, 0.12], "range": 0.07}]),
        find(root, "wheel").setProperty("selected", 0), stage(root).setProperty("reach", 1))),
    "pins-after": (BIG, photo, lambda root, session: (
        show(root, ""), session.setPanel("live", True),
        session.setEdit("pins", [{"from": [-0.09, 0.08], "to": [-0.03, 0.12], "range": 0.07},
                                 {"from": [0.1, -0.04], "to": [0.13, -0.1], "mode": "rotate"}]),
        find(root, "wheel").setProperty("selected", 1))),
    "saturation": (SMALL, photo, edits({"saturation": 140.0}, tool="saturation")),
    "eq": (BIG, photo, edits({"eq": [{"hue": 140, "gain": 160, "width": 45}, {"hue": 250, "gain": 40, "width": 30},
                                     {"hue": 20, "gain": 120, "width": 70}]}, tool="eq")),
    "eq-small": (SMALL, photo, edits({}, tool="eq")),
    "chroma": (BIG, photo, edits({"chroma": [[0.0, 0.0], [0.2, 0.36], [0.7, 0.8], [1.0, 1.0]]}, tool="chroma")),
    "test-chart": (BIG, lambda: ui.Session([Path("IMG_0412.ARW")], load=flat), lambda root, session: (
        session.setEdits({"exposure": 0.3, "saturation": 130.0}), session.commitEdit(), session.setPanel("chart", True),
        wait(lambda: root.property("photoReady")))),
    "test-chart-small": (SMALL, lambda: ui.Session([Path("IMG_0412.ARW")], load=flat), lambda root, session: (
        session.setPanel("chart", True), wait(lambda: root.property("photoReady")))),
    "library": (BIG, empty, lambda root, session: find(root, "box:library").setProperty("current", 4)),
    "library-small": (SMALL, empty, lambda root, session: (
        ON_SHELF[0].show("month:2026-07"), find(root, "box:library").setProperty("chosen", [0, 1]))),
    "library-filter": (BIG, empty, lambda root, session: ON_SHELF[0].setFilter("stars", 3)),
    "library-raw-only": (BIG, empty, None),  # as the program starts: finished pictures are not shown
    "export": (BIG, photo, lambda root, session: (
        call(root, "show", "export"), find(root, "box:export").setProperty("folder", "/home/you/Pictures/Photone"),
        setattr(session.exporter, "_recent", [
            {"path": f"/home/you/Pictures/Photone/{name}", "name": name, "folder": "/home/you/Pictures/Photone", "source": "/x.ARW", "thumb": export.small(export.picture(CHART_BYTES())),
             "when": 1795000000 - 400 * n, "wide": 7008, "high": 4672, "bytes": 9_400_000 - 700_000 * n}
            for n, name in enumerate(["IMG03692.jpg", "IMG03690.jpg", "IMG03682-2.jpg", "IMG03682.jpg"])]),
        session.exporter.changed.emit())),
    "export-mark": (BIG, photo, lambda root, session: (
        call(root, "show", "export"), find(root, "box:export").setProperty("folder", "/home/you/Pictures/Photone"),
        session.exporter.setMark({"kind": "text", "text": "© Morphox", "size": 6.0, "strength": 80.0}))),
    "mark": (BIG, photo, lambda root, session: session.exporter.setMark({"kind": "text", "text": "© Morphox", "size": 6.0, "strength": 80.0})),
    "export-small": (SMALL, photo, lambda root, session: (
        call(root, "show", "export"), find(root, "box:export").setProperty("resized", True),
        find(root, "box:export").setProperty("kind", "tiff16"))),
    "crop": (BIG, photo, edits({"crop": [0.15, 0.1, 0.8, 0.85], "angle": 5.0}, tool="crop")),
    "crop-small": (SMALL, photo, edits({"crop": [0.2, 0.3, 0.9, 0.9], "turns": 1.0}, tool="crop")),
    "crop-done": (SMALL, photo, edits({"crop": [0.15, 0.1, 0.8, 0.85], "angle": 5.0}, tool="")),
    "split": (BIG, photo, lambda root, session: (
        session.setEdits({"exposure": -0.8, "saturation": 30.0}), show(root, ""), stage(root).setProperty("split", 0.45))),
    "pins-mark": (SMALL, photo, lambda root, session: (
        show(root, ""), settle(), counted(root.window(), session),
        call(find(root, "wheel"), "mark", [0.06, 0.09]))),
    "masks": (BIG, photo, masked(OVAL)),
    "dock": (BIG, photo, lambda root, session: show(root, "eq")),
    "dock-small": (SMALL, photo, masked(OVAL)),
    "masks-linear": (SMALL, photo, masked(SLOPE)),
    "masks-show": (SMALL, photo, masked({**OVAL, "invert": True}, shown=True)),
    "masks-whole": (SMALL, photo, masked(SLOPE, chosen=0)),
    "sections": (BIG, photo, lambda root, session: (
        checkpoints(session, 2), masked(OVAL)(root, session), show(root, "pins"),
        find(root, "wheel").setProperty("selected", 0))),
    "strip": (BIG, empty, lambda root, session: (
        session.openAt(ON_SHELF[0].paths(), 3), call(root, "show", "edit"), wait(lambda: root.property("photoReady")),
        wait(lambda: ON_SHELF[0].making == 0 and not ON_SHELF[0].strip, 4000))),
    "strip-export": (BIG, empty, lambda root, session: (
        session.openAt(ON_SHELF[0].paths(), 1), call(root, "show", "edit"), wait(lambda: root.property("photoReady")),
        wait(lambda: ON_SHELF[0].making == 0 and not ON_SHELF[0].strip, 4000), view_of(root).grabWindow(),
        find(root, "box:strip").setProperty("chosen", [1, 2, 3, 4, 6, 8, 9]), call(root, "show", "export"), settle(), view_of(root).grabWindow(), settle())),
    "strip-chosen": (BIG, empty, lambda root, session: (
        session.openAt(ON_SHELF[0].paths(), 1), call(root, "show", "edit"), wait(lambda: root.property("photoReady")),
        wait(lambda: ON_SHELF[0].making == 0 and not ON_SHELF[0].strip, 4000), view_of(root).grabWindow(),
        find(root, "box:strip").setProperty("chosen", [1, 3, 4]))),
    "strip-marks": (BIG, empty, lambda root, session: (
        session.openAt(ON_SHELF[0].paths(), 1), call(root, "show", "edit"), wait(lambda: root.property("photoReady")),
        wait(lambda: ON_SHELF[0].making == 0 and not ON_SHELF[0].strip, 4000), view_of(root).grabWindow(),
        find(root, "strip-row").setProperty("contentX", 700), settle(), view_of(root).grabWindow(),
        QTest.mouseMove(view_of(root), box(find(root, "strip:8")).center().toPoint()))),
    "strip-pair": (BIG, empty, lambda root, session: (
        pair(3), pair(8), session.openAt(ON_SHELF[0].paths(), 4), call(root, "show", "edit"), wait(lambda: root.property("photoReady")),
        root.setProperty("stripHigh", 140), wait(lambda: ON_SHELF[0].making == 0 and not ON_SHELF[0].strip, 4000))),
    "strip-pair-usual": (BIG, empty, lambda root, session: (
        pair(3), pair(8), session.openAt(ON_SHELF[0].paths(), 3), call(root, "show", "edit"), wait(lambda: root.property("photoReady")),
        wait(lambda: ON_SHELF[0].making == 0 and not ON_SHELF[0].strip, 4000))),
    # Photos whose files are gone: a plain one, a denoised copy beside its original, and (in
    # the second) the open one, which can only be taken out; then the question before one goes.
    "strip-missing": (BIG, empty, lambda root, session: (
        pair(3), session.openAt(ON_SHELF[0].paths(), 2), call(root, "show", "edit"), wait(lambda: root.property("photoReady")),
        lose(1, 4, 7), wait(lambda: ON_SHELF[0].making == 0 and not ON_SHELF[0].strip, 4000))),
    "strip-missing-open": (BIG, empty, lambda root, session: (
        session.openAt(ON_SHELF[0].paths(), 2), call(root, "show", "edit"), wait(lambda: root.property("photoReady")),
        wait(lambda: ON_SHELF[0].making == 0 and not ON_SHELF[0].strip, 4000), lose(2, 7))),
    "strip-missing-ask": (BIG, empty, lambda root, session: (
        session.openAt(ON_SHELF[0].paths(), 2), call(root, "show", "edit"), wait(lambda: root.property("photoReady")),
        wait(lambda: ON_SHELF[0].making == 0 and not ON_SHELF[0].strip, 4000), lose(1, 7),
        call(root, "removeMissing", [ON_SHELF[0].paths()[1]]))),
    "strip-small": (SMALL, empty, lambda root, session: (
        session.openAt(ON_SHELF[0].paths(), 9), call(root, "show", "edit"), wait(lambda: root.property("photoReady")),
        root.setProperty("stripHigh", 72), wait(lambda: ON_SHELF[0].making == 0 and not ON_SHELF[0].strip, 4000))),
    "mono": (BIG, photo, lambda root, session: (session.setEdit("mono", True), show(root, "mono"))),
    "mono-red": (BIG, photo, lambda root, session: (
        session.setEdits({"mono": True, "bw": edit.FILMS["red"]}), show(root, "mono"))),
    "mono-own": (SMALL, photo, lambda root, session: (
        session.setEdits({"mono": True, "bw": [{"hue": 140, "gain": 30, "width": 60}, {"hue": 262, "gain": -35, "width": 50}]}),
        show(root, "mono"))),
    "mono-masks": (SMALL, photo, lambda root, session: (session.setEdit("mono", True), masked(OVAL)(root, session))),
    "detail": (BIG, photo, lambda root, session: (
        session.setEdits({"sharpen": 60.0, "sharpenMask": 40.0, "denoise": 50.0, "denoiseColor": 60.0}), show(root, "detail"))),
    "detail-denoised": (BIG, photo, lambda root, session: (
        setattr(session, "_denoised", True), session.refresh(), show(root, "detail"))),
    "denoise-working": (BIG, photo, lambda root, session: neural(root, session, "working", 0.42)),
    "denoise-warming": (BIG, photo, lambda root, session: neural(root, session, "working", 0.0, "Warming up")),
    "denoise-ready": (BIG, photo, lambda root, session: neural(root, session, "ready")),
    "denoise-small": (SMALL, photo, lambda root, session: neural(root, session, "ready")),
    "denoise-failed": (BIG, photo, lambda root, session: neural(root, session, "failed")),
    # Which net ran, on a real photo: the raw net; the older one for a sensor the raw net does
    # not take (X-Trans); and the older one because no raw net is installed (nothing said)
    "denoise-sensor": (BIG, *told("sensor")),
    "denoise-older": (BIG, *told("developed", "pattern")),
    "denoise-missing": (BIG, *told("developed", "missing")),
    "lens": (BIG, lambda: ui.Session([Path("test chart.arw")], load=lensed), lambda root, session: (
        session.setEdit("lensColor", False), show(root, "lens"))),
    "lens-none": (SMALL, photo, lambda root, session: show(root, "lens")),
    "lens-hand": (BIG, lambda: ui.Session([Path("test chart.arw")], load=lensed), lambda root, session: (
        session.setEdits({"defringe": 60.0}), show(root, "lens"))),
    "grain": (BIG, photo, lambda root, session: (
        session.setEdits({**edit.CHARACTERS["bold"], "grainSize": 4.0}), show(root, "grain"))),
    "grain-own": (SMALL, photo, lambda root, session: (
        session.setEdits({"grain": 80.0, "grainSize": 4.0, "grainColor": 60.0, "grainCurve": edit.RESPONSES["negative"], "mono": True}),
        show(root, "grain"))),
    "presets": (BIG, photo, lambda root, session: (
        session.setEdits({"contrast": 25.0, "saturation": 120.0}),
        [find_presets(root).save(name, look, list(edit.PARTS)) for name, look in (
            ("Warm fade", {"temperature": 6800.0, "contrast": -25.0, "blacks": 40.0, "saturation": 80.0}),
            ("Red filter", {"mono": True, "bw": edit.FILMS["red"], "contrast": 20.0}),
            ("Punch", {"contrast": 45.0, "vibrance": 50.0}),
            ("Pushed film", {**edit.CHARACTERS["pushed"], "grainSize": 4.0, "mono": True, "bw": edit.FILMS["orange"]}),
            ("Teal shadows", {"pins": [{"from": [0.0, 0.0], "to": [-0.03, -0.02], "range": 0.1, "high": 0.4}]}))],
        call(root, "go", "presets"))),
    "presets-none": (SMALL, photo, lambda root, session: call(root, "go", "presets")),
    "lut": (BIG, photo, lambda root, session: (
        session.installLuts([str(file) for file in some_luts()]), session.addLut(session.lutsInstalled[3]["id"]),
        session.addLut(session.lutsInstalled[0]["id"]), session.setLut(0, {"strength": 80.0}), session.setLut(1, {"off": True}),
        show(root, "lut"), wait(lambda: stage(root).property("lut") is not None, 3000))),
    "lut-drag": (BIG, photo, lambda root, session: (
        session.installLuts([str(file) for file in some_luts()]), session.addLut(session.lutsInstalled[3]["id"]),
        session.addLut(session.lutsInstalled[0]["id"]), show(root, "lut"), wait(lambda: stage(root).property("lut") is not None, 3000),
        find(root, "panel:lut").setProperty("carried", session.lutsInstalled[4]),
        find(root, "panel:lut").setProperty("carriedAt", QPointF(box(find(root, "lut-list")).center().x() - 60, box(find(root, "lut-list")).top() + 4 + 28 + 3)))),
    "lut-none": (SMALL, lambda: (setattr(ui.Session, "luts_home", Path(SHELF.name) / "luts-none"), photo())[1],
                 lambda root, session: (show(root, "lut"), setattr(ui.Session, "luts_home", Path(SHELF.name) / "luts"))),
    "lut-lost": (SMALL, photo, lambda root, session: (
        session.installLuts([str(file) for file in some_luts()]),
        session.setEdit("luts", [{"id": "gone", "name": "Fuji 3513", "size": 33, "strength": 100.0}]), show(root, "lut"),
        wait(lambda: stage(root).property("lut") is not None, 3000))),
    "checkpoints": (BIG, photo, lambda root, session: checkpoints(session)),
    "checkpoints-changed": (SMALL, photo, lambda root, session: (
        checkpoints(session), session.setEdit("highlights", -35))),
    "checkpoints-many": (BIG, photo, lambda root, session: checkpoints(session, 9)),
    "checkpoints-low": (SMALL, photo, lambda root, session: (
        show(root, "curve"), checkpoints(session, 9))),
    "checkpoints-confirm": (BIG, photo, lambda root, session: (
        checkpoints(session), session.setEdit("highlights", -35), settle(),
        click(root.window(), find(root, "checkpoint:1")))),
    "develop-jpeg": (BIG, lambda: ui.Session([Path("scan.jpg")], load=picture), None),
    "notice": (BIG, lambda: ui.Session([Path("IMG_0001.ARW")], load=damaged), None),
    "notice-no-hdr": (SMALL, photo, lambda root, session: session.setHdr(True)),
    "zoom-100": (BIG, photo, lambda root, session: call(stage(root), "actual")),
    "zoom-corner": (BIG, photo, lambda root, session: (call(stage(root), "zoomTo", 6.0, 40.0, 40.0),
                                                       call(stage(root), "panBy", -4000.0, -4000.0))),
    "zoom-out": (SMALL, photo, lambda root, session: call(stage(root), "zoomTo", 0.5, 300.0, 200.0)),
}

warnings: list[str] = []


def wait(done, ms: int = 8000) -> bool:
    clock = QElapsedTimer()
    clock.start()
    while not done() and clock.elapsed() < ms:
        QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
    return done()


def settle() -> None:
    """Let everything that is under way arrive, also a move of the mouse that is being held
    back until the next frame (app.PerFrameInput): a check does not wait for frames."""
    for window in QGuiApplication.allWindows():
        for held in window.findChildren(ui.PerFrameInput):
            held.send()
    for _ in range(3):
        QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
    for window in QGuiApplication.allWindows():  # (a zoom that glides is there at once)
        root = window.rootObject() if hasattr(window, "rootObject") else None
        if root:
            QMetaObject.invokeMethod(root, "land")
        view = root and root.findChild(QObject, "box:stage")
        if view:
            QMetaObject.invokeMethod(view, "land")
        wheel = root and root.findChild(QObject, "wheel")
        if wheel:
            QMetaObject.invokeMethod(wheel, "land")


def pictures_in(root) -> bool:
    """Every picture the interface has asked Qt to read from a file is in (thumbnails are)."""
    return all(i.property("implicitWidth") > 0 for i in items(root)
               if "QQuickImage" in i.metaObject().className() and i.property("source").toString())


def items(item):
    yield item
    for child in item.childItems():
        yield from items(child)


def box(item) -> QRectF:
    return item.mapRectToScene(QRectF(0, 0, item.width(), item.height()))


def label(item) -> str:
    return item.objectName() or item.metaObject().className()


def audit_layout(view) -> list[str]:
    faults = []
    seen = [i for i in items(view.contentItem()) if i.isVisible() and i.width() > 0 and i.height() > 0]
    boxes = [(i, box(i).adjusted(0.5, 0.5, -0.5, -0.5)) for i in seen if i.objectName().startswith("box:")]
    for n, (a, ra) in enumerate(boxes):
        faults += [f"overlap: {label(a)} and {label(b)}" for b, rb in boxes[n + 1:] if ra.intersects(rb)]
    floating = [(i, box(i)) for i in seen if i.objectName().startswith("screen:")]
    for n, (a, ra) in enumerate(floating):
        faults += [f"overlap: {label(a)} and {label(b)}" for b, rb in floating[n + 1:] if ra.intersects(rb)]
    window = QRectF(0, 0, view.width(), view.height()).adjusted(-0.5, -0.5, 0.5, 0.5)
    faults += [f"leaves the window: {label(i)}" for i in seen  # modal: above everything
               if i.objectName().startswith(("screen:", "modal:")) and not window.contains(box(i))]
    # A canvas is a picture of its own size in memory: one that grows with the zoomed photo
    # took 8 GB at 8x and a third of a second at every step of the zoom (the crop frame's).
    room = max(1.0, view.width() * view.height())
    faults += [f"a canvas of {i.width():.0f} x {i.height():.0f} is kept: {label(i)}" for i in items(view.contentItem())
               if "Canvas" in i.metaObject().className() and i.width() * i.height() > 1.2 * room]
    for i in seen:
        if "QQuickText" not in i.metaObject().className():
            continue
        clipped = i.property("truncated") or i.property("contentWidth") > i.width() + 1
        if clipped:
            faults.append(f'text clipped: "{str(i.property("text"))[:40]}"')
    return faults


def audit_sliders(root) -> list[str]:
    """The fill must run exactly between the tick at neutral and the middle of the thumb, and
    the track must be as long as the slider is wide: its length is how fine the control is."""
    faults = []
    for slider in (i for i in items(root) if i.objectName().startswith("slider:") and i.isVisible()):
        fill, tick, thumb = (box(slider.findChild(QObject, n)) for n in ("fill", "tick", "thumb"))
        if slider.findChild(QObject, "track").width() < slider.width() - 1:
            faults.append(f"{slider.objectName()}: the track is shorter than the slider is wide")
        ends = sorted((tick.center().x(), thumb.center().x()))
        if abs(fill.left() - ends[0]) > 1.5 or abs(fill.right() - ends[1]) > 1.5:
            faults.append(f"{slider.objectName()}: the fill is not between the tick and the thumb")
    return faults


def audit_photo(view, root, session) -> list[str]:
    """Look at the picture itself: is the photo there?"""
    if not session.photos or root.property("photoFailed") or root.property("photoMissing") or root.property("place") != "edit":
        return []
    if not root.property("photoReady"):
        return ["the photo never arrived"]
    area = box(root.findChild(QObject, "photo")).intersected(box(stage(root)))  # the part in view
    at = (int(area.left() + area.width() * 0.5), int(area.top() + area.height() * 0.3))
    if view.grabWindow().pixelColor(*at) == BACKGROUND:
        return ["the photo is not drawn (only the background where it should be)"]
    return []


TO_SRGB = np.array([[1.6605, -0.5876, -0.0728], [-0.1246, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]])


def audit_smooth(view, area, raw: bool) -> list[str]:
    """Down a dark gradient of the untouched chart, every pixel on screen is what the chart's
    own numbers give, to within one level. A picture that lost its floats on the way to the
    GPU shows steps here instead (it did, in Qt's texture atlas, for small pictures)."""
    chart = np.frombuffer(decode.synthetic(480, 320)[1].buf, np.float16).reshape(320, 480, 4)
    chart = chart.astype(np.float64)
    tone = edit.uniforms({}, None, raw)
    seen = view.grabWindow()
    worst = 0
    for y in range(int(area.top() + area.height() * 0.62), int(area.top() + area.height() * 0.78), 3):
        x = int(area.left() + area.width() * 0.5)
        sy = (y + 0.5 - area.top()) / area.height() * 320 - 0.5
        sx = (x + 0.5 - area.left()) / area.width() * 480 - 0.5
        y0, x0 = int(sy), int(sx)
        fy, fx = sy - y0, sx - x0
        c = ((chart[y0, x0] * (1 - fx) + chart[y0, x0 + 1] * fx) * (1 - fy)
             + (chart[y0 + 1, x0] * (1 - fx) + chart[y0 + 1, x0 + 1] * fx) * fy)[:3]
        if raw:  # a finished picture gets no profile
            p = c ** tone["curveC"]
            c = tone["curveK"] * p / (p + tone["curveS"])
        green = np.clip(TO_SRGB @ c, 0, 1)[1]
        level = 12.92 * green if green <= 0.0031308 else 1.055 * green ** (1 / 2.4) - 0.055
        worst = max(worst, abs(seen.pixelColor(x, y).green() - round(level * 255)))
    return [f"a smooth gradient is drawn in steps, up to {worst} levels off: the picture is "
            "not float on the GPU"] if worst > 1 else []


def audit_develop(view, root, session) -> list[str]:
    """Each setting does to the picture what its name says, measured on the chart's gray wedge
    (eight steps from black to white along the bottom), and undo takes it back."""
    if not root.property("photoReady"):
        return []
    was = (stage(root).property("fitted"), stage(root).property("zoom"), stage(root).property("center"))
    kept, session.history = session.history, edit.History()  # a clean edit to measure from
    session.refresh()
    call(stage(root), "fit")
    settle()
    area = box(root.findChild(QObject, "photo"))

    def wedge(step: int) -> QColor:  # what is on screen in the middle of that gray step
        return view.grabWindow().pixelColor(int(area.left() + area.width() * (step + 0.5) / 8),
                                            int(area.top() + area.height() * 0.92))

    def after(key: str, value, step: int) -> QColor:
        session.setEdit(key, value)
        session.commitEdit()
        settle()
        seen = wedge(step)
        session.undo()
        settle()
        return seen

    faults = audit_smooth(view, area, session.develops)
    mid, dark, light, high = wedge(3), wedge(1), wedge(6), wedge(5)
    if abs(mid.red() - mid.blue()) > 2 or abs(mid.red() - mid.green()) > 2:
        faults.append(f"untouched, a gray is not gray: {mid.red()} {mid.green()} {mid.blue()}")
    checks = (
        ("exposure", 1.0, 3, lambda c: c.green() > mid.green() + 20, "Exposure +1 does not brighten"),
        ("exposure", -1.0, 3, lambda c: c.green() < mid.green() - 20, "Exposure -1 does not darken"),
        ("shadows", 100.0, 1, lambda c: c.green() > dark.green() + 8, "Shadows +100 does not lift the darks"),
        ("shadows", 100.0, 6, lambda c: abs(c.green() - light.green()) < 3, "Shadows moves the lights"),
        ("highlights", -100.0, 6, lambda c: c.green() < light.green() - 8, "Highlights -100 does not hold the lights back"),
        ("highlights", -100.0, 1, lambda c: abs(c.green() - dark.green()) < 3, "Highlights moves the darks"),
        ("temperature", session.shot["temperature"] + 2000, 3, lambda c: c.red() > c.blue() + 12, "a higher Temperature does not warm"),
        ("temperature", session.shot["temperature"] - 1500, 3, lambda c: c.blue() > c.red() + 12, "a lower Temperature does not cool"),
        ("tint", session.shot["tint"] + 60, 3, lambda c: c.green() < min(c.red(), c.blue()) - 6, "a higher Tint does not go magenta"),
        ("profile", "flat", 3, lambda c: c.green() < mid.green() - 4, "the Flat profile is not flatter than Standard"),
        ("contrast", 100.0, 1, lambda c: c.green() < dark.green() - 4, "Contrast +100 does not darken the darks"),
        ("contrast", 100.0, 5, lambda c: c.green() > high.green() + 4, "Contrast +100 does not lighten the lights"),
        ("contrast", -100.0, 1, lambda c: c.green() > dark.green() + 4, "Contrast -100 does not lift the darks"),
        ("brightness", 100.0, 3, lambda c: c.green() > mid.green() + 20, "Brightness +100 does not lighten the middle"),
        ("brightness", 100.0, 0, lambda c: c.green() < 2, "Brightness moves black"),
        ("brightness", -100.0, 7, lambda c: c.green() > 253, "Brightness moves white"),
        ("curve", LIFT, 3, lambda c: abs(c.green() - curved(mid.green(), LIFT)) <= 2 and abs(c.red() - c.blue()) <= 2,
         f"under a tone curve a gray is not where the line says ({curved(mid.green(), LIFT)}), or not gray"),
        ("curve", LIFT, 7, lambda c: c.green() > 253, "the tone curve moves white"),
    )
    for key, value, step, good, what in checks:
        if key == "profile" and not session.develops:
            continue  # not offered for this photo
        seen = after(key, value, step)
        if not good(seen):
            faults.append(f"{what}: {seen.red()} {seen.green()} {seen.blue()} "
                          f"(untouched: dark {dark.green()}, mid {mid.green()}, light {light.green()})")
    back = wedge(3)
    if back != mid or session.canUndo:
        faults.append("undo does not bring the untouched picture back")
    session.setOriginal(True)
    session.setEdit("exposure", 2.0)
    settle()
    if wedge(3) != mid:
        faults.append("the original (B held) is not the picture without its edit")
    session.setOriginal(False)
    session.history = kept
    session.refresh()
    for key, value in zip(("fitted", "zoom", "center"), was):
        stage(root).setProperty(key, value)
    settle()
    return faults


def audit_view(view, root, session) -> list[str]:
    """Zoom and pan: the middle of the view is always on the photo, 100 % is one pixel for
    one, and zooming keeps the point under the pointer where it is."""
    if not root.property("photoReady"):
        return []
    faults = []
    pane, photo = stage(root), root.findChild(QObject, "photo")
    room, area = box(pane), box(photo)
    if not area.adjusted(-1, -1, 1, 1).contains(room.center()):
        faults.append("the photo was moved so far that the middle of the view is off it")
    if pane.property("shown") == 1 and round(photo.width()) != session.fullWidth:
        faults.append("at 100 % the photo is not drawn one pixel for one")
    # Zoom about a point of the view and see that the photo's point under it has not moved.
    was = (pane.property("fitted"), pane.property("zoom"), pane.property("center"))
    at = QPointF(pane.width() * 0.3, pane.height() * 0.6)
    under = photo.mapFromItem(pane, at)
    before = (under.x() / photo.width(), under.y() / photo.height())
    call(pane, "zoomTo", pane.property("shown") * 3, at.x(), at.y())
    settle()
    under = photo.mapFromItem(pane, at)
    after = (under.x() / photo.width(), under.y() / photo.height())
    inside = all(0.02 < v < 0.98 for v in before)  # a point off the photo has nothing to hold
    if inside and max(abs(a - b) for a, b in zip(before, after)) > 2 / min(photo.width(), photo.height()):
        faults.append(f"zooming moved the point under the pointer: {before} -> {after}")
    for key, value in zip(("fitted", "zoom", "center"), was):
        pane.setProperty(key, value)
    settle()
    return faults


def audit_gestures(view, root, session) -> list[str]:
    """The real events, sent to the window as a mouse and a keyboard would: the wheel zooms
    about the pointer, a drag moves the photo at any zoom, as far as a corner in the middle of
    the view, a double click goes to 1:1 and back, B shows the original."""
    if not root.property("photoReady"):
        return []
    faults = []
    pane, photo = stage(root), root.findChild(QObject, "photo")
    was = (pane.property("fitted"), pane.property("zoom"), pane.property("center"))
    origin = box(pane).topLeft()
    at = QPointF(origin.x() + pane.width() * 0.4, origin.y() + pane.height() * 0.5)

    def send(event) -> None:
        QCoreApplication.sendEvent(view, event)
        settle()

    def mouse(kind, where, button, held) -> None:
        send(QMouseEvent(kind, where, where, button, held, Qt.KeyboardModifier.NoModifier))

    def drag(button, by=QPointF(70, 50)) -> None:
        mouse(QEvent.Type.MouseButtonPress, at, button, button)
        mouse(QEvent.Type.MouseMove, at + by, Qt.MouseButton.NoButton, button)
        mouse(QEvent.Type.MouseButtonRelease, at + by, button, Qt.MouseButton.NoButton)

    def under() -> tuple[float, float]:
        point = photo.mapFromScene(at)
        return point.x() / photo.width(), point.y() / photo.height()

    call(pane, "fit")
    settle()
    before, size = photo.x(), pane.property("shown")
    drag(Qt.MouseButton.LeftButton)
    if abs((photo.x() - before) - 70) > 1.5 or pane.property("shown") != size:
        faults.append("dragging a fitted photo does not move it as it is")
    # As far as it goes: a corner of the photo in the middle of the view, and no further.
    for by, corner in ((QPointF(9000, 9000), "topLeft"), (QPointF(-20000, -20000), "bottomRight")):
        drag(Qt.MouseButton.LeftButton, by)
        gap = getattr(box(photo), corner)() - box(pane).center()
        if abs(gap.x()) > 1.5 or abs(gap.y()) > 1.5:
            faults.append(f"dragged to the end, the photo's {corner} corner is {gap.x():.0f}, {gap.y():.0f} px "
                          "from the middle of the view")
    call(pane, "fit")
    settle()
    held, zoom = under(), pane.property("shown")
    send(QWheelEvent(at, at, QPoint(0, 0), QPoint(0, 240), Qt.MouseButton.NoButton,
                     Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
    if not pane.property("shown") > zoom * 1.2:
        faults.append("turning the wheel does not zoom in")
    elif max(abs(a - b) for a, b in zip(held, under())) > 2 / photo.width():
        faults.append("the wheel zoomed, but not about the pointer")

    # The wheel goes by steps, the same ones each way: what was left is come back to exactly.
    def turn(by: int) -> float:
        send(QWheelEvent(at, at, QPoint(0, 0), QPoint(0, by), Qt.MouseButton.NoButton,
                         Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
        return pane.property("shown")

    # Zooming out is about the pointer, as zooming in is: the point of the photo under the
    # pointer stays under it, and nothing else moves the photo (the owner's wish; it used to
    # be gathered toward the middle on the way out).
    side = QPointF(origin.x() + pane.width() * 0.6, origin.y() + pane.height() * 0.45)

    def under() -> float:
        return (side.x() - origin.x() - photo.x()) / photo.width()

    stood = under()
    send(QWheelEvent(side, side, QPoint(0, 0), QPoint(0, -120), Qt.MouseButton.NoButton,
                     Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
    if abs(under() - stood) > 2 / photo.width():
        faults.append(f"zoomed out at one side, the point under the pointer went from {stood:.3f} to {under():.3f} of the photo")
    send(QWheelEvent(at, at, QPoint(0, 0), QPoint(0, 120), Qt.MouseButton.NoButton,
                     Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
    turn(-240)
    if pane.property("fitted") or abs(pane.property("shown") - zoom) > 1e-9:
        faults.append(f"two notches in from Fit and two out is not the size of Fit again, or the wheel put the view on Fit by itself: {pane.property('shown'):.4f}, fitted {pane.property('fitted')}")
    call(pane, "actual")
    settle()
    # (Fit is one of the steps, wherever it lies; a step a hair from it is left out.)
    steps = sorted([z for z in plain(pane.property("steps")) if abs(z / zoom - 1) > 0.04 and z >= pane.property("minZoom") * 0.999] + [zoom])
    above, under_it = [z for z in steps if z > 1.0005][:5], [z for z in reversed(steps) if z < 0.9995][:3]
    there = [turn(120) for _ in above] + [turn(-120) for _ in above]
    near = lambda got, want: len(got) == len(want) and all(abs(a - b) < 1e-9 for a, b in zip(got, want))  # noqa: E731
    if not near(there, above + above[-2::-1] + [1.0]):
        faults.append(f"from 1:1, five notches in and five out went {there}: not the steps {above}, or not back at 1:1")
    below = [turn(-120) for _ in under_it]
    back = [turn(120) for _ in under_it]
    if not near(below, under_it) or not near(back[-1:], [1.0]):
        faults.append(f"from 1:1, three notches out and three in went {below} then {back}: not the steps {under_it}, or not back at 1:1")
    parts = [turn(40), turn(40), turn(40)]
    if not near(parts, [1.0, 1.0, above[0]]):
        faults.append(f"three thirds of a notch should be one step, at the third: {parts}")
    call(pane, "fit")  # (and from where the wheel first left it, for what follows)
    settle()
    turn(240)
    call(pane, "zoomTo", 6.0, at.x() - origin.x(), at.y() - origin.y())
    settle()
    before = photo.x()
    drag(Qt.MouseButton.LeftButton)
    if abs((photo.x() - before) - 70) > 1.5:
        faults.append(f"a left-button drag of 70 px moved the zoomed photo {photo.x() - before:.0f} px")
    drag(Qt.MouseButton.LeftButton, QPointF(-70, -50))
    # The middle button fits the photo as it goes down; dragged on, it does not move it.
    call(pane, "fit")
    settle()
    fitted_at = photo.x()
    call(pane, "zoomTo", 6.0, at.x() - origin.x(), at.y() - origin.y())
    settle()
    drag(Qt.MouseButton.MiddleButton)
    if not pane.property("fitted") or abs(photo.x() - fitted_at) > 2:
        faults.append(f"a middle-button drag of 70 px from a zoomed photo: fitted {pane.property('fitted')}, "
                      f"{photo.x() - fitted_at:.0f} px from where Fit has it")
    call(pane, "fit")
    settle()
    for expected in (1.0, None):
        # Qt's own test click: a double click built by hand from events was not taken for one.
        QTest.mouseDClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at.toPoint())
        settle()
        if expected and pane.property("shown") != expected:
            faults.append("a double click on a fitted photo does not go to 1:1")
        if not expected and not pane.property("fitted"):
            faults.append("a second double click does not go back to Fit")
    # A middle click, not a drag, puts the photo back to Fit.
    call(pane, "zoomTo", 2.0, at.x() - origin.x(), at.y() - origin.y())
    settle()
    QTest.mouseClick(view, Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, at.toPoint(), 300)
    settle()
    if not pane.property("fitted"):
        faults.append("a middle click on the photo does not put it back to Fit")
    # Pressed again and again, fast, it stays on Fit (as a double click it went to 1:1).
    QTest.mouseDClick(view, Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, at.toPoint())
    settle()
    if not pane.property("fitted"):
        faults.append(f"a fast second middle click left Fit: {pane.property('shown'):.2f}")

    def key(kind, which) -> None:  # to the window's root, as if it had the keys
        QCoreApplication.sendEvent(root, QKeyEvent(kind, which, Qt.KeyboardModifier.NoModifier))
        settle()

    key(QEvent.Type.KeyPress, Qt.Key.Key_B)
    if not session.original:
        faults.append("holding B does not show the original")
    key(QEvent.Type.KeyRelease, Qt.Key.Key_B)
    if session.original:
        faults.append("letting go of B does not bring the edit back")
    # While the next photo is on its way it has no size: a drag or a notch then must leave the
    # view as it is (they put it at 100 % in a corner, and the photo came there).
    call(pane, "fit")
    pane.setProperty("missing", True)   # (what `ready` reads; `have` follows the pictures)
    settle()
    drag(Qt.MouseButton.LeftButton)
    turn(120)
    if not pane.property("fitted") or pane.property("center") != QPointF(0.5, 0.5):
        faults.append(f"a drag and a notch while no photo is shown moved the view: "
                      f"fitted {pane.property('fitted')}, center {pane.property('center')}")
    pane.setProperty("missing", False)
    settle()
    for key_, name in zip(("fitted", "zoom", "center"), was):
        pane.setProperty(key_, name)
    settle()
    return faults


def audit_modal(view, root) -> list[str]:
    """While a question is open, the photo behind it takes no wheel and no drag."""
    pane, photo = stage(root), root.findChild(QObject, "photo")
    at = QPointF(box(pane).left() + 60, box(pane).top() + 60)  # beside the dialog, on the photo
    zoom, place = pane.property("shown"), photo.x()
    QCoreApplication.sendEvent(view, QWheelEvent(
        at, at, QPoint(0, 0), QPoint(0, 240), Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
    QTest.mousePress(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at.toPoint())
    QTest.mouseMove(view, (at + QPointF(80, 40)).toPoint())
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                       (at + QPointF(80, 40)).toPoint())
    settle()
    if pane.property("shown") != zoom or photo.x() != place:
        return ["with a question open, the photo behind it still zooms or moves"]
    # The arrows are the question's own: from one button to the other, and not to the photos.
    source = root.property("openPath")
    yes, no = find(root, "confirm-yes"), find(root, "confirm-no")
    if find(root, "modal:denoise").isVisible():  # (the Neural Denoise window: its own two)
        yes, no = find(root, "denoise-confirm"), find(root, "denoise-cancel")
        if not yes.isEnabled():  # (nothing to confirm yet: the keys stay on Cancel)
            QTest.keyClick(view, Qt.Key.Key_Right)
            settle()
            return [] if no.hasActiveFocus() and root.property("openPath") == source else [
                "with Neural Denoise open and nothing to confirm, the right arrow leaves Cancel or steps to another photo"]
    QTest.keyClick(view, Qt.Key.Key_Right)
    settle()
    if not yes.hasActiveFocus() or root.property("openPath") != source:
        return [f"with a question open, the right arrow does not go to its other button (it has the keys: {yes.hasActiveFocus()}), or steps to another photo"]
    QTest.keyClick(view, Qt.Key.Key_Left)
    settle()
    if not no.hasActiveFocus():
        return ["with a question open, the left arrow does not go back to Cancel"]
    return []


def audit_checkpoints() -> list[str]:
    """Checkpoints as the owner asked for them, with real clicks: saving lists and lights one;
    a change turns its mark to "changed"; going to another then asks first, Cancel leaves the
    edit alone, the other answer discards it; with nothing unsaved it goes without asking."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session)
    faults = []

    def states() -> dict:
        return {c["id"]: c["state"] for c in session.checkpoints}

    def asking() -> bool:
        return find(root, "modal:confirm").isVisible()

    def expect(what: bool, fault: str) -> None:
        if not what:
            faults.append(fault)

    session.setEdit("exposure", 0.5)
    session.commitEdit()
    expect(session.unsaved, "an edit in no checkpoint does not count as unsaved")
    click(view, find(root, "checkpoint-save"))
    session.setEdit("exposure", 1.5)
    session.commitEdit()
    click(view, find(root, "checkpoint-save"))
    expect(states() == {1: "", 2: "here"} and not session.unsaved, f"after two saves: {states()}")
    expect(find(root, "checkpoint:2") is not None, "a saved checkpoint is not in the list")

    session.setEdit("shadows", 30)
    session.commitEdit()
    settle()
    expect(states() == {1: "", 2: "changed"} and session.unsaved, f"after a change: {states()}")
    click(view, find(root, "checkpoint:1"))
    expect(asking(), "going to a checkpoint with unsaved changes does not ask")
    expect(session.edit["exposure"] == 1.5, "the edit was replaced before the question was answered")
    click(view, find(root, "confirm-no"))
    expect(not asking() and session.edit["shadows"] == 30 and states()[2] == "changed", "Cancel did not leave everything as it was")
    click(view, find(root, "checkpoint:1"))
    click(view, find(root, "confirm-yes"))
    expect(not asking() and session.history.edit == {"exposure": 0.5} and states() == {1: "here", 2: ""},
           f"discarding did not go to the checkpoint: {session.history.edit} {states()}")
    click(view, find(root, "checkpoint:2"))
    expect(not asking() and session.edit["exposure"] == 1.5 and states() == {1: "", 2: "here"},
           "with nothing unsaved, going to a checkpoint asked or did not go")
    session.undo()
    settle()
    expect(states() == {1: "here", 2: ""}, f"undo after going to a checkpoint: {states()}")
    ui.close_view(view)
    return faults


def audit_column() -> list[str]:
    """A low window: Checkpoints stands at the foot of the left panel, the cards over it clear
    of it; the right column scrolls by the wheel, and so does the checkpoints' list; a drag on
    the tone curve moves the point and scrolls nothing."""
    session = ui.Session([Path("first.arw")], load=chart)
    checkpoints(session, 9)
    session.resetAll()
    view = ui.make_view(session)
    session.surface_known(False)
    view.resize(*SMALL)
    root = view.rootObject()
    root.setSize(view.size())
    wait(lambda: root.property("photoReady"))
    view.grabWindow()
    show(root, "curve")
    faults = Faults()
    left, foot, folders = (box(find(root, name)) for name in ("box:left", "checkpoints", "panel:folders"))
    upper, rows = find(root, "tools-scroll"), find(root, "checkpoint-list")
    faults.expect(abs(left.bottom() - foot.bottom() - 12) < 1, f"Checkpoints is not at the foot of the left panel: {foot.getRect()} in {left.getRect()}")
    # (A card's ground ends 16 px over the foot of a folded panel, and Checkpoints' begins 8 px
    # under its top: between the two grounds there are the 8 px that part every pair of cards.)
    faults.expect(folders.bottom() - 16 <= foot.top() + 1, f"the cards over Checkpoints reach into it: Folders ends at {folders.bottom()}, Checkpoints begins at {foot.top()}")
    faults.expect(upper.property("contentHeight") > upper.height(), "(the window is not low enough to try scrolling)")

    def wheel(over, notches: int, glide=400) -> None:
        at = box(over).center()
        QCoreApplication.sendEvent(view, QWheelEvent(
            at, at, QPoint(0, 0), QPoint(0, -120 * notches), Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
        wait(lambda: False, glide)  # the scroll glides to its place

    area, was = box(find(root, "curve-plot")), upper.property("contentY")
    drag(view, area.center().toPoint(), QPoint(0, -40))
    faults.expect(upper.property("contentY") == was and len(session.history.edit.get("curve", [])) == 3,
                  f"a drag on the curve scrolled the column or made no point: {was} -> {upper.property('contentY')}")
    upper.setProperty("contentY", 0)
    wheel(upper, 3)
    faults.expect(upper.property("contentY") > 20, "the wheel does not scroll the column")
    # Notches that come during the glide add up: a wheel turned fast goes as far as it says.
    # (Each went on from where the column was, not from where it was going: the owner,
    # "never getting faster if i really scroll faster".)
    upper.setProperty("contentY", 0)
    wheel(upper, 1)
    one = upper.property("contentY")
    upper.setProperty("contentY", 0)
    wheel(upper, 1, 20)
    wheel(upper, 1, 20)
    wheel(upper, 1)
    reach = upper.property("contentHeight") - upper.height()
    faults.expect(reach > 3 * one and abs(upper.property("contentY") - 3 * one) < 2,
                  f"three fast notches scrolled the column {upper.property('contentY'):.0f} px, one goes {one:.0f} (of {reach:.0f})")
    rows.setProperty("contentY", rows.property("originY"))
    wheel(rows, 3)
    faults.expect(rows.property("contentY") - rows.property("originY") > 20, "the wheel does not scroll the checkpoints")
    faults.expect(stage(root).property("fitted"), "a wheel over the column zoomed the photo")
    ui.close_view(view)
    return faults


def oklab_of(color: QColor) -> np.ndarray:
    """(lightness, a, b) of a color on screen."""
    coded = np.array([color.red(), color.green(), color.blue()]) / 255
    linear = np.where(coded <= 0.04045, coded / 12.92, ((coded + 0.055) / 1.055) ** 2.4)
    return np.linalg.inv(scopes.LAB_TO_ROOTS) @ np.cbrt(np.linalg.inv(scopes.LMS_TO_SRGB) @ linear)


def audit_wheel_data() -> list[str]:
    """What the wheel is fed is the picture itself: the color the session gives for a spot of
    the photo is the color on screen there, a gray is in the middle of the wheel, and every
    dot is drawn where its a and b say, in its own color."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "")
    session.setPanel("live", False)  # (Before: what this audit starts from; After is where a new session starts)
    faults = Faults()
    faults.expect(counted(view, session), "the photo's colors are never counted")
    area = box(root.findChild(QObject, "photo"))

    def agrees(when: str) -> None:
        shot, compared = view.grabWindow(), 0
        for u, v in ((0.44, 0.92), (0.3, 0.7), (0.62, 0.55), (0.12, 0.45), (0.85, 0.3), (0.5, 0.75)):
            seen = shot.pixelColor(int(area.left() + u * area.width()), int(area.top() + v * area.height()))
            if not all(4 < c < 251 for c in (seen.red(), seen.green(), seen.blue())):
                continue  # cut off by the screen's own range: nothing to compare
            a, b, light = session.colorAt(u, v)
            off = np.abs(np.array([light, a, b]) - oklab_of(seen)).max()
            compared += 1
            faults.expect(off < 0.02, f"{when}, at {u}, {v} the photo shows {seen.name()} but the wheel is "
                                      f"told L {light:.3f} a {a:.3f} b {b:.3f} ({off:.3f} off)")
        faults.expect(compared >= 2, f"{when}, too few spots of the chart could be compared")

    agrees("untouched")
    session.setEdits({"curve": LIFT, "exposure": -0.6, "temperature": 6400})
    session.commitEdit()
    settle()
    counted(view, session)
    agrees("with an edit and a tone curve")
    session.resetAll()
    settle()
    counted(view, session)
    # With a pin: "before" stays the photo's own colors, "after" is what is on screen.
    a, b, lit = session.colorAt(0.2, 0.62)
    was = session.cloud["box"]
    session.setEdits({"pins": [{"from": [a, b], "to": [a + 0.05, b - 0.04], "range": 0.09}], "saturation": 40.0})
    session.commitEdit()
    settle()
    counted(view, session)
    faults.expect(np.allclose(session.colorAt(0.2, 0.62), [a, b, lit], atol=0.004), "a pin changed the colors it takes hold of")
    faults.expect(np.allclose(session.cloud["box"], was, atol=1e-4), "showing the colors before the pins, the dots moved with a pin")
    click(view, find(root, "wheel-after"))
    seen = oklab_of(view.grabWindow().pixelColor(int(area.left() + 0.2 * area.width()), int(area.top() + 0.62 * area.height())))
    after = [float(v[int(0.62 * 255), int(0.2 * 255)]) for v in session._after]  # lightness, a, b
    faults.expect(np.abs(np.array(after) - seen).max() < 0.02, f"after the pins the wheel is told {np.round(after, 3)}, the photo shows {np.round(seen, 3)}")
    faults.expect(session.panels["live"] and not np.allclose(session.cloud["box"], was, atol=1e-3), "After does not show the dots as the pins left them")
    click(view, find(root, "wheel-before"))
    faults.expect(np.allclose(session.cloud["box"], was, atol=1e-4), "Before does not bring the photo's own colors back")
    session.resetAll()
    settle()
    counted(view, session)
    gray = session.colorAt(0.44, 0.92)
    faults.expect(abs(gray[0]) < 0.004 and abs(gray[1]) < 0.004, f"a gray is not in the wheel's middle: {gray}")
    wheel, cloud = find(root, "wheel"), session.cloud
    faults.expect(len(cloud["colors"]) > 200, f"the chart gives only {len(cloud['colors'])} dots")
    call(wheel, "showAll")
    settle()
    shot, place = view.grabWindow(), box(wheel)
    per = place.height() / wheel.property("span")

    # The disc itself: red to the right, yellow up, green to the left, blue down.
    def disc(a: float, b: float) -> QColor:
        return shot.pixelColor(round(place.center().x() + a * per), round(place.center().y() - b * per))

    # (Down and a little to the left: straight down, Fit and All lie over the disc.)
    right, up, left, down = disc(0.29, 0.0), disc(0.0, 0.29), disc(-0.29, 0.0), disc(-0.17, -0.23)
    faults.expect(right.red() > right.green() + 30 and up.blue() < min(up.red(), up.green()) - 30
                  and left.red() < left.green() - 30 and down.blue() > down.red() + 30,
                  f"the wheel's colors are not where they belong: right {right.name()} up {up.name()} "
                  f"left {left.name()} down {down.name()}")
    # The dots: drawn by the GPU from the photo itself (dots.vert), so a dot lies where a
    # color of the photo lies, and has that hue. Without them the wheel is only its disc.
    spots = [(u / 13, v / 9) for v in range(1, 7) for u in range(1, 13)]

    def wheel_as_drawn(with_dots: bool) -> np.ndarray:
        """The wheel with its dots or without them. (Without the dark ground under them
        either way: it is not a color of the photo.)"""
        find(root, "wheel-dots").setVisible(with_dots)
        find(root, "wheel-ink").setVisible(False)
        QCoreApplication.processEvents()  # no waiting: they are drawn with the frame
        seen = export.bytes_of(view.grabWindow().copy(place.toRect())).astype(int)
        find(root, "wheel-dots").setVisible(True)
        find(root, "wheel-ink").setVisible(True)
        return seen

    def dots(span: float, middle: tuple, when: str, look: dict | None = None) -> int:
        """How much of the wheel is dots; and every spot's color must have one, of its hue."""
        wheel.setProperty("span", span)
        wheel.setProperty("middle", QPointF(*middle))
        bare, drawn = wheel_as_drawn(False), wheel_as_drawn(True)
        is_dot = np.abs(drawn - bare).max(axis=2) > 8
        per, tried, wrong, notes = place.height() / span, 0, 0, []
        reach = max(5, round(0.004 * per))  # the photo's colors are sampled: that far to the next dot at most
        for u, v in spots:
            a, b, light = session.colorAt(u, v)
            if look:
                light, a, b = edit.colors_at(look, [light, a, b])
            x, y = round(place.width() / 2 + (a - middle[0]) * per), round(place.height() / 2 - (b - middle[1]) * per)
            if not (reach + 4 <= x < place.width() - reach - 4 and 44 + reach <= y < place.height() - 64 - reach):
                continue  # out of view, or under the wheel's buttons
            # (Of the dots in reach, the pixel most like the color: other colors' dots lie there
            # too, darker ones that a screen cannot show as vivid, and every dot has a soft edge.)
            near = [oklab_of(QColor(*(int(c) for c in drawn[y + dy, x + dx])))
                    for dy in range(-reach, reach + 1, 2) for dx in range(-reach, reach + 1, 2) if is_dot[y + dy, x + dx]]
            tried += 1
            off = min((np.hypot(seen[1] - a, seen[2] - b) for seen in near), default=9.0)
            if off > 0.012:
                wrong += 1
                notes.append(f"{u:.2f},{v:.2f}: a {a:.3f} b {b:.3f} L {light:.2f}, the nearest in color of {len(near)} dot pixels is {off:.3f} off")
        faults.expect(tried >= 8 and wrong <= tried // 8, f"{when}: {wrong} of {tried} colors of the photo have no dot of their hue where they lie ({'; '.join(notes[:5])})")
        return int(is_dot.sum())

    whole = dots(wheel.property("span"), (0.0, 0.0), "the whole wheel")
    faults.expect(whole > 2000, f"the dots cover only {whole} pixels of the wheel")
    vivid = session.colorAt(0.2, 0.3)
    close = dots(0.12, vivid[:2], "a close view, straight after it moved")
    faults.expect(close < 0.3 * place.width() * place.height(), f"seen closely the dots still cover {close} pixels: what ties them together is drawn")
    # In the same frame as the photo: nothing is counted in between.
    call(wheel, "showAll")
    click(view, find(root, "wheel-before")) if find(root, "wheel-before").isVisible() else None
    session.setEdits({"saturation": 45.0})
    session.setPanel("live", True)
    less = dots(wheel.property("span"), (0.0, 0.0), "after the look, saturation at 45", {"saturation": 45.0})
    faults.expect(less < 0.6 * whole, f"with less saturation the dots cover {less} pixels, {whole} before: they did not follow the photo")
    session.setPanel("live", False)
    back = dots(wheel.property("span"), (0.0, 0.0), "before the pins, whatever the look")
    faults.expect(abs(back - whole) < 0.05 * whole, f"Before does not show the photo's own colors: {back} pixels of dots, {whole} untouched")
    session.resetAll()
    session.setRange("shadows")
    QCoreApplication.processEvents()
    dark = int((np.abs(wheel_as_drawn(True) - wheel_as_drawn(False)).max(axis=2) > 8).sum())
    faults.expect(0 < dark < 0.7 * whole, f"with Shadows chosen the dots cover {dark} pixels of the wheel, {whole} with all tones")
    ui.close_view(view)
    return faults


def audit_dots_place() -> list[str]:
    """Seen as closely as the wheel goes, a dot lies where its color does, to a pixel: for
    photos of one color each, the middle of what the dots cover is where the color's a and b
    say. (Kept too coarsely, as they were for a while, the dots stood in a grid some pixels
    wide: the owner, "why are the points in a grid? they used to not be in a grid".)"""
    colors = [(0.30, 0.21, 0.13), (0.12, 0.25, 0.31), (0.33, 0.30, 0.09), (0.10, 0.19, 0.11), (0.27, 0.12, 0.24), (0.21, 0.33, 0.36)]

    def one_color(path: Path, half: bool):
        rgb = np.full((64, 96, 3), colors[int(path.stem)], np.float32)
        return decode.meta(path.name, 96, 64), decode.Pixels(decode.pack(rgb))

    session = ui.Session([Path(f"{n}.png") for n in range(len(colors))], load=one_color)
    view, root = opened(session, "")
    faults, wheel, telling = Faults(), find(root, "wheel"), 0
    place = box(wheel)

    def drawn(with_dots: bool) -> np.ndarray:
        find(root, "wheel-dots").setVisible(with_dots)
        find(root, "wheel-ink").setVisible(False)
        QCoreApplication.processEvents()
        seen = export.bytes_of(view.grabWindow().copy(place.toRect())).astype(int)
        find(root, "wheel-dots").setVisible(True)
        find(root, "wheel-ink").setVisible(True)
        return seen

    for n in range(len(colors)):
        wait(lambda: root.property("photoReady"))
        settle()
        faults.expect(counted(view, session), "the photo's colors are never counted")
        a, b, _ = session.colorAt(0.5, 0.5)
        span = wheel.property("whole") / 24
        wheel.setProperty("span", span)
        wheel.setProperty("middle", QPointF(a - 0.21 * span, b + 0.13 * span))
        settle()
        view.grabWindow()
        covered = np.abs(drawn(True) - drawn(False)).max(axis=2) > 8
        rows, columns = np.nonzero(covered)
        if len(rows) < 12:
            faults.append(f"a photo of one color has no dot on the wheel ({len(rows)} pixels)")
        else:
            per = place.height() / span
            want = (place.width() / 2 + 0.21 * span * per, place.height() / 2 + 0.13 * span * per)
            off = float(np.hypot(columns.mean() + 0.5 - want[0], rows.mean() + 0.5 - want[1]))
            # How far off it would be with a and b kept to 0.0005: a color that would be near
            # enough either way tells nothing.
            coarse = max(abs(round((v + 0.5) / 2 ** -11) * 2 ** -11 - 0.5 - v) for v in (a, b)) * per
            telling += coarse > 1.2
            faults.expect(off <= 1.0, f"seen closely, the dot of a {a:.4f} b {b:.4f} lies {off:.1f} px from where its color does")
        session.step(1)
    faults.expect(telling >= 2, f"only {telling} of the colors tried would show a coarse place: the audit proves too little")
    ui.close_view(view)
    return faults


def audit_pixels() -> list[str]:
    """From 200 % on, the photo shows its own pixels: of a picture that is a checkerboard of
    two grays, a pixel wide each, nearly every pixel of the screen is one of the two at 400 %,
    300 % and 200 % (squares; a blur has nearly none); below 200 % nothing is changed. For a
    finished picture and for a raw file's full-size one alike."""
    board = np.indices((300, 400)).sum(axis=0) % 2  # 0 or 1, turn about
    levels = np.where(board[..., None] > 0, 0.35, 0.03).astype(np.float32) * np.ones(3, np.float32)

    def finished(path: Path, half: bool):
        return decode.meta(path.name, 400, 300), decode.Pixels(decode.pack(levels))

    def sensed(path: Path, half: bool):  # a raw file: a quick picture of half the size (one gray: the two mixed), and the full one
        about = {"raw": True, "camera": decode.synthetic(8, 4)[0]["camera"]}
        if half:
            return {**decode.meta(path.name, 200, 150, (400, 300)), **about}, decode.Pixels(decode.pack(np.full((150, 200, 3), 0.19, np.float32)))
        return {**decode.meta(path.name, 400, 300), **about}, decode.Pixels(decode.pack(levels))

    faults = Faults()
    for kind, load, name in (("a finished picture", finished, "board.png"), ("a raw file", sensed, "board.arw")):
        session = ui.Session([Path(name)], load=load)
        view, root = opened(session, "")
        pane, photo = stage(root), root.findChild(QObject, "photo")
        wait(lambda: pane.property("sharpReady") or kind != "a raw file", 4000)

        def middle(zoom: float) -> np.ndarray:  # the green of the screen's pixels in the middle of the view
            call(pane, "zoomTo", zoom, pane.width() / 2, pane.height() / 2)
            settle()
            at = box(pane).center().toPoint()
            return export.bytes_of(view.grabWindow().copy(at.x() - 60, at.y() - 60, 120, 120))[..., 1].astype(int)

        low, high = int(middle(4.0).min()), int(middle(4.0).max())  # the two grays as they are shown
        faults.expect(high - low > 60, f"{kind} at 400 %: its two grays are shown as {low} and {high}: no board to be seen")
        for zoom in (4.0, 3.0, 2.0):
            seen = middle(zoom)
            share = float(((np.abs(seen - low) < 4) | (np.abs(seen - high) < 4)).mean())
            faults.expect(share > 0.9, f"{kind} at {zoom:.0%}: {share:.0%} of the screen's pixels are one of the photo's two grays (squares expected)")
        for zoom in (1.5, 1.0, 2 / 3):
            middle(zoom)
            faults.expect(bool(photo.property("crisp")) is False, f"{kind}: the pixels' squares stay on at {zoom:.0%}")
        ui.close_view(view)
    return faults


def audit_colors_data() -> list[str]:
    """What is done to colors is on screen what the arithmetic says (edit.colors_at): for each
    kind of pin, for saturation, the chroma curve and the color EQ, and for all at once, the
    pixel after is the pixel before, changed as predicted; and each really changes something,
    or leaves alone what it must."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "")
    faults = Faults()
    area = box(root.findChild(QObject, "photo"))
    spots = [(0.2, 0.62)] + [(u / 16, v) for v in (0.3, 0.4, 0.5, 0.62, 0.74) for u in range(1, 16)] + [(0.44, 0.92)]

    def shown(shot, spot) -> QColor:
        return shot.pixelColor(int(area.left() + spot[0] * area.width()), int(area.top() + spot[1] * area.height()))

    def seen() -> list[np.ndarray]:
        shot = view.grabWindow()
        return [oklab_of(shown(shot, spot)) for spot in spots]

    # Only spots the screen shows in full: a color cut off at its limit is not the color it was.
    first = view.grabWindow()
    spots = [spot for spot in spots if 3 <= min(shown(first, spot).getRgb()[:3]) and max(shown(first, spot).getRgb()[:3]) <= 252]
    before = seen()
    lit, a, b = before[0]  # the color every pin here is put on
    turn = np.array([[np.cos(0.7), -np.sin(0.7)], [np.sin(0.7), np.cos(0.7)]]) @ [a, b] * 0.8
    here = {"from": [a, b], "range": 0.1, "feather": 60}
    hue, guard = float(np.degrees(np.arctan2(b, a))) % 360, {**here, "range": 0.03, "feather": 0, "mode": "protect"}
    cases = {  # name: (the settings, whether the spot they are aimed at must change)
        "offset": ({"pins": [{**here, "to": [a + 0.06, b - 0.045]}]}, True),
        "rotate": ({"pins": [{**here, "to": turn.tolist(), "mode": "rotate"}]}, True),
        "lighter": ({"pins": [{**here, "light": 0.08}]}, True),
        "variance": ({"pins": [{**here, "range": 0.2, "variance": -90}]}, False),
        "only in the lights": ({"pins": [{**here, "range": 0.2, "to": [a + 0.06, b - 0.045], "low": lit + 0.12}]}, False),
        "protected": ({"pins": [{**here, "to": [a + 0.06, b - 0.045]}, guard]}, False),
        "two pins": ({"pins": [{**here, "to": [a + 0.03, b]}, {**here, "to": [a, b + 0.03], "light": -0.05}]}, True),
        "more saturation": ({"saturation": 135.0}, True),
        "less saturation": ({"saturation": 40.0}, True),
        "saturation, but protected": ({"saturation": 40.0, "pins": [guard]}, False),
        "more vibrance": ({"vibrance": 80.0}, True),
        "less vibrance": ({"vibrance": -70.0}, True),
        "vibrance, but protected": ({"vibrance": 80.0, "pins": [guard]}, False),
        "chroma curve": ({"chroma": [[0.0, 0.0], [0.2, 0.36], [1.0, 1.0]]}, True),
        "color EQ on that hue": ({"eq": [{"hue": hue, "gain": 45, "width": 50}]}, True),
        "color EQ on the hue across": ({"eq": [{"hue": hue + 180, "gain": 45, "width": 50}]}, False),
        "all at once": ({"pins": [{**here, "to": [a + 0.03, b - 0.02]}], "saturation": 60.0,
                         "chroma": [[0.0, 0.0], [0.3, 0.4], [1.0, 1.0]], "eq": [{"hue": hue + 30, "gain": 140, "width": 70}]}, True),
    }
    for name, (values, moves) in cases.items():
        session.setEdits(values)
        session.commitEdit()
        settle()
        after, compared, changed = seen(), 0, 0
        for was, now in zip(before, after):
            want = edit.colors_at(edit.clean(values), was)
            if scopes.srgb_of(*want).min() < 3 or scopes.srgb_of(*want).max() > 252:
                continue  # at the screen's limit: cut off, nothing to compare
            if any(pin.get("feather") == 0 and abs(np.hypot(*(was[1:] - pin["from"])) - pin["range"]) < 0.006
                   for pin in values.get("pins", [])):
                continue  # on a hard edge of a pin: what the screen's 8 bits say is not which side it is on
            compared += 1
            changed += np.abs(now - was).max() > 0.01
            off = np.abs(now - want).max()
            faults.expect(off < 0.01, f"{name}: a color {np.round(was, 3)} should become {np.round(want, 3)} "
                                       f"and is {np.round(now, 3)} on screen")
        faults.expect(compared >= 8 and changed >= 1, f"{name}: too little to go by ({compared} spots, {changed} changed)")
        moved = np.abs(after[0] - before[0]).max()
        faults.expect((moved > 0.02) == moves, f"{name}: the pinned color itself moved by {moved:.3f}")
        session.undo()
    faults.expect(session.history.edit == {}, "undoing each of them does not leave the photo untouched")
    ui.close_view(view)
    return faults


def audit_pins() -> list[str]:
    """Pins under a real mouse and keyboard: a click marks a color and changes nothing, dragging
    the mark makes a pin and moves it, the pin and its origin can each be dragged, the mouse
    wheel on a pin sets its range (Shift: feather, Alt: variance) and elsewhere still zooms, a
    click on the photo marks that color, Escape lets go, Delete and a double click remove."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "")
    faults = Faults()
    counted(view, session)
    wheel = find(root, "wheel")
    call(wheel, "showAll")
    settle()
    place = box(wheel)
    per = place.height() / wheel.property("span")

    def at(a: float, b: float) -> QPoint:
        return QPoint(round(place.center().x() + a * per), round(place.center().y() - b * per))

    def pins() -> list:
        return session.history.edit.get("pins", [])

    def near(found, want, by=0.003) -> bool:
        return bool(found) and all(abs(f - w) < by for f, w in zip(found, want))

    def turn(where: QPoint, held=Qt.KeyboardModifier.NoModifier) -> None:
        spot = QPointF(where)
        QCoreApplication.sendEvent(view, QWheelEvent(spot, spot, QPoint(0, 0), QPoint(0, 240), Qt.MouseButton.NoButton,
                                                     held, Qt.ScrollPhase.NoScrollPhase, False))
        settle()

    def key(which) -> None:
        QCoreApplication.sendEvent(root, QKeyEvent(QEvent.Type.KeyPress, which, Qt.KeyboardModifier.NoModifier))
        settle()

    tap(view, at(0.05, 0.10))
    fresh = plain(wheel.property("fresh"))
    faults.expect(fresh and near(fresh["from"], (0.05, 0.10)), f"a click on the wheel does not mark that color: {fresh}")
    faults.expect(pins() == [] and not session.canUndo, f"a click on the wheel changed the photo: {pins()}")
    drag(view, at(0.05, 0.10), QPoint(40, -30))
    made = pins()
    faults.expect(len(made) == 1 and near(made[0]["from"], (0.05, 0.10)) and near(made[0]["to"], (0.05 + 40 / per, 0.10 + 30 / per)),
                  f"dragging the mark by 40, -30 px gave {made}")
    faults.expect(wheel.property("selected") == 0 and not plain(wheel.property("fresh")), "the new pin is not the one being worked on")
    faults.expect(session.uniforms["pinCount"] == 1, "the pin does not reach the picture")
    session.undo()
    faults.expect(pins() == [], "making a pin is more than one step of undo")
    session.redo()
    to = made[0]["to"] if made else [0.1, 0.15]
    drag(view, at(*to), QPoint(-20, 10))
    faults.expect(len(pins()) == 1 and near(pins()[0]["to"], (to[0] - 20 / per, to[1] - 10 / per)) and near(pins()[0]["from"], (0.05, 0.10)),
                  f"dragging the pin by -20, 10 px gave {pins()}")
    to = pins()[0]["to"] if pins() else to
    drag(view, at(0.05, 0.10), QPoint(12, 12))
    faults.expect(len(pins()) == 1 and near(pins()[0]["from"], (0.05 + 12 / per, 0.10 - 12 / per)) and near(pins()[0]["to"], to),
                  f"dragging the origin marker gave {pins()}")
    span, was = wheel.property("span"), dict(pins()[0]) if pins() else {}
    turn(at(*to))
    turn(at(*to), Qt.KeyboardModifier.ShiftModifier)
    turn(at(*to), Qt.KeyboardModifier.AltModifier)
    now = pins()[0] if pins() else {}
    faults.expect(now and now["range"] > was["range"] * 1.1 and now["feather"] == was["feather"] + 10
                  and now["variance"] == was["variance"] + 20 and wheel.property("span") == span,
                  f"the mouse wheel on a pin (plain, Shift, Alt) gave {now} from {was}")
    turn(at(-0.2, -0.2))
    faults.expect(wheel.property("span") < span and pins()[:1] == [now], "the mouse wheel off a pin does not zoom the wheel")
    call(wheel, "showAll")
    settle()
    area = box(root.findChild(QObject, "photo"))
    tap(view, QPoint(int(area.left() + area.width() * 0.3), int(area.top() + area.height() * 0.6)))
    fresh = plain(wheel.property("fresh"))
    faults.expect(fresh and near(fresh["from"], session.colorAt(0.3, 0.6)[:2], 0.01), f"a click on the photo does not mark its color: {fresh}")
    key(Qt.Key.Key_Escape)
    faults.expect(not plain(wheel.property("fresh")) and wheel.property("selected") == -1, "Escape does not let go of the mark")
    tap(view, at(*to))
    faults.expect(wheel.property("selected") == 0 and pins()[:1] == [now], "a click on a pin does not take it up, or changes it")
    key(Qt.Key.Key_Delete)
    faults.expect(pins() == [], "Delete does not remove the pin")
    session.undo()
    double_click(view, at(*to))
    faults.expect(pins() == [], f"a double click does not remove the pin: {pins()}")
    faults.expect(not plain(wheel.property("fresh")), "a double click on a pin leaves a mark where the pin was")
    session.undo()
    QTest.mouseMove(view, at(*to), 500)
    QTest.mouseClick(view, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, at(*to))
    settle()
    faults.expect(pins() == [] and not plain(wheel.property("fresh")), f"a right click does not remove the pin: {pins()}")
    session.undo()
    faults.expect(len(pins()) == 1, "a pin removed by a right click does not come back with undo")
    # A right click while a pin is being dragged does not take it away, and the drag goes on.
    count, held, looked = len(pins()), pins()[0]["to"], wheel.property("middle")
    QTest.mousePress(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at(*held), 500)
    both = Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton
    gone = 0
    for kind, button, buttons in ((QEvent.Type.MouseMove, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton),
                                  (QEvent.Type.MouseButtonPress, Qt.MouseButton.RightButton, both),
                                  (QEvent.Type.MouseButtonRelease, Qt.MouseButton.RightButton, Qt.MouseButton.LeftButton),
                                  (QEvent.Type.MouseMove, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)):
        gone += 12 if kind == QEvent.Type.MouseMove else 0  # (12 px before the right click, 24 after)
        to = QPointF(at(*held) + QPoint(gone, 0))
        QCoreApplication.sendEvent(view, QMouseEvent(kind, to, to, button, buttons, Qt.KeyboardModifier.NoModifier))
        QTest.qWait(40)
        settle()
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at(*held) + QPoint(24, 0))
    settle()
    faults.expect(len(pins()) == count and near(pins()[0]["to"], (held[0] + 24 / per, held[1])) and wheel.property("middle") == looked,
                  f"a right click in the middle of dragging a pin: {len(pins())} pins of {count}, the pin at {pins()[0]['to'] if pins() else None}, "
                  f"the wheel's view moved from {looked} to {wheel.property('middle')}")
    session.undo()
    # A mark (a color clicked, no pin yet): a click and a quick second one leave it there; a
    # double click on a mark that was there already takes it away.
    double_click(view, at(0.14, 0.12))
    faults.expect(plain(wheel.property("fresh")), "a color clicked twice quickly is not marked: the second click took the mark away")
    double_click(view, at(0.14, 0.12))
    faults.expect(not plain(wheel.property("fresh")), "a double click on a mark does not take it away")
    # A mark is a pin as soon as the wheel is turned on it, where it stands.
    tap(view, at(-0.08, 0.02))
    turn(at(-0.08, 0.02))
    last = pins()[-1] if len(pins()) == 2 else {}
    faults.expect(last and near(last["to"], (-0.08, 0.02)) and last["to"] == last["from"] and last["range"] > 0.08 * 1.1 and wheel.property("selected") == 1,
                  f"the mouse wheel on a color just marked does not make it a pin with a wider range: {pins()[1:]}")
    # A pin made for other tones than the wheel is showing is not taken hold of there.
    session.setRange("highlights")
    settle()
    tap(view, at(0.12, -0.06))
    drag(view, at(0.12, -0.06), QPoint(10, 10))
    session.setRange("shadows")
    settle()
    faults.expect(wheel.property("selected") == -1, "the pin of the highlights is still the one worked on when the wheel shows the shadows")
    high = pins()[-1]["to"] if len(pins()) == 3 else [0.13, -0.07]
    tap(view, at(*high))
    faults.expect(wheel.property("selected") == -1 and plain(wheel.property("fresh")), "a pin of the highlights can be taken up while the wheel shows the shadows")
    # All tones is the overview: there every pin is at hand, whatever tones it was made for.
    session.setRange("full")
    settle()
    key(Qt.Key.Key_Escape)
    tap(view, at(*high))
    faults.expect(wheel.property("selected") == 2, "with the wheel showing all tones, a pin of the highlights cannot be taken up")
    # And a pin of All tones is not at hand in Midtones (though its middle lies there).
    key(Qt.Key.Key_Escape)
    session.setRange("midtones")
    settle()
    whole = pins()[0]["to"]
    tap(view, at(*whole))
    faults.expect(wheel.property("selected") == -1, "a pin of All tones can be taken up while the wheel shows the midtones")
    ui.close_view(view)
    return faults


def audit_masks_data() -> list[str]:
    """Masks on screen are what the arithmetic says (edit.colors_at with a place): a pin kept
    to a mask changes each pixel by as much as the mask covers it, for an oval, an oval inside
    out and a slope, also in a photo that is turned, straightened and cropped; H shows the
    mask; and an export has it."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "")
    session.exporter.side = 256
    faults = Faults()
    spots = [(u / 12, v / 8) for v in range(1, 7) for u in range(1, 12)]
    pin = {"from": [0.0, 0.0], "to": [0.03, -0.02], "range": 0.5, "feather": 0, "light": -0.12, "mask": 1}

    def seen() -> list:
        area, shot = box(root.findChild(QObject, "photo")), view.grabWindow()
        return [shot.pixelColor(int(area.left() + u * area.width()), int(area.top() + v * area.height()))
                for u, v in spots]

    cases = {"an oval": ({}, OVAL), "an oval inside out": ({}, {**OVAL, "invert": True}),
             "a slope": ({}, SLOPE), "a hard-edged circle": ({}, {**OVAL, "ratio": 1.0, "feather": 0.0}),
             "an oval in a turned, cropped photo": ({"turns": 1.0, "crop": [0.1, 0.05, 0.9, 0.8]}, OVAL),
             "a slope in a straightened photo": ({"angle": 7.0, "turns": 3.0}, SLOPE)}
    for name, (frame, mask) in cases.items():
        session.setEdits(frame)
        settle()
        before = seen()
        look = {**frame, "masks": [mask], "pins": [pin]}
        session.setEdits(look)
        settle()
        matrix, compared, changed, kept = session._frame()["matrix"], 0, 0, 0
        for spot, was, now in zip(spots, before, seen()):
            want = edit.colors_at(edit.clean(look), oklab_of(was), edit.framed(matrix, *spot), 1.5)
            limits = scopes.srgb_of(*want).min(), scopes.srgb_of(*want).max(), *was.getRgb()[:3]
            if min(limits) < 3 or max(limits) > 252:
                continue  # at the screen's limit: cut off, nothing to compare
            compared += 1
            moved = np.abs(oklab_of(now) - oklab_of(was)).max()
            changed, kept = changed + (moved > 0.1), kept + (moved < 0.004)
            off = np.abs(oklab_of(now) - want).max()
            faults.expect(off < 0.012, f"{name}: at {spot[0]:.2f}, {spot[1]:.2f} a color {np.round(oklab_of(was), 3)} "
                                       f"should become {np.round(want, 3)} and is {np.round(oklab_of(now), 3)}")
        faults.expect(compared >= 20 and changed >= 3 and kept >= 3,
                      f"{name}: too little to go by ({compared} spots, {changed} changed, {kept} left alone)")
        session.history = edit.History()
        session.refresh()
        settle()

    session.setEdits({"masks": [OVAL], "pins": [pin]})
    settle()
    folder = Path(SHELF.name) / "masks"  # its own: another audit exports a first.png too
    folder.mkdir()
    session.exporter.start(["first.arw"], {"kind": "png8", "quality": 90, "longest": 0, "folder": str(folder)})
    carry_out(view, session)
    file = export.bytes_of(QImage(str(folder / "first.png"))).astype(int)
    screen = shown_at_full_size(view, root).astype(int)
    off = np.abs(file - screen).max() if file.shape == screen.shape else 999
    faults.expect(off <= 1, f"the export of a masked photo is not what the screen shows: up to {off} levels off")
    faults.expect(np.abs(screen[134, 288] - screen[20, 20]).max() > 20, "the mask does nothing at 1:1 either")
    # H: the mask instead of the picture, white where it is.
    area = box(root.findChild(QObject, "photo"))
    stage(root).setProperty("showMask", 1)
    settle()
    shot = view.grabWindow()
    inside = shot.pixelColor(int(area.left() + 0.6 * area.width()), int(area.top() + 0.42 * area.height()))
    outside = shot.pixelColor(int(area.left() + 0.1 * area.width()), int(area.top() + 0.1 * area.height()))
    faults.expect(min(inside.getRgb()[:3]) >= 250 and max(outside.getRgb()[:3]) < 100,
                  f"shown, the mask is {inside.name()} inside and {outside.name()} outside")
    ui.close_view(view)
    return faults


def audit_masks() -> list[str]:
    """Masks under a real mouse and keyboard, in the Masks tab: "+ Radial" makes one and chooses
    it, its shape is on the photo, each handle does its own thing in one step of undo, a pin
    made meanwhile is kept to it and cannot be taken hold of from the Adjust tab (the whole
    image), Inverted and the row's bin do what they say (the bin takes the mask's pins along,
    undo brings all back), H shows the mask, and there is no ninth."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "masks")
    faults = Faults()
    counted(view, session)
    wheel, frame = find(root, "wheel"), find(root, "mask-frame")
    area = box(root.findChild(QObject, "photo"))

    def masks() -> list:
        return session.masks

    def middle(name: str) -> QPoint:
        return box(find(root, name)).center().toPoint()

    def near(found, want, by=0.004) -> bool:
        return all(abs(f - w) < by for f, w in zip(found, want))

    def key(which, kind=QEvent.Type.KeyPress) -> None:
        QCoreApplication.sendEvent(root, QKeyEvent(kind, which, Qt.KeyboardModifier.NoModifier))
        settle()

    def press(name: str) -> None:
        click(view, find(root, name))
        view.grabWindow()  # a frame: the list's rows are only put in their places when one is drawn

    def tab(name: str) -> None:  # that tab of the column, the Tone Wheel in sight
        bring(root, "modes")
        press("mode:" + name)
        if name == "masks":
            press("mask-item:1")
        bring(root, "wheel-block")

    faults.expect(not frame.isVisible() and wheel.property("mask") == 0, "a mask is on show before there is one")
    press("add-mask:radial")
    faults.expect(len(masks()) == 1 and wheel.property("mask") == 1 and frame.isVisible(),
                  f"+ Radial gave {masks()}, chosen: {wheel.property('mask')}")
    faults.expect(masks() and near(masks()[0]["at"], (0.5, 0.5)), "the new mask is not in the middle of the photo")
    if not (masks() and frame.isVisible()):  # nothing to take hold of: the rest would say nothing
        ui.close_view(view)
        return faults
    steps = session.steps
    at = middle("mask:at")
    faults.expect(abs(at.x() - area.center().x()) <= 2 and abs(at.y() - area.center().y()) <= 2,
                  f"the mask's middle handle is at {at}, the photo's middle at {area.center()}")
    was = masks()[0]
    drag(view, at, QPoint(60, -40))
    now = masks()[0]
    by = (60 / area.width(), -40 / area.height())
    faults.expect(near(now["at"], (was["at"][0] + by[0], was["at"][1] + by[1]))
                  and near(now["to"], (was["to"][0] + by[0], was["to"][1] + by[1])) and session.steps == steps + 1,
                  f"dragging the middle by 60, -40 px took the mask from {was} to {now}")
    was = now
    drag(view, middle("mask:to"), QPoint(-30, 45))
    now = masks()[0]
    faults.expect(near(now["to"], (was["to"][0] - 30 / area.width(), was["to"][1] + 45 / area.height()))
                  and near(now["at"], was["at"]) and now["ratio"] == was["ratio"],
                  f"dragging the end of the axis by -30, 45 px took the mask from {was} to {now}")
    was, side = now, middle("mask:side")
    out = np.array([side.x() - middle("mask:at").x(), side.y() - middle("mask:at").y()], float)
    drag(view, side, QPoint(*np.round(out * 0.5).astype(int)))
    now = masks()[0]
    faults.expect(abs(now["ratio"] / was["ratio"] - 1.5) < 0.03 and near(now["to"], was["to"]) and near(now["at"], was["at"]),
                  f"dragging the side handle half as far out again took the mask from {was} to {now}")
    session.undo()
    faults.expect(masks()[0]["ratio"] == was["ratio"] and near(masks()[0]["to"], was["to"]),
                  f"undo after three drags of handles left {masks()[0]}: each is a step of its own")
    session.redo()
    slider = find(root, "slider:mask-feather")  # kept: its wrapper going would take the track's
    track = slider.findChild(QObject, "track")
    drag(view, box(slider.findChild(QObject, "thumb")).center().toPoint(), QPoint(-60, 0))  # (by its thumb)
    want = 0.5 - 60 / (track.width() - 12)
    faults.expect(abs(masks()[0]["feather"] - want) < 0.011, f"60 px to the left took Feather to {masks()[0]['feather']}, not {want:.2f}")
    # Turned, the mask is still on what it covered: its handle goes with the picture.
    u, v = masks()[0]["at"]
    session.turn(1)
    settle()
    turned, at = box(root.findChild(QObject, "photo")), middle("mask:at")
    faults.expect(near(masks()[0]["at"], (1 - v, u)) and abs(at.x() - turned.left() - (1 - v) * turned.width()) <= 2
                  and abs(at.y() - turned.top() - u * turned.height()) <= 2,
                  f"turned to the right, a mask at {u:.2f}, {v:.2f} is drawn at {masks()[0]['at']}")
    session.undo()
    settle()

    # A pin made while the mask is chosen is the mask's.
    bring(root, "wheel-block")
    call(wheel, "showAll")
    settle()
    per = box(wheel).height() / wheel.property("span")

    def spot() -> QPoint:  # a color on the wheel, wherever the wheel is just now
        return QPoint(round(box(wheel).center().x() + 0.05 * per), round(box(wheel).center().y() - 0.10 * per))

    tap(view, spot())
    drag(view, spot(), QPoint(40, -30))
    pins = session.history.edit.get("pins", [])
    faults.expect(len(pins) == 1 and pins[0]["mask"] == 1, f"a pin made with the mask chosen: {pins}")
    key(Qt.Key.Key_Escape)
    tab("edit")
    faults.expect(wheel.property("mask") == 0 and not frame.isVisible(), "the Adjust tab leaves the mask on the photo")
    tap(view, spot() + QPoint(40, -30))
    faults.expect(wheel.property("selected") == -1, "a mask's pin can be taken hold of from the Adjust tab")
    key(Qt.Key.Key_Escape)
    tab("masks")
    faults.expect(wheel.property("mask") == 1 and frame.isVisible(), "a click on a mask's row does not choose it")
    tap(view, spot() + QPoint(40, -30))
    faults.expect(wheel.property("selected") == 0, "the mask's pin cannot be taken hold of with its mask chosen")
    key(Qt.Key.Key_H)
    faults.expect(stage(root).property("reach") == 1 and stage(root).property("showMask") == 0, "H with a pin taken up does not show its reach")
    key(Qt.Key.Key_H, QEvent.Type.KeyRelease)
    key(Qt.Key.Key_Escape)
    key(Qt.Key.Key_H)
    faults.expect(stage(root).property("showMask") == 1 and stage(root).property("reach") == 0, "H with a mask chosen does not show it")
    key(Qt.Key.Key_H, QEvent.Type.KeyRelease)
    faults.expect(stage(root).property("showMask") == 0, "letting go of H leaves the mask on show")

    bring(root, "panel:masks")
    press("mask:inverted")
    faults.expect(masks()[0]["invert"] is True, "Inverted does not turn the mask inside out")
    press("mask:inverted")
    faults.expect(masks()[0]["invert"] is False, "Inverted a second time does not turn it back")
    steps = session.steps
    QTest.mouseMove(view, middle("mask-item:1"), 300)  # (the row's bin is there while the pointer is on the row)
    settle()
    view.grabWindow()
    press("mask-delete:1")
    faults.expect(masks() == [] and session.history.edit.get("pins", []) == [] and wheel.property("mask") == 0
                  and not frame.isVisible(), f"Delete left {session.history.edit}")
    session.undo()
    faults.expect(len(masks()) == 1 and len(session.history.edit.get("pins", [])) == 1,
                  "undoing Delete does not bring back the mask and its pin at once")
    press("add-mask:linear")
    faults.expect(len(masks()) == 2 and masks()[1]["kind"] == "linear" and wheel.property("mask") == 2
                  and find(root, "mask-middle").isVisible() and not find(root, "mask:side").isVisible(),
                  f"+ Linear gave {masks()[1:]}")
    was = masks()[1]
    drag(view, middle("mask:to"), QPoint(50, 20))
    faults.expect(near(masks()[1]["to"], (was["to"][0] + 50 / area.width(), was["to"][1] + 20 / area.height())),
                  f"dragging the far line of a slope took it from {was} to {masks()[1]}")
    # A drag just outside a radial mask turns it about its middle: its axis keeps its length
    # and goes a quarter round; and a press well inside it is still the photo's.
    wheel.setProperty("mask", 1)
    settle()
    was, per = masks()[0], (area.width(), area.height())
    arm = [(was["to"][n] - was["at"][n]) * per[n] for n in (0, 1)]
    long_ = math.hypot(*arm)
    centre = QPointF(area.left() + was["at"][0] * per[0], area.top() + was["at"][1] * per[1])
    out = QPointF(arm[0], arm[1]) * ((long_ + 20) / long_)
    drag(view, (centre + out).toPoint(), (QPointF(-out.y(), out.x()) - out).toPoint())
    now = masks()[0]
    turned = [(now["to"][n] - now["at"][n]) * per[n] for n in (0, 1)]
    faults.expect(abs(turned[0] + arm[1]) < 3 and abs(turned[1] - arm[0]) < 3 and near(now["at"], was["at"]),
                  f"a drag a quarter round outside the mask took its axis from {arm} to {turned}")
    drag(view, (centre + QPointF(arm[0], arm[1]) * 0.4).toPoint(), QPoint(30, 0))
    faults.expect(masks()[0] == now, "a drag inside the mask, off its handles, changed the mask")
    # On its rim the pointer is the turning one, a real cursor; inside, it is not.
    turner = find(root, "mask:turn")
    now_arm = [(now["to"][n] - now["at"][n]) * per[n] for n in (0, 1)]
    rim = centre + QPointF(now_arm[0], now_arm[1]) * ((math.hypot(*now_arm) + 20) / math.hypot(*now_arm))
    for nudge in (0, 2, 4):
        QTest.mouseMove(view, rim.toPoint() + QPoint(nudge, 0), 100)
        QTest.qWait(120)
    over = QGuiApplication.overrideCursor()
    faults.expect(over is not None and over.shape() == Qt.CursorShape.BitmapCursor and turner.property("turnStep") >= 0,
                  f"on a mask's rim the pointer is not the turning one: {over and over.shape()}")
    for nudge in (0, 2, 4):
        QTest.mouseMove(view, centre.toPoint() + QPoint(nudge, 0), 100)
        QTest.qWait(120)
    faults.expect(QGuiApplication.overrideCursor() is None, "inside the mask the pointer is still the turning one")
    stage(root).setProperty("fitted", True)
    settle()
    wheel.setProperty("mask", 2)
    for _ in range(6):
        session.addMask("linear")
    settle()
    faults.expect(len(masks()) == 8 and not find(root, "add-mask:radial").isVisible(), "eight masks, and still a + to press")
    faults.expect(session.addMask("radial") == 0 and len(masks()) == 8 and "eight" in session.error, "a ninth mask")
    faults += audit_layout(view)
    ui.close_view(view)
    return faults


def audit_pin_panel() -> list[str]:
    """A pin's settings under a real mouse: the handle on the lightness strip, H held for what
    the pin reaches, and the Pin panel: its hint while no pin is taken up, the kind, a slider,
    Duplicate and Delete."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "")
    faults = Faults()
    counted(view, session)
    area = box(root.findChild(QObject, "photo"))
    spot = QPoint(int(area.left() + area.width() * 0.2), int(area.top() + area.height() * 0.62))
    source = session.colorAt(0.2, 0.62)[:2]
    wheel = find(root, "wheel")

    def pins() -> list:
        return session.history.edit.get("pins", [])

    def key(kind, which) -> None:
        QCoreApplication.sendEvent(root, QKeyEvent(kind, which, Qt.KeyboardModifier.NoModifier))
        settle()

    handle = find(root, "lightness-handle")
    faults.expect(not handle.isVisible(), "the lightness strip has a handle with no pin taken up")
    session.setEdit("pins", [{"from": source, "to": [source[0] + 0.05, source[1]], "range": 0.06}])
    session.commitEdit()
    wheel.setProperty("selected", 0)
    settle()
    strip = box(find(root, "lightness"))
    drag(view, box(handle).center().toPoint() + QPoint(-90, 0), QPoint(60, 0))  # from beside the handle
    faults.expect(len(pins()) == 1 and abs(pins()[0]["light"] - 60 / strip.width()) < 0.003,
                  f"dragging the lightness handle 60 px gave {pins()[0]['light'] if pins() else None}, not {60 / strip.width():.3f}")
    double_click(view, box(handle).center().toPoint())
    faults.expect(pins() and pins()[0]["light"] == 0, "a double click on the lightness strip does not put the pin's lightness back")

    far = QPoint(int(area.left() + area.width() * 0.75), int(area.top() + area.height() * 0.62))
    usual = view.grabWindow().pixelColor(spot)
    key(QEvent.Type.KeyPress, Qt.Key.Key_H)
    shown = view.grabWindow()
    faults.expect(min(shown.pixelColor(spot).getRgb()[:3]) > 235, f"with H held the pinned color is not shown white: {shown.pixelColor(spot).name()}")
    reached = shown.pixelColor(far)
    faults.expect(reached.red() == reached.green() == reached.blue() and reached.red() < 110, f"with H held, what the pin does not reach is not a dim gray: {reached.name()}")
    key(QEvent.Type.KeyRelease, Qt.Key.Key_H)
    faults.expect(view.grabWindow().pixelColor(spot) == usual, "letting go of H does not bring the picture back")

    show(root, "pins")
    faults.expect(find(root, "panel:pins").isVisible() and find(root, "pin:range").property("enabled"), "the pin's settings are not under the wheel, or not for the pin taken up")
    view.grabWindow()  # a frame: the panel's rows take their places
    tall, stood = find(root, "panel:pins").height(), box(wheel).top()
    click(view, find(root, "pin-kind:rotate"))
    faults.expect(pins() and pins()[0]["mode"] == "rotate", "choosing Rotate does not change the pin's kind")
    slider = find(root, "pin:feather")
    track = slider.findChild(QObject, "track")
    drag(view, box(slider.findChild(QObject, "thumb")).center().toPoint(), QPoint(-50, 0))  # (by its thumb)
    faults.expect(pins() and abs(pins()[0]["feather"] - (70 - 50 * 100 / (track.width() - 12))) < 1.01,
                  f"dragging Feather 50 px to the left gave {pins()[0]['feather'] if pins() else None}")
    key(QEvent.Type.KeyPress, Qt.Key.Key_Escape)
    view.grabWindow()
    # With no pin taken up the section is there all the same, dimmed and as tall, so that the
    # wheel under it stays where it is; a color marked is shown as the pin it would be, and is
    # one as soon as a slider is moved.
    faults.expect(find(root, "pin-hint").isVisible() and not find(root, "pin:range").property("enabled") and find(root, "pin:range").property("opacity") < 0.5,
                  "with no pin taken up the panel's settings are not dimmed")
    faults.expect(find(root, "panel:pins").height() == tall and box(wheel).top() == stood,
                  f"with no pin taken up the Pin section is {find(root, 'panel:pins').height():.0f} px tall, not {tall:.0f}: the wheel moves")
    tap(view, box(wheel).center().toPoint() + QPoint(-40, 50))
    view.grabWindow()
    faults.expect(len(pins()) == 1 and find(root, "pin:range").property("enabled") and box(wheel).top() == stood,
                  f"a color marked on the wheel is not shown in the Pin section, or made a pin by itself ({len(pins())}), or moved the wheel")
    drag(view, box(track).center().toPoint(), QPoint(-30, 0))
    faults.expect(len(pins()) == 2 and wheel.property("selected") == 1 and pins()[1]["feather"] < 70 and pins()[1]["from"] == pins()[1]["to"],
                  f"moving Feather with a color marked does not make it a pin: {pins()[1:]}")
    session.undo()
    faults.expect(len(pins()) == 1, "a pin made from the Pin section is more than one step of undo")
    show(root, "")
    click(view, find(root, "reset:wheel"))
    faults.expect(pins() == [], "Reset on the Tone Wheel's card does not remove every pin")
    session.undo()
    faults.expect(len(pins()) == 1, "removing every pin is not one step of undo")
    ui.close_view(view)
    return faults


def audit_color_tools() -> list[str]:
    """Saturation, the color EQ and the chroma curve under a real mouse: their cards, and S;
    the saturation slider; a point made and moved on the chroma curve; on the EQ a
    click that changes nothing, a point made by dragging the line, moved, widened by the mouse
    wheel, removed by a double click."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "")
    faults = Faults()
    counted(view, session)

    QCoreApplication.sendEvent(root, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_S, Qt.KeyboardModifier.NoModifier))
    settle()
    view.grabWindow()
    faults.expect(find(root, "panel:saturation").property("open") and find(root, "slider:saturation").isVisible(), "S does not open Saturation")
    slider = find(root, "slider:saturation")
    track = slider.findChild(QObject, "track")
    drag(view, box(slider.findChild(QObject, "thumb")).center().toPoint(), QPoint(40, 0))  # (by its thumb)
    want = 100 + 40 * 200 / (track.width() - 12)
    faults.expect(abs(session.edit["saturation"] - want) < 1.01, f"dragging Saturation 40 px gave {session.edit['saturation']}, not {want:.0f}")
    faults.expect(session.adjusted["saturation"] and session.uniforms["saturation"] > 1.1, "Saturation does not reach the picture")
    click(view, find(root, "reset:saturation"))
    faults.expect(session.edit["saturation"] == 100, "Reset does not put Saturation back to 100 %")

    show(root, "chroma")
    area = box(find(root, "chroma-plot"))
    w, h = area.width() - 16, area.height() - 16
    drag(view, QPoint(round(area.left() + 8 + 0.25 * w), round(area.top() + 8 + 0.75 * h)), QPoint(0, -round(0.2 * h)))
    curve = session.history.edit.get("chroma")
    faults.expect(curve and len(curve) == 3 and abs(curve[1][0] - 0.25) < 0.01 and abs(curve[1][1] - 0.45) < 0.01,
                  f"dragging the chroma curve up at a quarter gave {curve}")
    faults.expect("curve" not in session.history.edit, "the chroma curve's editor changed the tone curve")
    session.undo()

    show(root, "eq")
    area = box(find(root, "eq-plot"))
    w, h = area.width() - 16, area.height() - 16

    def at(hue: float, gain: float) -> QPoint:
        return QPoint(round(area.left() + 8 + hue / 360 * w), round(area.top() + 8 + (1 - gain / 200) * h))

    def eq() -> list:
        return session.history.edit.get("eq", [])

    steps = session.history.at
    tap(view, at(140, 100))
    faults.expect(eq() == [] and session.history.at == steps, f"a click on the EQ's line changed it: {eq()}")
    drag(view, at(140, 100), QPoint(0, -round(0.2 * h)))
    faults.expect(len(eq()) == 1 and abs(eq()[0]["hue"] - 140) < 2 and abs(eq()[0]["gain"] - 140) < 2 and eq()[0]["width"] == 40,
                  f"dragging the EQ's line up at 140 degrees gave {eq()}")
    faults.expect(session.uniforms["eqCount"] == 1, "the EQ does not reach the picture")
    session.undo()
    faults.expect(eq() == [], "making an EQ point is more than one step of undo")
    session.redo()
    drag(view, at(140, 140), QPoint(round(40 / 360 * w), round(0.1 * h)))
    faults.expect(len(eq()) == 1 and abs(eq()[0]["hue"] - 180) < 2 and abs(eq()[0]["gain"] - 120) < 2, f"dragging the point along and down gave {eq()}")
    spot = QPointF(at(180, 120))
    QCoreApplication.sendEvent(view, QWheelEvent(spot, spot, QPoint(0, 0), QPoint(0, 240), Qt.MouseButton.NoButton,
                                                 Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
    settle()
    faults.expect(len(eq()) == 1 and eq()[0]["width"] > 48, f"the mouse wheel on an EQ point does not widen it: {eq()}")
    line = session.eqLine
    faults.expect(abs(line[90] - 1.2) < 0.03 and abs(line[75] - line[105]) < 0.01 and line[0] == 1 and line[150] == 1,
                  f"the EQ's line is not what the point says: {line[75]:.2f}, {line[90]:.2f}, {line[105]:.2f} at 150, 180, 210 degrees")
    double_click(view, at(180, 120))
    faults.expect(eq() == [], f"a double click does not take the EQ point away: {eq()}")
    ui.close_view(view)
    return faults


def audit_mask_wheel() -> list[str]:
    """With a mask chosen the wheel shows only the colors that mask reaches: what is counted
    of them (Fit, the lightness strip) and the dots themselves; with none chosen, all again."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "masks")
    faults = Faults()
    wheel = find(root, "wheel")
    session.setEdits({"masks": [{"kind": "radial", "at": [0.2, 0.3], "to": [0.26, 0.3], "ratio": 1.0, "feather": 0.2}]})
    session.commitEdit()

    def seen(mask: int) -> tuple[dict, QImage]:
        wheel.setProperty("mask", mask)
        settle()
        bring(root, "wheel-block")
        faults.expect(counted(view, session), "the scopes were not counted")
        area = box(wheel).toRect()
        return session.cloud, view.grabWindow().copy(area)

    def wide(box_: list) -> float:
        return (box_[1] - box_[0]) * (box_[3] - box_[2])

    whole = seen(0)[0]  # (the wheel is not on show in this tab until a mask is chosen)
    bars = session.histogram["g"]
    part, drawn = seen(1)
    faults.expect(0 < len(part["colors"]) < len(whole["colors"]) / 3,
                  f"a small mask chosen: {len(part['colors'])} of the photo's {len(whole['colors'])} dots are counted")
    faults.expect(session.histogram["g"] != bars, "the histogram is the same with a small mask chosen as for the whole photo")
    faults.expect(wide(part["box"]) < wide(whole["box"]) / 3, f"Fit would frame {part['box']} for the mask, {whole['box']} for the photo")
    session.setMask(1, {"invert": True})
    session.commitEdit()
    rest, rest_drawn = seen(1)
    faults.expect(drawn != rest_drawn, "the dots drawn on the wheel are the same for a mask and for all that is outside it")
    drawn.save(str(OUT / "mask-wheel-inside.png")), rest_drawn.save(str(OUT / "mask-wheel-outside.png"))
    faults.expect(len(rest["colors"]) > len(part["colors"]) * 2, f"the mask turned inside out counts {len(rest['colors'])} dots, the mask itself {len(part['colors'])}")
    back = seen(0)[0]
    faults.expect(len(back["colors"]) == len(whole["colors"]), f"no mask chosen any more: {len(back['colors'])} dots, not the photo's {len(whole['colors'])}")
    # And on the photo the chosen mask lies in red while the pointer is on the photo, a radial
    # one as a painted one does; it fades, so it is looked at a moment later.
    session.setMask(1, {"invert": False})
    session.commitEdit()
    wheel.setProperty("mask", 1)
    area = box(find(root, "photo"))
    place = lambda u, w: QPoint(int(area.left() + u * area.width()), int(area.top() + w * area.height()))  # noqa: E731

    def red(pointer: QPoint) -> float:  # how much redder than green and blue the photo is inside the mask
        QTest.mouseMove(view, pointer)
        wait(lambda: False, 400)
        r, g, b = view.grabWindow().pixelColor(place(0.225, 0.275)).getRgb()[:3]
        return r - (g + b) / 2

    off, on = red(QPoint(30, 30)), red(place(0.8, 0.8))
    view.grabWindow().save(str(OUT / "mask-red.png"))  # (to be looked at)
    gone = red(QPoint(30, 30))
    faults.expect(on > off + 25, f"with the pointer on the photo a radial mask is not laid over it in red: {off:.0f}, then {on:.0f}")
    faults.expect(abs(gone - off) < 3, f"with the pointer off the photo again the red stays: {gone:.0f}, before {off:.0f}")
    ui.close_view(view)
    return faults


def audit_test_chart() -> list[str]:
    """The test chart in the photo's place, by its switch: the chart is what is shown and said
    loudly, the photo's edit works on it, the scopes count it, the way back works, and another
    photo never opens with the chart still on."""
    session = ui.Session([Path("IMG_0412.ARW"), Path("IMG_0413.ARW")], load=flat)
    view, root = opened(session, "")
    faults = Faults()

    def wedge(step: int) -> QColor:
        area = box(root.findChild(QObject, "photo"))
        return view.grabWindow().pixelColor(int(area.left() + area.width() * (step + 0.5) / 8),
                                            int(area.top() + area.height() * 0.92))

    faults.expect(not find(root, "box:chart").isVisible() and session.fullWidth == 300, "the test chart is on show without being asked for")
    usual = wedge(7)
    click(view, find(root, "show-chart"))
    wait(lambda: root.property("photoReady"))
    view.grabWindow()
    faults.expect(session.panels["chart"] and session.fullWidth == 1200, "the switch does not put the test chart in the photo's place")
    words = find(root, "text:chart")
    faults.expect(find(root, "box:chart").isVisible() and "TEST CHART" in words.property("text") and "IMG_0412.ARW" in words.property("text"),
                  "nothing says that this is a test chart and whose edit it shows")
    faults.expect(box(find(root, "box:stage")).top() >= box(find(root, "box:chart")).bottom() - 1, "the notice lies over the picture")
    white, gray = wedge(7), wedge(3)
    faults.expect(white.green() > 250 and white != usual and gray.red() == gray.green() == gray.blue(), f"the chart is not what is shown: {white.name()}, {gray.name()}")
    session.setEdit("exposure", -1.0)
    session.commitEdit()
    settle()
    faults.expect(wedge(7).green() < white.green() - 10, "the photo's edit does not work on the chart")
    faults.expect(session.history.edit == {"exposure": -1.0}, "an edit made on the chart is not the photo's edit")
    counted(view, session)
    faults.expect(len(session.cloud["colors"]) > 200, f"the wheel does not show the chart's colors: {len(session.cloud['colors'])} dots")
    click(view, find(root, "chart-back"))
    wait(lambda: root.property("photoReady"))
    faults.expect(not session.panels["chart"] and session.fullWidth == 300 and not find(root, "box:chart").isVisible(),
                  "Back to my photo does not bring the photo back")
    faults.expect(session.history.edit == {"exposure": -1.0}, "going back to the photo lost the edit")
    click(view, find(root, "show-chart"))
    QCoreApplication.sendEvent(root, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier))
    settle()
    faults.expect(not session.panels["chart"] and session.index == 1 and "chart" not in session.source, "the next photo opens with the test chart still on")
    ui.close_view(view)
    # The photo's lens corrections are not for the chart, which no lens drew: neither the
    # camera's figures nor Defringe.
    session = ui.Session([Path("test chart.arw")], load=lensed)
    view, root = opened(session, "")
    session.setEdits({"defringe": 80.0, **catalog.LENS_ON})
    session.commitEdit()
    faults.expect(session.uniforms["lensOn"] == 1 and session.uniforms["defringe"] > 0, "the lens corrections are not on for the photo itself")
    session.setPanel("chart", True)
    faults.expect(session.uniforms["lensOn"] == 0 and session.uniforms["defringe"] == 0, "the photo's lens corrections are applied to the test chart")
    session.setPanel("chart", False)
    faults.expect(session.uniforms["lensOn"] == 1, "back on the photo, its lens corrections are gone")
    ui.close_view(view)
    return faults


def audit_library() -> list[str]:
    """The library under a real mouse and keyboard: the places, selecting with Ctrl and Shift,
    the marking keys on a selection (the same key again takes the mark away), the filter, the
    arrows, Enter into Edit and marking there, back with Ctrl+1; a folder taken out and added
    back keeps its marks; and nothing is ever written where the photos are."""
    shown = shelf("audit", marked=False)
    session = ui.Session([])
    view = ui.make_view(session, shown)
    session.surface_known(False)
    root = view.rootObject()
    root.setSize(view.size())
    settle()
    view.grabWindow()
    faults = Faults()
    photos = Path(shown.catalog.folders()[0]).parent
    before = sorted(str(p) for p in photos.rglob("*"))
    here = find(root, "box:library")

    def key(which, held=Qt.KeyboardModifier.NoModifier) -> None:
        QCoreApplication.sendEvent(root, QKeyEvent(QEvent.Type.KeyPress, which, held))
        settle()
        view.grabWindow()

    def press(row: int, held=Qt.KeyboardModifier.NoModifier) -> None:
        QTest.mouseMove(view, box(find(root, f"photo:{row}")).center().toPoint(), 500)
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, held, box(find(root, f"photo:{row}")).center().toPoint())
        settle()

    def marks() -> dict:
        return {p["name"]: (p["flag"], p["stars"], p["label"]) for p in shown.catalog.photos()}

    def chosen() -> list:
        return [int(r) for r in plain(here.property("chosen"))]

    names = [p["name"] for p in shown.model.rows]
    faults.expect(root.property("place") == "library" and here.isVisible() and shown.count == 12,
                  f"with no photo to edit the program does not start in the library: {root.property('place')}, {shown.count} photos")
    wait(lambda: pictures_in(root), 4000)
    middle = view.grabWindow().pixelColor(box(find(root, "photo:0")).center().toPoint())
    faults.expect(max(abs(middle.red() - 24), abs(middle.green() - 24), abs(middle.blue() - 26)) > 12,
                  f"a photo's cell shows no thumbnail: {middle.name()} in its middle")
    click(view, find(root, "place:month:2026-07"))
    faults.expect(shown.count == 3 and find(root, "text:place").property("text") == "July 2026", f"July 2026 shows {shown.count} photos")
    click(view, find(root, "place:all"))
    view.grabWindow()
    press(1)
    press(3, Qt.KeyboardModifier.ControlModifier)
    faults.expect(chosen() == [1, 3], f"a click, then Ctrl and a click, selected {chosen()}")
    press(5, Qt.KeyboardModifier.ShiftModifier)
    faults.expect(chosen() == [3, 4, 5], f"Shift and a click selected {chosen()}")
    key(Qt.Key.Key_5)
    faults.expect([marks()[names[r]][1] for r in range(6)] == [0, 0, 0, 5, 5, 5], f"5 on three photos gave stars {[marks()[n][1] for n in names[:6]]}")
    cell = find(root, "photo:4")  # kept: its wrapper going would take its child's
    faults.expect(len(cell.findChild(QObject, "stars").property("text")) == 5, "the stars do not show on the photo")
    key(Qt.Key.Key_5)
    faults.expect(all(marks()[names[r]][1] == 0 for r in range(6)), "the same key again does not take the stars away")
    press(2)
    key(Qt.Key.Key_P)
    key(Qt.Key.Key_7)
    faults.expect(marks()[names[2]] == (1, 0, 2), f"P and 7 gave {marks()[names[2]]}")
    key(Qt.Key.Key_X)
    faults.expect(marks()[names[2]][0] == -1, "X does not reject")
    key(Qt.Key.Key_U)
    key(Qt.Key.Key_P)
    # Raw only (which the program starts with: here it was turned off, the photos are JPEGs).
    click(view, find(root, "filter:raw"))
    view.grabWindow()
    faults.expect(shown.filter["raw"] and shown.count == 0 and shown.places["all"] == 0 and shown.total == 12
                  and find(root, "text:raw-only").isVisible() and not shown.places["months"]
                  and find(root, "place:all").property("count") == 0,
                  f"Raw only shows {shown.count} finished pictures, counts {shown.places['all']} of them")
    click(view, find(root, "filter:raw"))
    view.grabWindow()
    faults.expect(not shown.filter["raw"] and shown.count == 12 and shown.places["all"] == 12, f"Raw only turned off shows {shown.count} of 12 photos")
    click(view, find(root, "filter:pick"))
    faults.expect(shown.count == 1 and shown.model.rows[0]["name"] == names[2], f"Picked shows {shown.count} photos")
    click(view, find(root, "filter:pick"))
    view.grabWindow()
    press(0)
    key(Qt.Key.Key_Right)
    key(Qt.Key.Key_Down)
    across = find(root, "library-grid").property("across")
    faults.expect(here.property("current") == 1 + across, f"Right, then Down, went to photo {here.property('current')}, not {1 + across}")
    key(Qt.Key.Key_Return)
    wait(lambda: root.property("photoReady"))
    faults.expect(root.property("place") == "edit" and session.path == shown.model.rows[1 + across]["path"] and session.count == 12,
                  f"Enter does not open that photo in Edit: {root.property('place')}, {session.path}")
    key(Qt.Key.Key_3)
    faults.expect(marks()[names[1 + across]][1] == 3,
                  f"3 in Edit does not give the photo three stars: {marks()[names[1 + across]]}")
    key(Qt.Key.Key_Right)
    faults.expect(session.path == shown.model.rows[2 + across]["path"], "the arrows in Edit do not step through the library's photos")
    faults.expect(shown.resting, "while a photo is being edited the library does not rest")
    key(Qt.Key.Key_1, Qt.KeyboardModifier.ControlModifier)
    faults.expect(root.property("place") == "library" and here.property("current") == 2 + across, "Ctrl+1 does not come back to the library at that photo")
    faults.expect(not shown.resting, "back in the library, it still rests")
    garden = next(p for p in shown.places["folders"] if p["name"] == "2025 Garden")
    shown.mark([n for n, p in enumerate(shown.model.rows) if p["path"].startswith(garden["path"])], "stars", 4)
    shown.removeFolder(garden["path"])
    faults.expect(shown.total == 7 and Path(garden["path"]).is_dir(), f"taking a folder out left {shown.total} photos")
    shown.addFolder(garden["path"])
    wait(lambda: not shown.busy and shown.total == 12, 5000)
    faults.expect(shown.total == 12 and sum(1 for mark in marks().values() if mark[1] == 4) == 5, "a folder added back has lost its marks")
    faults.expect(sorted(str(p) for p in photos.rglob("*")) == before, "something was written, moved or removed where the photos are")
    # Ctrl and the wheel make the thumbnails larger or smaller, a step a notch.
    grid = find(root, "library-grid")
    cell, at = grid.property("cell"), QPointF(box(grid).center())
    for by in (120, 120, -120):
        QCoreApplication.sendEvent(view, QWheelEvent(at, at, QPoint(0, 0), QPoint(0, by), Qt.MouseButton.NoButton,
                                                     Qt.KeyboardModifier.ControlModifier, Qt.ScrollPhase.NoScrollPhase, False))
        settle()
    faults.expect(grid.property("cell") == cell + 40, f"Ctrl and two notches up, one down: thumbnails {cell} -> {grid.property('cell')}, not {cell + 40}")
    shown.close()
    ui.close_view(view)
    return faults


def carry_out(view, session) -> None:
    """Carry an export through. The program has Qt read each tile back from the bench, which
    Qt refuses for a window that is not on screen; so here the bench is put in the corner of
    the window, tile after tile, and the exporter is handed the same pictures."""
    exporter, bench = session.exporter, find(view.rootObject(), "bench")
    while wait(lambda: not exporter.running or (bool(exporter.job) and bench.property("loaded")), 8000) and exporter.running:
        job = exporter.job
        bench.setVisible(True)
        for n, (_, _, wide, high, _) in enumerate(job["tiles"]):
            call(bench, "aim", n)
            settle()
            exporter.tile(view.grabWindow().copy(int(box(bench).left()), int(box(bench).top()), wide, high))
        bench.setVisible(False)
        exporter.rendered()
        wait(lambda: exporter.job != job, 8000)


def shown_at_full_size(view, root) -> np.ndarray:
    """The photo as the window shows it at 1:1, as rows x columns x 3 bytes."""
    call(stage(root), "actual")
    settle()
    area = box(root.findChild(QObject, "photo"))
    seen = view.grabWindow().copy(int(area.left()), int(area.top()), int(area.width()), int(area.height()))
    call(stage(root), "fit")
    settle()
    return export.bytes_of(seen)


def audit_export() -> list[str]:
    """An export is the picture the screen shows, pixel for pixel, in 8 bits and in 16; smaller
    if asked; never over a file that is there; a photo that cannot be read costs a sentence
    and not the others; Stop ends it after the photo under way; and the Export place does it."""
    def some(path: Path, half: bool):
        if "broken" in path.name:
            raise decode.Unreadable(f"{path.name}: the file is damaged and cannot be read")
        return chart(path, half)

    out = Path(tempfile.mkdtemp(prefix="photone-export-", dir=SHELF.name))
    session = ui.Session([Path("first.arw")], load=some)
    view, root = opened(session, "")
    session.exporter.side = 256  # several tiles of the small chart, each fitting the window
    session.setEdits({"exposure": 0.35, "saturation": 125.0, "curve": LIFT,
                      "pins": [{"from": [-0.09, 0.08], "to": [-0.04, 0.11], "range": 0.08}],
                      # (And a mask with settings of its own: what it changes inside it is exported too.)
                      "masks": [{"kind": "radial", "at": [0.3, 0.4], "to": [0.5, 0.4], "feather": 0.4,
                                 "edit": {"exposure": 0.8, "saturation": -40.0, "contrast": 20.0}}]})
    session.commitEdit()
    settle()
    faults = Faults()
    screen = shown_at_full_size(view, root)
    faults.expect(len(session.maskLayers) == 1, "a mask with settings of its own is not drawn as a layer")

    def run(photos: list, kind: str, longest: int = 0) -> None:
        session.exporter.start([str(p) for p in photos], {"kind": kind, "quality": 92, "longest": longest, "folder": str(out)})
        carry_out(view, session)

    def read(name: str) -> np.ndarray:
        image = QImage(str(out / name)).convertToFormat(QImage.Format.Format_RGBA64)
        return np.frombuffer(image.constBits(), np.uint16).reshape(image.height(), -1, 4)[..., :3].astype(int)

    run(session.photos, "png8")
    faults.expect((out / "first.png").exists() and session.exporter.finished == 1 and not session.exporter.failures,
                  f"no file was exported: {sorted(p.name for p in out.iterdir())}, {session.exporter.failures}")
    if (out / "first.png").exists():
        file = read("first.png") // 257
        off = np.abs(file - screen).max() if file.shape == screen.shape else 999
        faults.expect(off <= 1, f"the exported picture is not the one on screen: {file.shape} against {screen.shape}, up to {off} levels off")
    run(session.photos, "png16")
    faults.expect((out / "first-2.png").exists() and (out / "first.png").stat().st_size > 0, "a second export took the first one's place")
    if (out / "first-2.png").exists():
        deep = read("first-2.png")
        gap = np.abs(deep // 257 - screen)
        # (Two levels along a mask's soft edge: there the screen mixes two pictures of 8 bits,
        # the export two of 16, and they round apart; elsewhere one, as ever.)
        faults.expect(gap.max() <= 2 and (gap > 1).mean() < 0.01 and len(np.unique(deep[..., 1])) > 300,
                      f"the 16-bit export is not the screen's picture in finer steps: {len(np.unique(deep[..., 1]))} levels of green, "
                      f"up to {gap.max()} levels off the screen's, more than one in {(gap > 1).mean():.1%} of it")
    run(session.photos, "jpeg", 200)
    small = QImage(str(out / "first.jpg"))
    faults.expect((small.width(), small.height()) == (200, 133), f"an export asked to be 200 px is {small.width()} x {small.height()}")
    run([Path("broken.arw"), Path("first.arw")], "tiff16")
    faults.expect(session.exporter.finished == 2 and len(session.exporter.failures) == 1 and "broken.arw" in session.exporter.failures[0]
                  and (out / "first.tif").exists(), f"a photo that cannot be read: {session.exporter.failures}, {sorted(p.name for p in out.iterdir())}")
    session.exporter.start(["first.arw"] * 3, {"kind": "png8", "quality": 90, "longest": 0, "folder": str(out)})
    session.exporter.stop()
    carry_out(view, session)
    faults.expect(not session.exporter.running and len(list(out.glob("first*.png"))) == 2, f"Stop did not stop: running {session.exporter.running}, {sorted(p.name for p in out.glob('first*.png'))}")
    # (Also with the photo on the bench, half rendered: nothing is written, and the next export is whole.)
    session.exporter.start(["first.arw"], {"kind": "png8", "quality": 90, "longest": 0, "folder": str(out)})
    wait(lambda: bool(session.exporter.job), 8000)
    session.exporter.stop()
    session.exporter.rendered()
    settle()
    faults.expect(not session.exporter.running and len(list(out.glob("first*.png"))) == 2, "Stop during the render left a file or an export running")
    session.exporter.start(["first.arw"], {"kind": "png8", "quality": 90, "longest": 0, "folder": str(out / "nowhere")})
    faults.expect(not session.exporter.running and "no folder" in " ".join(session.exporter.failures), "exporting to a folder that is not there says nothing")

    QCoreApplication.sendEvent(root, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_3, Qt.KeyboardModifier.ControlModifier))
    settle()
    view.grabWindow()
    desk = find(root, "box:export")
    faults.expect(root.property("place") == "export" and desk.isVisible() and find(root, "export-preview").isVisible()
                  and "first.jpg" in find(root, "export-file").property("text"), "Ctrl+3 does not show Export with the open photo to export")
    desk.setProperty("folder", str(out))
    click(view, find(root, "export-kind:png16"))
    view.grabWindow()
    click(view, find(root, "export-go"))
    faults.expect(session.exporter.running, "the Export button starts nothing")
    carry_out(view, session)
    settle()
    faults.expect(len(list(out.glob("first*.png"))) == 3 and session.exporter.finished == 1 and find(root, "export-said").property("text") == "",
                  f"the Export place: {find(root, 'export-said').property('text')!r}")
    last = plain(session.exporter.property("recent"))
    faults.expect(len(last) == 5 and last[0]["name"] == "first-3.png" and last[0]["thumb"].startswith("data:image/jpeg") and find(root, "export-row:0") is not None,
                  f"the last files exported: {[f['name'] for f in last]}")
    click(view, find(root, "export-edit:0"))
    faults.expect(root.property("place") == "edit" and session.path == "first.arw", f"Edit in the log of exports: at {root.property('place')}, {session.path!r}")
    ui.close_view(view)
    return faults


def audit_frame() -> list[str]:
    """The photo's frame on screen: a quarter turn, a crop and both are the untouched picture
    turned and cut, pixel for pixel; straightened, the middle stays the middle and the frame
    is smaller; the wheel counts only what is in the frame; Pick neutral reads the spot that
    is clicked; the original keeps the frame; and an export has it."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "")
    session.exporter.side = 256
    faults = Faults()
    plain_ = shown_at_full_size(view, root).astype(int)

    def framed(values: dict) -> np.ndarray:
        session.setEdits(values)
        session.commitEdit()
        settle()
        return shown_at_full_size(view, root).astype(int)

    def same(seen: np.ndarray, want: np.ndarray, what: str) -> None:
        off = np.abs(seen - want).max() if seen.shape == want.shape else 999
        faults.expect(off <= 1, f"{what}: {seen.shape} against {want.shape}, up to {off} levels off")

    same(framed({"turns": 1.0}), np.rot90(plain_, -1), "a quarter turn to the right")
    faults.expect((session.fullWidth, session.fullHeight) == (320, 480), f"turned, the photo is {session.fullWidth} x {session.fullHeight}")
    session.undo()
    same(framed({"crop": [0.25, 0.25, 0.75, 0.75]}), plain_[80:240, 120:360], "a crop of the middle")
    session.undo()
    same(framed({"turns": 3.0, "crop": [0.0, 0.0, 0.5, 1.0]}), np.rot90(plain_, 1)[:, :160], "a turn to the left and its left half")
    counted(view, session)
    session.setOriginal(True)
    faults.expect((session.fullWidth, session.fullHeight) == (160, 480), "holding B, the photo jumps out of its frame")
    session.setOriginal(False)
    session.exporter.start(["first.arw"], {"kind": "png8", "quality": 90, "longest": 0, "folder": SHELF.name})
    carry_out(view, session)
    file = QImage(str(Path(SHELF.name) / "first.png"))
    same(export.bytes_of(file).astype(int), shown_at_full_size(view, root).astype(int), "the export of a turned and cropped photo")
    session.undo()

    tilted = framed({"angle": 8.0})
    faults.expect(tilted.shape[1] < 440 and abs(tilted.shape[1] / tilted.shape[0] - 1.5) < 0.02, f"straightened by 8 degrees, the photo is {tilted.shape[1]} x {tilted.shape[0]}")
    faults.expect(np.abs(tilted[tilted.shape[0] // 2, tilted.shape[1] // 2] - plain_[160, 240]).max() <= 2, "straightened, the middle is no longer the middle")

    def grays_from(column: int) -> int:  # the row at which the chart's row of grays begins there
        gray = np.ptp(tilted[:, column], axis=1) <= 2
        return int(np.argmax(gray[tilted.shape[0] // 2:])) + tilted.shape[0] // 2

    # Turned to the right, the level line where the grays begin runs downhill to the right.
    left, right = grays_from(tilted.shape[1] // 5), grays_from(tilted.shape[1] * 4 // 5)
    faults.expect(right > left + 15, f"an angle to the right does not turn the picture to the right: the grays begin at row {left} on the left, {right} on the right")
    session.undo()

    framed({"crop": [0.0, 0.86, 1.0, 1.0]})  # the row of grays alone
    counted(view, session)
    box_ = session.cloud["box"]
    faults.expect(max(abs(v) for v in box_) < 0.01, f"cropped to the grays, the wheel still shows colors: {box_}")
    session.undo()
    framed({"crop": [0.5, 0.0, 1.0, 0.5], "turns": 1.0})  # (a spot whose color a white balance can take out)
    session.pickNeutral(0.3, 0.14)
    settle()
    area = box(root.findChild(QObject, "photo"))
    spot = view.grabWindow().pixelColor(int(area.left() + 0.3 * area.width()), int(area.top() + 0.14 * area.height()))
    faults.expect(max(spot.getRgb()[:3]) - min(spot.getRgb()[:3]) <= 5, f"Pick neutral in a cropped, turned photo left the spot {spot.name()}")
    ui.close_view(view)
    return faults


def audit_crop() -> list[str]:
    """The Crop tool and the split under a real mouse and keyboard: C shows the whole photo
    with its frame; the frame is dragged by a corner and by its middle; a shape is kept; the
    turns turn the crop along; Esc undoes what the tool did and Enter keeps it; the split
    shows the photo without its edit on the left of a line that can be dragged."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "")
    faults = Faults()

    def key(which) -> None:
        QCoreApplication.sendEvent(root, QKeyEvent(QEvent.Type.KeyPress, which, Qt.KeyboardModifier.NoModifier))
        settle()
        view.grabWindow()

    def crop() -> list:
        return session.edit["crop"]

    def grip(name: str) -> QPoint:
        return box(find(root, f"crop:{name}")).center().toPoint()

    session.setEdit("crop", [0.1, 0.1, 0.9, 0.9])
    session.commitEdit()
    settle()
    faults.expect(session.fullWidth == 384, f"a crop of four fifths leaves the photo {session.fullWidth} px wide")
    key(Qt.Key.Key_C)
    faults.expect(find(root, "panel:crop").isVisible() and find(root, "crop-frame").isVisible() and session.fullWidth == 480,
                  f"C does not open the Crop tool on the whole photo: {session.fullWidth} px wide")
    area = box(root.findChild(QObject, "photo"))
    drag(view, grip("bottom right"), QPoint(-48, -32))
    want = [0.1, 0.1, 0.9 - 48 / area.width(), 0.9 - 32 / area.height()]
    faults.expect(np.allclose(crop(), want, atol=0.004), f"dragging the bottom right corner gave {crop()}, not {np.round(want, 3)}")
    was = crop()
    drag(view, grip("move"), QPoint(20, 12))
    want = [was[0] + 20 / area.width(), was[1] + 12 / area.height(), was[2] + 20 / area.width(), was[3] + 12 / area.height()]
    faults.expect(np.allclose(crop(), want, atol=0.004), f"dragging the frame by its middle gave {crop()}")
    click(view, find(root, "crop-shape:1 : 1"))
    wide, high = (crop()[2] - crop()[0]) * 480, (crop()[3] - crop()[1]) * 320
    faults.expect(abs(wide - high) < 1.5, f"1 : 1 gave a frame of {wide:.0f} x {high:.0f} pixels")
    drag(view, grip("top left"), QPoint(30, 4))
    wide, high = (crop()[2] - crop()[0]) * 480, (crop()[3] - crop()[1]) * 320
    faults.expect(abs(wide - high) < 1.5 and wide < 250, f"dragged by a corner, a 1 : 1 frame became {wide:.0f} x {high:.0f}")
    was = crop()
    click(view, find(root, "turn-right"))
    faults.expect(session.edit["turns"] == 1 and np.allclose(crop(), [1 - was[3], was[0], 1 - was[1], was[2]], atol=1e-4)
                  and (session.fullWidth, session.fullHeight) == (320, 480), f"Turn right gave turns {session.edit['turns']}, crop {crop()}")
    key(Qt.Key.Key_BracketLeft)
    faults.expect(session.edit["turns"] == 0 and np.allclose(crop(), was, atol=1e-4), "[ does not turn the photo and its crop back")
    key(Qt.Key.Key_Escape)
    faults.expect(not find(root, "panel:crop").isVisible() and crop() == [0.1, 0.1, 0.9, 0.9], f"Esc does not undo what the tool did: {crop()}")
    key(Qt.Key.Key_C)
    drag(view, grip("right"), QPoint(-40, 0))
    kept = crop()
    key(Qt.Key.Key_Return)
    faults.expect(not find(root, "panel:crop").isVisible() and crop() == kept and session.fullWidth == round((kept[2] - kept[0]) * 480),
                  f"Enter does not keep the crop and show the photo in it: {crop()}, {session.fullWidth} px")

    # Straightening: the picture turns under a frame that stays level, and what is in the
    # frame is what the photo is once the tool is shut.
    session.resetAll()
    settle()
    key(Qt.Key.Key_C)
    # (1 : 1 is still chosen from before, and the frame, reset, is not square: a drag makes it so.)
    drag(view, grip("bottom right"), QPoint(-20, -10))
    wide, high = (crop()[2] - crop()[0]) * 480, (crop()[3] - crop()[1]) * 320
    faults.expect(abs(wide - high) < 1.5 and 250 < wide < 320, f"a frame that had lost its shape became {wide:.0f} x {high:.0f} when dragged, not 1 : 1")
    session.undo()
    click(view, find(root, "crop-shape:Free"))
    slider = find(root, "slider:angle")  # kept: its wrapper going would take the track's
    track = slider.findChild(QObject, "track")
    drag(view, box(slider.findChild(QObject, "thumb")).center().toPoint(), QPoint(round(8 * (track.width() - 12) / 30), 0))  # (by its thumb)
    view.grabWindow()
    faults.expect(abs(session.edit["angle"] - 8) < 0.15, f"dragging Straighten by 8 degrees' worth gave {session.edit['angle']}")
    area, frame = box(root.findChild(QObject, "photo")), box(find(root, "crop-box"))
    lean = np.radians(session.edit["angle"])
    wide = 480 * np.cos(lean) + 320 * np.sin(lean)
    faults.expect(abs(session.fullWidth - wide) <= 1, f"straightening, the tool shows {session.fullWidth} px across, not the turned picture's {wide:.0f}")
    shot = view.grabWindow()
    # (The frame fills the view, so the turned picture may reach under the left panel: the
    # corner looked at is the first of it that is in the view.)
    corner = shot.pixelColor(int(max(area.left(), box(stage(root)).left())) + 3, int(area.top()) + 3)
    faults.expect(max(corner.getRgb()[:3]) <= 15, f"beyond the turned picture's corner there is {corner.name()}, not nothing")

    def grays_from(share: float) -> int:  # the row at which the chart's grays begin, that far across the view
        x = int(area.left() + area.width() * share)
        column = [shot.pixelColor(x, y) for y in range(int(area.center().y()), int(area.bottom()))]
        return next((n for n, c in enumerate(column) if max(c.getRgb()[:3]) - min(c.getRgb()[:3]) <= 2 and c != BACKGROUND), -1)

    left, right = grays_from(0.3), grays_from(0.7)
    want = np.tan(lean) * 0.4 * area.width()
    faults.expect(abs((right - left) - want) <= 3, f"the picture is not turned by the angle under the frame: the grays begin {right - left} px lower on the right, not {want:.0f}")
    inside = [QPoint(int(frame.left() + frame.width() * u), int(frame.top() + frame.height() * v)) for u, v in ((0.5, 0.5), (0.1, 0.1), (0.9, 0.85), (0.15, 0.9))]
    before = [shot.pixelColor(p) for p in inside]
    counted(view, session)
    black = session.histogram["under"]  # what the scopes count while the tool shows more than the frame
    want = [frame.left() / 1, frame.top() / 1]
    drag(view, grip("top left"), QPoint(40, 30))
    view.grabWindow()
    now = box(find(root, "crop-box"))
    faults.expect(abs(now.left() - want[0] - 40) <= 1.5 and abs(now.top() - want[1] - 30) <= 1.5 and abs(now.right() - frame.right()) <= 1.5 and abs(now.bottom() - frame.bottom()) <= 1.5,
                  f"dragging a corner of the level frame by 40, 30 px moved it from {frame} to {now}")
    session.undo()
    drag(view, grip("bottom right"), QPoint(400, 300))
    view.grabWindow()
    out = edit.frame(session.history.edit, 480, 320)["matrix"]
    ends = [edit.framed(out, *p) for p in ((0, 0), (1, 0), (1, 1), (0, 1))]
    now = box(find(root, "crop-box"))
    faults.expect(all(-1e-3 <= v <= 1.001 for end in ends for v in end) and abs(now.left() - frame.left()) <= 1.5 and abs(now.top() - frame.top()) <= 1.5,
                  f"dragged far out of the picture, the frame is at {np.round(ends, 3).tolist()} of it and its other corner moved to {now.topLeft()}")
    session.undo()
    settle()
    key(Qt.Key.Key_Return)
    area = box(root.findChild(QObject, "photo"))
    shot = view.grabWindow()
    after = [shot.pixelColor(int(area.left() + area.width() * u), int(area.top() + area.height() * v)) for u, v in ((0.5, 0.5), (0.1, 0.1), (0.9, 0.85), (0.15, 0.9))]
    off = max(abs(a - b) for was, now_ in zip(before, after) for a, b in zip(was.getRgb()[:3], now_.getRgb()[:3]))
    faults.expect(off <= 6, f"what was in the frame is not what the photo is once the tool is shut: up to {off} levels off")
    counted(view, session)
    faults.expect(abs(session.histogram["under"] - black) < 0.005,
                  f"with the Crop tool open the scopes do not count the frame: {black:.3f} black then, {session.histogram['under']:.3f} once it is shut")

    session.resetAll()
    session.setEdit("exposure", -1.5)
    session.commitEdit()
    # A drag just outside the frame turns the picture under it, the way the pointer goes.
    show(root, "crop")
    settle()
    tip = find(root, "screen:tip")
    faults.expect(tip.isVisible() and "straighten" in tip.property("says"), f"opening Crop shows no tip at the top of the photo: {tip.property('says')!r}")
    session.setEdit("angle", 0.0)
    session.commitEdit()
    settle()
    view.grabWindow()
    frame_ = box(find(root, "crop-box"))
    beside = QPointF(frame_.center().x(), frame_.bottom() + 20)
    reach_ = beside.y() - frame_.center().y()
    drag(view, beside.toPoint(), QPoint(-int(reach_ * math.tan(math.radians(8))), 0))
    faults.expect(abs(session.history.edit.get("angle", 0) - 8) < 0.6, f"a drag of 8° round the frame, outside it, left the angle at {session.history.edit.get('angle', 0)}")
    # The pointer for turning shows on the rim only, and lies round the frame's middle.
    turn_area = find(root, "crop:turn")
    frame_ = box(find(root, "crop-box"))  # (smaller now: turned, less of the picture is level)
    beside = QPointF(frame_.center().x(), frame_.bottom() + 20)
    QTest.mouseMove(view, frame_.center().toPoint(), 100)
    settle()
    faults.expect(QGuiApplication.overrideCursor() is None, "the pointer for turning shows inside the crop frame")
    for nudge in (0, 3, 6):  # (a move that follows another is handed on with the next frame)
        QTest.mouseMove(view, beside.toPoint() + QPoint(nudge, 0), 100)
        QTest.qWait(120)
    settle()
    over = QGuiApplication.overrideCursor()
    faults.expect(over is not None and over.shape() == Qt.CursorShape.BitmapCursor and turn_area.property("turnStep") == 18,
                  f"on the frame's rim, under it, the pointer is not the turning one bulging downward: {over and over.shape()}, step {turn_area.property('turnStep')}")
    # A shape of one's own: Custom lights up and opens its field; what is typed is used without
    # Enter, and Enter only leaves the field (it used to leave the Crop tool as well).
    custom = find(root, "crop-shape:Custom")
    click(view, custom)
    entry = find(root, "crop-custom-entry")
    faults.expect(custom.property("on") and entry.isVisible() and entry.hasActiveFocus(), "Custom does not light up, or its field is not open with the cursor in it")
    entry.setProperty("text", "")
    for letter in "16:10":
        QTest.keyClick(view, letter)
    QTest.qWait(500)
    settle()
    got = find(root, "crop-box")
    faults.expect(abs(stage(root).property("cropRatio") - 1.6) < 1e-6 and abs(got.width() / got.height() - 1.6) < 0.02 and custom.property("on"),
                  f"16:10 typed (no Enter) gave a frame of {got.width() / got.height():.3f} (asked {stage(root).property('cropRatio')}), Custom lit {custom.property('on')}")
    QTest.keyClick(view, Qt.Key.Key_Return)
    settle()
    faults.expect(session.panels["tool"] == "crop" and custom.property("on"), f"Enter in the shape's field left the Crop tool: {session.panels['tool']}")
    # Typed, then a click on the card's empty ground: used at once, without waiting.
    click(view, custom)
    entry.setProperty("text", "")
    for letter in "2:1":
        QTest.keyClick(view, letter)
    tap(view, box(find(root, "panel:crop")).bottomLeft().toPoint() + QPoint(4, -6))
    faults.expect(abs(stage(root).property("cropRatio") - 2.0) < 1e-6 and not entry.hasActiveFocus(),
                  f"2:1 typed, then a click elsewhere: the shape is {stage(root).property('cropRatio')}, the field still has the keys {entry.hasActiveFocus()}")
    click(view, find(root, "crop-portrait"))
    faults.expect(abs(stage(root).property("cropRatio") - 1 / 2.0) < 1e-6, f"Portrait does not stand the shape on end: {stage(root).property('cropRatio')}")
    click(view, find(root, "crop-portrait"))
    click(view, find(root, "crop-shape:3 : 2"))
    faults.expect(not custom.property("on") and find(root, "crop-shape:3 : 2").property("on") and not entry.isVisible(),
                  "choosing 3 : 2 after Custom leaves Custom lit or its field open")
    call(stage(root), "reshape", 0.0)
    session.setEdit("angle", 0.0)
    session.commitEdit()
    show(root, "")
    settle()
    area = box(root.findChild(QObject, "photo"))
    at = [QPoint(int(area.left() + area.width() * share), int(area.top() + area.height() * 0.92)) for share in (0.31, 0.69)]
    edited = [view.grabWindow().pixelColor(p).green() for p in at]
    key(Qt.Key.Key_Backslash)
    shown = [view.grabWindow().pixelColor(p).green() for p in at]
    faults.expect(stage(root).property("split") == 0.5 and shown[0] > edited[0] + 20 and shown[1] == edited[1],
                  f"the split does not show the photo without its edit on the left: {shown} against {edited}")
    drag(view, box(find(root, "split-line")).center().toPoint(), QPoint(int(area.width() * 0.3), 0))
    moved = [view.grabWindow().pixelColor(p).green() for p in at]
    faults.expect(abs(stage(root).property("split") - 0.8) < 0.02 and moved[1] > edited[1] + 20, f"dragging the line to the right: split {stage(root).property('split'):.2f}, {moved}")
    key(Qt.Key.Key_Backslash)
    faults.expect(stage(root).property("split") < 0 and view.grabWindow().pixelColor(at[0]).green() == edited[0], "the backslash key again does not take the split away")
    ui.close_view(view)
    return faults


def audit_wheel() -> list[str]:
    """The wheel under a real mouse and keyboard: the wheel zooms about the pointer, a drag
    moves the view, Fit frames the dots and All shows the whole disc (F and 0 too); the working
    range limits the dots, Pick takes it from the photo; the pointer on the photo puts a cross
    on the wheel and a tick on the lightness strip."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "")
    faults = Faults()
    counted(view, session)
    wheel = find(root, "wheel")
    place = box(wheel)

    def under(at: QPointF) -> tuple[float, float]:  # the a, b drawn at that point of the window
        span, middle = wheel.property("span"), wheel.property("middle")
        return (middle.x() + (at.x() - place.center().x()) / place.height() * span,
                middle.y() - (at.y() - place.center().y()) / place.height() * span)

    def key(which) -> None:
        QCoreApplication.sendEvent(root, QKeyEvent(QEvent.Type.KeyPress, which, Qt.KeyboardModifier.NoModifier))
        settle()

    box_ = session.cloud["box"]
    fitted = wheel.property("span")
    faults.expect(fitted < wheel.property("whole") and all(
        abs(v - c) < fitted / 2 for v, c in zip(box_, (*[wheel.property("middle").x()] * 2, *[wheel.property("middle").y()] * 2))),
        f"a photo's colors are not framed when it opens: span {fitted:.3f}, box {box_}")
    at = QPointF(place.left() + place.width() * 0.3, place.top() + place.height() * 0.35)
    held, tools_at, photo_zoom = under(at), find(root, "tools-scroll").property("contentY"), stage(root).property("shown")
    QCoreApplication.sendEvent(view, QWheelEvent(at, at, QPoint(0, 0), QPoint(0, 360), Qt.MouseButton.NoButton,
                                                 Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
    settle()
    faults.expect(wheel.property("span") < fitted * 0.7, "turning the mouse wheel over the color wheel does not zoom it")
    faults.expect(max(abs(a - b) for a, b in zip(held, under(at))) < 1e-6, "the wheel zoomed, but not about the pointer")
    faults.expect(find(root, "tools-scroll").property("contentY") == tools_at and stage(root).property("shown") == photo_zoom,
                  "zooming the wheel also scrolled the column or zoomed the photo")
    before, span = wheel.property("middle"), wheel.property("span")
    drag(view, at.toPoint(), QPoint(40, 30))
    moved = wheel.property("middle")
    faults.expect(abs(moved.x() - before.x() + 40 / place.height() * span) < 1e-6
                  and abs(moved.y() - before.y() - 30 / place.height() * span) < 1e-6, "a drag does not move the wheel's view with the pointer")
    key(Qt.Key.Key_0)
    faults.expect(wheel.property("all"), "0 does not show the whole wheel")
    key(Qt.Key.Key_F)
    faults.expect(abs(wheel.property("span") - fitted) < 1e-9, "F does not frame the colors again")
    click(view, find(root, "wheel-all"))
    faults.expect(wheel.property("all"), "All does not show the whole wheel")

    dots = len(session.cloud["colors"])
    click(view, find(root, "range:shadows"))
    dark = session.cloud
    strip = box(find(root, "lightness-strip"))
    faults.expect(session.range["name"] == "shadows" and 0 < len(dark["colors"]) < dots,
                  f"Shadows does not limit the dots: {len(dark['colors'])} of {dots}")
    faults.expect(max(oklab_of(QColor(c))[0] for c in dark["colors"]) < 0.45, "with Shadows chosen, light colors are still on the wheel")
    faults.expect(abs(box(find(root, "range-above")).left() - strip.left() - 0.4 * strip.width()) < 1.5,
                  "the lightness strip does not show where the range ends")
    area = box(root.findChild(QObject, "photo"))
    click(view, find(root, "range:full"))
    faults.expect(len(session.cloud["colors"]) == dots, "All tones does not bring every dot back")

    spot = QPointF(area.left() + area.width() * 0.3, area.top() + area.height() * 0.6)
    QCoreApplication.sendEvent(view, QMouseEvent(QEvent.Type.MouseMove, spot, spot, Qt.MouseButton.NoButton,
                                                 Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))
    settle()
    probe, want = wheel.property("probe"), session.colorAt(0.3, 0.6)
    probe = probe.toVariant() if hasattr(probe, "toVariant") else probe  # an empty list comes as a JS value
    faults.expect(len(probe) == 3 and np.allclose(probe, want, atol=0.01), f"the pointer on the photo gives the wheel {probe}, not {want}")
    if len(probe) == 3:
        per = place.height() / wheel.property("span")
        cross = view.grabWindow().pixelColor(round(place.center().x() + probe[0] * per) + 9, round(place.center().y() - probe[1] * per))
        faults.expect(min(cross.red(), cross.green(), cross.blue()) > 190, f"no cross on the wheel at that color: {cross.name()}")
        tick = box(find(root, "lightness-tick"))
        faults.expect(find(root, "lightness-tick").isVisible() and abs(tick.left() - strip.left() - probe[2] * strip.width()) < 2,
                      "no tick on the lightness strip at that lightness")
    ui.close_view(view)
    return faults


def audit_stepping() -> list[str]:
    """With two photos open, the arrow keys go to the next and back, and stop at the ends."""
    session = ui.Session([Path("first.arw"), Path("second.arw")], load=chart)
    view = ui.make_view(session)
    session.surface_known(False)
    root = view.rootObject()
    root.setSize(view.size())
    seen = []
    for which in (Qt.Key.Key_Right, Qt.Key.Key_Right, Qt.Key.Key_Left):
        QCoreApplication.sendEvent(root, QKeyEvent(QEvent.Type.KeyPress, which, Qt.KeyboardModifier.NoModifier))
        settle()
        seen.append(session.source.rsplit("/", 2)[1])
    faults = [] if seen == ["1", "1", "0"] else [f"the arrow keys stepped through photos as {seen}, not 1, 1, 0"]
    # The zoom, and the place in the photo, stay from one photo to the next; so does Fit.
    pane = stage(root)
    wait(lambda: root.property("photoReady"))
    if not pane.property("fitted"):
        faults.append("stepping to another photo took a fitted photo out of Fit")
    call(pane, "zoomTo", 2.0, 100.0, 80.0)
    settle()
    held = (pane.property("zoom"), pane.property("center"))
    QCoreApplication.sendEvent(root, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier))
    settle()
    wait(lambda: root.property("photoReady"))
    if session.source.rsplit("/", 2)[1] != "1" or pane.property("fitted") or (pane.property("zoom"), pane.property("center")) != held:
        faults.append(f"stepping to another photo changed the zoom or the place: {held} to {pane.property('zoom')}, {pane.property('center')}, "
                      f"fitted {pane.property('fitted')}")
    ui.close_view(view)
    return faults


def audit_held() -> list[str]:
    """A step to a photo beside the open one, whose first picture is held (`session.held`),
    shows that photo in the first frame drawn after it: not its thumbnail, not nothing. Read
    off a frame the GPU draws, not off `ready` (PLAN.md §4.1 once timed that: "28 ms", where
    the screen showed the thumbnail for 40 to 130 ms)."""
    from concurrent.futures import Future

    def one_color(path: Path, half: bool):
        rgb = np.full((64, 96, 3), 0.05, np.float32)
        rgb[..., int(path.stem) % 3] = 0.7
        return decode.meta(path.name, 96, 64), decode.Pixels(decode.pack(rgb))

    session = ui.Session([Path(f"{n}.png") for n in range(5)], load=one_color)
    done = Future()
    done.set_result(None)
    session.ahead = lambda wanted, begin=True: dict.fromkeys(wanted, done)  # all decoded ahead at once
    view, root = opened(session, "")
    middle = box(stage(root)).center().toPoint()
    faults = Faults()

    def drawn() -> int:  # which photo the frame drawn now shows in the middle (its color), or -1
        rgb = export.bytes_of(view.grabWindow().copy(middle.x() - 2, middle.y() - 2, 4, 4))[..., :3].mean(axis=(0, 1))
        return int(rgb.argmax()) if rgb.max() > 2 * np.median(rgb) + 20 else -1

    def held() -> bool:
        there = {i.property("source").toString() for i in stage(root).childItems() if isinstance(i.property("source"), QUrl) and i.property("implicitWidth") > 0}
        return len(session.held) > 0 and set(session.held) <= there

    session.goTo(1)
    for way in (1, -1, -1):  # on, back to the one before (held as the one behind), and on again
        faults.expect(wait(held), f"the photos beside photo {session.index} are never held")
        settle()
        session.step(way)
        faults.expect(drawn() == session.index % 3, f"a step to photo {session.index}, beside the one before, does not show it in the next frame")
    # (Proof that the audit can tell: a photo that is not held is not on the next frame.)
    wait(held)
    session.goTo(4)
    faults.expect(drawn() == -1, "a photo two along, not held, was drawn at once: the audit proves nothing")
    # Other photos opened (another folder): what was held of these is let go, and its GPU memory.
    session.openAt([f"{n}.png" for n in (5, 6)], 0)
    settle()
    kept = {a for i in stage(root).childItems() if isinstance(i.property("source"), QUrl) and (a := i.property("source").toString()).endswith("/half")}
    faults.expect(kept <= {session.source, *session.held}, f"other photos opened, pictures of the ones before are still held: {kept}")
    ui.close_view(view)
    return faults


class Faults(list):
    """What an audit found: `expect(what should hold, the fault if it does not)`."""

    def expect(self, holds, fault: str) -> None:
        if not holds:
            self.append(fault)


def audit_slider() -> list[str]:
    """A slider under a real mouse: pressing its thumb changes nothing, a drag moves the value by
    how far the pointer went (Shift: a tenth of that), a double click puts it back to rest, and
    a press off the thumb brings the thumb there at once."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "tone")
    slider = find(root, "slider:exposure")  # kept: its wrapper going would take the track's
    track = slider.findChild(QObject, "track")
    at = box(track).center().toPoint()                   # on the thumb, which rests in the middle
    per_px = 10 / (track.width() - 12)                   # -5..+5 EV over the thumb's travel
    faults = Faults()
    drag(view, at, QPoint(100, 0), down=lambda: faults.expect(
        session.edit["exposure"] == 0, f"pressing the track changed the value to {session.edit['exposure']}"))
    moved = session.edit["exposure"]
    if abs(moved - 100 * per_px) > 0.011:
        faults.append(f"a drag of 100 px moved Exposure by {moved:.2f}, not {100 * per_px:.2f}")
    drag(view, at + QPoint(100, 0), QPoint(100, 0), Qt.KeyboardModifier.ShiftModifier)
    fine = session.edit["exposure"] - moved
    if abs(fine - 10 * per_px) > 0.011:
        faults.append(f"with Shift, 100 px moved Exposure by {fine:.2f}, not {10 * per_px:.2f}")
    double_click(view, at)
    if session.edit["exposure"] != 0:
        faults.append(f"a double click left Exposure at {session.edit['exposure']}")
    # A press off the thumb brings the thumb there at once, and the drag goes on from there:
    # pressed 120 px left of the middle the value is that of 120 px left, and dragged 30 px
    # right, that of 90 px left. All of it is one step of undo.
    drag(view, at + QPoint(-120, 0), QPoint(30, 0), down=lambda: faults.expect(
        abs(session.edit["exposure"] + 120 * per_px) < 0.011, f"a press 120 px left of the thumb gave {session.edit['exposure']:.2f}, not {-120 * per_px:.2f}"))
    faults.expect(abs(session.edit["exposure"] + 90 * per_px) < 0.011, f"a drag begun 120 px left of the thumb and taken 30 px right gave {session.edit['exposure']:.2f}, not {-90 * per_px:.2f}")
    session.undo()
    faults.expect(session.edit["exposure"] == 0, "a press off the thumb and its drag are more than one step of undo")
    # A number typed and then left by a click elsewhere (on the empty ground of the photo's
    # view) is taken, as Enter would; Escape throws it away (docs/PLAN.md §5).
    def typed(value: str, leave) -> None:
        tap(view, box(find(slider, "value")).center().toPoint())
        entry = find(slider, "entry")
        entry.setProperty("text", "")
        for letter in value:
            QTest.keyClick(view, letter)
        leave()
        settle()

    corner = box(stage(root)).topLeft().toPoint() + QPoint(6, 6)
    typed("1.25", lambda: tap(view, corner))
    faults.expect(abs(session.edit["exposure"] - 1.25) < 1e-6, f"1.25 typed, then a click elsewhere: Exposure is {session.edit['exposure']}")
    typed("2", lambda: QTest.keyClick(view, Qt.Key.Key_Escape))
    faults.expect(abs(session.edit["exposure"] - 1.25) < 1e-6, f"2 typed, then Escape: Exposure is {session.edit['exposure']}")
    ui.close_view(view)
    return faults


def audit_moves() -> list[str]:
    """A mouse that reports far more often than frames are shown (app.PerFrameInput): of a
    burst of moves the interface is told the first and then only the latest, the last one is
    never lost, and a release that follows a burst finds the slider where the pointer is."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "tone")
    slider = find(root, "slider:exposure")  # kept: its wrapper going would take the track's
    track = slider.findChild(QObject, "track")
    at, per_px = box(track).center().toPoint(), 10 / (track.width() - 12)  # (on the thumb)
    faults, told = Faults(), [0]
    session.changed.connect(lambda: told.__setitem__(0, told[0] + 1))

    def reports(first: int, last: int) -> None:  # the pointer going right a pixel a report, all at once
        for n in range(first, last + 1):
            to = QPointF(at + QPoint(n, 0))
            QCoreApplication.sendEvent(view, QMouseEvent(QEvent.Type.MouseMove, to, to, Qt.MouseButton.NoButton,
                                                         Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))

    QTest.mousePress(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at, 500)
    settle()
    QTest.qWait(30)  # (whatever the press set going is over)
    before = told[0]
    reports(1, 60)
    faults.expect(told[0] - before <= 2, f"60 reports at once changed the photo {told[0] - before} times")
    QTest.qWait(60)
    faults.expect(abs(session.edit["exposure"] - 60 * per_px) < 0.011 and told[0] - before <= 4,
                  f"after the burst Exposure is {session.edit['exposure']:.2f}, not {60 * per_px:.2f}, in {told[0] - before} changes")
    reports(61, 100)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at + QPoint(100, 0))
    faults.expect(abs(session.edit["exposure"] - 100 * per_px) < 0.011,
                  f"let go straight after a burst, Exposure is {session.edit['exposure']:.2f}, not {100 * per_px:.2f}: the last move was lost")
    settle()
    session.undo()
    faults.expect(session.edit["exposure"] == 0, "a drag in bursts is more than one step of undo")
    # Turns of the wheel likewise: forty small turns at once on a pin (each a whole edit) are
    # added up and change the photo once or twice, and the pin ends where forty would take it.
    session.setEdit("pins", [{"from": [0.05, 0.02], "to": [0.05, 0.02], "range": 0.08}])
    session.commitEdit()
    show(root, "")
    wheel = find(root, "wheel")
    middle, span, side = wheel.property("middle"), wheel.property("span"), wheel.height()  # (Wheel.px, Wheel.py: by its height)
    spot = QPointF(box(wheel).center().x() + (0.05 - middle.x()) / span * side,
                   box(wheel).y() + wheel.height() / 2 - (0.02 - middle.y()) / span * side)
    QTest.qWait(60)
    before = told[0]
    for _ in range(40):
        QCoreApplication.sendEvent(view, QWheelEvent(spot, spot, QPoint(0, 0), QPoint(0, 15), Qt.MouseButton.NoButton,
                                                     Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
    QTest.qWait(60)
    reach = session.edit["pins"][0]["range"]
    faults.expect(told[0] - before <= 3 and abs(reach - 0.08 * 1.001 ** 600) < 1e-5,
                  f"forty turns of the wheel on a pin at once: {told[0] - before} changes, its range {reach:.4f} (not {0.08 * 1.001 ** 600:.4f})")
    ui.close_view(view)
    return faults


def audit_resume() -> list[str]:
    """Started without a photo named, the program is where it was left: the library at the
    place that was on show with the photo last looked at selected, or that photo open in
    Edit if that is where the window was; and it keeps track as one goes on."""
    first = shelf("resume", marked=False)
    settings = QSettings(str(Path(SHELF.name) / "resume.ini"), QSettings.Format.IniFormat)
    settings.setValue("library/raw_only", False)  # (the shelf's photos are JPEGs)
    first.show("month:2026-07")
    wanted = first.model.rows[1]["path"]
    settings.setValue("library/where", "month:2026-07")
    settings.setValue("library/photo", wanted)
    faults = Faults()

    def started(place: str):
        settings.setValue("view/place", place)
        shown = library.Library(first.catalog, first.cache, settings)
        session = ui.Session([], settings=settings)
        view = ui.make_view(session, shown)
        session.surface_known(False)
        root = view.rootObject()
        root.setSize(view.size())
        settle()
        view.grabWindow()
        settle()
        return shown, session, view, root

    shown, session, view, root = started("library")
    here = find(root, "box:library")
    faults.expect(root.property("place") == "library" and shown.where == "month:2026-07" and shown.count == 3,
                  f"left in the library at July 2026, the program starts at {root.property('place')}, {shown.where}, {shown.count} photos")
    faults.expect(here.property("current") == 1 and [int(r) for r in plain(here.property("chosen"))] == [1],
                  f"the photo last looked at is not the one selected: row {here.property('current')}")
    ui.close_view(view)
    shown.close()
    shown, session, view, root = started("edit")
    wait(lambda: root.property("photoReady"), 4000)
    faults.expect(root.property("place") == "edit" and session.path == wanted and len(session.photos) == 3,
                  f"left in Edit on a photo, the program starts in {root.property('place')} with {session.path or 'no photo'} open")
    QCoreApplication.sendEvent(root, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier))
    settle()
    faults.expect(session.path != wanted and settings.value("library/photo") == session.path,
                  f"going on to another photo is not remembered: {settings.value('library/photo')}")
    QCoreApplication.sendEvent(root, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_1, Qt.KeyboardModifier.ControlModifier))
    settle()
    faults.expect(root.property("place") == "library" and settings.value("view/place") == "library", "going back to the library is not remembered")
    click(view, find(root, "place:all"))
    faults.expect(settings.value("library/where") == "all", f"another place chosen in the library is not remembered: {settings.value('library/where')}")
    ui.close_view(view)
    shown.close()
    # The strip is as tall as it was dragged, and as far along as it was left in each place
    # (the owner: "the scroll amount on that strip (per folder ...)").
    shown, session, view, root = started("edit")
    wait(lambda: root.property("photoReady"), 4000)
    drag(view, box(find(root, "strip-edge")).center().toPoint(), QPoint(0, -120))
    view.grabWindow()
    strip, row = find(root, "box:strip"), find(root, "strip-row")
    high = strip.height()

    def along() -> float:
        return row.property("contentX") - row.property("originX")

    far = min(333.0, row.property("contentWidth") - row.width())
    row.setProperty("contentX", row.property("originX") + far)
    QTest.qWait(400)
    faults.expect(far > 50 and "all" in str(settings.value("view/strip_at")), f"scrolled {far:.0f} px along, the strip kept {settings.value('view/strip_at')}")
    shown.show("month:2026-07")
    session.openAt(shown.paths(), 0)
    QTest.qWait(400)
    shown.show("all")
    session.openAt(shown.paths(), 0)
    QTest.qWait(400)
    faults.expect(abs(along() - far) <= 2, f"back in a place left {far:.0f} px along, the strip is {along():.0f} px along")
    ui.close_view(view)
    shown.close()
    shown, session, view, root = started("edit")
    wait(lambda: root.property("photoReady"), 4000)
    view.grabWindow()
    strip, row = find(root, "box:strip"), find(root, "strip-row")
    faults.expect(abs(strip.height() - high) <= 1, f"dragged {high:.0f} px tall, the strip starts {strip.height():.0f} px tall")
    faults.expect(abs(along() - far) <= 2, f"left {far:.0f} px along, the strip starts {along():.0f} px along")
    ui.close_view(view)
    shown.close()
    return faults


def audit_scroll() -> list[str]:
    """Scrolling the library: a notch of the mouse wheel goes a row, the scrollbar is wide
    enough to take hold of, dragging its thumb scrolls by as much, a press beside the thumb
    goes there. (The owner: "pretty slow and i cant even grab the scrollbar, its also very thin".)"""
    shown = shelf("scroll", marked=False)
    session = ui.Session([])
    view = ui.make_view(session, shown)
    session.surface_known(False)
    root = view.rootObject()
    root.setSize(view.size())
    settle()
    view.grabWindow()
    faults = Faults()
    for _ in range(5):  # larger thumbnails: more rows than the window holds
        click(view, find(root, "larger"))
    view.grabWindow()
    grid, bar = find(root, "library-grid"), find(root, "library-bar")
    thumb = bar.findChild(QObject, "thumb")
    reach, row = grid.property("contentHeight") - grid.height(), grid.property("cellHeight")
    faults.expect(reach > row and bar.isVisible() and bar.width() >= 12 and thumb.width() >= 5,
                  f"nothing to scroll by, or too thin: {reach:.0f} px to go, a bar {bar.width():.0f} px wide, a thumb {thumb.width():.0f}")
    bar.setVisible(False)  # and it is the only one: without it, nothing is drawn where it was
    QCoreApplication.processEvents()
    under = export.bytes_of(view.grabWindow().copy(int(box(bar).left()), int(box(grid).top()), int(view.width() - box(bar).left()), int(grid.height())))
    bar.setVisible(True)
    faults.expect(len(np.unique(under.reshape(-1, 3), axis=0)) == 1, "under the grid's scrollbar something else is drawn: a second scrollbar (the owner saw two)")

    def turn(notches: int) -> None:
        spot = QPointF(box(grid).center())
        QCoreApplication.sendEvent(view, QWheelEvent(spot, spot, QPoint(0, 0), QPoint(0, -120 * notches), Qt.MouseButton.NoButton,
                                                     Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
        QTest.qWait(250)  # it glides there

    turn(1)
    faults.expect(abs(grid.property("contentY") - min(reach, row)) < 2, f"a notch of the wheel scrolled {grid.property('contentY'):.0f} px, a row is {row:.0f}")
    turn(-1)
    faults.expect(abs(grid.property("contentY")) < 2, "a notch back does not come back")
    room = bar.height() - thumb.height()
    drag(view, box(thumb).center().toPoint(), QPoint(0, 60))
    want = min(reach, 60 / room * reach)
    faults.expect(abs(grid.property("contentY") - want) < 3, f"dragging the thumb 60 px scrolled {grid.property('contentY'):.0f} px, not {want:.0f}")
    tap(view, QPoint(int(box(bar).center().x()), int(box(bar).bottom()) - 4))
    faults.expect(abs(grid.property("contentY") - reach) < 3, f"a press at the foot of the bar goes to {grid.property('contentY'):.0f}, the end is {reach:.0f}")
    faults += audit_layout(view)
    shown.close()
    ui.close_view(view)
    return faults


def audit_strip() -> list[str]:
    """The filmstrip under a real mouse: the photos at hand in a row under the photo, their
    thumbnails there although the library rests, the open one marked; a click shows another,
    the wheel goes along the row, the top edge is dragged to make it taller or put it away,
    and a double click on it brings it back."""
    shown = shelf("strip-audit", marked=False)
    session = ui.Session([])
    view = ui.make_view(session, shown)
    session.surface_known(False)
    root = view.rootObject()
    root.setSize(view.size())
    settle()
    faults = Faults()
    for made in shown.cache.glob("*.jpg"):  # no thumbnail is there: the strip has to ask
        made.unlink()
    shown._news()
    session.openAt(shown.paths(), 3)
    call(root, "show", "edit")
    wait(lambda: root.property("photoReady"))
    view.grabWindow()
    strip, row = find(root, "box:strip"), find(root, "strip-row")
    faults.expect(strip.isVisible() and shown.resting, "no filmstrip under the photo, or the library is not resting")
    wait(lambda: shown.making == 0 and not shown.strip, 5000)
    view.grabWindow()
    wait(lambda: pictures_in(root), 3000)
    cells = [i for i in items(row) if i.objectName().startswith("strip:")]
    drawn = [c for c in cells if c.findChild(QObject, "strip-thumb").property("paintedWidth") > 20]
    faults.expect(len(cells) >= 6 and len(drawn) == len(cells), f"{len(drawn)} of {len(cells)} cells of the strip have their thumbnail")
    faults.expect(len(list(shown.cache.glob("*.jpg"))) <= len(cells) + 1,
                  f"resting, the library made {len(list(shown.cache.glob('*.jpg')))} thumbnails for a strip of {len(cells)}")

    def marked() -> list:
        return [c.property("index") for c in cells if c.property("open")]

    faults.expect(marked() == [3], f"the open photo is not the one marked in the strip: {marked()}")
    click(view, find(root, "strip:5"))
    wait(lambda: root.property("photoReady"))
    faults.expect(session.index == 5 and session.path == shown.paths()[5] and marked() == [5],
                  f"a click on the sixth cell opened photo {session.index}, marked {marked()}")
    area, high = box(stage(root)), strip.height()
    edge = box(find(root, "strip-edge")).center().toPoint()
    stood = box(find(root, "strip:5")).center().x()
    drag(view, edge, QPoint(0, -40))
    view.grabWindow()
    faults.expect(abs(box(find(root, "strip:5")).center().x() - stood) <= 2,
                  f"made taller, the strip moved the open photo from {stood:.0f} to {box(find(root, 'strip:5')).center().x():.0f} px across")
    faults.expect(abs(strip.height() - high - 40) <= 1 and abs(box(stage(root)).height() - area.height() + 40) <= 1,
                  f"dragging its edge up by 40 px made the strip {strip.height():.0f} px (it was {high:.0f}), the photo's room {box(stage(root)).height():.0f}")
    turned = QPointF(box(row).center())
    QCoreApplication.sendEvent(view, QWheelEvent(turned, turned, QPoint(0, 0), QPoint(0, -120), Qt.MouseButton.NoButton,
                                                 Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
    QTest.qWait(250)
    faults.expect(row.property("contentX") > 100, f"the wheel on the strip moved it {row.property('contentX'):.0f} px along")
    # The open photo's marks, set by the keys, show on its cell; scrolled out of sight, the
    # open photo is marked on the scrollbar and a button brings it back.
    cell = find(root, "strip:5")
    root.forceActiveFocus()
    QTest.keyClick(view, Qt.Key.Key_P)
    QTest.keyClick(view, Qt.Key.Key_3)
    settle()
    marks = shown.marksOf(session.path)
    faults.expect(marks.get("flag") == 1 and marks.get("stars") == 3 and cell.property("flag") == 1 and cell.property("stars") == 3,
                  f"picked and given three stars by the keys, the photo has {marks}, its cell flag {cell.property('flag')}, stars {cell.property('stars')}")
    faults.expect(cell.findChild(QObject, "strip-pick").property("on"), "a picked photo has no check in the strip")
    QTest.keyClick(view, Qt.Key.Key_X)
    settle()
    faults.expect(shown.marksOf(session.path).get("flag") == -1 and cell.findChild(QObject, "strip-thumb").property("opacity") < 0.5,
                  "a rejected photo is not dimmed in the strip")
    QTest.keyClick(view, Qt.Key.Key_X)
    settle()
    faults.expect(shown.marksOf(session.path).get("flag") == 0, "pressing X again does not take the mark away")
    # On a cell itself: nothing to click on an ungraded photo until the pointer is on it; then
    # five stars large enough to hit, a check and a cross; a click sets, the same again unsets.
    row.setProperty("contentX", row.property("originX"))  # (the third cell in sight)
    settle()
    other = find(root, "strip:2")
    stars, pick = other.findChild(QObject, "strip-stars"), other.findChild(QObject, "strip-pick")
    faults.expect(not stars.isVisible() and not pick.isVisible(), "an ungraded photo's cell shows stars before the pointer is on it")
    QTest.mouseMove(view, box(other).center().toPoint())
    settle()
    view.grabWindow()
    fourth = find(other, "strip-star4")
    faults.expect(stars.isVisible() and pick.isVisible() and fourth.width() >= 14 and fourth.height() >= 14,
                  f"with the pointer on a cell its stars are not offered, or too small to hit: {fourth.width():.0f} x {fourth.height():.0f} px")
    opened_ = session.index
    click(view, fourth)
    faults.expect(shown.marksOf(shown.paths()[2]).get("stars") == 4 and other.property("stars") == 4 and session.index == opened_,
                  f"a click on the fourth star of a cell gave {shown.marksOf(shown.paths()[2])}, the open photo is {session.index}")
    click(view, pick)
    faults.expect(shown.marksOf(shown.paths()[2]).get("flag") == 1 and session.index == opened_, "a click on a cell's check does not pick that photo")
    QTest.mouseMove(view, box(stage(root)).center().toPoint())
    settle()
    faults.expect(stars.isVisible() and pick.isVisible(), "what a photo has been given is gone from its cell when the pointer leaves")
    QTest.mouseMove(view, box(other).center().toPoint())
    click(view, fourth)
    click(view, pick)
    faults.expect(shown.marksOf(shown.paths()[2]) == {"flag": 0, "stars": 0, "label": 0, "edited": False}, f"the same clicks again do not take the marks away: {shown.marksOf(shown.paths()[2])}")
    QTest.mouseMove(view, box(stage(root)).center().toPoint())
    settle()
    call(strip, "bringBack")  # (the open photo in sight again: the row was put at its start)
    settle()
    view.grabWindow()
    bar, back = find(root, "strip-bar"), find(root, "strip-back")
    faults.expect(bar.isVisible() and not back.isVisible(), f"no scrollbar under the strip, or a way back while the open photo is in sight: bar {bar.isVisible()}, back {back.isVisible()}, at {row.property('contentX'):.0f} of {row.property('contentWidth'):.0f}, open {session.index}")
    session.goTo(1)  # (one that the end of the row leaves out of sight)
    wait(lambda: root.property("photoReady"))
    grip = bar.findChild(QObject, "thumb")
    drag(view, box(grip).center().toPoint(), QPoint(int(box(bar).width()), 0))
    view.grabWindow()
    faults.expect(row.property("along") > 300 and back.isVisible() and box(back).center().x() < box(row).center().x(),
                  f"dragged to the end ({row.property('contentX'):.0f} px), the strip does not offer the way back on the left")
    if back.isVisible():
        click(view, back)
    view.grabWindow()
    seen_ = box(find(root, "strip:1"))
    faults.expect(not back.isVisible() and box(row).left() <= seen_.center().x() <= box(row).right(), "the way back does not bring the open photo into sight")
    # Until a photo's first picture is there, its thumbnail stands in for it.
    session.goTo(7)
    stand = find(root, "stand-in")
    faults.expect(stand.isVisible() and stand.property("source").toString() == shown.thumbFor(session.path) != "",
                  "a photo just asked for shows nothing until it is decoded: its thumbnail does not stand in")
    wait(lambda: root.property("photoReady"))
    faults.expect(not stand.isVisible(), "the thumbnail stays in the photo's place once the photo is there")
    # Not on Fit either is the thumbnail larger or smaller than the photo that follows it.
    pane, photo = stage(root), find(root, "photo")
    call(pane, "zoomTo", pane.property("fitZoom") * 0.5, pane.width() / 2, pane.height() / 2)
    settle()
    session.goTo(8)
    wait(lambda: stand.property("paintedWidth") > 0, 2000)
    drawn = stand.property("paintedWidth")
    wait(lambda: root.property("photoReady"))
    settle()
    faults.expect(abs(drawn - photo.width()) <= 2, f"zoomed out, the thumbnail standing in is {drawn:.0f} px wide and the photo after it {photo.width():.0f}")
    call(pane, "fit")
    session.goTo(5)
    wait(lambda: root.property("photoReady"))
    view.grabWindow()
    # A thumbnail shows the photo as it is edited. (The program reads a small render back with
    # Qt's item grab, which a window that is not on screen is refused: here that render is put
    # in the window for one grab and the library handed the same picture.)
    def kept() -> QColor:
        keeper = find(root, "keeper")
        QMetaObject.invokeMethod(keeper, "take")
        keeper.setVisible(True)
        seen = view.grabWindow().copy(box(keeper).toRect())
        keeper.setVisible(False)
        cell = find(root, "strip:5")
        was = cell.property("thumb")
        shown.keep(session.path, seen)
        wait(lambda: cell.property("thumb") != was, 3000)  # another address: Qt reads it anew
        picture = cell.findChild(QObject, "strip-thumb")
        settle()
        wait(lambda: picture.property("progress") == 1.0, 3000)
        view.grabWindow()
        settle()
        return view.grabWindow().pixelColor(box(cell).center().toPoint())

    call(strip, "bringBack")
    settle()
    view.grabWindow()
    plain_ = view.grabWindow().pixelColor(box(find(root, "strip:5")).center().toPoint())
    session.setEdits({"exposure": -2.5, "crop": [0.0, 0.0, 0.4, 1.0]})
    session.commitEdit()
    settle()
    edited = kept()
    file = QImage(str(shown.cache / f"{shown.model.rows[5]['thumb']}.jpg"))
    faults.expect(edited.green() < plain_.green() - 40, f"the open photo's cell in the strip is {edited.name()} after it was darkened (it was {plain_.name()})")
    faults.expect(file.height() == 512 and abs(file.width() / file.height() - 0.4 * 1.5) < 0.02,
                  f"the thumbnail kept is {file.width()} x {file.height()}: not the photo as it is cropped")
    session.resetAll()
    settle()
    again = kept()
    faults.expect(abs(again.green() - plain_.green()) <= 6, f"reset, the photo's thumbnail is {again.name()}, the photo's own was {plain_.name()}")
    edge = box(find(root, "strip-edge")).center().toPoint()
    double_click(view, edge)
    view.grabWindow()
    faults.expect(strip.height() <= 8 and box(stage(root)).height() > area.height() + 60, f"a double click on the edge leaves the strip {strip.height():.0f} px tall")
    # (Put away, its cells keep a width: with none, one was made for every photo of the folder.)
    faults.expect(row.property("wide") >= 40 and not row.property("visible"),
                  f"put away, the strip's row is still shown, or its cells are {row.property('wide'):.0f} px wide")
    double_click(view, box(find(root, "strip-edge")).center().toPoint())
    view.grabWindow()
    faults.expect(abs(strip.height() - high - 40) <= 1, f"a second double click brings it back {strip.height():.0f} px tall, not {high + 40:.0f}")
    # Dragged down as far as it goes, it is put away as well, and dragged up it is back.
    drag(view, box(find(root, "strip-edge")).center().toPoint(), QPoint(0, 400))
    view.grabWindow()
    faults.expect(strip.height() <= 8, f"dragged all the way down, the strip is {strip.height():.0f} px tall")
    drag(view, box(find(root, "strip-edge")).center().toPoint(), QPoint(0, -90))
    view.grabWindow()
    faults.expect(abs(strip.height() - 90) <= 1 and row.property("visible"), f"dragged up again by 90 px, the strip is {strip.height():.0f} px tall")
    # A denoised copy just written is shown beside its original: the row stays there, and
    # does not start over at its left end.
    session.goTo(10)
    wait(lambda: root.property("photoReady"))
    call(strip, "bringBack")
    QTest.qWait(400)  # (where the row is, is kept a moment after it has moved)
    settle()
    copy = Path(session.path).with_name("IMG_4110-denoised.jpg")
    copy.write_bytes(Path(session.path).read_bytes())
    session.openBeside(str(copy))
    wait(lambda: root.property("photoReady"))
    settle()
    faults.expect(session.index == 11 and row.property("away") == 0,
                  f"a file opened beside the open photo is out of sight in the strip (the row is {row.property('contentX') - row.property('originX'):.0f} px along)")
    faults += audit_layout(view)
    # The open photo far out of sight to the left: the button that brings it back stands at
    # the left end wherever the row is, also with a gap between two cells in its middle.
    session.goTo(0)
    wait(lambda: root.property("photoReady"))
    root.setProperty("stripHigh", 260)
    settle()
    view.grabWindow()
    end, sides = row.property("contentWidth") + row.property("originX") - row.width(), set()
    for back in range(0, 700, 2):
        row.setProperty("contentX", end - back)
        sides.add(row.property("away"))
    faults.expect(sides == {-1}, f"the open photo out of sight to the left, the button to bring it back stands at {sides} as the row moves")
    shown.close()
    ui.close_view(view)
    return faults


def audit_strip_wheel() -> list[str]:
    """Every notch of the wheel moves the filmstrip by one notch's way, also in a long folder
    where denoised copies stand as wider pairs: there the view guesses anew where the row
    begins as it makes cells, and a glide put back on that guess at every frame jumped (the
    owner: "the scroll jumps somewhere"; up to twice a notch, measured)."""
    home = Path(SHELF.name) / "strip-wheel"
    folder = home / "photos" / "long"
    folder.mkdir(parents=True)
    for n in range(80):
        fixtures.picture_file(folder / f"IMG_{4100 + n}.jpg", fixtures.exif("SONY", "ILCE-7M4", f"2026:06:10 10:{n // 60:02}:{n % 60:02}", 1),
                              (600, 900) if n % 7 == 3 else (900, 600))
    kept = catalog.Catalog(home / "catalog.db")
    kept.add_folder(folder)
    kept.found(*library.scan(folder, kept.known(folder)))
    for photo in kept.photos():
        library.thumbnail(Path(photo["path"]), photo["key"], home / "thumbs")
    shown = library.Library(kept, home / "thumbs")
    shown.setFilter("raw", False)
    for n in (2, 9, 20, 30, 31, 45, 60, 70):  # (as `pair` does)
        shown.made_here[shown.paths()[n + 1]] = {"of": shown.model.rows[n]["key"]}
    session = ui.Session([])
    view = ui.make_view(session, shown)
    session.surface_known(False)
    view.resize(1000, 700)
    root = view.rootObject()
    root.setSize(view.size())
    session.openAt(shown.paths(), 5)
    call(root, "show", "edit")
    wait(lambda: root.property("photoReady"))
    settle()
    view.grabWindow()
    faults = Faults()
    row, step = find(root, "strip-row"), find(root, "strip-bar").property("step")
    left, right = box(row).left(), box(row).right()

    def seen() -> dict:
        return {i.property("index"): box(i).x() for i in items(row) if i.objectName().startswith("strip:") and left - 300 < box(i).x() < right}

    def turn(notches: int) -> None:
        spot = QPointF(box(row).center())
        QCoreApplication.sendEvent(view, QWheelEvent(spot, spot, QPoint(0, 0), QPoint(0, notches), Qt.MouseButton.NoButton,
                                                     Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
        clock = QElapsedTimer()
        clock.start()
        while clock.elapsed() < 300:  # (frame by frame: the view makes its cells as it glides)
            QTest.qWait(10)
            view.grabWindow()

    moves = []
    for notches, times in ((-120, 14), (120, 6), (-120, 12), (120, 10)):
        for _ in range(times):
            was, start = seen(), row.property("contentX") - row.property("originX")
            turn(notches)
            now, both = seen(), set(was) & set(seen())
            moved = sorted(now[k] - was[k] for k in both)[len(both) // 2] if both else 0
            ends = start < step + 2 or start > row.property("contentWidth") - row.width() - step - 2
            # (A few px is a pair's cell settling once its thumbnail is in, not the wheel.)
            if not ends and abs(abs(moved) - step) > 4:
                moves.append(round(moved))
    faults.expect(not moves, f"notches of the wheel that did not move the strip {step:.0f} px: {moves}")
    shown.close()
    ui.close_view(view)
    return faults


def audit_menu() -> list[str]:
    """A right click on a photo, with a real mouse: in the library it is for what is selected
    (or selects the photo), in the filmstrip for that photo; Reset edits asks first, takes the
    edit out of the catalog and leaves a step that undo takes back; Export offers that photo."""
    shown = shelf("menu-audit")  # (its ninth photo has an edit: exposure 0.5)
    session = ui.Session([], catalog=shown.catalog)
    view = ui.make_view(session, shown)
    session.surface_known(False)
    root = view.rootObject()
    root.setSize(view.size())
    settle()
    view.grabWindow()
    faults, kept, menu = Faults(), shown.catalog, find(root, "modal:menu")

    def right(item) -> None:
        QTest.mouseMove(view, box(item).center().toPoint(), 500)
        QTest.mouseClick(view, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, box(item).center().toPoint())
        settle()
        view.grabWindow()

    def edit_of(row: int) -> dict:
        photo = shown.model.rows[row]
        return kept.load(photo["key"], Path(photo["path"]))

    row = next(n for n, photo in enumerate(shown.model.rows) if photo["edited"])
    faults.expect(not menu.isVisible(), "the photo menu is open before anything was clicked")
    whole = menu.parentItem()  # asked for past the window's corners, the card stays inside it
    for corner in (99999, -99999):
        whole.setProperty("atX", corner), whole.setProperty("atY", corner), whole.setProperty("visible", True)
        settle()
        view.grabWindow()
        faults.expect(QRectF(0, 0, root.width(), root.height()).contains(box(menu)), f"the menu asked for at {corner} is cut by the window: {box(menu)}")
    whole.setProperty("visible", False)
    right(find(root, f"photo:{row}"))
    lib = find(root, "box:library")
    faults.expect(menu.isVisible() and plain(lib.property("chosen")) == [row], f"a right click on a photo of the library opens no menu, or does not select it: {plain(lib.property('chosen'))}")
    faults += audit_layout(view)
    click(view, find(root, "menu-reset"))
    faults.expect(not menu.isVisible() and find(root, "modal:confirm").isVisible() and edit_of(row), "Reset edits does not ask first")
    click(view, find(root, "confirm-no"))
    faults.expect(edit_of(row) == {"exposure": 0.5}, f"Cancel reset the photo all the same: {edit_of(row)}")
    right(find(root, f"photo:{row}"))
    click(view, find(root, "menu-reset"))
    click(view, find(root, "confirm-yes"))
    wait(lambda: shown.making == 0, 4000)
    steps = kept.steps(shown.model.rows[row]["key"])
    faults.expect(edit_of(row) == {} and not shown.model.rows[row]["edited"], f"reset, the photo still has {edit_of(row)}")
    faults.expect(steps == ([{"exposure": 0.5}, {}], 1), f"the reset is no step that undo could take back: {steps}")
    # In the filmstrip: reset of a photo that is not the open one, and of the open one; Export.
    for n in (row, 1):
        kept.save(shown.model.rows[n]["key"], Path(shown.model.rows[n]["path"]), {"contrast": 0.3})
    session.openAt(shown.paths(), 1)
    call(root, "show", "edit")
    wait(lambda: root.property("photoReady"))
    view.grabWindow()
    along = find(root, "strip-row")  # (to its end: the cell of a photo that far along is there)
    along.setProperty("contentX", along.property("originX") + along.property("contentWidth") - along.width())
    settle()
    view.grabWindow()
    right(find(root, f"strip:{row}"))
    faults.expect(menu.isVisible(), "a right click on a photo of the filmstrip opens no menu")
    click(view, find(root, "menu-reset"))
    click(view, find(root, "confirm-yes"))
    faults.expect(edit_of(row) == {} and session.index == 1 and session.history.edit == {"contrast": 0.3},
                  f"reset from the filmstrip, the photo has {edit_of(row)}; the open one {session.history.edit}")
    call(find(root, "box:strip"), "bringBack")  # (the open photo's cell, and those beside it, in sight)
    settle()
    view.grabWindow()
    right(find(root, "strip:1"))
    click(view, find(root, "menu-reset"))
    click(view, find(root, "confirm-yes"))
    faults.expect(session.history.edit == {} and edit_of(1) == {}, f"the open photo reset from its cell still has {session.history.edit}, in the catalog {edit_of(1)}")
    session.undo()
    faults.expect(session.history.edit == {"contrast": 0.3}, "undo does not bring the open photo's edit back after a reset")
    # Ctrl and a click selects in the filmstrip (the open photo with it), Shift up to there; a
    # right click on one of them exports them all at once, as the Export page is set up.
    strip = find(root, "box:strip")
    held = lambda cell, key: (QTest.mouseClick(view, Qt.MouseButton.LeftButton, key, box(find(root, cell)).center().toPoint()), settle())  # noqa: E731
    held("strip:2", Qt.KeyboardModifier.ControlModifier)
    faults.expect(plain(strip.property("chosen")) == [1, 2] and session.index == 1, f"Ctrl+click in the filmstrip selects {plain(strip.property('chosen'))}")
    held("strip:4", Qt.KeyboardModifier.ShiftModifier)
    faults.expect(plain(strip.property("chosen")) == [2, 3, 4], f"Shift+click in the filmstrip selects {plain(strip.property('chosen'))}")
    call(root, "show", "export")
    settle()
    faults.expect(len(plain(find(root, "box:export").property("photos"))) == 3 and find(root, "export-go").property("label") == "Export 3 photos"
                  and find(root, "export-sheet").isVisible() and not find(root, "export-preview").isVisible(),
                  f"the Export page with three photos selected: {find(root, 'export-go').property('label')!r}")
    call(root, "show", "edit")
    settle()
    out = Path(tempfile.mkdtemp(prefix="photone-menu-", dir=SHELF.name))
    find(root, "box:export").setProperty("folder", str(out))
    right(find(root, "strip:3"))
    faults.expect(find(root, "menu-export").property("label") == "Export all (3)", f"the menu of three selected photos offers {find(root, 'menu-export').property('label')!r}")
    click(view, find(root, "menu-export"))
    faults.expect(session.exporter.running and session.exporter.total == 3 and root.property("place") == "edit",
                  f"Export all from the menu: running {session.exporter.running}, {session.exporter.total} photos, at {root.property('place')}")
    faults.expect("Exporting" in find(root, "foot-export").property("text"), "the footer does not say that an export runs")
    session.exporter.stop()
    carry_out(view, session)
    click(view, find(root, "strip:0"))
    faults.expect(plain(strip.property("chosen")) == [] and session.index == 0, "a plain click in the filmstrip does not open the photo and end the selection")
    shown.close()
    ui.close_view(view)
    return faults


def audit_look() -> list[str]:
    """A look copied from one photo and laid on others, with a real mouse and keys: from the
    menu in the library onto those selected, each in the catalog as a step of its own; by
    Ctrl+Shift+V onto the open photo as one step of undo, its frame and masks left alone."""
    shown = shelf("look-audit")  # (its ninth photo has an edit: exposure 0.5)
    session = ui.Session([], catalog=shown.catalog)
    view = ui.make_view(session, shown)
    session.surface_known(False)
    root = view.rootObject()
    root.setSize(view.size())
    settle()
    view.grabWindow()
    faults, kept, menu = Faults(), shown.catalog, find(root, "modal:menu")

    def right(item) -> None:
        QTest.mouseMove(view, box(item).center().toPoint(), 500)
        QTest.mouseClick(view, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, box(item).center().toPoint())
        settle()
        view.grabWindow()

    def edit_of(row: int) -> dict:
        photo = shown.model.rows[row]
        return kept.load(photo["key"], Path(photo["path"]))

    def keys(letter) -> None:
        QTest.keyClick(view, letter, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
        settle()

    source = next(n for n, photo in enumerate(shown.model.rows) if photo["edited"])
    others = [n for n in range(len(shown.model.rows)) if n != source][:2]
    kept.save(shown.model.rows[others[1]]["key"], Path(shown.model.rows[others[1]]["path"]), {"contrast": 30.0, "turns": 1.0})
    right(find(root, f"photo:{others[0]}"))
    faults.expect(find(root, "menu-copy-look").isVisible() and not find(root, "menu-paste-look").isVisible(), "Paste look is offered before a look was copied")
    click(view, find(root, "menu-copy-look"))
    faults.expect(shown.hasLook and shown.look == {}, f"the look of a photo without an edit is {shown.look}")
    right(find(root, f"photo:{source}"))
    click(view, find(root, "menu-copy-look"))
    faults.expect(shown.look == {"exposure": 0.5}, f"Copy look took {shown.look}")
    tap(view, box(find(root, f"photo:{others[0]}")).center().toPoint())
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, box(find(root, f"photo:{others[1]}")).center().toPoint())
    settle()
    right(find(root, f"photo:{others[1]}"))
    faults.expect(find(root, "menu-paste-look").property("label") == "Paste look on all (2)" and not find(root, "menu-copy-look").isVisible(),
                  f"the menu of two selected photos offers {find(root, 'menu-paste-look').property('label')!r}")
    click(view, find(root, "menu-paste-look"))
    wait(lambda: shown.making == 0, 4000)
    faults.expect(edit_of(others[0]) == {"exposure": 0.5} and edit_of(others[1]) == {"exposure": 0.5, "turns": 1.0},
                  f"pasted on two photos: {edit_of(others[0])} and {edit_of(others[1])} (the second had Contrast and a turn)")
    steps = kept.steps(shown.model.rows[others[1]]["key"])
    faults.expect(steps == ([{"contrast": 30.0, "turns": 1.0}, {"exposure": 0.5, "turns": 1.0}], 1), f"a pasted look is no step that undo could take back: {steps}")
    faults.expect(shown.model.rows[others[0]]["edited"], "a photo with a pasted look is not marked as edited")
    # In Edit: onto the open photo by the keys, and from it.
    session.openAt(shown.paths(), others[1])
    call(root, "show", "edit")
    wait(lambda: root.property("photoReady"))
    view.grabWindow()
    session.setEdits({"saturation": 140.0, "masks": [OVAL], "crop": [0.1, 0.1, 0.9, 0.9]})
    session.commitEdit()
    was, at = dict(session.history.edit), session.history.at
    keys(Qt.Key.Key_V)
    faults.expect(session.history.edit == {"exposure": 0.5, "turns": 1.0, "masks": was["masks"], "crop": was["crop"]} and session.history.at == at + 1,
                  f"Ctrl+Shift+V on the open photo gave {session.history.edit} in {session.history.at - at} steps")
    session.undo()
    faults.expect(session.history.edit == was, "undo does not take a pasted look back")
    keys(Qt.Key.Key_C)
    faults.expect(shown.look == {"exposure": 0.5, "saturation": 140.0}, f"Ctrl+Shift+C on the open photo took {shown.look}")
    faults.extend(audit_layout(view))
    shown.close()
    ui.close_view(view)
    return faults


def audit_presets() -> list[str]:
    """Presets under a real mouse and keyboard: the look of the open photo kept under a name
    with the parts chosen, shown as a picture of the photo with it, laid on a photo by a
    click as one step of undo and only in those parts, renamed, and deleted after a question."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session)
    faults, kept = Faults(), find_presets(root)
    session.setEdits({"contrast": 30.0, "temperature": 6400.0, "saturation": 140.0, "crop": [0.1, 0.1, 0.9, 0.9]})
    session.commitEdit()
    click(view, find(root, "mode:presets"))
    view.grabWindow()
    faults.expect(find(root, "panel:presets").isVisible() and not find(root, "panel:tone").isVisible(), "the Presets tab does not show")
    click(view, find(root, "preset-keep"))
    faults.expect(kept.all == [], "a preset was saved without a name")
    faults.expect(not find(root, "preset-part:balance").property("on") and find(root, "preset-part:light").property("on"), "the parts as they start: all but the white balance")
    click(view, find(root, "preset-part:color"))
    click(view, find(root, "preset-name"))
    for letter in "Hard":
        QTest.keyClick(view, letter)
    QTest.keyClick(view, Qt.Key.Key_Return)
    settle()
    view.grabWindow()
    faults.expect(kept.all == [{"name": "Hard", "parts": ["light", "curve", "mono", "detail", "lens", "grain"], "look": {"contrast": 30.0}}], f"saved as Hard without Color: {kept.all}")
    # The same name again is refused, and said to be taken.
    click(view, find(root, "preset-name"))
    for letter in "Hard":
        QTest.keyClick(view, letter)
    settle()
    view.grabWindow()
    faults.expect(find(root, "preset-taken").isVisible(), "a name that is taken is not said to be")
    QTest.keyClick(view, Qt.Key.Key_Return)
    click(view, find(root, "preset-keep"))
    faults.expect(len(kept.all) == 1, f"a second preset of the same name was saved: {[p['name'] for p in kept.all]}")
    QTest.keyClick(view, "A", Qt.KeyboardModifier.ControlModifier)
    QTest.keyClick(view, Qt.Key.Key_Backspace)
    QTest.keyClick(view, Qt.Key.Key_Escape)
    settle()
    faults.expect(session.index == 0 and session.history.edit.get("contrast") == 30.0, "typing a name changed the photo")
    tile = find(root, "preset:0")
    faults.expect(tile is not None and tile.isVisible(), "the preset saved is not shown")
    if tile is None:
        ui.close_view(view)
        return faults
    # Its picture is the photo with the preset: with the photo's edit taken away it is the
    # only thing left with contrast, so its darks are darker than the photo's.
    session.resetAll()
    settle()
    wait(lambda: False, 200)
    shot, area, small = view.grabWindow(), box(root.findChild(QObject, "photo")), box(tile)
    photo_dark = shot.pixelColor(int(area.left() + 0.5 * area.width()), int(area.top() + 0.7 * area.height())).lightness()
    tile_dark = shot.pixelColor(int(small.left() + 0.5 * small.width()), int(small.top() + 0.7 * (small.height() - 30))).lightness()
    faults.expect(tile_dark < photo_dark - 6, f"the preset's picture is not the photo with its contrast: {tile_dark} against {photo_dark} in the darks")
    # Its picture has the photo's shape, also when the photo is upright (it was stretched).
    session.setEdit("turns", 1.0)
    settle()
    view.grabWindow()
    small_ = next(i for i in items(find(root, "preset:0")) if "Pipeline" in i.metaObject().className())
    faults.expect(abs(small_.width() / small_.height() - 320 / 480) < 0.02, f"an upright photo's preset picture is {small_.width():.0f} x {small_.height():.0f}, not 2 : 3")
    session.undo()
    settle()
    view.grabWindow()
    tile = find(root, "preset:0")
    session.setEdits({"saturation": 60.0, "highlights": -20.0})
    session.commitEdit()
    at = session.history.at
    click(view, tile)
    faults.expect(session.history.edit == {"contrast": 30.0, "saturation": 60.0} and session.history.at == at + 1,
                  f"a click on the preset gave {session.history.edit} in {session.history.at - at} steps (Color was not part of it)")
    QTest.mouseMove(view, box(tile).center().toPoint(), 300)
    settle()
    view.grabWindow()
    click(view, find(root, "preset-rename:0"))
    QTest.keyClick(view, "A", Qt.KeyboardModifier.ControlModifier)
    for letter in "Soft":
        QTest.keyClick(view, letter)
    QTest.keyClick(view, Qt.Key.Key_Return)
    settle()
    faults.expect(kept.all[0]["name"] == "Soft", f"renamed to Soft it is {kept.all[0]['name']!r}")
    faults.extend(audit_layout(view))
    QTest.mouseMove(view, box(find(root, "preset:0")).center().toPoint(), 300)
    settle()
    view.grabWindow()
    click(view, find(root, "preset-delete:0"))
    view.grabWindow()
    faults.expect(find(root, "modal:confirm").isVisible() and len(kept.all) == 1, "deleting a preset does not ask first")
    click(view, find(root, "confirm-yes"))
    faults.expect(kept.all == [], "nor delete it when told to")
    ui.close_view(view)
    return faults


def audit_lut() -> list[str]:
    """LUTs installed in Photone and laid on a photo, with a real mouse, measured in the
    picture: installing .cube and .clf files (one that is none costs a sentence, one that is there
    already is not there twice), an installed one dragged into the photo's list is laid on
    at the place it is let go (a mere press on it does nothing); one that changes nothing changes nothing, one of opposites and
    one that swaps red and blue do exactly that, one after the other; Strength, the eye, the
    bin, a drag to another place; one that is uninstalled changes nothing and says so; no
    fifth; and the exported file is the picture on screen."""
    from photone import lut
    ui.Session.luts_home = Path(tempfile.mkdtemp(prefix="photone-luts-", dir=SHELF.name))
    warnings.clear()
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "lut")
    faults = Faults()
    folder = Path(tempfile.mkdtemp(prefix="photone-lut-", dir=SHELF.name))
    same = lut.identity(9)
    for name, table in (("Same", lut.identity(17)), ("Opposite", 1 - same)):
        (folder / f"{name}.cube").write_text(lut.cube(table, name), "utf-8")
    # (The one that swaps red and blue as a .clf file: XML, blue running fastest.)
    rows = "\n".join(" ".join(map(str, row)) for row in same[..., ::-1].transpose(2, 1, 0, 3).reshape(-1, 3).tolist())
    (folder / "Swapped.clf").write_text('<?xml version="1.0"?>\n<ProcessList id="s" compCLFversion="3.0"><LUT3D inBitDepth="32f" '
                                        f'outBitDepth="32f"><Array dim="9 9 9 3">\n{rows}\n</Array></LUT3D></ProcessList>\n', "utf-8")
    (folder / "Broken.cube").write_text("LUT_3D_SIZE 4\n0 0 0\n", "utf-8")
    area = box(root.findChild(QObject, "photo"))
    spots = [QPoint(int(area.left() + u * area.width()), int(area.top() + v * area.height()))
             for u in (0.1, 0.3, 0.5, 0.7, 0.9) for v in (0.15, 0.4, 0.65, 0.92)]

    def seen() -> np.ndarray:
        settle()
        wait(lambda: False, 300)  # (the LUTs' picture is fetched anew when their list is another)
        view.grabWindow()
        wait(lambda: False, 100)
        shot = view.grabWindow()
        return np.array([shot.pixelColor(spot).getRgb()[:3] for spot in spots], int)

    def names() -> list[str]:
        return [one["name"] + (" off" if one.get("off") else "") for one in session.history.edit.get("luts", [])]

    def shelf_row(name: str):
        view.grabWindow()
        return find(root, f"lut-shelf:{[one['name'] for one in session.lutsInstalled].index(name)}")

    def lay(name: str, place: int | None = None) -> None:
        """Drag that installed LUT into the photo's list: to that place, or to its end."""
        row, into = box(shelf_row(name)), box(find(root, "lut-list"))
        count = len(session.history.edit.get("luts", []))
        y = into.top() + 4 + (count if place is None else place) * 28 + (2 if place is None else 0)
        to = QPoint(int(into.center().x()), int(min(y, into.bottom() - 3)))
        drag(view, row.center().toPoint(), to - row.center().toPoint())
        view.grabWindow()

    def over(item) -> None:  # the pointer on a row: its bin is there
        QTest.mouseMove(view, box(item).center().toPoint(), 300)
        settle()
        view.grabWindow()

    plain_ = seen()
    inside = ((plain_ > 6) & (plain_ < 249)).all(axis=1)  # (not cut off by the screen's own range)
    faults.expect(inside.sum() >= 8 and session.lutsInstalled == [], f"(only {inside.sum()} spots can be compared, or LUTs are installed already)")
    session.installLuts([str(folder / name) for name in ("Same.cube", "Opposite.cube", "Broken.cube", "Swapped.clf")])
    faults.expect([one["name"] for one in session.lutsInstalled] == ["Opposite.cube", "Same.cube", "Swapped.clf"] and "Broken.cube cannot be used as a LUT" in session.error
                  and names() == [], f"installed: {[one['name'] for one in session.lutsInstalled]}, said {session.error!r}, on the photo {names()}")
    # The same table again, in a file of another name and written otherwise: not twice.
    (folder / "Copy of it.cube").write_text("# a copy\r\n" + lut.cube(1 - same, "Renamed").replace("\n", "\r\n"), "utf-8")
    session.installLuts([str(folder / "Copy of it.cube"), str(folder / "Same.cube")])
    faults.expect(len(session.lutsInstalled) == 3 and "Copy of it.cube is installed already, as “Opposite.cube”" in session.error and "Same.cube is installed already" in session.error
                  and len(list(ui.Session.luts_home.glob("*.cube"))) == 3, f"a LUT installed twice: {[one['name'] for one in session.lutsInstalled]}, said {session.error!r}")
    session.say("")
    steps = session.steps
    click(view, shelf_row("Same.cube"))
    faults.expect(names() == [] and session.steps == steps, f"a mere press on an installed LUT laid it on: {names()}")
    row = box(shelf_row("Same.cube"))
    drag(view, row.center().toPoint(), QPoint(0, 60))
    faults.expect(names() == [] and not find(root, "lut-carried").isVisible(), f"an installed LUT let go beside the photo's list: {names()}")
    lay("Same.cube")
    faults.expect(names() == ["Same.cube"] and session.steps == steps + 1 and find(root, "lut-item:0") is not None and not find(root, "lut-carried").isVisible(),
                  f"an installed LUT dragged into the photo's list: {names()} in {session.steps - steps} steps")
    off = np.abs(seen() - plain_)[inside].max()
    faults.expect(off <= 1, f"a LUT that changes nothing changed the picture by up to {off} levels")
    lay("Opposite.cube")
    turned = seen()
    off = np.abs(turned - (255 - plain_))[inside].max()
    faults.expect(names() == ["Same.cube", "Opposite.cube"] and off <= 2, f"a LUT of opposites after it: up to {off} levels from the opposite color")
    slider = find(root, "slider:lut")
    track = slider.findChild(QObject, "track")
    drag(view, box(slider.findChild(QObject, "thumb")).center().toPoint(), QPoint(-round(0.5 * (track.width() - 12)), 0))
    half = seen()
    low, high = np.minimum(plain_, turned), np.maximum(plain_, turned)
    much = session.history.edit["luts"][1]["strength"]
    faults.expect(abs(much - 50) <= 1 and session.history.edit["luts"][0]["strength"] == 100 and ((half >= low - 1) & (half <= high + 1))[inside].all()
                  and np.abs(half - turned)[inside].max() > 20, f"the chosen LUT at half strength ({much}): the picture is not between the two")
    session.undo()
    click(view, find(root, "lut-eye:1"))
    faults.expect(names() == ["Same.cube", "Opposite.cube off"] and np.abs(seen() - plain_)[inside].max() <= 1, f"the eye does not switch a LUT off: {names()}")
    click(view, find(root, "lut-eye:1"))
    # While it is held over the list, the rows below its place stand a row lower: a gap for it.
    row, into, shown_ = box(shelf_row("Swapped.clf")), box(find(root, "lut-list")), []
    over_second = QPoint(int(into.center().x()), int(into.top() + 4 + 28)) - row.center().toPoint()
    moves(view, row.center().toPoint(), [over_second, over_second],
          lambda: shown_.append((find(root, "lut-item:0").property("place"), find(root, "lut-item:1").property("place"), find(root, "lut-gap").isVisible())))
    faults.expect(shown_ == [(0, 2, True)], f"held over the second place, the rows stand at {shown_} (the second should make room)")
    faults.expect(names() == ["Same.cube", "Swapped.clf", "Opposite.cube"] and not find(root, "lut-gap").isVisible() and find(root, "lut-item:2").property("place") == 2,
                  f"let go between the first and the second, it is at {names()}")
    session.moveLut(1, 2)
    off = np.abs(seen() - (255 - plain_)[:, ::-1])[inside].max()
    faults.expect(names() == ["Same.cube", "Opposite.cube", "Swapped.clf"] and off <= 2, f"three LUTs one after the other: up to {off} levels from the opposite with red and blue swapped")
    # Dragged to the top, it is the first; one step of undo.
    # (And while it is dragged the two it passes stand a row lower, as they will.)
    steps, row, shown_ = session.steps, box(find(root, "lut-item:2")), []
    # (Dragged far past the top of the list, it stays in it: at the first row's place.)
    up, into = QPoint(0, -int(6 * row.height())), box(find(root, "lut-list"))
    moves(view, QPoint(int(row.left() + 0.5 * row.width()), int(row.center().y())), [up, up],
          lambda: shown_.append((find(root, "lut-item:0").property("place"), find(root, "lut-item:1").property("place"),
                                 round(box(find(root, "lut-item:2")).top() - into.top() - 4))))
    view.grabWindow()
    faults.expect(shown_ == [(1, 2, 0)] and [find(root, f"lut-item:{n}").property("place") for n in range(3)] == [0, 1, 2]
                  and [round(box(find(root, f"lut-item:{n}")).top() - box(find(root, "lut-list")).top() - 4) for n in range(3)] == [0, 28, 56],
                  f"a row dragged past the top: the others and it stood at {shown_} meanwhile (it should stop at the first place), and after it the rows are at {[find(root, f'lut-item:{n}').property('place') for n in range(3)]}")
    faults.expect(names() == ["Swapped.clf", "Same.cube", "Opposite.cube"] and session.steps == steps + 1, f"a LUT dragged from third to first place: {names()} in {session.steps - steps} steps")
    row, shown_ = box(find(root, "lut-item:0")), []
    down = QPoint(0, int(7 * row.height()))
    moves(view, row.center().toPoint(), [down, down], lambda: shown_.append(round(box(find(root, "lut-item:0")).top() - into.top() - 4)))
    faults.expect(shown_ == [56] and names() == ["Same.cube", "Opposite.cube", "Swapped.clf"], f"a row dragged far past the bottom stood at {shown_} px (the last place is 56), then: {names()}")
    session.undo()
    settle()
    view.grabWindow()
    # The file is the picture on screen.
    screen = shown_at_full_size(view, root)
    out = Path(tempfile.mkdtemp(prefix="photone-lut-out-", dir=SHELF.name))
    session.exporter.side = 256
    session.exporter.start([str(p) for p in session.photos], {"kind": "png8", "quality": 92, "longest": 0, "folder": str(out)})
    carry_out(view, session)
    file = QImage(str(out / "first.png"))
    if file.isNull():
        faults.append(f"a photo with LUTs was not exported: {session.exporter.failures}")
    else:
        written = export.bytes_of(file).astype(int)
        off = np.abs(written - screen).max() if written.shape == screen.shape else 999
        faults.expect(off <= 1, f"the exported picture is not the one on screen with its LUTs: up to {off} levels off")
    # Uninstalled (after a question), it changes nothing on the photo and its row says so.
    over(shelf_row("Opposite.cube"))
    click(view, find(root, f"lut-uninstall:{[one['name'] for one in session.lutsInstalled].index('Opposite.cube')}"))
    view.grabWindow()
    faults.expect(find(root, "modal:confirm").isVisible() and len(session.lutsInstalled) == 3, "uninstalling a LUT does not ask first")
    click(view, find(root, "confirm-yes"))
    off = np.abs(seen() - plain_[:, ::-1])[inside].max()
    faults.expect([one["name"] for one in session.lutsInstalled] == ["Same.cube", "Swapped.clf"] and session.lutsOn[2]["lost"] and off <= 2 and names() == ["Swapped.clf", "Same.cube", "Opposite.cube"],
                  f"a LUT uninstalled: still laid on ({off} levels from the photo with red and blue swapped), or not marked as gone")
    # Installed again from a file of another name, it is found again by the photo, under
    # its new name: by its colors, and also where the photo names it as the first day's
    # Photone did (by its file's text), as the owner's photos do.
    (folder / "Negative.cube").write_text((folder / "Opposite.cube").read_text("utf-8"), "utf-8")
    session.installLuts([str(folder / "Negative.cube")])
    off = np.abs(seen() - (255 - plain_)[:, ::-1])[inside].max()
    faults.expect(not session.lutsOn[2]["lost"] and session.lutsOn[2]["name"] == "Negative.cube" and off <= 2,
                  f"installed again as Negative: the photo's entry is {session.lutsOn[2]}, the picture {off} levels from what it was with it")
    old = lut.written((folder / "Opposite.cube").read_text("utf-8"))
    session.setEdit("luts", [{"id": old, "name": "Opposite of old", "size": 9, "strength": 100.0}])
    off = np.abs(seen() - (255 - plain_))[inside].max()
    faults.expect(old != session.lutsInstalled[0]["id"] and not session.lutsOn[0]["lost"] and session.lutsOn[0]["name"] == "Negative.cube" and off <= 2,
                  f"a LUT named as the first day named it: {session.lutsOn[0]}, the picture {off} levels from the opposite")
    session.undo()
    settle()
    # The bin takes one off the photo; the fifth is refused.
    over(find(root, "lut-item:2"))
    click(view, find(root, "lut-delete:2"))
    faults.expect(names() == ["Swapped.clf", "Same.cube"], f"the bin on the third row left {names()}")
    for _ in range(3):
        lay("Same.cube", 0)
    faults.expect(len(names()) == 5 and names()[0] == "Same.cube", f"dragged to the top three times, a photo with two has {names()}")
    # More than the four rows in sight: the wheel scrolls the list, which says there is more.
    scroll, into = find(root, "lut-scroll"), box(find(root, "lut-list"))
    faults.expect(scroll.property("contentHeight") == 5 * 28 and find(root, "luts-more").isVisible(), "five LUTs: the list does not say that there is more below")
    QCoreApplication.sendEvent(view, QWheelEvent(QPointF(into.center()), QPointF(into.center()), QPoint(), QPoint(0, -120), Qt.MouseButton.NoButton,
                                                 Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
    wait(lambda: False, 300)
    view.grabWindow()
    faults.expect(scroll.property("contentY") > 10 and into.contains(box(find(root, "lut-item:4"))), f"the wheel over the list scrolled it by {scroll.property('contentY'):.0f} px")
    # A scroll of the column that passes over the list is not caught by it; a moment later the
    # wheel is the list's again. And a list with nothing to scroll leaves it to the column.
    column = find(root, "tools-scroll")

    def turn(at: QPointF) -> None:
        QCoreApplication.sendEvent(view, QWheelEvent(at, at, QPoint(), QPoint(0, -120), Qt.MouseButton.NoButton,
                                                     Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
        wait(lambda: False, 150)

    scroll.setProperty("contentY", 0)
    column.setProperty("contentY", 0)
    wait(lambda: False, 500)
    view.grabWindow()
    faults.expect(column.property("contentHeight") > column.height() + 40, "(the column has nothing to scroll: the window is too high for this)")
    turn(QPointF(box(find(root, "panel:tone")).center()))  # (on the column itself)
    was = column.property("contentY")
    turn(QPointF(box(find(root, "lut-list")).center()))
    faults.expect(was > 10 and column.property("contentY") > was + 10 and scroll.property("contentY") == 0,
                  f"a scroll of the column passing over the LUT list: the column went {was:.0f} to {column.property('contentY'):.0f}, the list to {scroll.property('contentY'):.0f}")
    wait(lambda: False, 600)
    was = column.property("contentY")
    turn(QPointF(box(find(root, "lut-list")).center()))
    faults.expect(scroll.property("contentY") > 10 and column.property("contentY") == was, "a moment later the wheel over the list does not scroll the list alone")
    column.setProperty("contentY", 0)
    wait(lambda: False, 600)
    view.grabWindow()
    turn(QPointF(box(find(root, "lut-installed")).center()))  # (three installed, room for five: nothing to scroll)
    faults.expect(column.property("contentY") > 10, "over a list with nothing to scroll the wheel does not scroll the column")
    column.setProperty("contentY", 0)
    bring(root, "panel:lut")
    wait(lambda: False, 500)
    into = box(find(root, "lut-list"))
    # Held at the foot of the box, a dragged row has the list scroll on, and can be let go at the end.
    scroll.setProperty("contentY", 0)
    settle()
    view.grabWindow()
    top = box(find(root, "lut-item:0"))
    foot = QPoint(0, int(into.bottom() - 8 - top.center().y()))
    moves(view, top.center().toPoint(), [foot, foot], lambda: wait(lambda: scroll.property("contentY") >= 27, 2000))
    faults.expect(names() == ["Same.cube", "Same.cube", "Swapped.clf", "Same.cube", "Same.cube"], f"the first row dragged to the box's foot and held there: {names()} (it should be the last of five)")
    session.undo()
    # Sixteen at most.
    while session.addLut(session.lutsInstalled[0]["id"]) >= 0:
        pass
    faults.expect(len(names()) == edit.LUTS == 16 and "16 LUTs" in session.error, f"a seventeenth LUT: {len(names())} on the photo, said {session.error!r}")
    lay("Same.cube", 0)
    faults.expect(len(names()) == 16, "an installed LUT can be dragged onto a photo that has sixteen")
    # A .cube dropped on the photo is installed and laid on.
    session.setEdit("luts", [])
    (folder / "Dropped.cube").write_text(lut.cube(same[..., [1, 2, 0]], "Dropped"), "utf-8")
    session.open([str(folder / "Dropped.cube")])
    faults.expect(names() == ["Dropped.cube"] and "Dropped.cube" in [one["name"] for one in session.lutsInstalled] and session.index == 0,
                  f"a .cube dropped on the photo: on it {names()}, installed {[one['name'] for one in session.lutsInstalled]}")
    faults.extend(audit_layout(view))
    faults.extend(f"the interface warned: {said}" for said in warnings if "lut" in said.lower())
    ui.close_view(view)
    ui.Session.luts_home = Path(SHELF.name) / "luts"
    return faults


def audit_table() -> list[str]:
    """A look written out as a .cube file: of an untouched photo it is the table that changes
    nothing; and brought in again on the photo without its edit, it shows the picture the
    edit showed (a finished picture: every setting; a raw file: what comes after its develop)."""
    from photone import lut
    faults = Faults()
    folder = Path(tempfile.mkdtemp(prefix="photone-table-", dir=SHELF.name))

    def written(view, session, name: str) -> np.ndarray | None:
        session.exporter.side = 1024
        session.exporter.startTable(str(folder / name))
        carry_out(view, session)
        wait(lambda: not session.exporter.running, 8000)
        file = folder / f"{name}.cube"
        faults.expect(file.exists() and not session.exporter.failures, f"{name}.cube was not written: {session.exporter.failures}")
        return lut.parse(file.read_text("utf-8")) if file.exists() else None

    for kind, load, look in (("a finished picture", picture, {"exposure": 0.4, "contrast": 20.0, "saturation": 70.0, "curve": LIFT}),
                             ("a raw file", chart, {"curve": LIFT, "saturation": 60.0, "eq": [{"hue": 140, "gain": 160, "width": 60}],
                                                    "pins": [{"from": [0.1, 0.0], "to": [0.1, 0.06], "range": 0.1}]})):
        session = ui.Session([Path("first.arw")], load=load)
        view, root = opened(session, "")
        area = box(root.findChild(QObject, "photo"))
        spots = [QPoint(int(area.left() + u * area.width()), int(area.top() + v * area.height()))
                 for u in (0.1, 0.3, 0.5, 0.7, 0.9) for v in (0.15, 0.4, 0.65, 0.92)]

        def seen(view=view, spots=spots) -> np.ndarray:
            wait(lambda: False, 150)
            shot = view.grabWindow()
            return np.array([shot.pixelColor(spot).getRgb()[:3] for spot in spots], int)

        if load is picture:
            same = written(view, session, "nothing")
            if same is not None:
                off = np.abs(same - lut.identity(len(same))).max()
                # (To 0.008: a small red beside a full green and blue is kept that finely by the half
                # floats a picture is handed over in; on average the table is 0.0002 off.)
                faults.expect(len(same) == 65 and off < 0.008 and np.abs(same - lut.identity(len(same))).mean() < 0.0005, f"the table of an untouched photo is not the one that changes nothing: {len(same)} points, up to {off:.4f} off")
        untouched = seen()
        session.setEdits({**look, "grain": 60.0, "crop": [0.0, 0.0, 1.0, 1.0], "masks": [OVAL]})
        session.commitEdit()
        session.setEdit("grain", 0.0)  # (seen without the grain, which no table holds)
        want = seen()
        session.setEdit("grain", 60.0)
        name = "look-" + kind.split()[-1]
        if written(view, session, name) is not None:
            session.resetAll()
            session.importLut(str(folder / f"{name}.cube"))
            settle()
            wait(lambda: stage(root).property("lut") is not None, 3000)
            view.grabWindow()
            wait(lambda: False, 300)
            got = seen()
            # (Where the photo itself is a color the screen can show: a table holds those and
            # no others, so a color beyond them is not what the look made of it.)
            inside = ((want > 6) & (want < 249) & (untouched > 6) & (untouched < 249)).all(axis=1)
            off = np.abs(got - want)[inside].max()
            faults.expect(inside.sum() >= 8 and off <= 3, f"{kind}: its look as a table, brought in again, is up to {off} levels from the look itself ({inside.sum()} spots)")
        ui.close_view(view)
    return faults


def audit_local() -> list[str]:
    """What a mask changes inside it (edit.LOCAL, Local.qml): where the mask is whole, the
    picture is the one the whole image would be with the mask's amounts added (sliders) or
    with the mask's own curve; where the mask is not, it is untouched; half way, between the
    two; and taking the mask's settings away brings the picture back."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "")
    faults = Faults()
    session.setEdits({"exposure": 0.2, "masks": [{"kind": "radial", "at": [0.5, 0.45], "to": [0.75, 0.45], "feather": 0.5}]})
    session.commitEdit()
    settle()
    area = box(root.findChild(QObject, "photo"))

    def at(u: float, v: float) -> QPoint:
        return QPoint(int(area.left() + u * area.width()), int(area.top() + v * area.height()))

    spots = {"inside": at(0.5, 0.45), "edge": at(0.5 + 0.25 * 0.8, 0.45), "outside": at(0.05, 0.1)}

    def seen() -> dict:
        image = view.grabWindow()
        return {name: np.array(image.pixelColor(spot).getRgb()[:3]) for name, spot in spots.items()}

    plain = seen()
    bent = [[0.0, 0.0], [0.4, 0.62], [1.0, 1.0]]
    for name, own, whole in (("Exposure +0.8 and Saturation −60", {"exposure": 0.8, "saturation": -60.0}, {"exposure": 1.0, "saturation": 40.0}),
                             ("Temperature −1500", {"temperature": -1500.0}, {"exposure": 0.2, "temperature": session.shot["temperature"] - 1500.0}),
                             ("a curve of its own", {"curve": bent}, {"exposure": 0.2, "curve": bent})):
        for key, value in own.items():
            (session.setMaskWhole if key in edit.LOCAL_WHOLE else session.setMaskEdit)(1, key, value)
        session.commitEdit()
        settle()
        masked = seen()
        was = dict(session.history.edit)
        session.setEdits({**{k: edit.DEFAULTS[k] for k in whole}, **whole, "masks": []})
        settle()
        all_over = seen()
        session.history.replace(was)
        session.refresh()
        session.resetMaskEdit(1, list(own))
        settle()
        back = seen()
        faults.expect(np.abs(masked["inside"] - all_over["inside"]).max() <= 2,
                      f"{name}: inside the mask the picture is {masked['inside']}, the whole image so changed is {all_over['inside']}")
        faults.expect(np.abs(masked["outside"] - plain["outside"]).max() <= 1, f"{name}: outside the mask the picture changed: {plain['outside']} to {masked['outside']}")
        low, high = np.minimum(plain["edge"], all_over["edge"]) - 2, np.maximum(plain["edge"], all_over["edge"]) + 2
        faults.expect(bool(((masked["edge"] >= low) & (masked["edge"] <= high)).all()) and np.abs(all_over["inside"] - plain["inside"]).max() > 8,
                      f"{name}: on the mask's soft edge the picture is {masked['edge']}, not between {plain['edge']} and {all_over['edge']}; or the change is too small to tell")
        faults.expect(all(np.abs(back[k] - plain[k]).max() <= 1 for k in plain), f"{name}: taken away again, the picture is not as it was")
    ui.close_view(view)
    return faults


def moves(view, at: QPoint, steps: list, between=None, held=Qt.KeyboardModifier.NoModifier) -> None:
    """A real drag through several places (each a QPoint from `at`); `between` is called
    after every move but the last; a tuple (QPoint, modifier) changes what is held."""
    QTest.mousePress(view, Qt.MouseButton.LeftButton, held, at, 500)
    settle()
    for n, step in enumerate(steps):
        by, held = step if isinstance(step, tuple) else (step, held)
        to = QPointF(at + by)
        QCoreApplication.sendEvent(view, QMouseEvent(
            QEvent.Type.MouseMove, to, to, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, held))
        settle()
        if between and n < len(steps) - 1:
            between()
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, held, (QPointF(at) + QPointF(to - QPointF(at))).toPoint())
    settle()


def fourth_drags(view, root, session, faults) -> None:
    """Drags that used to break: a pin's slider while the colors are counted, Shift on the way."""
    wheel, scroll = find(root, "wheel"), find(root, "tools-scroll")
    scroll.setProperty("contentY", scroll.property("contentHeight") - scroll.height())
    settle()
    view.grabWindow()
    tap(view, box(wheel).center().toPoint() + QPoint(40, -30))
    slider = find(root, "pin:feather")
    at = box(slider.findChild(QObject, "thumb")).center().toPoint()
    moves(view, at, [QPoint(-10 * k, 0) for k in range(1, 5)], lambda: (session.scoped.emit(), settle()))
    faults.expect(slider.property("value") < 64, f"a pin's Feather dragged 40 px while the colors were counted stopped at {slider.property('value')}")
    root.setProperty("unfolded", {"wheel": True})  # (the wheel's head in sight, whatever is over it)
    settle()
    view.grabWindow()
    scroll.setProperty("contentY", 0)
    settle()
    view.grabWindow()
    click(view, find(root, "reset:wheel"))
    faults.expect(len(session.pins) == 0, f"the Tone Wheel's Reset left {len(session.pins)} pins")
    session.undo()
    faults.expect(len(session.pins) == 1, "and Undo does not bring them back")
    root.setProperty("unfolded", {"tone": True, "wheel": True})
    settle()
    view.grabWindow()
    scroll.setProperty("contentY", 0)
    settle()
    view.grabWindow()
    slider = find(root, "slider:contrast")
    at = box(slider.findChild(QObject, "thumb")).center().toPoint()
    shift = Qt.KeyboardModifier.ShiftModifier
    seen = []
    moves(view, at, [QPoint(40, 0), (QPoint(41, 0), shift), (QPoint(61, 0), shift)], lambda: seen.append(slider.property("value")))
    faults.expect(len(seen) == 2 and abs(seen[1] - seen[0]) <= 1 and 0 < slider.property("value") - seen[1] < (seen[0] or 1) * 0.3,
                  f"Shift pressed in the middle of a drag: the slider went {seen} and then {slider.property('value')}")
    session.resetAll()
    settle()


def fourth_masks(view, root, session, faults) -> None:
    """A painted mask, the eye, a name."""
    click(view, find(root, "mode:masks"))
    view.grabWindow()
    click(view, find(root, "add-mask:brush"))
    view.grabWindow()
    settle()
    area = box(find(root, "photo"))
    place = lambda u, w: QPoint(int(area.left() + u * area.width()), int(area.top() + w * area.height()))  # noqa: E731
    moves(view, place(0.2, 0.3), [QPoint(int(0.02 * k * area.width()), 0) for k in range(1, 16)])
    kept = session.history.edit["masks"][0]["strokes"]
    faults.expect(len(kept) == 1 and len(kept[0]["points"]) > 5, f"a stroke of the brush kept {[len(k['points']) for k in kept]} places")
    # (All of its places handed to the interface at every change of anything: a curve dragged
    # on a photo with a painted mask ran at 30 frames a second.)
    faults.expect(session.edit["masks"][0]["strokes"] == 1, "the interface is handed the strokes of a painted mask, not how many there are")
    QTest.mouseMove(view, QPoint(30, 30))  # (off the photo: the mask's red is gone)
    settle()

    def seen() -> tuple:
        wait(lambda: False, 250)
        shot = view.grabWindow()
        return tuple(shot.pixelColor(place(*at)).getRgb()[:3] for at in ((0.35, 0.3), (0.8, 0.8)))

    before = seen()
    session.setMaskEdit(1, "exposure", 1.5)
    session.commitEdit()
    lit = seen()
    faults.expect(sum(lit[0]) > sum(before[0]) + 40 and lit[1] == before[1], f"a painted mask's Exposure: under the stroke {before[0]} -> {lit[0]}, away from it {before[1]} -> {lit[1]}")
    click(view, find(root, "mask-eye:1"))
    faults.expect(session.masks[0].get("off") is True and seen() == before, "the eye does not turn the mask off")
    click(view, find(root, "mask-eye:1"))
    faults.expect(seen() == lit, "nor on again")
    click(view, find(root, "brush:erase"))
    moves(view, place(0.33, 0.3), [QPoint(int(0.04 * area.width()), 0)])
    QTest.mouseMove(view, QPoint(30, 30))
    faults.expect(seen()[0] == before[0], "the eraser does not rub the mask out")
    double_click(view, box(find(root, "mask-item:1")).center().toPoint())
    for letter in "Sky":
        QTest.keyClick(view, letter)
    QTest.keyClick(view, Qt.Key.Key_Return)
    settle()
    faults.expect(session.masks[0].get("name") == "Sky" and find(root, "panel:mask-settings").property("title") == "Sky",
                  f"a mask named Sky is called {session.masks[0].get('name')!r}")
    faults.expect(session.path != "" and session.index == 0, "typing a name moved to another photo")
    faults.extend(audit_layout(view))


def fourth_crop(view, root, session, faults) -> None:
    """Portrait and back is the frame it was; the frame stays put while the picture turns."""
    click(view, find(root, "mode:crop"))
    view.grabWindow()
    session.setFrame([0, 0, 1, 1], [0.1, 0.15, 0.6, 0.55], False)
    session.commitEdit()
    settle()
    frame = find(root, "crop-box")

    def at() -> list:
        view.grabWindow()
        settle()
        return [box(frame).x(), box(frame).y(), box(frame).width(), box(frame).height()]

    still, crop = at(), list(session.edit["crop"])
    for angle in (3.0, 7.5, -6.0):
        session.setEdit("angle", angle)
        settle()
        faults.expect(max(abs(a - b) for a, b in zip(at(), still, strict=True)) < 2, f"turned to {angle}°, the frame moved on the screen: {still} -> {at()}")
    session.setEdit("angle", 0.0)
    session.commitEdit()
    wait(lambda: False, 900)
    for flip in range(4):
        click(view, find(root, "crop-portrait"))
        wide = at()[2] > at()[3]
        faults.expect(wide == (flip % 2 == 1), f"flip {flip + 1}: the frame is {'wide' if wide else 'tall'}")
    faults.expect(max(abs(a - b) for a, b in zip(crop, session.edit["crop"], strict=True)) < 1e-9, f"Portrait four times: the crop went from {crop} to {session.edit['crop']}")
    # One shape after another and back: the frame it was, not one cut out of each in turn.
    click(view, find(root, "crop-shape:3 : 2"))
    first = list(session.edit["crop"])
    for shape in ("1 : 1", "16 : 9", "5 : 4", "3 : 2"):
        click(view, find(root, "crop-shape:" + shape))
    faults.expect(max(abs(a - b) for a, b in zip(first, session.edit["crop"], strict=True)) < 1e-9,
                  f"3:2, three other shapes and 3:2 again: the crop went from {first} to {session.edit['crop']}")
    click(view, find(root, "mode:edit"))


def fourth_rest(view, root, session, shown, faults) -> None:
    """A checkpoint renamed, a folder's menu, the zoom gliding, the foot."""
    click(view, find(root, "checkpoint-save"))
    view.grabWindow()
    number = session.checkpoints[0]["id"]
    QTest.mouseMove(view, box(find(root, f"checkpoint:{number}")).center().toPoint(), 300)
    settle()
    view.grabWindow()
    click(view, find(root, f"checkpoint-rename:{number}"))
    QTest.keyClick(view, "A", Qt.KeyboardModifier.ControlModifier)
    for letter in "Warm":
        QTest.keyClick(view, letter)
    tap(view, box(find(root, "photo")).center().toPoint())  # (left by a click elsewhere: taken)
    faults.expect(session.checkpoints[0]["name"] == "Warm", f"a checkpoint renamed Warm is {session.checkpoints[0]['name']!r}")
    stage_ = stage(root)
    was = stage_.property("shown")
    QCoreApplication.sendEvent(view, QWheelEvent(QPointF(box(stage_).center()), QPointF(box(stage_).center()), QPoint(), QPoint(0, 120),
                                                 Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
    for window in QGuiApplication.allWindows():
        for held in window.findChildren(ui.PerFrameInput):
            held.send()
    QCoreApplication.processEvents()
    mid = stage_.property("shown")
    settle()
    faults.expect(was <= mid < stage_.property("shown") and not stage_.property("fitted"),
                  f"a notch of the wheel: the zoom went {was:.3f}, {mid:.3f}, {stage_.property('shown'):.3f} (it should glide, and not be Fit)")
    faults.expect(find(root, "text:version").isVisible() and find(root, "foot-github").isVisible(), "the foot does not say the version")
    # An arrow held goes on through the photos: the repeats of the key each step once more.
    tap(view, box(find(root, "photo")).center().toPoint())
    session.goTo(0)
    settle()
    for repeat in (False, True, True):
        QCoreApplication.sendEvent(view, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier, "", repeat))
    settle()
    faults.expect(session.index == 3, f"Right held for two repeats: at photo {session.index}, not 3")
    session.goTo(0)
    wait(lambda: root.property("photoReady"))
    settle()
    call(root, "show", "library")
    settle()
    view.grabWindow()
    folder = shown.places["folders"][0]
    QTest.mouseClick(view, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, box(find(root, "place:" + folder["id"])).center().toPoint())
    settle()
    view.grabWindow()  # (a frame: the menu's entries are placed when one is drawn)
    faults.expect(find(root, "menu-reveal").isVisible() and not find(root, "menu-export").isVisible(), "a right click on a folder does not offer Show in file manager")
    click(view, find(root, "menu-remove-folder"))
    view.grabWindow()
    faults.expect(find(root, "modal:confirm").isVisible() and len(shown.places["folders"]) == 2, f"removing a folder does not ask first: asked {find(root, 'modal:confirm').isVisible()}, folders {len(shown.places['folders'])}, menu {find(root, 'modal:menu').isVisible()}")
    click(view, find(root, "confirm-yes"))
    faults.expect(len(shown.places["folders"]) == 1, "nor remove it when told to")
    faults.extend(audit_layout(view))


def audit_fourth() -> list[str]:
    """The owner's fourth list from use (M18), on the Edit screen as the program starts with
    it: under real clicks and drags."""
    shown = shelf("fourth")
    session = ui.Session([], load=chart)
    view = ui.make_view(session, shown)
    session.surface_known(False)
    root = view.rootObject()
    root.setSize(view.size())
    shown.show(shown.places["folders"][0]["id"])
    session.openAt(shown.paths(), 0)
    call(root, "show", "edit")
    wait(lambda: root.property("photoReady"))
    settle()
    view.grabWindow()
    settle()
    faults = Faults()
    faults.extend(audit_layout(view))
    faults.expect(find(root, "slider:exposure").isVisible() and not find(root, "balance-pad").isVisible() and not find(root, "wheel").isVisible(),
                  "at start, Light is not the one card open")
    # Over a card's head the pointer is a hand, over a panel's edge two arrows: nothing
    # laid over the window takes the pointer's shape from what is under it.
    for name, by, shape in (("panel:tone", QPoint(80, 14), Qt.CursorShape.PointingHandCursor),
                            ("left-edge", QPoint(3, 200), Qt.CursorShape.SizeHorCursor)):
        QTest.mouseMove(view, box(find(root, name)).topLeft().toPoint() + by, 100)
        settle()
        faults.expect(view.cursor().shape() == shape, f"over {name} the pointer is {view.cursor().shape()}, not {shape}")
    root.setProperty("unfolded", {"tone": True, "wheel": True})
    settle()
    view.grabWindow()
    fourth_drags(view, root, session, faults)
    fourth_masks(view, root, session, faults)
    fourth_crop(view, root, session, faults)
    fourth_rest(view, root, session, shown, faults)
    ui.close_view(view)
    return faults


def audit_mono() -> list[str]:
    """Black and white under real clicks and keys: the switch puts its card in the color
    cards' place and M brings them back, the color grade untouched; the picture is gray, and
    as light at every spot as edit.mono_at says; a look is its points and is lit while the EQ
    is exactly it; a drag on the line makes a point; Reset is the look a photo starts with."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "saturation")
    faults = Faults()
    counted(view, session)
    grade = {"saturation": 140.0, "pins": [{"from": [0.1, 0.0], "to": [0.1, 0.06]}]}
    session.setEdits(grade)
    session.commitEdit()
    kept = dict(session.history.edit)
    area = box(root.findChild(QObject, "photo"))

    def there(*names) -> list[str]:
        return [n for n in names if find(root, n).isVisible()]

    def compared(when: str) -> None:
        shot, tried = view.grabWindow(), 0
        for u, v in ((0.3, 0.7), (0.62, 0.55), (0.12, 0.45), (0.85, 0.3), (0.5, 0.75), (0.2, 0.2), (0.75, 0.6)):
            seen = shot.pixelColor(int(area.left() + u * area.width()), int(area.top() + v * area.height()))
            if not all(4 < c < 251 for c in seen.getRgb()[:3]):
                continue  # cut off by the screen's own range: nothing to compare
            a, b, light = session.colorAt(u, v)
            want = edit.mono_at(session.bw, [light, a, b])[0]
            tried += 1
            faults.expect(max(seen.getRgb()[:3]) - min(seen.getRgb()[:3]) <= 2 and abs(oklab_of(seen)[0] - want) < 0.02,
                          f"{when}, at {u}, {v} the photo shows {seen.name()}, not a gray of lightness {want:.3f}")
        faults.expect(tried >= 3, f"{when}, too few spots of the chart could be compared ({tried})")

    faults.expect(there("panel:saturation", "panel:eq", "panel:chroma", "wheel-block", "panel:mono") == ["panel:saturation", "panel:eq", "panel:chroma", "wheel-block"],
                  f"a photo in color shows {there('panel:saturation', 'panel:eq', 'panel:chroma', 'wheel-block', 'panel:mono')}")
    bring(root, "treatment")
    click(view, find(root, "treatment:mono"))
    view.grabWindow()
    faults.expect(session.history.edit == {**kept, "mono": True}, f"Black & white changed more than the switch: {session.history.edit}")
    faults.expect(there("panel:saturation", "panel:eq", "panel:chroma", "wheel-block", "panel:mono") == ["panel:mono"] and find(root, "bw-plot").isVisible(),
                  f"in black and white the column shows {there('panel:saturation', 'panel:eq', 'panel:chroma', 'wheel-block', 'panel:mono')}")
    faults.expect(find(root, "film:panchromatic").property("on") and session.bw == edit.FILMS["panchromatic"], "a photo does not start black and white as Panchromatic")
    compared("as Panchromatic")
    bring(root, "panel:mono")
    click(view, find(root, "film:red"))
    faults.expect(session.bw == edit.FILMS["red"] and find(root, "film:red").property("on") and not find(root, "film:panchromatic").property("on"),
                  f"Red filter gave {session.bw}")
    compared("with the red filter")
    plot = box(find(root, "bw-plot"))
    w, h = plot.width() - 16, plot.height() - 16
    steps = session.steps
    drag(view, QPoint(round(plot.left() + 8 + 330 / 360 * w), round(plot.top() + 8 + 0.5 * h)), QPoint(0, -round(0.25 * h)))
    made = session.bw[-1] if session.bw else {}
    faults.expect(len(session.bw) == len(edit.FILMS["red"]) + 1 and abs(made["hue"] - 330) < 2 and abs(made["gain"] - 20) < 1.5 and session.steps == steps + 1,
                  f"dragging the line up a quarter at 330 degrees gave {made} in {session.steps - steps} steps")
    faults.expect(not any(find(root, f"film:{name}").property("on") for name in edit.FILMS), "a look changed by hand is still lit")
    compared("with a point of my own")
    click(view, find(root, "reset:mono"))
    faults.expect(session.history.edit == {**kept, "mono": True}, f"Reset on the card left {session.history.edit}")
    QCoreApplication.sendEvent(root, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_M, Qt.KeyboardModifier.NoModifier))
    settle()
    view.grabWindow()
    faults.expect(session.history.edit == kept and find(root, "panel:saturation").isVisible() and not find(root, "panel:mono").isVisible(),
                  f"M does not bring the color grade back as it was: {session.history.edit}")
    seen = view.grabWindow().pixelColor(int(area.left() + 0.3 * area.width()), int(area.top() + 0.7 * area.height()))
    faults.expect(max(seen.getRgb()[:3]) - min(seen.getRgb()[:3]) > 20, f"back in color the photo is still gray: {seen.name()}")
    # An export of it is gray as well: the file is made by the same shader.
    session.setEdits({"mono": True, "bw": edit.FILMS["orange"]})
    session.commitEdit()
    settle()
    out = Path(tempfile.mkdtemp(prefix="photone-mono-", dir=SHELF.name))
    session.exporter.side = 256
    session.exporter.start([str(p) for p in session.photos], {"kind": "png8", "quality": 92, "longest": 0, "folder": str(out)})
    carry_out(view, session)
    file = QImage(str(out / "first.png")).convertToFormat(QImage.Format.Format_RGB888)
    if file.isNull():
        faults.append(f"a black and white photo was not exported: {session.exporter.failures}")
    else:
        levels = np.frombuffer(file.constBits(), np.uint8)[:file.sizeInBytes()].reshape(file.height(), file.bytesPerLine())[:, :file.width() * 3].reshape(file.height(), -1, 3).astype(int)
        spread = (levels.max(axis=2) - levels.min(axis=2)).max()
        faults.expect(spread <= 1 and levels.std() > 20, f"the exported black and white photo has colors: up to {spread} levels between red, green and blue")
    # Inside a mask too, whose cards for colors are not there.
    session.setEdits({"mono": True, "masks": [OVAL]})
    session.commitEdit()
    show(root, "masks")
    click(view, find(root, "mask-item:1"))
    view.grabWindow()
    faults.expect(there("panel:local-light", "panel:local-saturation", "panel:local-eq", "panel:local-chroma", "wheel-block") == ["panel:local-light"],
                  f"a mask of a black and white photo shows {there('panel:local-light', 'panel:local-saturation', 'panel:local-eq', 'panel:local-chroma', 'wheel-block')}")
    session.setMaskEdit(1, "exposure", 1.0)
    session.commitEdit()
    QTest.mouseMove(view, QPoint(30, 30))
    wait(lambda: False, 250)
    inside = view.grabWindow().pixelColor(int(area.left() + 0.6 * area.width()), int(area.top() + 0.42 * area.height()))
    faults.expect(max(inside.getRgb()[:3]) - min(inside.getRgb()[:3]) <= 2, f"inside a mask with its own Exposure the photo is not gray: {inside.name()}")
    faults.extend(audit_layout(view))
    ui.close_view(view)
    return faults


def audit_grain() -> list[str]:
    """Film grain, measured in the picture: there in the middle grays and, as on a print, not
    at black or white; the same pattern when the photo is moved, and in an exported file;
    weaker where a pixel of the screen covers several grains; colored only when asked, and
    never in a black and white photo."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "")
    faults = Faults()

    def wedge(step: int) -> np.ndarray:  # the levels of that step of the gray wedge, as far as it is in sight
        area, room = box(root.findChild(QObject, "photo")), box(stage(root)).adjusted(4, 4, -4, -60)
        part = QRectF(area.left() + (step + 0.2) / 8 * area.width(), area.top() + 0.87 * area.height(),
                      0.6 / 8 * area.width(), 0.1 * area.height()).intersected(room).toRect()
        return export.bytes_of(view.grabWindow().copy(part)).astype(float)

    # (The chart is 480 px across and fitted at 170 %: at size 4 there are 1000 grains across
    # it, about one to a pixel of the screen, so each is seen for what it is.)
    faults.expect(wedge(3).std() < 0.6, f"without grain a gray of the wedge is not even: {wedge(3).std():.2f} levels")
    session.setEdits({"grain": 100.0, "grainSize": 4.0, "grainCurve": edit.RESPONSES["even"]})
    session.commitEdit()
    settle()
    even = wedge(3)
    faults.expect(even.std() > 4 and abs(even.mean() - wedge(3).mean()) < 0.01 and (even.max(axis=2) - even.min(axis=2)).max() <= 2,
                  f"grain of 100 on a middle gray: {even.std():.2f} levels of it, up to {(even.max(axis=2) - even.min(axis=2)).max():.0f} between red, green and blue")
    # The pattern is the picture's: seen closely and then moved 24 px across the screen, the
    # same part of the photo has the same grain.
    area = box(root.findChild(QObject, "photo"))
    call(stage(root), "zoomTo", 3.0, area.left() + 3.5 / 8 * area.width(), area.top() + 0.92 * area.height())
    settle()
    near = wedge(3)
    call(stage(root), "panBy", 24.0, 0.0)
    settle()
    moved = wedge(3)
    same = np.abs(moved - near).max() if moved.shape == near.shape else 999
    faults.expect(same <= 1 and near.std() > 4, f"the photo moved by 24 px, its grain is another: up to {same} levels")
    call(stage(root), "fit")
    settle()
    # A response: as on a print, none at black and at white.
    session.setEdit("grainCurve", edit.RESPONSES["print"])
    session.commitEdit()
    found = {}
    for step in (0, 3, 7):
        found[step] = round(float(wedge(step).std()), 2)
    faults.expect(found[0] < 0.8 and found[7] < 0.8 and found[3] > 3, f"with the Print response the grain is {found} levels at black, a middle gray and white")
    # Finer than the screen's pixels it is their average: weaker, not a coarser pattern.
    session.setEdits({"grainSize": 0.4, "grainCurve": edit.RESPONSES["even"]})
    settle()
    faults.expect(0.2 < wedge(3).std() < 0.3 * even.std(), f"twelve grains to a pixel of the screen: {wedge(3).std():.2f} levels, {even.std():.2f} with each grain seen")
    # Colored when asked; never in black and white.
    session.setEdits({"grainSize": 4.0, "grainColor": 100.0})
    settle()
    tinted = wedge(3)
    session.setEdit("mono", True)
    settle()
    gray = wedge(3)
    faults.expect((tinted.max(axis=2) - tinted.min(axis=2)).max() > 6 and (gray.max(axis=2) - gray.min(axis=2)).max() <= 2 and gray.std() > 4,
                  f"colored grain: up to {(tinted.max(axis=2) - tinted.min(axis=2)).max():.0f} levels between red, green and blue; in black and white {(gray.max(axis=2) - gray.min(axis=2)).max():.0f}")
    # And the file has the grain the screen shows at 100 %.
    session.setEdits({"mono": False, "grainColor": 30.0})
    session.commitEdit()
    settle()
    screen = shown_at_full_size(view, root)
    out = Path(tempfile.mkdtemp(prefix="photone-grain-", dir=SHELF.name))
    session.exporter.side = 256
    session.exporter.start([str(p) for p in session.photos], {"kind": "png8", "quality": 92, "longest": 0, "folder": str(out)})
    carry_out(view, session)
    file = QImage(str(out / "first.png"))
    if file.isNull():
        faults.append(f"a photo with grain was not exported: {session.exporter.failures}")
    else:
        written = export.bytes_of(file).astype(int)
        # (But the picture's outermost row and column, where a pixel has no neighbor to
        # tell how many grains it covers: there the two differ.)
        off = np.abs(written - screen)[1:-1, 1:-1].max() if written.shape == screen.shape else 999
        faults.expect(off <= 1 and written[-40:-10, 180:220].std() > 2, f"the exported grain is not the grain on screen: up to {off} levels off")
    # The card: a character is an amount, a size and a roughness, lit while the edit is
    # exactly it; a response is the curve's points; no Color for a black and white photo; and
    # the loupe shows the photo at 100 %, which a drag moves.
    session.resetAll()
    show(root, "grain")
    click(view, find(root, "grain-character:bold"))
    faults.expect(session.history.edit == edit.clean(edit.CHARACTERS["bold"]) and find(root, "grain-character:bold").property("on")
                  and not find(root, "grain-character:classic").property("on"), f"Bold gave {session.history.edit}")
    slider = find(root, "slider:grain")
    drag(view, box(slider.findChild(QObject, "thumb")).center().toPoint(), QPoint(-30, 0))
    faults.expect(session.edit["grain"] < 55 and not find(root, "grain-character:bold").property("on"), "an amount changed by hand leaves Bold lit")
    session.undo()
    bring(root, "grain-response:negative")
    click(view, find(root, "grain-response:negative"))
    faults.expect(session.grainCurve == edit.RESPONSES["negative"] and find(root, "grain-response:negative").property("on"), f"Shadows gave {session.grainCurve}")
    session.undo()
    faults.expect(session.grainCurve == edit.RESPONSES["print"] and session.edit["grain"] == 55, "a response chosen is more than one step of undo")
    plot = box(find(root, "grainCurve-plot"))
    drag(view, QPoint(round(plot.left() + 8 + 0.25 * (plot.width() - 16)), round(plot.top() + 8 + 0.5 * (plot.height() - 16))), QPoint(0, 30))
    faults.expect(len(session.grainCurve) == 4 and not any(find(root, f"grain-response:{name}").property("on") for name in edit.RESPONSES),
                  f"a drag on the response curve gave {session.grainCurve}")
    bring(root, "panel:grain")
    loupe, shown_ = find(root, "grain-loupe"), find(root, "grain-loupe-photo")
    faults.expect(shown_.isVisible() and shown_.width() == session.fullWidth and shown_.height() == session.fullHeight,
                  f"the loupe shows the photo {shown_.width():.0f} px across, not its own {session.fullWidth}")
    was = box(shown_).topLeft()
    drag(view, box(loupe).center().toPoint(), QPoint(-10, -25))  # (the chart is hardly wider than the loupe: it stops at the photo's edge)
    now = box(shown_).topLeft()
    faults.expect(abs(now.x() - was.x() + 10) <= 1 and abs(now.y() - was.y() + 25) <= 1, f"a drag of -10, -25 px in the loupe moved the photo by {now.x() - was.x():.0f}, {now.y() - was.y():.0f}")
    drag(view, box(loupe).center().toPoint(), QPoint(-300, 0))
    faults.expect(box(shown_).right() >= box(loupe).right() - 1, "dragged far, the loupe shows what is beside the photo")
    faults.expect(find(root, "slider:grainColor").isVisible(), "no Color for a photo in color")
    session.setEdit("mono", True)
    settle()
    view.grabWindow()
    faults.expect(find(root, "panel:grain").isVisible() and not find(root, "slider:grainColor").isVisible(), "a black and white photo's grain card: not there, or with Color")
    click(view, find(root, "reset:grain"))
    faults.expect(session.history.edit == {"mono": True}, f"Reset on the card left {session.history.edit}")
    faults.extend(audit_layout(view))
    ui.close_view(view)
    return faults


def audit_detail() -> list[str]:
    """Sharpening and noise reduction, measured in the picture at 100 %: noise reduction for
    lightness makes a noisy gray even and leaves an edge where it is, the one for color takes
    the color out of the noise and leaves its lightness; sharpening makes an edge steeper
    and, with Masking, leaves noise alone; seen from far none of it is done; and the file is
    the picture on screen."""
    session = ui.Session([Path("first.png")], load=noisy)
    view, root = opened(session, "")
    faults = Faults()
    call(stage(root), "actual")
    settle()

    def part(first: int, last: int) -> np.ndarray:  # those columns of the photo, rows 40 to 280, as levels
        area = box(root.findChild(QObject, "photo"))
        wait(lambda: False, 60)
        grab = view.grabWindow().copy(int(area.left()) + first, int(area.top()) + 40, last - first, 240)
        return export.bytes_of(grab).astype(float)

    def rise() -> float:  # how many pixels the noisy edge takes from a tenth to nine tenths of its step
        across = part(200, 280).mean(axis=(0, 2))
        low, high = across[:20].mean(), across[-20:].mean()
        return float(((across > low + 0.1 * (high - low)) & (across < low + 0.9 * (high - low))).sum())

    def set_(**values) -> None:
        # (From nothing of it, as a photo starts.)
        session.setEdits({"sharpen": 0.0, "sharpenRadius": 1.0, "sharpenMask": 0.0, "denoise": 0.0, "denoiseColor": 0.0, **values})
        settle()

    faults.expect(session.uniforms["sharpen"] == 0 and session.uniforms["denoise"] == 0 and session.uniforms["denoiseColor"] == 0 and session.history.edit == {},
                  "a photo starts with sharpening or noise reduction")
    set_()
    gray = part(20, 140)
    light, tint, edge = gray.mean(axis=2).std(), (gray[..., 0] - gray[..., 1]).std(), rise()
    faults.expect(light > 3 and tint > 4 and edge <= 3, f"(the photo to try it on: noise of {light:.1f} levels, {tint:.1f} of color, an edge of {edge:.0f} px)")
    set_(denoise=100.0)
    calm = part(20, 140)
    faults.expect(calm.mean(axis=2).std() < 0.5 * light and abs(calm.mean() - gray.mean()) < 1 and rise() <= edge + 1,
                  f"noise reduction at 100: the gray's noise went from {light:.2f} to {calm.mean(axis=2).std():.2f} levels, the edge from {edge:.0f} to {rise():.0f} px")
    # A lone dark pixel has no neighbor like it: it is taken for its neighbors all the same.
    # (The owner, of Luminance at 100: "these tiny few pixel spots didnt get reduced".)
    def specks(of: np.ndarray) -> float:  # how much darker than the gray the dark pixels are, in levels
        return float(of.mean() - of.mean(axis=2)[SPECKS[0] - 40, SPECKS[1] - 20].mean())

    faults.expect(specks(gray) > 60 and specks(calm) < 3, f"noise reduction at 100 leaves the lone dark pixels: {specks(gray):.0f} levels darker than the gray, then {specks(calm):.1f}")
    set_(denoiseColor=10.0)
    some = part(20, 140)
    set_(denoiseColor=100.0)
    plain_ = part(20, 140)
    # (Past 10 it is taken from further out: before, 100 did what 10 does.)
    faults.expect((plain_[..., 0] - plain_[..., 1]).std() < 0.8 * (some[..., 0] - some[..., 1]).std(),
                  f"color noise reduction at 100 does no more than at 10: {(plain_[..., 0] - plain_[..., 1]).std():.2f} levels of color left, {(some[..., 0] - some[..., 1]).std():.2f} at 10")
    faults.expect((plain_[..., 0] - plain_[..., 1]).std() < 0.4 * tint and plain_.mean(axis=2).std() > 0.8 * light and rise() <= edge + 1,
                  f"color noise reduction at 100: the noise's color went from {tint:.2f} to {(plain_[..., 0] - plain_[..., 1]).std():.2f} levels, its lightness from {light:.2f} to {plain_.mean(axis=2).std():.2f}")
    # Sharpening, on the clean edge: steeper, and lighter than the light side just beyond it.
    set_()
    soft = part(360, 440).mean(axis=(0, 2))
    set_(sharpen=100.0)
    crisp = part(360, 440).mean(axis=(0, 2))
    faults.expect(crisp.max() > soft.max() + 3 and crisp.min() < soft.min() - 1 and abs(crisp[:20].mean() - soft[:20].mean()) < 0.5,
                  f"sharpening at 100: the edge goes from {soft.min():.0f}..{soft.max():.0f} to {crisp.min():.0f}..{crisp.max():.0f} levels")
    rough = part(20, 140).mean(axis=2).std()
    set_(sharpen=100.0, sharpenMask=100.0)
    masked_ = part(20, 140).mean(axis=2).std()
    kept = part(360, 440).mean(axis=(0, 2))
    faults.expect(rough > 1.3 * light and masked_ < 1.15 * light and kept.max() > soft.max() + 3,
                  f"sharpening makes the noise {rough:.2f} levels (from {light:.2f}); with Masking {masked_:.2f}, the edge still to {kept.max():.0f}")
    # The file is the picture on screen.
    set_(sharpen=60.0, denoise=50.0, denoiseColor=50.0)
    session.commitEdit()
    screen = shown_at_full_size(view, root)
    out = Path(tempfile.mkdtemp(prefix="photone-detail-", dir=SHELF.name))
    session.exporter.side = 256
    session.exporter.start([str(p) for p in session.photos], {"kind": "png8", "quality": 92, "longest": 0, "folder": str(out)})
    carry_out(view, session)
    file = QImage(str(out / "first.png"))
    if file.isNull():
        faults.append(f"a sharpened photo was not exported: {session.exporter.failures}")
    else:
        written = export.bytes_of(file).astype(int)
        off = np.abs(written - screen)[4:-4, 4:-4].max() if written.shape == screen.shape else 999
        faults.expect(off <= 1, f"the exported picture is not the one on screen: up to {off} levels off")
    # From far as well, as an export made smaller (until 2026-11-20 none of it was done there;
    # the owner: "it can be VERY visible"): the noise a view at 40 % still shows is taken out.
    call(stage(root), "zoomTo", 0.4, box(stage(root)).center().x(), box(stage(root)).center().y())
    settle()
    area = box(root.findChild(QObject, "photo"))
    small = lambda: export.bytes_of(view.grabWindow().copy(int(area.left()) + 8, int(area.top()) + 16, 48, 96)).astype(float)  # noqa: E731
    set_()
    far = small().mean(axis=2).std()
    # (At 0 the full-size picture they are kept in is not made: no memory for nothing.)
    kept_ = lambda: root.findChild(QObject, "detailed").property("on")  # noqa: E731
    faults.expect(not kept_(), "all of Detail at 0, and its full-size picture is made all the same")
    set_(denoise=100.0)
    faults.expect(kept_(), "Detail turned up from 0 while seen from far, and no full-size picture of it is made")
    faults.expect(small().mean(axis=2).std() < 0.8 * far, f"seen at 40 %, noise reduction at 100 leaves {small().mean(axis=2).std():.2f} levels of noise of {far:.2f}")
    ui.close_view(view)
    return faults


def audit_lens() -> list[str]:
    """Lens corrections from a file's own figures, measured in the picture: vignetting makes
    the corners as much lighter as the figures say and leaves the middle; distortion puts
    each thing where the figures say, and no corner is left empty; chromatic aberration
    moves red and blue apart from green by what they say; the switches of the card; a file
    without figures has nothing to switch; and the file is the picture on screen."""
    from photone import lens
    session = ui.Session([Path("first.arw")], load=lensed)
    view, root = opened(session, "lens")
    faults = Faults()
    area = box(root.findChild(QObject, "photo"))

    def seen() -> np.ndarray:  # the photo as it is on screen, rows x columns x 3
        settle()
        wait(lambda: False, 80)
        return export.bytes_of(view.grabWindow().copy(area.toRect())).astype(float)

    def steps(shot: np.ndarray, channel: int = 1) -> list[float]:
        """Where the gray wedge's steps begin, in pixels across the shot, for one color:
        the places along a row near the foot where it gets lighter by a step."""
        row = shot[int(0.93 * len(shot)) - 3:int(0.93 * len(shot)) + 3, :, channel].mean(axis=0)
        rise = np.diff(row)
        found, n = [], 1
        while n < len(rise) - 1:
            if rise[n] > 4:
                last = n
                while last + 1 < len(rise) and rise[last + 1] > 1:
                    last += 1
                weights = rise[n:last + 1]
                found.append(float((np.arange(n, last + 1) * weights).sum() / weights.sum()) + 1)
                n = last + 1
            n += 1
        return found

    def where(table: list, x: float) -> float:
        """Where a thing that is at `x` (0..1 across the stored picture, on the wedge's row)
        is drawn: the place whose pixel reads the picture there."""
        low, high = 0.0, 1.0
        for _ in range(40):
            at = (low + high) / 2
            out = np.array([(at - 0.5) * 1.5, 0.93 - 0.5])
            r = float(np.hypot(*out) / np.hypot(0.75, 0.5))
            read = 0.5 + (at - 0.5) * lens._at_table(table, r)
            low, high = (at, high) if read < x else (low, at)
        return (low + high) / 2

    # All three are off for a photo that says nothing else (the owner, 2026-11-20: "make
    # these off by default"); each is then tried alone.
    faults.expect(session.lensHas == {"distortion": True, "color": True, "vignetting": True} and session.uniforms["lensOn"] == 0
                  and not any(find(root, f"lens:{name}").property("on") for name in ("distortion", "color", "vignetting")),
                  f"a file with figures: {session.lensHas}; its corrections are not off from the start")
    off = {"lensDistortion": False, "lensColor": False, "lensVignetting": False}
    # (With the sharpening a photo started with until 2026-11-22, when `steps` was set:
    # without it, blue's last step is too soft to be found.)
    session.setEdits({**off, **catalog.DETAIL_ON})
    session.commitEdit()
    plain_ = seen()
    faults.expect(session.uniforms["lensOn"] == 0, "switched off, something is still done")
    edges = steps(plain_)
    # (Six of the wedge's seven steps: the last, to white, is too small a step to find.)
    faults.expect(len(edges) == 6, f"(the wedge's steps are found at {edges})")
    # Vignetting, by its switch: the corner lighter, the middle as it was.
    click(view, find(root, "lens:vignetting"))
    lit = seen()
    high, wide = lit.shape[:2]
    faults.expect(session.history.edit == {"lensVignetting": True, **catalog.DETAIL_ON} and np.abs(lit[high // 2 - 4:high // 2 + 4, wide // 2 - 4:wide // 2 + 4] - plain_[high // 2 - 4:high // 2 + 4, wide // 2 - 4:wide // 2 + 4]).max() <= 1,
                  f"Vignetting: {session.history.edit}, or the picture's middle changed")
    corner = lambda shot: shot[int(0.7 * high):int(0.76 * high), int(0.03 * wide):int(0.06 * wide)].mean()  # noqa: E731
    faults.expect(corner(lit) > corner(plain_) * 1.1, f"Vignetting: a corner went from {corner(plain_):.1f} to {corner(lit):.1f} levels")
    click(view, find(root, "lens:vignetting"))
    # Distortion: each step of the wedge where the figures put it; no corner left empty.
    click(view, find(root, "lens:distortion"))
    bent = seen()
    green = session.history.edit and edit.uniforms(session.history.edit, session._camera, True, shape=1.5)["lensGreen"]
    want = [where(green, (n + 1) / 8) * wide for n in range(6)]
    got = steps(bent)[:6]  # (read by a curve, a step stays steep enough that the last may be found too)
    off = max(abs(a - b) for a, b in zip(got, want, strict=False)) if len(got) == 6 else 999
    faults.expect(off <= 1.5 and max(abs(a - b) for a, b in zip(got, edges, strict=False)) > 4,
                  f"Distortion: the wedge's steps are at {[round(v) for v in got]}, the figures put them at {[round(v) for v in want]} (they were at {[round(v) for v in edges]})")
    faults.expect(min(bent[2, 2].max(), bent[-3, -3].max(), bent[2, -3].max()) > 8, "Distortion leaves a corner of the frame empty")
    click(view, find(root, "lens:distortion"))
    # Chromatic aberration: the lens drew red larger and blue smaller than green (`lensed`),
    # so on the right half a step lies further out for red and further in for blue, by what
    # the figures say; corrected, the three are in one place, green where it was, and a
    # step is as steep as it was (the owner: "making the entire photo a bit soft").
    near = lambda found, x: min(found, key=lambda at: abs(at - x)) if found else 0.0  # noqa: E731
    last = lambda shot: [near(steps(shot, color), edges[-1]) for color in (0, 1, 2)]  # noqa: E731 - where the last step is for red, green, blue
    steep = lambda shot: float(np.diff(shot[int(0.93 * len(shot)) - 3:int(0.93 * len(shot)) + 3].mean(axis=(0, 2)))[int(edges[-1]) - 6:int(edges[-1]) + 6].max())  # noqa: E731 - the most the lightness rises by from one pixel to the next at that step
    was = last(plain_)
    click(view, find(root, "lens:color"))
    fringed = seen()
    red, mid, blue = last(fringed)
    numbers = edit.uniforms(session.history.edit, session._camera, True, shape=1.5)
    expect = (where(numbers["lensBlue"], 6 / 8) - where(numbers["lensRed"], 6 / 8)) * wide
    faults.expect(abs(was[0] - was[2] - expect) < 1.0 and abs(expect) > 2, f"(the chart's last step: red and blue are {was[0] - was[2]:.1f} px apart as the lens drew them, the figures say {expect:.1f})")
    faults.expect(abs(mid - edges[-1]) < 0.3 and abs(red - mid) < 0.5 and abs(blue - mid) < 0.5,
                  f"Chromatic aberration: corrected, the last step's red is {red - mid:.2f} px and its blue {blue - mid:.2f} px from its green, which moved {mid - edges[-1]:.2f}")
    faults.expect(steep(fringed) > 0.95 * steep(plain_), f"Chromatic aberration: corrected, the last step rises by {steep(fringed):.1f} levels a pixel at most, uncorrected by {steep(plain_):.1f}: softer")
    # Reset all switches them off again; the file is the picture on screen, with all three.
    session.resetAll()
    faults.expect(session.history.edit == {} and session.uniforms["lensOn"] == 0, "Reset all does not switch the three corrections off")
    session.setEdits({**catalog.LENS_ON, **catalog.DETAIL_ON})
    session.commitEdit()
    # Seen from close, the photo's pixels are squares also with the corrections (and some
    # sharpening and noise reduction) on: along a row, four pixels of
    # the screen to one of the photo, the color changes every fourth pixel and not between.
    centre = box(stage(root)).center()
    call(stage(root), "zoomTo", 4.0, centre.x(), centre.y())
    settle()
    wait(lambda: False, 150)
    close = export.bytes_of(view.grabWindow().copy(int(centre.x()) - 60, int(centre.y()) - 40, 120, 1)).astype(int)[0]
    # (Of green: red and blue have squares of their own, a little apart from green's, which
    # is what the fringe correction is. Between two squares one pixel is the mix of both.)
    same = float((np.diff(close[:, 1]) == 0).mean())
    faults.expect(len(set(close[:, 1].tolist())) >= 8 and same > 0.45,
                  f"at 400 % with the lens corrections on, {same:.0%} of a row's pixels are as their neighbor (the photo's own pixels as squares: half and more)")
    call(stage(root), "fit")
    settle()
    screen = shown_at_full_size(view, root)
    out = Path(tempfile.mkdtemp(prefix="photone-lens-", dir=SHELF.name))
    session.exporter.side = 256
    session.exporter.start([str(p) for p in session.photos], {"kind": "png8", "quality": 92, "longest": 0, "folder": str(out)})
    carry_out(view, session)
    file = QImage(str(out / "first.png"))
    if file.isNull():
        faults.append(f"a photo with lens corrections was not exported: {session.exporter.failures}")
    else:
        written = export.bytes_of(file).astype(int)
        # (All but a few pixels along the chart's one razor edge, between its colors and its
        # wedge: read between two pixels there, the picture is as the GPU mixes the two, which
        # it does to a 256th of a pixel, and the screen's place and the file's are not the same
        # 256th. And but the frame's outermost four pixels.)
        gap = np.abs(written - screen)[4:-4, 4:-4] if written.shape == screen.shape else np.full((1, 1, 3), 999)
        faults.expect((gap > 1).mean() < 0.002 and gap.mean() < 0.05, f"the exported picture is not the one on screen: {(gap > 1).mean():.2%} of it differs, by {gap.mean():.3f} levels on average")
    faults.extend(audit_layout(view))
    ui.close_view(view)
    # A file without figures: nothing to switch, and nothing is done.
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "lens")
    settle()
    faults.expect(not find(root, "lens:distortion").property("enabled") and session.uniforms["lensOn"] == 0 and "no figures" in find(root, "lens-words").property("text"),
                  "a file without lens figures: its switches can be pressed, or something is done to it")
    ui.close_view(view)
    return faults


def audit_fringes() -> list[str]:
    """Defringe, measured in the picture, for a file that carries no figures of its lens: it
    takes the seam of color off an edge, purple or green, and leaves a purple or green thing
    that is no seam as it is."""
    session = ui.Session([Path("first.png")], load=fringed)
    view, root = opened(session, "lens")
    faults = Faults()
    call(stage(root), "actual")
    settle()

    def part(x: int, y: int, wide: int, high: int) -> np.ndarray:
        area = box(root.findChild(QObject, "photo"))
        wait(lambda: False, 60)
        return export.bytes_of(view.grabWindow().copy(int(area.left()) + x, int(area.top()) + y, wide, high)).astype(float).mean(axis=(0, 1))

    purple = lambda c: min(c[0], c[2]) - c[1]  # noqa: E731 - how purple a color is, in levels
    green = lambda c: c[1] - max(c[0], c[2])  # noqa: E731
    seam, thing = part(239, 40, 2, 80), part(70, 50, 40, 40)
    low, lump = part(239, 200, 2, 80), part(70, 230, 40, 40)
    faults.expect(purple(seam) > 60 and green(low) > 60, f"(the photo to try it on: a seam of {seam} and one of {low})")
    faults.expect(not find(root, "lens:distortion").property("enabled") and find(root, "slider:defringe").isVisible(), "a file without figures: Defringe is not there")
    slider = find(root, "slider:defringe")
    bring(root, "slider:defringe")
    track = slider.findChild(QObject, "track")
    drag(view, box(slider.findChild(QObject, "thumb")).center().toPoint(), QPoint(int(track.width()), 0))
    faults.expect(session.history.edit == {"defringe": 100.0}, f"Defringe dragged to its end: {session.history.edit}")
    faults.expect(purple(part(239, 40, 2, 80)) < 0.25 * purple(seam) and green(part(239, 200, 2, 80)) < 0.25 * green(low),
                  f"Defringe at 100: the purple seam went from {purple(seam):.0f} to {purple(part(239, 40, 2, 80)):.0f} levels of purple, the green one from {green(low):.0f} to {green(part(239, 200, 2, 80)):.0f} of green")
    faults.expect(abs(purple(part(70, 50, 40, 40)) - purple(thing)) <= 2 and abs(green(part(70, 230, 40, 40)) - green(lump)) <= 2,
                  f"Defringe at 100: the purple square went from {purple(thing):.0f} to {purple(part(70, 50, 40, 40)):.0f} levels of purple, the green one from {green(lump):.0f} to {green(part(70, 230, 40, 40)):.0f} of green: things, no seams")
    faults.extend(audit_layout(view))
    ui.close_view(view)
    return faults


def audit_mark() -> list[str]:
    """A watermark, measured in the picture: text set on the Export page is on the photo
    there and in Edit, in the place chosen and as large as Size says; the switch hides it in
    Edit only; the exported file has it exactly as Edit shows it at 100 %, also when hidden
    there; a picture for a watermark; and with None there is none."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "")
    faults = Faults()
    exporter, out = session.exporter, Path(tempfile.mkdtemp(prefix="photone-mark-", dir=SHELF.name))

    def photo_now() -> np.ndarray:
        settle()
        wait(lambda: False, 250)
        view.grabWindow()
        return shown_at_full_size(view, root).astype(int)

    def written(name: str) -> np.ndarray:
        session.exporter.side = 256
        session.exporter.start([str(p) for p in session.photos], {"kind": "png8", "quality": 92, "longest": 0, "folder": str(out)})
        carry_out(view, session)
        wait(lambda: not session.exporter.running, 8000)
        return export.bytes_of(QImage(str(out / name))).astype(int)

    bare = photo_now()
    call(root, "show", "export")
    settle()
    view.grabWindow()
    faults.expect(not find(root, "mark-text").isVisible() and exporter.mark["kind"] == "none", "a watermark is set before one was asked for")
    click(view, find(root, "mark-kind:text"))
    view.grabWindow()
    click(view, find(root, "mark-text"))
    for letter in "PHOTONE":
        QTest.keyClick(view, letter)
    QTest.keyClick(view, Qt.Key.Key_Return)
    click(view, find(root, "mark-place:0"))
    exporter.setMark({"size": 10.0, "strength": 100.0})
    settle()
    view.grabWindow()
    faults.expect(exporter.mark == {"kind": "text", "text": "PHOTONE", "file": "", "place": 0, "size": 10.0, "strength": 100.0}, f"set on the Export page: {exporter.mark}")
    faults.extend(audit_layout(view))
    call(root, "show", "edit")
    marked = photo_now()
    changed = np.abs(marked - bare).max(axis=2) > 12
    ys, xs = np.nonzero(changed)
    edge = export.mark_edge(480, 320)
    # (Its letters, which stand a little inside the line they are written on: capitals of a
    # line 32 px high are some 24 px, and begin some 10 px under the line's top.)
    faults.expect(len(ys) > 200 and edge <= ys.min() <= edge + 16 and edge <= xs.min() <= edge + 8 and 18 <= ys.max() - ys.min() <= 34 and xs.max() < 300,
                  f"in Edit the watermark covers rows {ys.min() if len(ys) else 0} to {ys.max() if len(ys) else 0} and columns {xs.min() if len(xs) else 0} to {xs.max() if len(xs) else 0} (top left, a line 32 px high, {edge} px from the edges)")
    file = written("first.png")
    gap = np.abs(file - marked)
    # (Edit scales the watermark's picture on the GPU and the file's is scaled by Qt: along
    # the letters' edges the two differ by some levels, nowhere by a letter.)
    faults.expect(file.shape == marked.shape and (gap > 40).mean() < 0.002 and gap.mean() < 0.5 and np.abs(file - bare).max() > 100,
                  f"the exported file's watermark is not Edit's: {(gap > 40).mean():.2%} of the picture differs much, {gap.mean():.2f} levels on average")
    # Bottom right; hidden in Edit, the file has it all the same.
    exporter.setMark({"place": 8})
    low = photo_now()
    ys, xs = np.nonzero(np.abs(low - bare).max(axis=2) > 12)
    faults.expect(len(ys) > 200 and 319 - edge - 14 <= ys.max() <= 319 - edge and 479 - edge - 8 <= xs.max() <= 479 - edge and ys.min() > 200,
                  f"Bottom right: it covers rows {ys.min() if len(ys) else 0} to {ys.max() if len(ys) else 0}, columns {xs.min() if len(xs) else 0} to {xs.max() if len(xs) else 0}")
    exporter.setMarkShown(False)
    hidden = photo_now()
    faults.expect(np.abs(hidden - bare).max() <= 1, "hidden in Edit, the watermark is still on the photo there")
    file = written("first-2.png")
    faults.expect((np.abs(file - low) > 40).mean() < 0.002 and np.abs(file - bare).max() > 100, "hidden in Edit, the exported file has no watermark, or another")
    # Its scale, opacity and place are no other picture: nothing is fetched or drawn anew
    # for a step of a slider (it lagged when each step was).
    serial, made = exporter.markSerial, exporter.mark_picture()
    for step in range(20):
        exporter.setMark({"size": 5.0 + step, "strength": 40.0 + step, "place": step % 9})
    faults.expect(exporter.markSerial == serial and exporter.mark_picture() is made, "a change of scale, opacity or place makes the watermark's picture anew")
    # And a slider being dragged reaches the watermark on the page without Python being told
    # at all, which hears of it once, when the slider is let go.
    call(root, "show", "export")
    settle()
    view.grabWindow()
    told = [0]
    exporter.markChanged.connect(lambda: told.__setitem__(0, told[0] + 1))
    drawn = next(i for i in items(find(root, "box:export")) if i.objectName() == "watermark")
    slider = find(root, "mark-size")
    at, was, during = box(slider.findChild(QObject, "thumb")).center().toPoint(), exporter.mark["size"], []
    moves(view, at, [QPoint(-20, 0), QPoint(-40, 0), QPoint(-60, 0)], lambda: during.append((told[0], round(drawn.width(), 1))))
    faults.expect([d[0] for d in during] == [0, 0] and during[0][1] > during[1][1] > drawn.width(),
                  f"dragging Scale: Python was told {[d[0] for d in during]} times on the way, the watermark was {[d[1] for d in during]} and then {drawn.width():.1f} px wide")
    faults.expect(told[0] == 1 and exporter.mark["size"] < was, f"let go, Python was told {told[0]} times; Scale went from {was} to {exporter.mark['size']}")
    call(root, "show", "edit")
    exporter.setMark({"size": 10.0, "strength": 100.0, "place": 8})
    exporter.setMarkShown(True)
    # A picture: a red square, half as strong; one that is no picture is none.
    square = QImage(40, 40, QImage.Format.Format_RGB32)
    square.fill(QColor(255, 0, 0))
    square.save(str(out / "logo.png"))
    exporter.setMark({"kind": "picture", "file": str(out / "logo.png"), "place": 4, "size": 25.0, "strength": 50.0})
    logo = photo_now()
    middle, was = logo[160, 240], bare[160, 240]
    ys, xs = np.nonzero(np.abs(logo - bare).max(axis=2) > 6)
    faults.expect(abs(int(middle[0]) - (int(was[0]) + 255) // 2) <= 4 and abs(int(middle[1]) - int(was[1]) // 2) <= 4 and abs((ys.max() - ys.min() + 1) - 80) <= 2,
                  f"a red square at half strength in the middle: {middle.tolist()} over {was.tolist()}, {ys.max() - ys.min() + 1 if len(ys) else 0} px high (80)")
    (out / "logo.png").write_text("no picture any more", "utf-8")
    exporter.setMark({"file": str(out / "logo.png")})  # (chosen again: read again)
    faults.expect(np.abs(photo_now() - bare).max() <= 1 and exporter.markPlace(480, 320) == [], "a watermark's file that is no picture is still drawn")
    exporter.setMark({"kind": "none"})
    faults.expect(np.abs(written("first-3.png") - bare).max() <= 1, "with None the exported file is not the photo alone")
    ui.close_view(view)
    return faults


def audit_tools() -> list[str]:
    """The cards of the Adjust tab, with real clicks and keys: Light alone open at the start
    (the owner's choice), a card's head folds and opens it, T and W open theirs and bring them in sight,
    a card's A/B (held) and Reset, Reset all."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session)
    faults = Faults()
    cards = ("tone", "balance", "saturation", "eq", "chroma")
    column = box(find(root, "tools-scroll"))

    def shown() -> list[str]:
        return [n for n in cards if find(root, f"panel:{n}").property("open")]

    def head(name: str) -> None:  # a click on a card's head
        bring(root, f"panel:{name}")
        tap(view, box(find(root, f"panel:{name}")).topLeft().toPoint() + QPoint(80, 14))
        view.grabWindow()  # (a frame: the card has its height)
        wait(lambda: False, 60)  # (opened, it is brought into the column's middle a moment later)
        settle()
        view.grabWindow()

    def key(which) -> None:
        QCoreApplication.sendEvent(root, QKeyEvent(QEvent.Type.KeyPress, which, Qt.KeyboardModifier.NoModifier))
        settle()
        view.grabWindow()

    faults.expect(shown() == ["tone"], f"at the start the open cards are {shown()}, not Light alone")
    faults.expect(all(find(root, f"panel:{n}").isVisible() for n in cards), "not every card of Adjust is in the column")
    head("eq")
    plot = find(root, "curve-plot")
    faults.expect(shown() == ["tone", "eq"] and plot.isVisible(), f"a click on Color EQ's head: open {shown()}, the tone curve in Light {plot.isVisible()}")
    head("eq")
    faults.expect(shown() == ["tone"], f"a click on an open card's head does not fold it: {shown()}")
    head("tone")
    faults.expect(shown() == [] and not plot.isVisible(), f"folding Light left {shown()}, its curve {plot.isVisible()}")
    key(Qt.Key.Key_W)
    faults.expect(shown() == ["balance"] and column.contains(box(find(root, "balance-pad"))), f"W does not open White balance in sight: {shown()}")
    key(Qt.Key.Key_T)
    faults.expect(shown() == ["tone", "balance"] and column.contains(box(find(root, "slider:exposure"))), f"T does not open Light in sight: {shown()}")
    session.setEdits({"contrast": 40.0, "curve": LIFT, "saturation": 120.0})
    session.commitEdit()
    settle()
    at = box(find(root, "compare:tone")).center().toPoint()
    QTest.mousePress(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at)
    settle()
    held = session.uniforms
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at)
    settle()
    faults.expect(held["contrast"] == 0 and held["curveOn"] == 0 and session.uniforms["contrast"] > 0,
                  "holding A/B does not show the photo without that card, and only that card")
    click(view, find(root, "reset:tone"))
    faults.expect(session.history.edit == {"saturation": 120.0}, f"Reset on the Light card left {session.history.edit}")
    session.undo()
    bring(root, "reset-all")
    click(view, find(root, "reset-all"))
    faults.expect(session.history.edit == {}, f"Reset all left {session.history.edit}")
    session.undo()
    faults.expect(session.history.edit == {"contrast": 40.0, "curve": LIFT, "saturation": 120.0}, "Reset all is not one step of undo")
    ui.close_view(view)
    return faults


def audit_dock() -> list[str]:
    """The card dock beside the right column, with real clicks: a button for each card on show
    in Adjust, in the column's order; a click on a folded card's button opens it and brings it
    into the column; on an open one, brings it in and leaves it open; Masks has its own
    buttons, Crop none."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session)
    faults = Faults()
    column = box(find(root, "tools-scroll"))
    dock = find(root, "screen:dock")

    def buttons() -> list[str]:
        return [i.objectName()[5:] for i in sorted((i for i in items(dock) if i.objectName().startswith("dock:") and i.isVisible()),
                                                   key=lambda i: box(i).top())]

    def press(card: str) -> None:
        click(view, next(i for i in items(dock) if i.objectName() == "dock:" + card))
        view.grabWindow()  # (a frame: the card has its height)
        wait(lambda: False, 300)  # (brought in a moment later, and glides there)
        call(root, "land")
        settle()
        view.grabWindow()

    def seen(name: str) -> bool:  # its head in the column
        head = box(find(root, name))
        return column.contains(QRectF(head.left(), head.top(), head.width(), 30))

    adjust = ["light", "balance", "lut", "saturation", "eq", "chroma", "wheel", "detail", "lens", "grain"]
    faults.expect(dock.isVisible() and buttons() == adjust, f"Adjust's dock holds {buttons()}")
    faults.expect(box(dock).right() <= column.left(), "the dock is not beside the column")
    press("grain")
    faults.expect(find(root, "panel:grain").property("open") and seen("panel:grain"), "Film grain's button does not open it in sight")
    press("light")
    faults.expect(find(root, "panel:tone").property("open") and seen("panel:tone"), "Exposure's button folded it, or left it out of sight")
    call(root, "go", "masks")
    settle()
    view.grabWindow()
    faults.expect(buttons() == ["masks"], f"Masks with no mask chosen: the dock holds {buttons()}")
    session.setEdits({"masks": [{"kind": "radial", "at": [0.6, 0.42], "to": [0.8, 0.5], "ratio": 0.6, "feather": 0.4}]})
    settle()
    find(root, "wheel").setProperty("mask", 1)
    settle()
    view.grabWindow()
    faults.expect(buttons() == ["masks", "mask", "light", "balance", "saturation", "eq", "chroma", "wheel"], f"Masks with a mask: the dock holds {buttons()}")
    press("chroma")
    faults.expect(seen("panel:local-chroma"), "the mask's Chroma curve is not brought in")
    call(root, "go", "crop")
    settle()
    faults.expect(not dock.isVisible(), "the dock is there on Crop")
    ui.close_view(view)
    return faults


def audit_curve() -> list[str]:
    """The tone curve under a real mouse: a click on the line changes nothing, a drag from it
    makes a point and moves it, a point stops short of its neighbor, a double click removes it."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "curve")
    faults = Faults()
    area = box(find(root, "curve-plot"))
    w, h = area.width() - 16, area.height() - 16

    def at(x: float, y: float) -> QPoint:
        return QPoint(round(area.left() + 8 + x * w), round(area.top() + 8 + (1 - y) * h))

    def curve() -> list:
        return session.history.edit.get("curve")

    click(view, find(root, "curve-plot"))
    faults.expect(curve() is None and not session.canUndo, f"a click on the line changed the curve: {curve()}")
    drag(view, at(0.5, 0.5), QPoint(0, -round(0.2 * h)))
    faults.expect(curve() and len(curve()) == 3 and abs(curve()[1][0] - 0.5) < 0.01 and abs(curve()[1][1] - 0.7) < 0.01,
                  f"dragging the middle of the line up by a fifth gave {curve()}")
    session.undo()
    faults.expect(curve() is None, "making and moving a point is not one step of undo")
    session.redo()
    drag(view, at(0.5, 0.7), QPoint(round(w), 0))
    faults.expect(curve() and 0.9 < curve()[1][0] <= 0.98, f"a point dragged past the white point: {curve()}")
    drag(view, at(0.0, 0.0), QPoint(0, -round(0.1 * h)))
    faults.expect(curve() and abs(curve()[0][1] - 0.1) < 0.01, f"the black point does not move: {curve()}")
    double_click(view, at(0.0, 0.1))
    faults.expect(curve() and len(curve()) == 3 and curve()[0] == [0.0, 0.0], f"a double click on the black point: {curve()}")
    double_click(view, at(*(curve() or LIFT)[1]))
    faults.expect(curve() is None, f"a double click does not take a point away: {curve()}")
    ui.close_view(view)
    return faults


def audit_balance() -> list[str]:
    """The white-balance pad under a real mouse: up is warmer by octaves of kelvin, across is
    tint, a double click goes back to as shot. And Pick neutral: the spot clicked turns gray."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "balance")
    faults = Faults()
    middle = box(find(root, "balance-pad")).center().toPoint()
    shot = session.shot
    side = round(box(find(root, "balance-pad")).width())
    drag(view, middle, QPoint(0, -side // 4))  # a quarter of the pad up: half an octave
    faults.expect(abs(session.edit["temperature"] / shot["temperature"] - 2 ** 0.5) < 0.02 and session.edit["tint"] == shot["tint"],
                  f"dragging the pad up by a quarter gave {session.edit['temperature']:.0f} K, tint {session.edit['tint']}")
    drag(view, middle, QPoint(side // 8, 0))
    faults.expect(abs(session.edit["tint"] - shot["tint"] - 25) < 1.5, f"dragging across by an eighth gave tint {session.edit['tint']}")
    session.undo()
    session.undo()
    faults.expect(not session.canUndo, "a drag on the pad is more than one step of undo")
    session.redo()
    double_click(view, middle)
    faults.expect("temperature" not in session.history.edit, "a double click on the pad does not go back to as shot")
    pad = find(root, "balance-pad")
    pad.forceActiveFocus()
    QCoreApplication.sendEvent(pad, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier))
    settle()
    faults.expect(abs(session.edit["tint"] - shot["tint"] - 2) < 0.01, f"the right arrow on the pad gave tint {session.edit['tint']}")
    session.undo()
    # Pick neutral stays on: a click makes that spot gray; a press slid elsewhere makes each
    # spot gray as the pointer comes to it, all of it one step of undo; Esc ends it.
    click(view, find(root, "pick-neutral"))
    faults.expect(stage(root).property("picking") == "neutral", "Pick neutral does not start picking")
    area = box(root.findChild(QObject, "photo"))

    def at(u: float, v: float) -> QPoint:
        return QPoint(int(area.left() + area.width() * u), int(area.top() + area.height() * v))

    def spread(spot: QPoint) -> int:  # how far from gray the photo is there, in levels
        seen = view.grabWindow().pixelColor(spot)
        return max(seen.red(), seen.green(), seen.blue()) - min(seen.red(), seen.green(), seen.blue())

    # (Two spots with some color, but not more than a white balance can take out.)
    first = at(0.07, 0.35)
    second = next(spot for spot in (at(u, v) for v in (0.55, 0.65, 0.45, 0.3) for u in (0.3, 0.45, 0.6, 0.15))
                  if 25 < spread(spot) < 90)
    was = spread(first), spread(second)
    tap(view, first)
    faults.expect(spread(first) <= 4 and was[0] > 20, f"the spot clicked is {spread(first)} levels from gray (it was {was[0]})")
    faults.expect(stage(root).property("picking") == "neutral", "Pick neutral went off after one click")
    steps = session.steps
    drag(view, first, second - first, down=lambda: faults.expect(spread(first) <= 4, "pressing a spot does not make it gray at once"))
    faults.expect(spread(second) <= 4 and was[1] > 20, f"slid to another spot, that one is {spread(second)} levels from gray (it was {was[1]})")
    faults.expect(session.steps == steps + 1 and stage(root).property("fitted"),
                  f"a press slid over the photo made {session.steps - steps} steps of undo, or moved the photo")
    session.undo()
    faults.expect(spread(first) <= 4, "undo after sliding does not go back to the pick before")
    QCoreApplication.sendEvent(root, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    settle()
    faults.expect(not stage(root).property("picking"), "Esc does not end Pick neutral")
    # Its button out of sight, nothing is waiting for a pick: the card folded, another tab.
    for leave, how in ((lambda: tap(view, box(find(root, "panel:balance")).topLeft().toPoint() + QPoint(80, 14)), "folding the card"),
                       (lambda: (bring(root, "modes"), click(view, find(root, "mode:masks"))), "going to the Masks tab")):
        click(view, find(root, "pick-neutral"))
        faults.expect(stage(root).property("picking") == "neutral", "(Pick neutral did not start again)")
        leave()
        faults.expect(not stage(root).property("picking"), f"{how} leaves the photo waiting for a pick")
        show(root, "balance")
    ui.close_view(view)
    return faults


def audit_start() -> list[str]:
    """A raw photo starts at Exposure 0, with its own tone curve (edit.curve_of): the curve
    editor draws it, a double click on the Exposure slider keeps 0, Reset all brings the
    photo's own curve back, the Exposure card's A/B shows the curve straight, and the export
    is the picture on screen."""
    out = Path(tempfile.mkdtemp(prefix="photone-dark-", dir=SHELF.name))
    (out / "dark.arw").write_bytes(fixtures.exif("SONY", "ILCE-7M4", "2026:07:04 13:35:44", 1, -1.0))
    own = [[0.0, 0.0], [0.2, 0.34], [0.6, 0.72], [1.0, 1.0]]
    def matched(path: Path, half: bool):  # the chart, as a raw file whose camera showed it lighter
        meta, pixels = chart(path, half)
        return {**meta, "match": own}, pixels

    session = ui.Session([out / "dark.arw"], load=matched)
    view, root = opened(session, "tone")
    session.exporter.side = 256
    faults = Faults()
    faults.expect(session.edit["exposure"] == 0 and session.uniforms["gain"] == 1 and not session.history.edit,
                  f"the photo starts at {session.edit['exposure']} EV, not as shot")
    faults.expect(session.curve == own and session.uniforms["curveOn"] == 1, f"the photo's own curve is not on: {session.curve}")
    slider = find(root, "slider:exposure")  # kept: its wrapper going would take the track's
    at = box(slider.findChild(QObject, "thumb")).center().toPoint()  # (by its thumb: from where it is)
    drag(view, at, QPoint(60, 0))
    faults.expect(session.history.edit.get("exposure", 0) > 0, f"a drag to the right left Exposure at {session.edit['exposure']}")
    double_click(view, box(slider.findChild(QObject, "thumb")).center().toPoint())
    faults.expect(session.edit["exposure"] == 0 and "exposure" not in session.history.edit,
                  f"a double click left Exposure at {session.edit['exposure']}, not at 0")
    tick, thumb = (box(slider.findChild(QObject, n)).center().x() for n in ("tick", "thumb"))
    faults.expect(abs(tick - thumb) < 1.5, f"at 0, the slider's tick is {abs(tick - thumb):.0f} px from its thumb")
    session.setEdit("curve", edit.STRAIGHT)
    session.resetAll()
    settle()
    faults.expect(session.curve == own and not session.history.edit, f"Reset all left the curve at {session.curve}, not the photo's own")
    lighter = shown_at_full_size(view, root)
    session.setBypass("tone")
    settle()
    straight = shown_at_full_size(view, root)
    session.setBypass("")
    settle()
    faults.expect(lighter.mean() > straight.mean() + 5, f"held, the Exposure card's A/B does not show the curve straight ({lighter.mean():.0f} and {straight.mean():.0f})")
    screen = shown_at_full_size(view, root)
    session.exporter.start([str(out / "dark.arw")], {"kind": "png8", "quality": 92, "longest": 0, "folder": str(out)})
    carry_out(view, session)
    file = export.bytes_of(QImage(str(out / "dark.png"))) if (out / "dark.png").stat().st_size else np.zeros((1, 1, 3))
    off = np.abs(file.astype(int) - screen).max() if file.shape == screen.shape else 999
    faults.expect(off <= 1, f"the export of a photo with its own curve is not the picture on screen: up to {off} levels off")
    ui.close_view(view)
    return faults


def audit_tone() -> list[str]:
    """What is shown for a gray is what the arithmetic says (edit.tone_at), measured on the
    chart's gray wedge: Whites and Blacks move the two ends and leave mid gray, and with the
    switch on HDR the picture is the SDR one up to mid gray and climbs above it. The window
    here is an ordinary one, so what goes over white is cut off at white: up to there every
    step is compared, beyond it only that it is white."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session, "")
    session.surface_known(True)
    faults = Faults()
    area = box(root.findChild(QObject, "photo"))

    def wedge() -> list[int]:  # the eight grays on screen, black to white
        seen = view.grabWindow()
        return [seen.pixelColor(int(area.left() + area.width() * (step + 0.5) / 8), int(area.top() + area.height() * 0.92)).green()
                for step in range(8)]

    def level(light: float) -> int:  # what a share of the display's white is on screen
        light = min(max(light, 0.0), 1.0)
        return round(255 * (12.92 * light if light <= 0.0031308 else 1.055 * light ** (1 / 2.4) - 0.055))

    def shown(values: dict, hdr: bool) -> tuple[list[int], list[int]]:  # (on screen, by the arithmetic)
        session.history = edit.History()
        session.setEdits(values)
        session.commitEdit()
        session.setHdr(hdr)
        settle()
        peak = session.peak if hdr else 1.0
        numbers, knee = edit.uniforms(values, session._camera, True, peak), 0.9 * peak
        lights = [float(edit.tone_at(gray, numbers)) for gray in grays]
        if hdr:  # the shader's shoulder
            lights = [min(x, knee) + (peak - knee) * (1 - np.exp(-max(x - knee, 0) / (peak - knee))) for x in lights]
        return wedge(), [level(x) for x in lights]

    grays = [(step / 7) ** 2.2 for step in range(8)]  # the chart's own (decode.synthetic)
    # In HDR the colors' tools are handed the same picture as in SDR: a pin changes the same
    # pixels of the photo with the switch either way. (Lifted before them, it took hold of
    # other colors than the wheel showed: the owner, "the tone wheel pins are completely broken".)
    def photo_now() -> np.ndarray:
        return export.bytes_of(view.grabWindow().copy(area.toRect())).astype(int)

    def reached(hdr: bool) -> np.ndarray:  # the pixels a pin of the midtones changes
        pin = {**edit.PIN, "from": [0.06, 0.05], "to": [-0.02, 0.1], "range": 0.07, "feather": 20.0, "low": 0.35, "high": 0.75}
        shown({"exposure": -0.3}, hdr)
        without = photo_now()
        shown({"exposure": -0.3, "pins": [pin]}, hdr)
        return np.abs(photo_now() - without).max(axis=2) > 6

    in_sdr, in_hdr = reached(False), reached(True)
    faults.expect(in_sdr.mean() > 0.01 and (in_sdr != in_hdr).mean() < 0.01,
                  f"a pin reaches {in_sdr.mean():.1%} of the photo in SDR and {in_hdr.mean():.1%} in HDR, {(in_sdr != in_hdr).mean():.1%} of it not the same pixels")
    plain, _ = shown({"exposure": 0.0}, False)
    lifted = 0
    for stops in (-1.0, -0.5, 0.0, 0.6):
        sdr, _ = shown({"exposure": stops}, False)
        seen, wanted = shown({"exposure": stops}, True)
        told = root.findChild(QObject, "photo").property("peak")  # (what goes over white cannot be seen here)
        faults.expect(abs(told - session.peak) < 1e-6, f"in HDR the shader is told a peak of {told}, the display's is {session.peak}")
        for step, gray in enumerate(grays):
            lifted += wanted[step] > sdr[step] + 8 and wanted[step] < 252
            faults.expect(abs(seen[step] - wanted[step]) <= 2, f"HDR at {stops:+.1f} EV: gray {step} is {seen[step]} on screen, the arithmetic says {wanted[step]}")
            # (All of the picture climbs, from black on: never under its SDR level, black as it is.)
            faults.expect(seen[step] >= sdr[step] - 1 and (step > 0 or abs(seen[step] - sdr[step]) <= 1),
                          f"HDR at {stops:+.1f} EV: gray {step} is {seen[step]} where SDR has {sdr[step]}: under it, or black is not black")
    faults.expect(lifted >= 4, f"only {lifted} grays under white were measured clearly above their SDR level: the audit proves too little")
    cases = {  # what must be so of the eight grays on screen, beside being where the arithmetic says
        "whites +100": ({"whites": 100.0}, lambda g: g[5] > plain[5] + 12 and g[2] == plain[2], False),
        "whites -100": ({"whites": -100.0}, lambda g: g[7] < 225 and g[5] < plain[5] - 8 and g[2] == plain[2], False),
        "whites -60, a stop under": ({"whites": -60.0, "exposure": -1.0}, lambda g: True, False),
        "blacks -100": ({"blacks": -100.0}, lambda g: g[1] == 0 and g[2] < plain[2] - 5 and abs(g[5] - plain[5]) <= 1, False),
        "blacks +100": ({"blacks": 100.0}, lambda g: 45 <= g[0] <= 51 and g[1] > plain[1] + 12 and abs(g[5] - plain[5]) <= 1, False),
        "blacks -40": ({"blacks": -40.0}, lambda g: 0 < g[1] < plain[1] - 3, False),
        "both, and the others": ({"whites": 70.0, "blacks": -70.0, "contrast": 30.0, "shadows": 40.0, "highlights": -50.0}, lambda g: True, False),
        "the others alone": ({"contrast": -60.0, "shadows": -70.0, "highlights": 80.0, "brightness": 40.0, "exposure": -0.7}, lambda g: True, False),
        "midtones +100": ({"midtones": 100.0}, lambda g: g[3] > plain[3] + 30 and g[0] == plain[0] and g[1] <= plain[1] + 2 and abs(g[7] - plain[7]) <= 1, False),
        "midtones -100": ({"midtones": -100.0}, lambda g: g[3] < plain[3] - 30 and g[1] >= plain[1] - 2 and abs(g[7] - plain[7]) <= 1, False),
        "midtones and the others": ({"midtones": 60.0, "highlights": -80.0, "shadows": 50.0, "contrast": -40.0, "whites": 30.0}, lambda g: True, False),
        "midtones in HDR": ({"midtones": -50.0, "exposure": 0.4}, lambda g: True, True),
        "both in HDR": ({"whites": -80.0, "blacks": 60.0, "exposure": -0.5}, lambda g: g[0] > 15, True),
    }
    for name, (values, good, hdr) in cases.items():
        seen, wanted = shown(values, hdr)
        off = max(abs(a - b) for a, b in zip(seen, wanted))
        faults.expect(off <= 2, f"{name}: the grays on screen are {seen}, the arithmetic says {wanted}")
        faults.expect(good(seen), f"{name} does not do what its name says: {seen} (untouched: {plain})")
        faults.expect(all(a <= b for a, b in zip(seen, seen[1:])), f"{name}: a lighter gray is shown darker: {seen}")
    seen, _ = shown({"blacks": -100.0, "curve": [[0.0, 0.15], [1.0, 1.0]]}, False)
    faults.expect(seen[0] == seen[1] and seen[0] > 5, "what Blacks cuts off is not plain black to what comes after it: "
                  f"under a curve that lifts black, the two darkest grays are {seen[:2]}")
    back, _ = shown({"exposure": 0.0}, False)
    faults.expect(back == plain, "back on SDR and untouched, the picture is not what it was")
    # The three sliders are there, each in its panel, and a drag to the right raises its setting.
    session.history = edit.History()
    for tool, key in (("tone", "whites"), ("tone", "blacks"), ("tone", "midtones"), ("saturation", "vibrance")):
        show(root, tool)
        settle()
        view.grabWindow()
        slider = next((i for i in items(root) if i.objectName() == f"slider:{key}" and i.isVisible()), None)
        if slider is None:
            faults.append(f"the {tool} panel has no slider for {key}")
            continue
        drag(view, box(slider.findChild(QObject, "thumb")).center().toPoint(), QPoint(50, 0))  # (by its thumb)
        faults.expect(session.history.edit.get(key, 0) > 5, f"a drag to the right on the {key} slider left it at {session.edit[key]}")
    ui.close_view(view)
    return faults


def audit_scopes() -> list[str]:
    """The histogram is counted from the picture as displayed and says when pixels sit at
    white or black; Clipping paints those pixels, and only those."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session)
    faults = Faults()
    faults.expect(counted(view, session), "the histogram is never counted")
    seen = session.histogram or {"over": 0, "under": 0, "g": [0]}
    settle()
    faults.expect(seen["over"] > 0.01 and seen["under"] > 0.01 and len(seen["g"]) == 256,
                  f"the chart's white and black steps are not counted: over {seen['over']:.3f}, under {seen['under']:.3f}")
    faults.expect(find(root, "clip:white").property("lit") and find(root, "clip:black").property("lit"),
                  "the histogram's corners do not light up for pixels at white and black")
    # The graph itself is the GPU's (Counts.qml, Bars.qml): the same heights as numpy counts.
    show(root, "tone")
    view.grabWindow()
    settle()
    drawn = export.bytes_of(view.grabWindow().copy(box(find(root, "histogram-bars")).toRect()))[..., 0] > 180
    reds = drawn.mean(axis=0)  # how much of each column is red, or red under another color
    bins, spread = np.arange(1, 255), np.exp(-np.arange(-8, 9) ** 2 / (2 * 1.6 ** 2))
    smooth = {c: np.convolve(np.asarray(seen[c])[1:255], spread, "same")
              / np.convolve(np.ones(254), spread, "same") for c in "rgb"}
    want = np.interp(np.arange(drawn.shape[1]) / (drawn.shape[1] - 1) * 255, bins,
                     smooth["r"] / max(1e-9, *(v.max() for v in smooth.values())))
    off = float(np.abs(reds - want)[3:-3].mean()) if seen["g"] != [0] else 1.0
    faults.expect(off < 0.03, f"the histogram the GPU draws is not what numpy counts: {off:.3f} of its height off")
    session.setEdit("exposure", -2.0)
    counted(view, session)
    settle()
    faults.expect(session.histogram.get("over", 1) == 0 and not find(root, "clip:white").property("lit"),
                  "two stops darker, the histogram still says pixels are at white")
    session.undo()
    session.setPanel("clipping", True)
    settle()
    area = box(root.findChild(QObject, "photo"))
    shot = view.grabWindow()
    # (The paint is stripes that run: along a row of 30 pixels both of its colors are met.)
    def row(step: int) -> list:
        x, y = int(area.left() + area.width() * (step + 0.5) / 8), int(area.top() + area.height() * 0.92)
        return [shot.pixelColor(x + k, y) for k in range(30)]

    white, gray, black = row(7), row(3)[0], row(0)
    faults.expect(any(c.red() > 200 and c.green() < 90 for c in white) and any(c.red() < 90 and c.green() < 30 for c in white),
                  f"Clipping does not paint white in red stripes: {sorted({c.name() for c in white})[:4]}")
    faults.expect(any(c.blue() > 200 and c.red() < 90 for c in black) and any(c.blue() > 200 and c.red() > 150 for c in black),
                  f"Clipping does not paint black in blue stripes: {sorted({c.name() for c in black})[:4]}")
    faults.expect(gray.red() == gray.green() == gray.blue(), f"Clipping painted a mid gray: {gray.name()}")
    ui.close_view(view)
    return faults


def audit_precision() -> list[str]:
    """The photo must reach the GPU as floats: a light far below one 8-bit step comes back when
    Exposure is raised, and one above white comes back when it is lowered. (It once arrived
    as 8-bit without anything looking wrong at a glance.)"""
    def faint(path: Path, half: bool):
        rgb = np.full((64, 96, 3), 0.001, np.float32)  # a quarter of one 8-bit step
        rgb[:, 48:] = 1.5                              # and more than white
        return decode.meta("faint", 96, 64), decode.Pixels(decode.pack(rgb))  # a finished picture

    session = ui.Session([Path("faint.png")], load=faint)
    view, root = opened(session)
    area = box(root.findChild(QObject, "photo"))
    seen = {}
    for stops, x in ((5.0, 0.25), (-1.0, 0.75)):
        session.setEdit("exposure", stops)
        settle()
        seen[stops] = view.grabWindow().pixelColor(int(area.left() + area.width() * x),
                                                   int(area.center().y())).green()
    ui.close_view(view)
    faults = []
    if abs(seen[5.0] - 50) > 2:  # 0.001 x 32 = 0.032 linear
        faults.append(f"a light of 0.001 raised 5 stops shows as {seen[5.0]}, not 50: the picture is not float")
    if abs(seen[-1.0] - 225) > 2:  # 1.5 / 2 = 0.75 linear
        faults.append(f"a light of 1.5 lowered a stop shows as {seen[-1.0]}, not 225: clipped at white")
    return faults


def shoot(name: str) -> list[str]:
    size, make, after = SHOTS[name]
    session = make()
    ON_SHELF[0] = shelf(name) if name.startswith(("library", "strip")) else None
    warnings.clear()
    view = ui.make_view(session, ON_SHELF[0])
    session.surface_known(False)  # a shot is an ordinary surface: a float one cannot be grabbed
    view.resize(*size)
    root = view.rootObject()
    root.setSize(view.size())  # a window that is never shown does not resize its content
    wait(lambda: not session.photos or root.property("photoReady") or root.property("photoFailed"))
    if after:
        after(root, session)
    settle()
    view.grabWindow()  # a frame: what is on show asks for its pictures
    wait(lambda: pictures_in(root), 4000)
    if root.property("photoReady"):
        counted(view, session)  # the scopes and the wheel: let them be counted first
        settle()
    image = view.grabWindow()
    faults = [f"the interface warned: {w}" for w in warnings]
    if image.isNull():
        faults.append("nothing was rendered")
    else:
        image.save(str(OUT / f"{name}.png"))
        faults += audit_layout(view) + audit_sliders(root) + audit_photo(view, root, session)
        session.setPanel("clipping", False)  # its paint is not the picture the audits measure
        stage(root).setProperty("split", -1)  # nor is the unedited half of a split
        stage(root).setProperty("showMask", 0)  # nor a mask shown in its place
        stage(root).setProperty("picking", "")  # and a drag on the photo is to move it
        find(root, "screen:dock").setProperty("opacity", 0)  # nor the card dock over its right edge
        if session.panels["tool"] == "crop":  # nor the photo dimmed around its frame
            call(root, "go", "edit")
            settle()
        if any(i.isVisible() and i.objectName().startswith("modal:") for i in items(root)):
            faults += audit_modal(view, root)  # a question is open: nothing else may respond
        elif root.property("place") == "edit" and not name.startswith(("strip", "lens", "mark")):  # (in the library or
            # Export the photo is not on show; the filmstrip's shots show a photo that is no chart,
            # and the lens shots a chart that is bent and lit by its lens's figures, the watermark's one
            # with words on it: "(lens)" and "(mark)" measure those)
            faults += (audit_develop(view, root, session) + audit_view(view, root, session)
                       + audit_gestures(view, root, session))
    ON_SHELF[0] = None  # with its window, not when the next shot begins (see app.close_view)
    ui.close_view(view)
    return faults


def sheet(names: list[str], title: str) -> None:
    """All shots on one page: one image to look at."""
    cols, w, h, pad = 3, 590, 370, 10
    rows = -(-len(names) // cols)
    page = QImage(cols * (w + pad) + pad, rows * (h + 26 + pad) + pad, QImage.Format.Format_RGB32)
    page.fill(QColor("#111"))
    painter = QPainter(page)
    painter.setPen(QColor("#aaa"))
    for n, name in enumerate(names):
        x, y = pad + n % cols * (w + pad), pad + n // cols * (h + 26 + pad)
        shot = QImage(str(OUT / f"{name}.png")).scaled(
            w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        painter.drawImage(x, y, shot)
        painter.drawText(x, y + h + 18, name)
    painter.end()
    page.save(str(OUT / f"{title}.png"))


# What is tried with real events on a window of its own, beside the shots.
def brush_on_gpu() -> list[str]:
    """A stroke as the GPU paints it while it is made (ui/brush.frag) is the stroke numpy
    paints once it is over (edit.Painter), to a step of 255: over nothing, over an older
    stroke, rubbing out, begun before the last one's picture has come, many places at once.
    In a window that is shown (`audit_brush` says where): one that is not keeps no picture
    from a frame to the next, and the brush paints each step onto the one before."""
    session = ui.Session([Path("first.arw")], load=chart)
    view, root = opened(session)
    view.show()
    settle()
    if view.rendererInterface().graphicsApi() == QSGRendererInterface.GraphicsApi.Software:
        return []  # (no GPU to be had here: nothing to compare)
    session.addMask("brush")
    session.commitEdit()
    settle()
    brush, faults = find(root, "brush"), Faults()
    find(root, "screen:dock").setProperty("opacity", 0)  # (it lies over the photo, where the stroke is read)
    stage = box(find(root, "box:stage")).toRect().adjusted(8, 8, -8, -140)  # (clear of what lies over the photo)

    def painted_there() -> np.ndarray:
        wait(lambda: not brush.property("stepping") and not brush.property("waiting").toVariant(), 5000)
        settle()
        mix = find(root, "brush-mix")
        wide, high = int(mix.width()), int(mix.height())
        out = np.zeros((high, wide), dtype=int)
        brush.setProperty("clip", False), brush.setZ(99), mix.setVisible(True)
        origin = mix.mapToScene(QPointF(0, 0))
        for top in range(0, high, stage.height()):   # (read off the window, a tile at a time)
            for left in range(0, wide, stage.width()):
                mix.setX(stage.left() - origin.x() - left), mix.setY(stage.top() - origin.y() - top)
                view.grabWindow()
                tile = export.bytes_of(view.grabWindow().copy(stage))[:high - top, :wide - left, 0]
                out[top:top + tile.shape[0], left:left + tile.shape[1]] = tile
        mix.setVisible(False), brush.setProperty("clip", True), brush.setZ(0), mix.setX(0), mix.setY(0)
        return out

    def stroke(what: str, size: float, feather: float, erase: bool, places: list, each: int = 3) -> None:
        session.paint(1, [places[0]], size, feather, erase, True)
        for k in range(1, len(places), each):
            session.paint(1, places[k:k + each], size, feather, erase, False)
            if k % 9 == 1:
                settle()
        wide, high = session._stored_size()
        there = painted_there()
        off = np.abs(there - edit.painted(session.history.edit["masks"][0]["strokes"], wide / high).astype(int))
        faults.expect(there.max() == 255 and off.max() <= 1,
                      f"{what}: the GPU's stroke is not numpy's, up to {off.max()} of 255 off in {int((off > 1).sum())} pixels")
        session.commitEdit()

    stroke("a wave", 0.05, 0.5, False, [[0.1 + 0.8 * t, 0.5 + 0.3 * math.sin(9 * t)] for t in np.linspace(0, 1, 120)])
    faults.expect(wait(lambda: not brush.property("on"), 8000), "the stroke's picture never comes, once it is over")
    settle()
    stroke("a large soft stroke over it", 0.12, 0.9, False, [[0.2, 0.2], [0.8, 0.8], [0.8, 0.2], [0.2, 0.8]]
           + [[0.2 + 0.6 * t, 0.8 - 0.1 * t] for t in np.linspace(0, 1, 40)])
    stroke("rubbing out, begun at once", 0.03, 0.3, True, [[0.5 + 0.3 * math.cos(a), 0.5 + 0.3 * math.sin(a)] for a in np.linspace(0, 12, 200)])
    stroke("a thin stroke, thirty places at a time", 0.004, 0.5, False, np.random.default_rng(3).uniform(0.1, 0.9, (150, 2)).tolist(), each=30)
    faults.expect(wait(lambda: not brush.property("on"), 8000), "the last stroke's picture never comes")
    return faults


def audit_brush() -> list[str]:
    """`brush_on_gpu`, in a process of its own whose window is shown where nobody sees it (Qt's
    offscreen platform, with the GPU all the same)."""
    done = subprocess.run([sys.executable, __file__, "--brush"], capture_output=True, text=True, timeout=300, check=False,
                          env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "QT_QUICK_BACKEND": "rhi"})
    said = [line[7:] for line in done.stdout.splitlines() if line.startswith("FAULT: ")]
    return said or ([] if done.returncode == 0 else [f"the brush check ended badly: {done.stderr.strip()[-300:]}"])


AUDITS = {"(keys)": audit_stepping, "(held)": audit_held, "(dock)": audit_dock, "(column)": audit_column, "(wheel)": audit_wheel,
          "(wheel data)": audit_wheel_data, "(dots)": audit_dots_place, "(pixels)": audit_pixels, "(colors data)": audit_colors_data, "(pins)": audit_pins,
          "(pin panel)": audit_pin_panel,
          "(masks)": audit_masks, "(brush)": audit_brush, "(mask wheel)": audit_mask_wheel, "(masks data)": audit_masks_data, "(moves)": audit_moves, "(scroll)": audit_scroll, "(resume)": audit_resume, "(strip)": audit_strip, "(strip wheel)": audit_strip_wheel, "(local)": audit_local, "(fourth)": audit_fourth, "(mono)": audit_mono, "(detail)": audit_detail, "(lens)": audit_lens, "(fringes)": audit_fringes, "(mark)": audit_mark, "(grain)": audit_grain, "(look)": audit_look, "(presets)": audit_presets, "(lut)": audit_lut, "(table)": audit_table, "(menu)": audit_menu, "(color tools)": audit_color_tools,
          "(test chart)": audit_test_chart, "(library)": audit_library, "(export)": audit_export,
          "(frame)": audit_frame, "(crop)": audit_crop, "(slider)": audit_slider, "(tools)": audit_tools,
          "(curve)": audit_curve, "(balance)": audit_balance, "(start)": audit_start, "(tone)": audit_tone, "(scopes)": audit_scopes,
          "(precision)": audit_precision, "(checkpoints)": audit_checkpoints}


def chosen(args: list[str]) -> tuple[list[str], list[str], str]:
    """(the shots, the audits, the sheet's name) that the arguments ask for; all with none."""
    # A group: `shots photo` is every shot named photo or photo-…, with its own sheet.
    exact = {a[1:] for a in args if a.startswith("=")}
    groups = [a for a in args if not a.startswith("=") and any(n.startswith(a + "-") for n in SHOTS)]
    audits = [a for a in AUDITS if a in args]
    unknown = [a for a in args if a.lstrip("=") not in SHOTS and a not in groups and a not in AUDITS]
    if unknown:
        sys.exit(f"shots: unknown {unknown}; try --list")
    names = [n for n in SHOTS if n in exact or n in args
             or any(n == g or n.startswith(g + "-") for g in groups)]
    title = groups[0] + "-sheet" if len(groups) == 1 and len(args) == 1 else "sheet"
    return (names, audits, title) if args else (list(SHOTS), list(AUDITS), title)


def main() -> int:
    # Should it ever hang: say where, and end, rather than sit there (it did once, unexplained).
    # (Ten minutes: all of it takes three by now, and it once ended here in the middle of an
    # audit that was only slow.)
    faulthandler.dump_traceback_later(600, exit=True)
    args = sys.argv[1:]
    if "--list" in args:
        print("\n".join([*SHOTS, *AUDITS]))
        return 0
    if "--brush" in args:  # (audit_brush's own process: a GPU, and a window nobody sees)
        modern = QSurfaceFormat()
        modern.setVersion(4, 5), modern.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        QSurfaceFormat.setDefaultFormat(modern)
        application = QGuiApplication(sys.argv[:1])  # noqa: F841 - must outlive the view
        faults = brush_on_gpu()
        print("\n".join(f"FAULT: {fault}" for fault in faults))
        return len(faults)
    names, audits, title = chosen(args)
    OUT.mkdir(parents=True, exist_ok=True)
    application = QGuiApplication(sys.argv[:1])  # noqa: F841 - must outlive the views
    qInstallMessageHandler(lambda kind, context, text: warnings.append(text))
    results = [(name, shoot(name)) for name in names] + [(name, AUDITS[name]()) for name in audits]
    for name, faults in results:
        where = OUT / (name + '.png') if name in SHOTS else ""
        print(f"{'FAULT' if faults else 'ok   '} {name:<14} {where}")
        for fault in faults:
            print(f"        - {fault}")
    if len(names) > 1:
        sheet(names, title)
        print(f"sheet          {OUT / (title + '.png')}")
    return sum(1 for _, faults in results if faults)


if __name__ == "__main__":
    sys.exit(main())
