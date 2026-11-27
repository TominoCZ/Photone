"""photone: the editor's window.

    photone [photo ...]          open the window, on these photos
    photone --hdr [photo ...]    with the SDR | HDR switch on HDR (it is remembered otherwise)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def tell_where_stuck() -> None:
    """Should the program ever hang: `kill -USR1 <its process>` then writes down what every
    one of its threads was doing, to `photone-stuck.txt` in the temporary folder, without
    ending it. (It hung twice for the owner with nothing to say where. Not on Windows, which
    has no such signal.)"""
    import faulthandler
    import signal
    import tempfile
    from pathlib import Path

    if hasattr(signal, "SIGUSR1"):
        out = open(Path(tempfile.gettempdir()) / "photone-stuck.txt", "w", encoding="utf-8")  # noqa: SIM115 - kept open: written to from a signal
        faulthandler.register(signal.SIGUSR1, file=out, all_threads=True)


def arguments() -> argparse.Namespace:
    tell_where_stuck()  # (first of all: from here on a hang can be asked about)
    parser = argparse.ArgumentParser(prog="photone", description="A raw photo editor.")
    parser.add_argument("photos", nargs="*", type=Path, help="photos to open")
    parser.add_argument("--hdr", action="store_true", help="start with the switch on HDR")
    args = parser.parse_args()
    missing = [str(p) for p in args.photos if not p.is_file()]
    if missing:
        sys.exit(f"photone: no such file: {', '.join(missing)}")
    args.photos = [p.resolve() for p in args.photos]
    return args


def open_catalog():
    """The catalog, or None and why: without it Photone still runs, and says edits are lost."""
    from photone import catalog

    try:
        return catalog.Catalog(catalog.default_path()), ""
    except catalog.Unusable as e:
        return None, str(e).capitalize() + "."


def begin(ui):
    """Qt's application, named, with its surface chosen."""
    from PySide6.QtCore import QLoggingCategory
    from PySide6.QtGui import QFontDatabase, QGuiApplication, QIcon

    # Qt tells the desktop's portal the program's name, and the portal refuses: it has no
    # `photone.desktop` to go by (that comes with the install script), and started from a
    # terminal the connection already goes by the terminal's name. Nothing here depends on
    # it, so the line it printed at every start is left out. (QT_LOGGING_RULES brings it back.)
    QLoggingCategory.setFilterRules("qt.qpa.services.warning=false")
    application = QGuiApplication(sys.argv[:1])
    ui.choose_surface(application.platformName())  # before the window: it decides then
    application.setApplicationName("Photone")
    application.setDesktopFileName("photone")
    application.setWindowIcon(QIcon(str(ui.UI / "icon.svg")))
    # Its own typeface, the same on every system (ui/fonts, with its license): for text that
    # names a family (Theme.font) and, at the desktop's size, for text that names none.
    for file in sorted((ui.UI / "fonts").glob("*.ttf")):
        QFontDatabase.addApplicationFont(str(file))
    font = application.font()
    font.setFamily("Noto Sans")
    application.setFont(font)
    return application


def quit_on_ctrl_c(application) -> None:
    """Ctrl+C in the terminal ends the program as closing its window does, edit saved. (Python
    only sees the signal when it runs, and while Qt waits for events it does not: a timer lets
    it look in now and then. The owner: "doing ctrl+c in the terminal doesnt kill the app".)"""
    import signal

    from PySide6.QtCore import QTimer

    signal.signal(signal.SIGINT, lambda *_: application.quit())
    awake = QTimer(application, interval=200)
    awake.timeout.connect(lambda: None)
    awake.start()


def show(ui, session, library, presets=None):
    try:
        view = ui.make_view(session, library, presets)
    except RuntimeError as e:
        sys.exit(f"photone: {e}")
    library.attend(session)  # (a photo whose file is gone is shown as missing in the filmstrip)
    ui.watch_surface(view, session)
    view.show()
    ui.ask_desktop_for_headroom(session, view.screen().name())
    return view


def main() -> None:
    args = arguments()
    from photone import decode

    # First of all: the worker starts, and decodes the first photo, while Qt loads and starts.
    workers = decode.Workers(args.photos[0] if args.photos else None)

    # The desktop's own file dialog, on any desktop, through its portal. (Plasma names its
    # own Qt theme here, which the Qt that comes with PySide6 does not have.)
    if sys.platform == "linux" and os.environ.get("QT_QPA_PLATFORMTHEME", "kde") == "kde":
        os.environ["QT_QPA_PLATFORMTHEME"] = "xdgdesktopportal"

    from PySide6.QtCore import QSettings

    from photone import app as ui

    application = begin(ui)
    kept, trouble = open_catalog()
    settings = QSettings("Photone", "Photone")
    session = ui.Session(args.photos, load=workers.load,
                         hdr=True if args.hdr else None,
                         settings=settings, catalog=kept)
    session.say(trouble)
    session.ahead = workers.ahead
    library = ui.Library(kept, settings=settings)
    # (Beside the catalog; with no catalog to write to, presets last as long as the program.)
    presets = ui.Presets(kept.path.with_name("presets.json") if kept else None)
    session.say(presets.error)
    view = show(ui, session, library, presets)
    library.rescan()  # what is new in the library's folders, found while the window is up
    quit_on_ctrl_c(application)
    code = application.exec()
    session.flush()  # the edit, should it not have been written yet
    library.close()
    ui.close_view(view)
    workers.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
