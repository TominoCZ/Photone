# deps.md — what a machine needs to run Photone

Everything that has to be installed on the system, outside the Python packages `uv` fetches from
`pyproject.toml` (PySide6 with Qt inside it, rawpy with LibRaw inside it, numpy). Package names
are Ubuntu/Debian. **Keep this file current:** a change that calls a new program, loads a new
library or relies on a new service adds a row here in the same change.

## Required

| Dependency | Package | Used for | Without it |
|---|---|---|---|
| `uv` | astral.sh installer (`curl -LsSf https://astral.sh/uv/install.sh \| sh`) | Running the app; it fetches Python 3.13 and the packages itself | `run.sh` stops and says so |
| A desktop session, X11 or Wayland | any (`plasma-desktop`, `gnome-shell`, …) | The window | Does not start |
| What Qt's X11 plugin loads | `libxcb-cursor0`, `libxkbcommon-x11-0` | The window on X11 | On X11: "could not load the Qt platform plugin xcb" |
| OpenGL 3.3 / OpenGL ES 3.0 or Vulkan | the GPU vendor's driver (`nvidia-driver-*`, Mesa) | The picture pipeline (a shader) | Qt falls back to software drawing, which has no shaders: the photo is not shown |

On Windows `install.bat` does everything (it fetches uv for the build and removes it again).
The Windows build (`./build-windows.sh` or `build-windows.bat`, the folder `bin/`) brings everything with it and needs
Windows 10 version 1903 or later (Qt loads the system's `icuuc.dll`) and a GPU with Direct3D 11.

## Optional

| Dependency | Package | Used for | Without it |
|---|---|---|---|
| Vulkan with an HDR surface (`VK_EXT_swapchain_colorspace`) on a Wayland desktop with HDR turned on for the display | the GPU driver (`nvidia-driver-*`, `mesa-vulkan-drivers`); Plasma 6 or a recent GNOME | The HDR side of the SDR/HDR switch | The switch says HDR is not available and Photone stays in SDR |
| `update-desktop-database` | `desktop-file-utils` | `./install.sh`: telling the desktop about the new menu entry at once | The entry shows up when the desktop next reads its menu |
| `kscreen-doctor` | `libkscreen-bin` (part of Plasma) | Reading the display's HDR headroom, for the brightest the photo may get | A headroom of 2× white is assumed |

## Pinned versions

| Package | Pinned at | Why |
|---|---|---|
| rawpy | `==0.28.0` (LibRaw 0.22.2 inside) | Neural Denoise's net is trained on pictures developed by this LibRaw, in Neuro, from data built by Denoiser's `pairs_v2.py`. Its input in Photone must be developed the same way. A LibRaw with a changed demosaic or colour step would feed the net a picture it wasn't trained on. Neuro pins the same version, and its GPU copy of LibRaw's development matches 0.22.2 bit for bit. Upgrade both together, and only after Neuro's check against rawpy passes on the new version. |

## Development only (not shipped)

| Dependency | Package | Used for |
|---|---|---|
| Python 3.11 or later, with `uv` or `pip` | `python3`, `python3-pip` | `./build-windows.sh` / `build-windows.bat` (`scripts/build_windows.py`): making the Windows build in `bin/` |
| A running desktop session (Wayland or X11) with a GPU | — | `./run.sh shots`: it renders the real window without showing it, and that needs the graphics driver |
| mingw | `gcc-mingw-w64-x86-64` | `scripts/launcher/build.sh`: building `Photone.exe` again after `launcher.c` or the icon changed (the built file is in the repository) |
| `ruff` | fetched by `uv run --with ruff` | `./run.sh lint` |
