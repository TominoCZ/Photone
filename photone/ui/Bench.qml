import QtQuick
import "."

// The bench: where a photo is rendered at its full size for an export. Never on show. The
// photo (exporter.job) stands behind a window as large as one tile, moved so that the tile to
// be rendered shows through; that is read back and handed to the exporter, tile after tile,
// through the same shader as the photo on screen, so an export is what the screen shows.
Item {
    id: bench
    readonly property var job: exporter.job   // empty while there is nothing to export
    property int at: -1                       // the tile being rendered
    property int within: -1                   // the mask whose own picture is being rendered, or none
    visible: false
    clip: true
    width: 1; height: 1
    // Put tile `i` in the window: [x, y, wide, high, pass]. Pass 0 is the picture in 8 bits;
    // 1 and 2 carry 16 bits a color between them (pipeline.frag, scope 6 and 7).
    function aim(i) {
        const tile = job.tiles[i]
        at = i
        width = tile[2]; height = tile[3]
        render.x = -tile[0]; render.y = -tile[1]
        // (3 and on: a mask's own picture in 16 bits, and its weight: export.tiles.)
        const turn = tile[4], part = turn < 3 ? -1 : (turn - 3) % 3
        within = turn < 3 ? -1 : Math.floor((turn - 3) / 3)
        render.scope = turn < 3 ? [0, 6, 7][turn] : [6, 7, 0][part]
        render.onlyMask = part === 2 ? -job.layers[within].mask : 0
    }
    function go(i) {
        if (!job.tiles) return   // (stopped)
        if (i >= job.tiles.length) return exporter.rendered()
        aim(i)
        if (!exporter.grabTile(bench)) exporter.rendered()
    }
    Connections { target: exporter; function onTileTaken() { bench.go(bench.at + 1) } }
    Image {
        id: full
        source: bench.job.source || ""
        asynchronous: true; cache: false; visible: false
        smooth: !bench.job.exact && !developed.on
        onStatusChanged: {
            // (Qt reads a render back only from a window that is on screen; in the checks,
            // which have none, scripts/shots.py takes the tiles itself.)
            if (status === Image.Error) exporter.rendered()
            else if (status === Image.Ready && bench.job.source && bench.Window.window.visible) bench.go(0)
        }
    }
    Develop { id: developed; mosaic: full; figures: bench.loaded && bench.job.mosaic ? bench.job.mosaic : null }   // (a raw file's mosaic: developed as on screen)
    Image { id: brushed; source: bench.job.painted || ""; asynchronous: false; cache: false; visible: false }
    Image { id: table; source: bench.job.lut || ""; asynchronous: false; cache: false; visible: false }
    readonly property bool loaded: full.status === Image.Ready && job.source !== undefined
    Pipeline {
        id: render
        width: bench.job.width || 1; height: bench.job.height || 1
        source: developed.on ? developed.picture : full; painted: bench.job.painted ? brushed : full
        lutPicture: bench.job.lut && table.status === Image.Ready && table.sourceSize.width > 1 ? table : null
        follows: true
        edit: !bench.loaded ? null : bench.within >= 0 && bench.job.layers ? bench.job.layers[bench.within].numbers : bench.job.uniforms || session.uniforms
        peak: 1; hdrLift: 0; linearOut: 0; clipping: 0
        Local { over: render; layers: bench.job.layers || [] }
    }
}
