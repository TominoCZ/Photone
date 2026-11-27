import QtQuick
import "."

// The photo, and how it is looked at: fitted to the view, or at a zoom level and panned.
// `zoom` is screen pixels per pixel of the full-size photo, whichever size is loaded, so 1 is
// 100 %. `center` is the point of the photo (0..1 both ways) in the middle of the view.
Item {
    id: view
    clip: true

    // There is a picture of the current photo to draw: its first picture is in (or was held
    // already, beside the photo before: then from the frame after the step on).
    readonly property bool have: session.source !== "" && String(pixels.source) === session.source && pixels.status === Image.Ready
    property bool missing: false   // the photo's file is gone: nothing of it is shown, though a picture of it may still be held
    readonly property bool ready: have && !session.failed && !missing && session.fullWidth > 0
    Connections { target: session; function onPhotoChanged() { view.brushedOnce = false; view.sharpFor = "" } }

    property bool fitted: true
    property real zoom: 1
    property point center: Qt.point(0.5, 0.5)
    // (More while the Crop tool is open: room round the picture to take hold of it and turn it.)
    readonly property real margin: session.panels.tool === "crop" ? 48 : 14
    readonly property real fitZoom: ready
        ? Math.min((width - 2 * margin) / session.fullWidth, (height - 2 * margin) / session.fullHeight) : 1
    readonly property real shown: fitted ? fitZoom : zoom          // the zoom in effect
    readonly property real minZoom: Math.min(fitZoom, 0.1)
    readonly property real maxZoom: 8

    function fit() { still(); fitted = true; center = Qt.point(0.5, 0.5) }
    function actual() { glideTo(1, width / 2, height / 2, false) }

    // The wheel, Fit and 1:1 glide to their zoom: very short, toward wherever was asked for
    // last (a second notch during the glide goes on from where the picture is, to the step
    // after the one the first was going to). Anything else that sets the zoom or moves the
    // photo does it at once and ends a glide under way. (The owner: "a VERY subtle and fast
    // interpolation ... always tween toward the latest desired zoom level", "zoom levels
    // forced by anything else need to be instant".)
    property var going: null   // the glide under way: from zoom z0 to z1, about a point, or to Fit
    property real glide: 1     // how far along it is, 0..1
    readonly property real aim: going ? going.z1 : shown   // the zoom it will have
    NumberAnimation { id: zooming; target: view; property: "glide"; from: 0; to: 1; duration: 85; easing.type: Easing.OutCubic; onFinished: view.landed() }
    function glideFit() { if (!fitted) glideTo(fitZoom, width / 2, height / 2, true) }
    function still() { zooming.stop(); going = null }
    function land() { if (zooming.running) zooming.complete() }   // (for the audits: there at once)
    // While no photo is shown (the next one still on its way, or unreadable) it has no size:
    // a zoom or a move then measured by it put the middle of the view on a corner, at 100 %,
    // and the photo came there. Only Fit is taken; the rest waits for the photo.
    function glideTo(z, x, y, toFit) {
        if (!ready) return toFit ? fit() : undefined
        zooming.stop()
        // (Out of Fit first, looking the same: from here on it is a zoom and a place.)
        if (fitted) { zoom = fitZoom; fitted = false }
        z = toFit ? fitZoom : Math.max(minZoom, Math.min(maxZoom, z))
        going = { z0: zoom, z1: z, x: x, y: y, u: (x - photo.x) / photo.width, v: (y - photo.y) / photo.height, fit: toFit, from: center }
        zooming.start()
    }
    onGlideChanged: {
        if (!going || !zooming.running) return
        const g = going, z = g.z0 * Math.pow(g.z1 / g.z0, glide)
        if (!g.fit) return put(z, g.x, g.y, g.u, g.v, g.z1 < g.z0)
        zoom = z
        center = Qt.point(g.from.x + (0.5 - g.from.x) * glide, g.from.y + (0.5 - g.from.y) * glide)
    }
    function landed() {
        const g = going
        going = null
        if (!g) return
        if (g.fit) { fitted = true; center = Qt.point(0.5, 0.5) } else put(g.z1, g.x, g.y, g.u, g.v, g.z1 < g.z0)
    }

    // The wheel zooms by steps, the same ones every time, so that a zoom once left is come
    // back to by turning the other way (the owner: "i cant go back to 1:1 just by scrolling
    // around": the zoom used to be multiplied a little at every notch). Halves and thirds
    // below 1:1, more above, and Fit among them wherever it lies.
    readonly property var steps: [1 / 16, 1 / 12, 1 / 8, 1 / 6, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 1, 1.5, 2, 3, 4, 6, 8]
    property real turned: 0   // of the wheel, and not yet a whole notch (a touchpad sends many small parts)
    function stepZoom(by, x, y) {
        // Those that can be gone to: within the limits, and none a hair's breadth from Fit.
        const near = z => Math.abs(z / fitZoom - 1) < 0.04
        const all = steps.filter(z => z >= minZoom * 0.999 && z <= maxZoom && !near(z)).concat([fitZoom]).sort((a, b) => a - b)
        // (From the zoom it is going to, if it is on its way: two notches are two steps.)
        const next = by > 0 ? all.find(z => z > aim * 1.001) : all.reverse().find(z => z < aim * 0.999)
        if (next === undefined) return
        // (The size of Fit is one of the steps, but the wheel never puts the view on Fit: that
        // moved the photo to the middle by itself; the owner: "scrolling shouldnt actually ever
        // call Fit on its own". Fit is its button, a middle click or a double click.)
        glideTo(next, x, y, false)
    }

    // How far the photo can be moved, at any zoom: until one of its corners is in the middle
    // of the view, and no further. That is, the middle of the view is always on the photo.
    function held(c) { return Math.max(0, Math.min(1, c)) }

    // Zoom so that the point of the photo under (x, y) of the view stays under it. Zooming
    // out, the photo is also gathered in: no edge of it comes into the view while it is
    // larger than the view, and it stands in the middle once it is smaller. (Without that it
    // slid off to one side on the way out and jumped back at Fit; the owner: "the image
    // jumps around".)
    function zoomTo(z, x, y) {
        still()
        z = Math.max(minZoom, Math.min(maxZoom, z))
        put(z, x, y, (x - photo.x) / photo.width, (y - photo.y) / photo.height, z < shown)
    }
    // At that zoom, with the point (u, v) of the photo under (x, y) of the view; `out`: gathered in.
    // Out as in: about the pointer, and nothing else moves the photo (the owner: "the photo
    // should zoom out at the point where my cursor is"). It used to be gathered in on the
    // way out, so that no edge came into the view while the photo was larger than it:
    // that was for the wheel's step onto Fit, which put the photo in the middle with a
    // jump; the wheel no longer goes onto Fit, and the gathering was itself a shift.
    function put(z, x, y, u, v, out) {
        if (!ready) return
        const w = session.fullWidth * z, h = session.fullHeight * z
        const left = x - u * w, top = y - v * h
        fitted = false
        zoom = z
        center = Qt.point(held((width / 2 - left) / w), held((height / 2 - top) / h))
    }
    // Moving a fitted photo takes it out of Fit, at the size it has (Fit puts it back).
    function panBy(dx, dy) {
        still()
        if (!ready) return
        if (fitted) { zoom = fitZoom; fitted = false }
        center = Qt.point(held(center.x - dx / photo.width), held(center.y - dy / photo.height))
    }

    // Two pictures of the photo on the GPU. The first (half size for a raw file, with mipmaps)
    // is what a view of the whole photo is drawn from. The sharp one (full size) is fetched
    // straight after it and used as soon as the view is closer than the first can show, so
    // crossing that line costs nothing: both are already there.
    readonly property var picture: pixels   // for the scopes' renders
    // The open photo's LUTs as the picture the shader reads them from, once it is there; else none.
    readonly property var lut: lutPixels.status === Image.Ready && lutPixels.sourceSize.width > 1 && String(lutPixels.source) !== "" ? lutPixels : null
    // (While another is being fetched there is none: the numbers that say where each LUT lies
    // in it are already the new picture's. It is another only when a LUT is added or taken
    // away, not when their order, strength or eye changes.)
    Image { id: lutPixels; source: session.lutSource; asynchronous: true; cache: false; visible: false }
    readonly property var sharpest: sharpReady ? (developed.on ? developed.picture : sharpPixels) : pixels   // every pixel of the photo, once it is there (the grain's loupe)
    readonly property bool sharpThere: sharpReady
    // The first pictures of the open photo and of the photos either side of it (`session.held`),
    // each kept by one of these three while it is wanted: a step to a photo beside the open one
    // draws its picture, already on the GPU, in the first frame after the key, where its
    // thumbnail stood in for some 40 ms (the owner: "I would like switching between photos ...
    // to be instant"). The one that has the open photo's is `pixels`. Each draws one pixel just
    // outside this view's edge, so that Qt sends it to the GPU now, not at the step.
    // ponytail: two first pictures more on the GPU, 87 MB each for 33 MP (half size, floats,
    // mipmaps), more for a JPEG, which has only its full size. Fast stepping outruns them.
    readonly property string wanted: [session.source, ...session.held].join(" ")
    onWantedChanged: if (built) hold()
    // (Not before the three are whole: a source set while `asynchronous` was not yet, was
    // asked for at once, and the window, waiting for Qt's image thread, never opened.)
    property bool built: false
    Component.onCompleted: { built = true; hold() }
    function hold() {
        const want = wanted.split(" ").filter(a => a !== ""), holds = [held0, held1, held2]
        const free = holds.filter(h => !want.includes(String(h.source)))
        for (const a of want) if (!holds.some(h => String(h.source) === a)) free.shift().source = a
        for (const h of free) h.source = ""
    }
    readonly property Image pixels: [held0, held1, held2].find(h => String(h.source) === session.source) || held0
    component Held: Image { x: -2; y: -2; width: 1; height: 1; asynchronous: true; cache: false; mipmap: true }
    Held { id: held0 }
    Held { id: held1 }
    Held { id: held2 }
    Image {
        id: sharpPixels
        // Asked for in the frame after the first picture is shown: Qt fetches one at a time,
        // in no promised order, and the first must not wait behind this one; and the two sent
        // to the GPU in one frame made it 110 to 125 ms long, the photo shown only after it.
        // (Its decoding runs regardless.)
        source: view.sharpFor === session.source ? session.fullSource : ""
        asynchronous: true; cache: false; visible: false
        smooth: !developed.on   // (a mosaic is read value by value)
    }
    // A raw file's full size comes as its mosaic, and is developed here (decode.figures).
    Develop {
        id: developed
        mosaic: sharpPixels
        figures: sharpPixels.status === Image.Ready ? session.mosaicOf(String(sharpPixels.source)) : null
        // (Its first pixels read back and compared to LibRaw's: where they differ, the file is
        // developed by LibRaw instead.)
        onOnChanged: if (on) session.checkDeveloped(picture, String(sharpPixels.source))
    }
    property string sharpFor: ""   // the photo (its first picture) whose sharp one is asked for
    FrameAnimation { running: view.have && view.sharpFor !== session.source; onTriggered: view.sharpFor = session.source }
    // Sharpening and noise reduction done once over every pixel of the photo, and kept: seen
    // from further than 100 %, the photo is drawn from this (pipeline.frag, `detailKept`),
    // made smaller through its mipmaps as an export made smaller would be. Done there per pixel
    // of the screen, they were left out, the neighbors read being a coarser picture's (the
    // owner: "it can be VERY visible and i find myself zooming out a lot to see it"). A frame
    // that changes anything else (a slider, a pan) does not do them again.
    // From 100 % in the photo does them itself as it did: sharpened after it is enlarged,
    // an edge stays as steep as it was (done before, the lens audit's wedge went soft).
    // ponytail: while further than 100 %, one more picture of the photo's full size on the GPU
    // (16-bit floats and mipmaps: 256 MB at 24 MP), made anew on each going out past 100 % and
    // at each change of a Detail slider. The upgrade, should a pass over all of it be too slow
    // on a large file, is to do it over the part of the photo in view only.
    // (Not until the full-size picture is there: until then the photo is as it was. Nor while
    // all of them are at 0, as a photo starts: then there is nothing to keep, and no 350 MB
    // and depth buffer at 33 MP; the owner: "they should not take up VRAM at zero either".)
    Pipeline {
        id: detailed
        objectName: "detailed"
        readonly property bool on: layer.enabled
        // (The edit's other numbers stay as they first were: a change of one of them must not
        // make it draw all of this again. None of them is read: `detailKept` 2 stops before.)
        follows: false
        visible: false; blending: false
        width: view.sharpest.sourceSize.width; height: view.sharpest.sourceSize.height
        source: view.sharpest; sharp: view.sharpest; useSharp: 1; detailKept: 2
        frame: Qt.matrix4x4(); lensOn: 0   // (the stored picture as it is)
        detailX: photo.detailX; detailY: photo.detailY
        sharpen: photo.sharpen; sharpenRadius: photo.sharpenRadius; sharpenMask: photo.sharpenMask
        denoise: photo.denoise; denoiseColor: photo.denoiseColor; denoiseFar: photo.denoiseFar
        layer.enabled: view.ready && view.visible && photo.width < session.fullWidth && view.sharpest.sourceSize.width >= session.wide - 1
                       && (photo.sharpen > 0 || photo.denoise > 0 || photo.denoiseColor > 0)
        layer.textureSize: Qt.size(width, height); layer.format: ShaderEffectSource.RGBA16F
        layer.mipmap: true; layer.smooth: true
    }
    // Until the photo's first picture is there (0.03 s for a neighbor, 0.3 s for a photo that
    // was not decoded ahead), its thumbnail stands in for it, fitted as the photo will be:
    // the photo is on screen the moment it is asked for, soft, and sharp a moment later.
    // (The owner: "it takes each photo about a second". An empty view for that long felt so.)
    property string standIn: ""   // the address of the thumbnail's file, or none
    // (Where and as large as the photo will be, also when the view is not on Fit: the zoom and
    // the place stay from one photo to the next, and a thumbnail fitted to the view was then
    // larger or smaller than the photo that took its place; the owner: "the lower quality
    // image is bigger than the sharp one". The photo's size is not known yet: its longer side
    // is taken to be that of the photo before it, which is so for photos of one camera.)
    property real longest: 0   // of the last photo shown, in its own pixels
    onReadyChanged: if (ready) longest = Math.max(session.fullWidth, session.fullHeight)
    Image {
        id: standing
        objectName: "stand-in"
        visible: !view.ready && !session.failed && !view.missing && view.standIn !== ""
        readonly property bool placed: !view.fitted && view.longest > 0 && implicitWidth > 0
        readonly property real by: placed ? view.longest * view.zoom / Math.max(implicitWidth, implicitHeight) : 1
        x: placed ? Math.round(view.width / 2 - view.center.x * width) : view.margin
        y: placed ? Math.round(view.height / 2 - view.center.y * height) : view.margin
        width: placed ? implicitWidth * by : view.width - 2 * view.margin
        height: placed ? implicitHeight * by : view.height - 2 * view.margin
        source: view.standIn; fillMode: Image.PreserveAspectFit; cache: false
    }
    // The photo's painted masks, as the picture the shader reads them from (session.painted).
    // (Once it has been there it stays in use while the next one is fetched: the picture it
    // had is kept meanwhile. Going by "is it ready now", every stroke's new picture put the
    // photo's own colors in the masks' place for a frame; the owner: "the entire canvas
    // flickers crazily".)
    property bool brushedOnce: false
    readonly property bool brushedThere: brushedOnce && session.paintedSource !== ""
    readonly property var painted: brush.on ? brush.item.mix : brushedThere ? brushed : pixels
    Image {
        id: brushed
        source: session.paintedSource; asynchronous: true; cache: false; visible: false; retainWhileLoading: true
        onStatusChanged: {
            if (status !== Image.Ready) return
            view.brushedOnce = true
            if (String(source) !== brush.held) brush.over()   // the picture with the stroke in it has come
        }
        onSourceChanged: if (String(source) === "") { view.brushedOnce = false; brush.over() }
    }
    // A stroke while it is made is painted by the GPU (brush.frag), not into that picture: a
    // step each frame, of up to sixteen stretches of the way, onto the picture of the steps
    // before. `painted` is then the masks' picture with that in its mask's place (brushed.frag),
    // until the picture that has the stroke in it has come: that one is made once, when the
    // stroke is over, and not on this thread. (It was made at every move of the brush, at a
    // cost that grew with the brush: a large one moved fast ran at 50 frames a second.)
    // ponytail: one stroke's mask at a time. A stroke begun on another mask before the picture
    // of the last has come (some hundredths of a second) shows the last one late, not at once.
    Loader {
        id: brush
        objectName: "brush"
        // (On show, though nothing of it is seen: Qt paints a step only for an item that is,
        // and not for one that is clear. So one pixel of it, just outside this view's edge.)
        x: -2; y: -2; width: 1; height: 1; clip: true
        active: false                  // from the first stroke on: its pictures are some 60 MB of the GPU's
        property bool on: false        // `painted` is the GPU's
        property string held: ""       // the picture's address when the stroke under way began
        property var waiting: []       // what is not painted yet (session.brushed), oldest first
        property bool stepping: false  // a step is being painted
        property bool wanted: false    // a stroke is not in the picture yet
        property int steps: 0          // how many were (the checks wait for them)
        property size raster: Qt.size(16, 16)
        property real channel: 0
        property real erase: 0
        function over() { on = wanted = false; waiting = [] }
        function add(stroke) {
            active = wanted = true
            if (stroke.begin) held = session.paintedSource
            // (More of the stroke that waits: one with it, so that a step takes what it can hold.)
            if (!stroke.begin && waiting.length) waiting[waiting.length - 1].steps.push(...stroke.steps)
            else waiting.push(stroke)
            step()
        }
        function step() {
            if (stepping || !waiting.length || !item) return
            const stroke = waiting[0], now = stroke.steps.splice(0, 16), to = item.painter
            if (!stroke.steps.length) waiting.shift()
            // (On what was painted here, where that is not in the picture yet; else on the picture.)
            to.begin = !stroke.begin ? 0 : on && stroke.channel === channel && raster.width === stroke.wide ? 2 : 1
            to.erased = erase
            stroke.begin = false
            raster = Qt.size(stroke.wide, stroke.high); channel = stroke.channel; erase = stroke.erase ? 1 : 0
            to.radius = stroke.radius; to.core = stroke.core; to.count = now.length
            const places = [[], [], [], []], turns = new Array(16).fill(0)
            for (let i = 0; i < 16; i++) {
                for (let k = 0; k < 4; k++) places[Math.floor(i / 4)][k * 4 + i % 4] = i < now.length ? now[i][k] : 0
                turns[(i % 4) * 4 + Math.floor(i / 4)] = i < now.length && now[i][4] ? 1 : 0
            }
            to.s0 = Qt.matrix4x4(...places[0]); to.s1 = Qt.matrix4x4(...places[1]); to.s2 = Qt.matrix4x4(...places[2]); to.s3 = Qt.matrix4x4(...places[3])
            to.turns = Qt.matrix4x4(...turns)
            stepping = true
            item.kept.scheduleUpdate()
        }
        // (Not `on` where the picture came while this step was painted: it has the stroke.)
        function stepped() { stepping = false; steps++; on = wanted; step() }
        onLoaded: step()
        sourceComponent: Item {
            property alias painter: painter
            property alias kept: kept
            property alias mix: mix
            ShaderEffect {
                id: painter
                width: brush.raster.width; height: brush.raster.height
                property var was: kept
                property var base: brushed
                property real begin: 0
                property real channel: brush.channel
                property real count: 0
                property real radius: 1
                property real core: 0
                property real erase: brush.erase
                property real erased: 0   // of the stroke before, which `begin` 2 puts together
                property size size: brush.raster
                property matrix4x4 s0
                property matrix4x4 s1
                property matrix4x4 s2
                property matrix4x4 s3
                property matrix4x4 turns
                fragmentShader: brushShader; vertexShader: brushVertex
            }
            ShaderEffectSource {
                id: kept
                width: painter.width; height: painter.height
                sourceItem: painter; recursive: true; live: false; hideSource: true; smooth: false
                format: ShaderEffectSource.RGBA16F; textureSize: brush.raster
                onScheduledUpdateCompleted: brush.stepped()
            }
            ShaderEffect {
                id: mix
                objectName: "brush-mix"
                width: brush.raster.width; height: brush.raster.height; visible: false
                property var state: kept
                property var base: view.brushedThere ? brushed : view.pixels
                property real based: view.brushedThere ? 1 : 0
                property real channel: brush.channel
                property real erase: brush.erase
                fragmentShader: brushedShader; vertexShader: brushedVertex
                layer.enabled: true; layer.textureSize: brush.raster
            }
        }
    }
    Connections { target: session; function onBrushed(stroke) { brush.add(stroke) } }
    readonly property bool sharpReady: sharpPixels.status === Image.Ready && sharpPixels.sourceSize.width > 1
    readonly property bool useSharp: sharpReady && shown * session.wide > pixels.sourceSize.width * 1.01
    // From 200 % the photo shows its own pixels as squares (pipeline.frag, `squarely`): only
    // from the picture that has them all, not from the quick one while the other is awaited.
    // (From 100 % for a while: the owner found 200 % "looks better that way".)
    readonly property bool crisp: shown > 1.999 && (useSharp || pixels.sourceSize.width >= session.wide - 1)
    Pipeline {
        id: photo
        objectName: "photo"
        visible: view.ready
        width: Math.round(session.fullWidth * view.shown * view.cropBy); height: Math.round(session.fullHeight * view.shown * view.cropBy)
        x: Math.round(view.width / 2 - view.center.x * width + view.cropOff.x)
        y: Math.round(view.height / 2 - view.center.y * height + view.cropOff.y)
        source: view.pixels; painted: view.painted; lutPicture: view.lut
        sharp: detailed.on ? detailed : view.sharpest
        useSharp: view.useSharp || detailed.on ? 1 : 0
        detailKept: detailed.on ? 1 : 0
        crisp: view.crisp ? 1 : 0
        cropDim: session.panels.tool === "crop" ? 1 : 0
        cropBox: Qt.vector4d(session.framing.frame[0], session.framing.frame[1], session.framing.frame[2], session.framing.frame[3])
        clipping: session.panels.clipping ? 1 : 0
        // (The clipping paint's stripes run, a stripe every 0.6 s, only while it is on.)
        NumberAnimation on stripes { running: session.panels.clipping && view.visible && view.ready; from: 0; to: 1; duration: 600; loops: Animation.Infinite }
        reach: view.reach
        // The mask being worked on, laid over the photo in red while the pointer is on the
        // photo (MaskFrame.tinting); it fades in and out.
        showMask: view.showMask > 0 ? view.showMask : maskTint > 0 ? -view.mask : 0
        maskTint: maskFrame.tinting ? 1 : 0
        Behavior on maskTint { NumberAnimation { duration: 160 } }
        Local { over: photo; layers: session.maskLayers }
        // The watermark an export will have, where it will have it (not while the frame is
        // being set, when what is on show is not the frame).
        Mark { visible: exporter.markShown && session.panels.tool !== "crop"; wide: session.fullWidth; high: session.fullHeight }
    }

    // The photo without its edit, left of a line that can be dragged (`split`: where the
    // line is across the photo, 0..1; below zero there is none).
    property real split: -1
    Item {
        id: beforeSide
        visible: view.ready && view.split >= 0
        x: photo.x; y: photo.y; width: photo.width * Math.max(0, view.split); height: photo.height
        clip: true
        Pipeline {
            objectName: "photo-before"
            width: photo.width; height: photo.height
            source: view.pixels; sharp: view.sharpest; useSharp: view.useSharp ? 1 : 0; crisp: photo.crisp   // (the original: not `detailed`)
            edit: follows ? session.before : null
        }
    }
    signal clicked()   // the photo was pressed: the window's keys belong to it again
    property int reach: 0   // 1 + the pin whose reach is shown instead of the picture (H held)
    property int mask: 0       // the mask being worked on (1, 2, …): its shape is on the photo
    property int showMask: 0   // the mask that is shown instead of the picture (H held)
    // What the pointer does on the photo: "" (a click is just a click, a drag moves the
    // photo), "range" (the next click picks a lightness for the wheel) or "neutral" (Pick
    // neutral: it stays on, and a press on the photo, slid about, says at every move which
    // spot should be gray: `sampled`, the first of a press marked so; `sampleDone` when let go).
    property string picking: ""
    signal picked(real u, real v)
    signal sampled(real u, real v, bool first)
    signal sampleDone()
    signal maskLeft()   // a click away from the mask being worked on
    signal tapped(real u, real v)   // a plain click on the photo, at that spot of it
    property point hover: Qt.point(-1, -1)   // where the pointer is on the photo (0..1), or off it

    // Zoom and pan with the mouse. A MouseArea, not a WheelHandler: that one never fired here.
    MouseArea {
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton | Qt.MiddleButton
        hoverEnabled: true
        cursorShape: view.picking !== "" ? Qt.CrossCursor : (pressedButtons & Qt.LeftButton) ? Qt.ClosedHandCursor : Qt.ArrowCursor
        function place(mouse) { return Qt.point((mouse.x - photo.x) / photo.width, (mouse.y - photo.y) / photo.height) }
        onExited: view.hover = Qt.point(-1, -1)
        property point last
        property real travelled   // since the press: a click is a press that went nowhere
        property bool sampling: false   // this press is one of Pick neutral
        onPressed: (mouse) => {
            last = Qt.point(mouse.x, mouse.y); travelled = 0; view.clicked()
            sampling = view.picking === "neutral" && mouse.button === Qt.LeftButton
            if (sampling) view.sampled(place(mouse).x, place(mouse).y, true)
            // The middle button fits the photo, as it goes down (the owner: "on mouseDown,
            // not mouseUp"). That is all it does: dragged, it does not move the photo.
            if (mouse.button === Qt.MiddleButton) view.glideFit()
        }
        // Dragging moves the photo, with the left button only. (A plain click is kept free: it picks a
        // neutral gray while that is asked for, and will pick a color once the wheel exists.)
        onPositionChanged: (mouse) => {
            if (pressed) {
                travelled += Math.abs(mouse.x - last.x) + Math.abs(mouse.y - last.y)
                if (sampling) view.sampled(place(mouse).x, place(mouse).y, false)
                else if (mouse.buttons & Qt.LeftButton) view.panBy(mouse.x - last.x, mouse.y - last.y)
                last = Qt.point(mouse.x, mouse.y)
            }
            const at = place(mouse), on = view.ready && at.x >= 0 && at.x <= 1 && at.y >= 0 && at.y <= 1
            view.hover = on ? at : Qt.point(-1, -1)
        }
        onReleased: (mouse) => {
            if (sampling) { sampling = false; return view.sampleDone() }
            const at = place(mouse)
            if (mouse.button === Qt.MiddleButton) return   // (it fitted the photo as it went down)
            // A click away from the mask being worked on lets go of it (also off the photo).
            if (travelled <= 4 && view.mask > 0 && view.picking === "" && maskFrame.away(mouse.x - photo.x, mouse.y - photo.y)) return view.maskLeft()
            if (travelled > 4 || at.x < 0 || at.x > 1 || at.y < 0 || at.y > 1) return
            if (view.picking === "") view.tapped(at.x, at.y)
            else view.picked(at.x, at.y)
        }
        // (The left button's alone: the middle one, pressed fast, was a double click too, and
        // went to 1:1 and back; the owner: "spamming middle mouse ... zooms in".)
        onDoubleClicked: (mouse) => { if (mouse.button === Qt.LeftButton && view.picking === "") view.fitted ? view.glideTo(1, mouse.x, mouse.y, false) : view.glideFit() }
        // One notch of a wheel is 120, and a step of zoom (view.stepZoom). A touchpad, and a
        // wheel that turns without notches, send small parts: they add up to steps.
        onWheel: (wheel) => {
            view.turned += wheel.angleDelta.y !== 0 ? wheel.angleDelta.y : wheel.pixelDelta.y * 2
            const notches = Math.trunc(view.turned / 120)
            view.turned -= notches * 120
            for (let n = 0; n < Math.abs(notches); n++) view.stepZoom(Math.sign(notches), wheel.x, wheel.y)
        }
    }
    // Pick neutral: the pointer is a pipette (a real cursor: app.TurnCursors), and beside it
    // a square of the color it would take, the mean of a small patch there (the owner: "a
    // square as a preview of the color that would be picked, maybe sampled from more than
    // just one pixel").
    readonly property bool dropper: picking === "neutral" && visible && hover.x >= 0
    onDropperChanged: cursors.pipette(dropper)
    Rectangle {
        objectName: "neutral-spot"
        visible: view.dropper && color.a > 0
        x: photo.x + view.hover.x * photo.width + 16; y: photo.y + view.hover.y * photo.height + 14
        width: 36; height: 36; radius: 4
        color: view.dropper ? (session.histogram, session.shownAt(view.hover.x, view.hover.y)) || "transparent" : "transparent"
        border { width: 1.5; color: Theme.text }
        Rectangle { anchors { fill: parent; margins: -1.5 } radius: 5; color: "transparent"; border { width: 1; color: Theme.bg } }
    }
    // The split's line, to drag, and what is on either side of it.
    Item {
        id: divider
        objectName: "split-line"
        visible: beforeSide.visible
        x: photo.x + photo.width * view.split - 10; y: photo.y; width: 20; height: photo.height
        Rectangle { x: 9; width: 2; height: parent.height; color: Theme.text }
        Rectangle { x: 4; y: parent.height / 2 - 14; width: 12; height: 28; radius: 4; color: Theme.text; border { width: 1; color: Theme.bg } }
        MouseArea {
            anchors.fill: parent; cursorShape: Qt.SizeHorCursor
            onPositionChanged: (mouse) => view.split = Math.max(0, Math.min(1, (mapToItem(view, mouse.x, 0).x - photo.x) / photo.width))
        }
        Label { anchors { right: parent.left; rightMargin: 4 } y: 8; text: "Before"; color: Theme.text; style: Text.Outline; styleColor: Theme.bg }
        Label { anchors { left: parent.right; leftMargin: 4 } y: 8; text: "After"; color: Theme.text; style: Text.Outline; styleColor: Theme.bg }
    }
    // While the picture is being straightened, the frame stays where it is on the screen and
    // as large, and the picture turns under it (the owner: "when i rotate the crop rectangle,
    // the crop rectangle moves around on the screen"). Left to itself the view shows all of
    // the turned picture in the middle, which is larger at every angle, and the frame, which
    // keeps its place on the picture, went round the picture's middle with it. So at each
    // change of the angle the picture is scaled (`cropBy`) and shifted (`cropOff`) to put the
    // frame back where it was; a moment after the last change both glide back to rest.
    property real cropBy: 1
    property point cropOff: Qt.point(0, 0)
    property rect frameWas: Qt.rect(0, 0, 0, 0)   // the frame on the screen, at the last change that was no turn
    property real angleWas: 0
    function laid(by, off) {   // the frame on the screen, were the picture scaled and shifted so
        const z = fitted ? Math.min((width - 2 * margin) / session.fullWidth, (height - 2 * margin) / session.fullHeight) : zoom
        const w = Math.round(session.fullWidth * z * by), h = Math.round(session.fullHeight * z * by), f = session.framing.frame
        const x = Math.round(width / 2 - center.x * w + off.x), y = Math.round(height / 2 - center.y * h + off.y)
        return Qt.rect(x + f[0] * w, y + f[1] * h, (f[2] - f[0]) * w, (f[3] - f[1]) * h)
    }
    function follow() {
        const angle = session.framing.angle
        if (session.panels.tool !== "crop" || !ready) {
            resting.stop(); gliding.stop()
            cropBy = 1; cropOff = Qt.point(0, 0); frameWas = Qt.rect(0, 0, 0, 0)
        } else if (angle !== angleWas && frameWas.width > 0) {
            gliding.stop()
            const by = frameWas.width / Math.max(1, laid(1, Qt.point(0, 0)).width), scaled = laid(by, Qt.point(0, 0))
            cropBy = by
            cropOff = Qt.point(frameWas.x + frameWas.width / 2 - scaled.x - scaled.width / 2, frameWas.y + frameWas.height / 2 - scaled.y - scaled.height / 2)
            resting.restart()
        } else if (!resting.running) frameWas = laid(cropBy, cropOff)
        angleWas = angle
    }
    Connections { target: session; function onChanged() { view.follow() } }
    onWidthChanged: follow()
    onHeightChanged: follow()
    onCropByChanged: if (gliding.running) frameWas = laid(cropBy, cropOff)
    Timer { id: resting; interval: 400; onTriggered: gliding.start() }
    ParallelAnimation {
        id: gliding
        NumberAnimation { target: view; property: "cropBy"; to: 1; duration: 220; easing.type: Easing.OutCubic }
        PropertyAnimation { target: view; property: "cropOff"; to: Qt.point(0, 0); duration: 220; easing.type: Easing.OutCubic }
        onFinished: view.frameWas = view.laid(1, Qt.point(0, 0))
    }
    // The crop tool's frame, on the whole photo, while that tool is open.
    property alias cropRatio: cropFrame.ratio
    function reshape(ratio) { cropFrame.reshape(ratio) }
    function flipCrop() { cropFrame.flip() }
    CropFrame {
        id: cropFrame
        objectName: "crop-frame"
        visible: view.ready && session.panels.tool === "crop"
        x: photo.x; y: photo.y; width: photo.width; height: photo.height
        onDone: view.clicked()
    }
    MaskFrame {
        id: maskFrame
        objectName: "mask-frame"
        number: view.mask
        visible: view.ready && session.panels.tool !== "crop" && mask !== null
        x: photo.x; y: photo.y; width: photo.width; height: photo.height
        onDone: view.clicked()
        onSized: (size) => brushSize = Math.max(0.003, Math.min(0.3, size))
    }
    // The brush a painted mask is painted with (MaskFrame).
    property alias brushSize: maskFrame.brushSize
    property alias brushFeather: maskFrame.brushFeather
    property alias erasing: maskFrame.erasing
    property bool asking: false   // a question is open over the window: nothing is taken in here
    DropArea {
        anchors.fill: parent
        enabled: !view.asking
        onDropped: (drop) => { if (drop.hasUrls) session.open(drop.urls) }
    }
}
