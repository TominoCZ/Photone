import QtQuick
import QtQuick.Shapes
import "."

// Neural Denoise: a window over everything else, as a question is (nothing under it can be
// clicked or typed into while it is up). The open photo goes through the net when it opens
// (photone/denoise.py, `denoiser`); a ring over the picture says how far that is. Then the
// picture is the net's result with the photo's edit on it, to be looked at as the photo is in
// Edit: the wheel zooms, a drag moves it. Color noise and Grain say how much of what the net
// took stays away; they are mixed here, on the GPU (mixing.frag), so they follow the hand.
// Denoise writes the file (the same shares, mixed by denoise.py in full floats) and `done`
// names it; Cancel, Escape and a failure leave everything as it was.
Item {
    id: window
    anchors.fill: parent
    visible: false
    property var stage             // the Edit screen's photo (PhotoView): its pictures are used here
    // Percent of the color noise the net found that stays away, and of the grain: as they were
    // left the last time, in this run or an earlier one.
    property real colorShare: kept("denoise/color")
    property real grainShare: kept("denoise/grain")
    function kept(key) { const v = parseFloat(session.remembered(key, "100")); return isFinite(v) ? Math.max(0, Math.min(100, v)) : 100 }
    signal done(string path)       // the denoised file is written
    signal closed()

    // The net's result is on the GPU and the photo's own full-size picture is too.
    readonly property bool there: denoiser.state !== "" && denoiser.state !== "working" && denoiser.state !== "failed"
                                  && cleanPicture.status === Image.Ready && cleanPicture.sourceSize.width > 1 && stage.ready && stage.sharpThere
    readonly property bool busy: denoiser.state === "working" || denoiser.state === "saving" || (denoiser.state === "ready" && !there)
    // Which net made the result (photone/denoise.py, `chosen`), a quiet line under the photo's
    // name (the owner's pick of three rendered, 2026-11-21): the raw net's, and why the older
    // one ran where the photo's sensor is the reason. While no raw net is installed, nothing.
    readonly property string reason: ({ pattern: "this camera's sensor pattern isn't supported by the new one yet.",
                                        layout: "this file's sensor layout isn't supported by the new one yet." })[denoiser.why] || ""
    readonly property string which: denoiser.kind === "sensor" ? "Denoised from the sensor's data"
                                    : denoiser.kind === "developed" && reason !== "" ? "Older denoiser: " + reason : ""

    function open() {
        // Where the photo was being looked at, at 100 %: noise is judged from close.
        view.center = stage.center; view.zoom = 1; view.fitted = false
        visible = true
        cancel.forceActiveFocus()
        denoiser.start(session.path)
    }
    function close() {
        if (!visible) return
        visible = false
        denoiser.close()
        closed()
    }
    Connections {
        target: denoiser
        function onSaved(path) { window.visible = false; window.closed(); window.done(path) }
    }
    // Another photo under the window (it can only be the program's own doing): not this one's result.
    Connections { target: session; function onPhotoChanged() { window.close() } }

    // While it is up, the keys are its own: Escape is Cancel, Tab goes between its buttons.
    Keys.onPressed: (event) => {
        if (event.key === Qt.Key_Escape && denoiser.state !== "saving") close()
        else if ([Qt.Key_Tab, Qt.Key_Backtab, Qt.Key_Left, Qt.Key_Right].includes(event.key)) (confirm.activeFocus || !confirm.enabled ? cancel : confirm).forceActiveFocus()
        event.accepted = true
    }
    Keys.onReleased: (event) => event.accepted = true
    Rectangle { anchors.fill: parent; color: Theme.bg; opacity: 0.72 }
    MouseArea { anchors.fill: parent; acceptedButtons: Qt.AllButtons; hoverEnabled: true; onWheel: (wheel) => wheel.accepted = true }

    Rectangle {
        id: card
        objectName: "modal:denoise"
        anchors.centerIn: parent
        width: Math.min(parent.width - 60, 1500); height: Math.min(parent.height - 60, 960); radius: 10
        color: Theme.panel; border { width: 1; color: Theme.faint }
        Accessible.role: Accessible.Dialog; Accessible.name: "Neural Denoise"
        MouseArea { anchors.fill: parent; acceptedButtons: Qt.AllButtons }   // (a press on the card is not one beside it)

        // ---- the picture ---------------------------------------------------------------
        Rectangle {
            id: view
            objectName: "denoise-view"
            anchors { left: parent.left; top: parent.top; bottom: parent.bottom; right: side.left; margins: 12 }
            radius: 6; color: Theme.bg; clip: true
            property bool fitted: false
            property real zoom: 1
            property point center: Qt.point(0.5, 0.5)
            readonly property real fitZoom: session.fullWidth > 0 ? Math.min((width - 28) / session.fullWidth, (height - 28) / session.fullHeight) : 1
            readonly property real shown: fitted ? fitZoom : zoom
            // The Edit screen's steps of zoom, from Fit up; below Fit there is nothing to see here.
            readonly property var steps: [1 / 16, 1 / 12, 1 / 8, 1 / 6, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 1, 1.5, 2, 3, 4, 6, 8]
            function held(c) { return Math.max(0, Math.min(1, c)) }
            // At that zoom, the photo's point (u, v) under (x, y) of the view.
            function put(z, x, y, u, v) {
                const w = session.fullWidth * z, h = session.fullHeight * z
                fitted = false; zoom = z
                center = Qt.point(held((width / 2 - (x - u * w)) / w), held((height / 2 - (y - v * h)) / h))
            }
            function fit() { fitted = true; center = Qt.point(0.5, 0.5) }
            function actual(x, y) { put(1, x, y, (x - photo.x) / photo.width, (y - photo.y) / photo.height) }
            function step(by, x, y) {
                const all = steps.filter(z => z > fitZoom * 1.04)
                const next = by > 0 ? all.find(z => z > shown * 1.001) : all.slice().reverse().find(z => z < shown * 0.999)
                if (next === undefined) return by < 0 ? fit() : undefined
                put(next, x, y, (x - photo.x) / photo.width, (y - photo.y) / photo.height)
            }

            // The net's result as a picture, and the photo with the shares mixed back, every
            // pixel of it, for the pipeline to draw from as it draws from the photo's own.
            Image { id: cleanPicture; source: denoiser.source; asynchronous: true; cache: false; visible: false }
            ShaderEffect {
                id: mixed
                visible: false
                width: cleanPicture.sourceSize.width; height: cleanPicture.sourceSize.height
                property var noisy: window.stage.sharpest
                property var clean: cleanPicture
                // (A/B held down gives all of it back: the photo as it was.)
                property real color: compare.held ? 0 : window.colorShare / 100
                property real grain: compare.held ? 0 : window.grainShare / 100
                readonly property var terms: denoiser.terms
                property matrix4x4 toCam: terms.toCam || Qt.matrix4x4()
                property matrix4x4 fromCam: terms.fromCam || Qt.matrix4x4()
                property vector4d scale: terms.scale || Qt.vector4d(1, 1, 1, 0)
                property vector4d offset: terms.offset || Qt.vector4d(0, 0, 0, 0)
                fragmentShader: mixingShader; vertexShader: mixingVertex
                layer.enabled: window.there; layer.format: ShaderEffectSource.RGBA16F
                layer.textureSize: Qt.size(width, height); layer.mipmap: true; layer.smooth: true
            }
            Pipeline {
                id: photo
                objectName: "denoise-photo"
                visible: window.visible && window.stage.ready
                width: Math.round(session.fullWidth * view.shown); height: Math.round(session.fullHeight * view.shown)
                x: Math.round(view.width / 2 - view.center.x * width); y: Math.round(view.height / 2 - view.center.y * height)
                // Until the result is there: the photo as it is, to find the place to judge by.
                source: window.there ? mixed : window.stage.picture
                sharp: window.there ? mixed : window.stage.sharpest
                useSharp: window.there || window.stage.sharpThere ? 1 : 0
                crisp: view.shown > 1.999 ? 1 : 0
                painted: window.stage.painted; lutPicture: window.stage.lut
                // The photo's edit without its own noise reduction, as the file will be shown.
                edit: follows ? Object.assign({}, session.uniforms, { denoise: 0, denoiseColor: 0, denoiseFar: 0 }) : null
                Local { over: photo; layers: session.maskLayers }
            }
            MouseArea {
                anchors.fill: parent
                cursorShape: pressed ? Qt.ClosedHandCursor : Qt.OpenHandCursor
                property point last
                onPressed: (mouse) => last = Qt.point(mouse.x, mouse.y)
                onPositionChanged: (mouse) => {
                    if (view.fitted) { view.zoom = view.fitZoom; view.fitted = false }
                    view.center = Qt.point(view.held(view.center.x - (mouse.x - last.x) / photo.width), view.held(view.center.y - (mouse.y - last.y) / photo.height))
                    last = Qt.point(mouse.x, mouse.y)
                }
                onDoubleClicked: (mouse) => view.fitted ? view.actual(mouse.x, mouse.y) : view.fit()
                property real turned: 0
                onWheel: (wheel) => {
                    turned += wheel.angleDelta.y !== 0 ? wheel.angleDelta.y : wheel.pixelDelta.y * 2
                    const notches = Math.trunc(turned / 120)
                    turned -= notches * 120
                    for (let n = 0; n < Math.abs(notches); n++) view.step(Math.sign(notches), wheel.x, wheel.y)
                }
            }
            // How far the net is: a ring that fills, over the picture.
            Rectangle {
                id: ring
                objectName: "denoise-ring"
                visible: window.busy
                anchors.centerIn: parent
                width: 132; height: 132; radius: 66; color: Qt.rgba(0.05, 0.05, 0.06, 0.82)
                readonly property real share: denoiser.state === "working" ? denoiser.progress : 1
                Shape {
                    anchors.fill: parent; layer.enabled: true; layer.samples: 8
                    ShapePath {
                        strokeColor: Theme.line; strokeWidth: 6; fillColor: "transparent"
                        PathAngleArc { centerX: 66; centerY: 66; radiusX: 50; radiusY: 50; startAngle: 0; sweepAngle: 360 }
                    }
                    ShapePath {
                        strokeColor: Theme.mark; strokeWidth: 6; fillColor: "transparent"; capStyle: ShapePath.RoundCap
                        PathAngleArc { centerX: 66; centerY: 66; radiusX: 50; radiusY: 50; startAngle: -90; sweepAngle: Math.max(1, 360 * ring.share) }
                    }
                }
                Column {
                    anchors.centerIn: parent; spacing: 0
                    Label { anchors.horizontalCenter: parent.horizontalCenter; color: Theme.text; font.pixelSize: 20; text: denoiser.state === "working" && denoiser.progress > 0 ? Math.round(denoiser.progress * 100) + " %" : "…" }
                    Label { anchors.horizontalCenter: parent.horizontalCenter; font.pixelSize: Theme.smallSize; text: denoiser.state === "saving" ? "Writing" : denoiser.state === "working" ? denoiser.step : "Preview" }
                    // (Said while it runs: on the processor it takes minutes, and that must not be a riddle.)
                    Label { objectName: "denoise-where"; anchors.horizontalCenter: parent.horizontalCenter; visible: denoiser.state === "working" && denoiser.where.includes("processor"); font.pixelSize: Theme.smallSize; color: Theme.mark; text: "on the processor" }
                }
                Accessible.role: Accessible.ProgressBar; Accessible.name: denoiser.state === "working" ? denoiser.step : "Denoising"
            }
            Rectangle {
                visible: denoiser.state === "failed"
                anchors.centerIn: parent
                width: Math.min(parent.width - 60, 520); height: failure.implicitHeight + 32; radius: 8; color: Qt.rgba(0.05, 0.05, 0.06, 0.9); border { width: 1; color: Theme.red }
                Label { id: failure; objectName: "denoise-failure"; x: 16; y: 16; width: parent.width - 32; wrapMode: Text.WordWrap; color: Theme.text; text: denoiser.message }
            }
            // A/B, as every card has it, in the picture's corner: held, the photo as it was
            // (the owner: "i like the A/B we have on all cards ... in a corner of the image
            // previewer"). It says which of the two is on show while it is held.
            Rectangle {
                anchors { right: parent.right; top: parent.top; margins: 12 }
                width: compare.implicitWidth + (compare.held ? said.implicitWidth + 30 : 22); height: 30; radius: 8
                color: Qt.rgba(0.09, 0.09, 0.1, 0.92); border { width: 1; color: compare.held ? Theme.mark : Theme.faint }
                Label { id: said; visible: compare.held; x: 10; anchors.verticalCenter: parent.verticalCenter; text: "Original"; color: Theme.mark }
                TextButton {
                    id: compare
                    objectName: "denoise-compare"
                    anchors { right: parent.right; rightMargin: 11; verticalCenter: parent.verticalCenter }
                    text: "A/B"; says: "Compare with the original"; strong: true; font.weight: Font.DemiBold
                    font.pixelSize: Theme.titleSize; font.letterSpacing: 0.6
                    available: window.there && denoiser.state === "ready"
                }
            }
            Rectangle {   // where it is: Fit, 100 %, and the zoom in effect
                anchors { horizontalCenter: parent.horizontalCenter; bottom: parent.bottom; bottomMargin: 12 }
                width: zooms.width + 8; height: 30; radius: 8; color: Qt.rgba(0.09, 0.09, 0.1, 0.92); border { width: 1; color: Theme.faint }
                Row {
                    id: zooms
                    x: 4; y: 3
                    TextButton { objectName: "denoise-fit"; width: 44; height: 24; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter; text: "Fit"; strong: view.fitted; onPressed: view.fit() }
                    TextButton { objectName: "denoise-actual"; width: 44; height: 24; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter; text: "1:1"; strong: !view.fitted && Math.abs(view.zoom - 1) < 0.001; onPressed: view.actual(view.width / 2, view.height / 2) }
                    Label { width: 64; height: 24; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter; text: Math.round(view.shown * 100) + " %" }
                }
            }
        }

        // ---- what to keep away, and the two answers ---------------------------------------
        Item {
            id: side
            anchors { right: parent.right; top: parent.top; bottom: parent.bottom; margins: 16 }
            width: 280
            Column {
                width: parent.width; spacing: 6
                Label { text: "Neural Denoise"; color: Theme.text; font.pixelSize: 17; font.weight: Font.DemiBold }
                Label { width: parent.width; elide: Text.ElideMiddle; text: session.name }
                Label {
                    objectName: "denoise-which"
                    visible: window.which !== ""
                    width: parent.width; wrapMode: Text.WordWrap; font.pixelSize: 12; color: Theme.faint
                    text: window.which
                }
                Item { width: 1; height: 10 }
                ParamSlider {
                    objectName: "denoise-color"
                    width: parent.width; name: "Color noise"; unit: "%"; from: 0; to: 100; neutral: 100; step: 1; digits: 0; signed: false
                    enabled: window.there && denoiser.state === "ready"; opacity: enabled ? 1 : 0.4
                    value: window.colorShare
                    onMoved: (v) => window.colorShare = v
                    onReleased: session.remember("denoise/color", String(window.colorShare))
                    onReset: { window.colorShare = 100; session.remember("denoise/color", "100") }
                }
                ParamSlider {
                    objectName: "denoise-grain"
                    width: parent.width; name: "Grain"; unit: "%"; from: 0; to: 100; neutral: 100; step: 1; digits: 0; signed: false
                    enabled: window.there && denoiser.state === "ready"; opacity: enabled ? 1 : 0.4
                    value: window.grainShare
                    onMoved: (v) => window.grainShare = v
                    onReleased: session.remember("denoise/grain", String(window.grainShare))
                    onReset: { window.grainShare = 100; session.remember("denoise/grain", "100") }
                }
                Label {
                    width: parent.width; wrapMode: Text.WordWrap; font.pixelSize: 12; color: Theme.faint; topPadding: 4
                    text: "How much of what the net found stays away: the colored speckle, and the lighter and darker grain. At 100 the photo is as the net left it; lower gives some of it back."
                }
            }
            Column {
                anchors.bottom: parent.bottom; width: parent.width; spacing: 10
                Label {
                    objectName: "denoise-words"
                    width: parent.width; wrapMode: Text.WordWrap; font.pixelSize: 12; color: Theme.dim
                    text: (denoiser.where.includes("processor") ? "Running on " + denoiser.where + ": no graphics card that the net can use was found, or its memory was full. That takes minutes.\n\n" : "")
                          + (denoiser.target !== "" ? "Saved beside the original as " + denoiser.target + ", a raw file with this photo's edit. The original stays as it is." : "")
                }
                Row {
                    anchors.right: parent.right; spacing: 8
                    PushButton { id: cancel; objectName: "denoise-cancel"; label: denoiser.state === "failed" ? "Close" : "Cancel"; enabled: denoiser.state !== "saving"; opacity: enabled ? 1 : 0.4; onPressed: window.close() }
                    PushButton {
                        id: confirm
                        objectName: "denoise-confirm"
                        label: window.there ? "Accept" : "Denoise"; tint: Theme.green
                        enabled: window.there && denoiser.state === "ready"; opacity: enabled ? 1 : 0.4
                        onPressed: if (enabled) denoiser.save(window.colorShare / 100, window.grainShare / 100)
                    }
                }
            }
        }
    }
}
