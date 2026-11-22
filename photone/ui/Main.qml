import QtQuick
import QtQuick.Shapes
import QtQuick.Dialogs
import QtQuick.Effects
import "."

// The window. `session` (app.py) says what is open; everything drawn here follows from it.
Rectangle {
    id: root
    color: Theme.bg

    readonly property var edit: session.edit   // read once per change; the sliders share it
    // Which of the two places is on show: the library (the photos there are) or Edit (one of
    // them). Without a photo to edit, the program starts in the library.
    property string place: "edit"
    readonly property bool editing: place === "edit"
    Component.onCompleted: {
        place = session.source === "" ? "library" : "edit"
        library.rest(away)
        if (session.source === "") Qt.callLater(root.resume)
    }
    // Started without a photo named: where it was left. The library is at the place it was
    // at already (library.py); here the photo last looked at is selected, and opened in Edit
    // if that is where the window was.
    function resume() {
        const row = library.rowOf(library.lastPhoto)
        if (row < 0) return
        shelf.choose(row)
        if (session.panels.place === "edit") { session.openAt(library.paths(), row); root.show("edit") }
    }
    function show(where) {
        if (where === "library" && place !== "library") {
            session.flush()   // so that the library knows of the edit just made
            shelf.settle(library.paths().indexOf(session.path))
        }
        if (where === "export" && place !== "export") {
            // (What is selected where the page is come from, if that is more than one photo.)
            const all = library.paths()
            const taken = place === "library" ? shelf.chosen.map(row => all[row]) : filmstrip.chosen.map(i => session.fileAt(i))
            desk.photos = taken.length > 1 ? taken.filter(path => !!path) : []
        }
        place = where
        if (where !== "export") session.setPanel("place", where)   // (remembered for the next start)
        root.forceActiveFocus()
    }
    // The open photo's marks (the library keeps them), asked again whenever either changes.
    readonly property string openPath: session.path   // (asked at every change; another only with another photo)
    // While a photo is being edited the library rests: no scans, no thumbnails made.
    readonly property bool away: place !== "library"
    onAwayChanged: library.rest(away)
    readonly property bool photoReady: stage.ready
    readonly property bool photoFailed: session.failed
    // The open photo's file is gone from the disk: it can only be taken out (the owner:
    // "when i select it to view it, i only have an option to remove it").
    readonly property bool photoMissing: (library.busy, session.listing, library.missing(session.path))

    // Keys reach here from wherever the focus is, unless a control used them itself.
    focus: true
    Keys.onPressed: (event) => {
        const ctrl = event.modifiers & Qt.ControlModifier, shift = event.modifiers & Qt.ShiftModifier
        if (ctrl && event.key === Qt.Key_1) return root.show("library")
        if (ctrl && event.key === Qt.Key_2) return root.show("edit")
        if (ctrl && event.key === Qt.Key_3) return root.show("export")
        // A look, copied from a photo and laid on others: in the library for those selected,
        // in Edit for the open photo (and those selected in the filmstrip).
        if (ctrl && shift && (event.key === Qt.Key_C || event.key === Qt.Key_V) && root.place !== "export") {
            const all = library.paths()
            const these = root.place === "library" ? shelf.chosen.map(row => all[row]).filter(path => path !== undefined)
                        : filmstrip.chosen.length ? filmstrip.chosen.map(i => session.fileAt(i)) : session.path !== "" ? [session.path] : []
            if (event.key === Qt.Key_V) root.pasteLook(these)
            else if (these.length) root.copyLook(root.place === "edit" ? session.path : these[0])
            return
        }
        if (root.place === "library") { shelf.key(event); return }
        if (root.place !== "edit") return
        // The arrows, held, go on through the photos (the owner: "hold left/right arrows to
        // quickly list through the filmstrip"); every other key acts once however long held.
        // (A and D too, the owner's ask: the left hand's keys for the same.)
        if (!ctrl && (event.key === Qt.Key_Left || event.key === Qt.Key_A)) return session.step(-1)
        if (!ctrl && (event.key === Qt.Key_Right || event.key === Qt.Key_D)) return session.step(1)
        if (event.isAutoRepeat) return
        // Marking the photo on show, as in the library: Edit is where a photo is looked at closely.
        if (!ctrl && event.key === Qt.Key_P) return library.markPhoto(session.path, "flag", 1)
        if (!ctrl && event.key === Qt.Key_X) return library.markPhoto(session.path, "flag", -1)
        if (!ctrl && event.key === Qt.Key_U) return library.markPhoto(session.path, "flag", 0)
        if (event.key >= Qt.Key_1 && event.key <= Qt.Key_5) return library.markPhoto(session.path, "stars", event.key - Qt.Key_0)
        if (event.key >= Qt.Key_6 && event.key <= Qt.Key_9) return library.markPhoto(session.path, "label", event.key - Qt.Key_5)
        if (event.key === Qt.Key_B) session.setOriginal(true)
        else if (event.key === Qt.Key_H) { if (wheel.selected >= 0) stage.reach = wheel.selected + 1; else stage.showMask = wheel.mask }
        else if (ctrl && event.key === Qt.Key_Z) shift ? session.redo() : session.undo()
        else if (ctrl && event.key === Qt.Key_Y) session.redo()
        else if (ctrl && event.key === Qt.Key_O) opener.open()
        else if (event.key === Qt.Key_T) root.toggle("tone")
        else if (event.key === Qt.Key_W) root.toggle("balance")
        else if (event.key === Qt.Key_S && !ctrl) root.toggle("saturation")
        else if (event.key === Qt.Key_C && !ctrl) root.toggle("crop")
        else if (event.key === Qt.Key_M && !ctrl) root.setMono(!root.mono)
        else if (event.key === Qt.Key_BracketLeft) session.turn(-1)
        else if (event.key === Qt.Key_BracketRight) session.turn(1)
        else if (event.key === Qt.Key_Backslash) stage.split = stage.split < 0 ? 0.5 : -1
        else if ((event.key === Qt.Key_Return || event.key === Qt.Key_Enter) && session.panels.tool === "crop") session.setPanel("tool", "")
        else if (event.key === Qt.Key_F) wheel.glideFit()
        else if (event.key === Qt.Key_0) wheel.glideAll()
        else if (event.key === Qt.Key_Delete || event.key === Qt.Key_Backspace) wheel.remove(wheel.selected)
        else if (event.key === Qt.Key_Escape && session.panels.tool === "crop") { session.undoTo(root.cropFrom); session.setPanel("tool", "") }
        else if (event.key === Qt.Key_Escape) { stage.picking = ""; wheel.deselect() }
    }
    // The right column: every card of a tab in one scrolling column, each folded to its head
    // or open (the owner's choice of M16's trials; the rail that opened one panel at a time
    // over the wheel is gone).
    // (Open at start: Light only; the white balance starts folded, the owner's choice.)
    property var unfolded: ({ tone: true, about: true, folders: true })
    // Presets are a trial for now (the owner: "only as a trial build ... until its polished
    // and ready"): their tab is there only when the program is started with PHOTONE_TRIAL=presets.
    readonly property bool presetsOn: trial.split(",").includes("presets")
    property string tab: "edit"        // the tab on show when it is not Crop's: edit, masks or presets
    readonly property string mode: session.panels.tool === "crop" ? "crop" : tab
    // Black and white: one card in the place of those that work on colors (and of the Tone
    // Wheel). The color grade stays in the edit and is back with the switch.
    readonly property bool mono: root.edit.mono === true
    function setMono(on) {
        if (on === mono || session.source === "") return
        wheel.deselect()
        session.commitEdit(); session.setEdit("mono", on); session.commitEdit()
        if (on && !unfolded.mono) unfolded = Object.assign({}, unfolded, { mono: true })
    }
    function shows(tool) {
        if (tool === "crop") return mode === "crop"
        return mode === "edit" && (tool === "mono" ? mono : !(mono && ["saturation", "eq", "chroma"].includes(tool)))
    }
    function go(to) { if (to !== "crop") tab = to; session.setPanel("tool", to === "crop" ? "crop" : ""); if (to !== "masks") wheel.mask = 0 }
    function isOpen(tool) { return tool === "crop" ? session.panels.tool === "crop" : unfolded[tool] === true }
    function fold(tool) {   // a section's header was pressed
        if (tool === "crop") return session.setPanel("tool", session.panels.tool === "crop" ? "" : "crop")
        unfolded = Object.assign({}, unfolded, { [tool]: !unfolded[tool] })
        if (unfolded[tool]) { laidOut.tool = tool; laidOut.restart() }
    }
    // A card of the right column just opened is brought to the middle of the column, quickly
    // and smoothly; one higher than the column, to its top; never past the column's ends.
    // (A trial of the owner's: "i want to just test this".)
    function bringIn(tool) {
        bringCard(({ tone: tonePanel, balance: balancePanel, saturation: saturationPanel,
                     eq: eqPanel, chroma: chromaPanel, mono: monoPanel, wheel: wheelBlock, grain: grainPanel, lut: lutPanel, detail: detailPanel, lens: lensPanel })[tool])
    }
    function bringCard(card) {
        if (!card || !card.visible) return
        const top = card.mapToItem(develop, 0, 0).y, high = card.height - 24, room = scroller.height
        const to = high > room - 20 ? top - 10 : top + high / 2 - room / 2
        bringing.to = Math.max(0, Math.min(Math.max(0, scroller.contentHeight - room), to))
        bringing.restart()
    }
    // A card's button on the dock beside the column: a folded card opens as a click on its
    // head opens it (lit up and brought in); one that is open, or does not fold, is brought
    // in and lit up.
    function reveal(card) {
        if (card.folds && !card.open) return card.fold()
        bringCard(card); card.flash()
    }
    // (A moment later: the column has its new height only once the card is laid out.)
    Timer { id: laidOut; property string tool; interval: 30; onTriggered: root.bringIn(tool) }
    NumberAnimation { id: bringing; target: scroller; property: "contentY"; duration: 170; easing.type: Easing.OutCubic }
    // The mouse wheel over a list in one of the right column's cards (`bar`: what scrolls that
    // list). While the column itself was scrolled a moment ago the wheel stays the column's:
    // a scroll passing over a list is not caught by it, as it is not by the Tone Wheel (the
    // owner: "that panel scroll protection that tone wheel does ... all the scrollable lists").
    // And a list with nothing to scroll leaves the wheel to the column altogether.
    function listWheel(bar, wheel) {
        if (scrolling.running || bar.reach <= 0.5) { scrolling.restart(); toolsBar.turned(wheel) }
        else bar.turned(wheel)
    }
    // (For the checks, as PhotoView's and the wheel's: whatever glides here is there at once.)
    function land() { bringing.complete() }
    function toggle(tool) {   // a card was asked for by its key
        if (!isOpen(tool)) fold(tool)
        const at = ({ tone: tonePanel, balance: balancePanel, pins: pinPanel, crop: cropPanel,
                      saturation: saturationPanel, eq: eqPanel, chroma: chromaPanel })[tool]
        Qt.callLater(() => scroller.contentY = Math.max(0, Math.min(scroller.contentHeight - scroller.height, at.mapToItem(develop, 0, 0).y - 10)))
    }
    // Where undo stood when the Crop tool was opened: Esc goes back there.
    property int cropFrom: 0
    readonly property bool cropping: session.panels.tool === "crop"
    onCroppingChanged: if (cropping) cropFrom = session.steps
    Keys.onReleased: (event) => {
        if (event.isAutoRepeat) return
        if (event.key === Qt.Key_B) session.setOriginal(false)
        if (event.key === Qt.Key_H) { stage.reach = 0; stage.showMask = 0 }
    }
    // A key that is held when the window loses the keyboard is never let go here: let go for it.
    Window.onActiveChanged: if (!Window.active) { session.setOriginal(false); stage.reach = 0; stage.showMask = 0 }
    Connections {
        target: session
        // (The zoom, and where in the photo it is, stay as they are from one photo to the
        // next: two shots of the same thing are compared at the same place.)
        function onPhotoChanged() { stage.picking = ""; wheel.deselect(); wheel.mask = 0; library.lookingAt(session.path) }
        function onMasksChanged() { if (wheel.mask > session.masks.length) wheel.mask = 0 }
    }

    // Nothing is drawn until the surface is known (one frame): colors may depend on it. By
    // opacity, not visibility: an invisible item cannot hold the keyboard focus.
    opacity: session.ready ? 1 : 0
    Binding { target: Theme; property: "linear"; value: session.linearSurface }

    component NavButton: Rectangle {
        property string label
        property bool current: false
        property bool built: true
        signal pressed()
        objectName: "nav:" + label
        width: caption.implicitWidth + 24; height: 30; radius: 6
        color: current ? Theme.raised : "transparent"
        Accessible.role: Accessible.PageTab; Accessible.name: label
        Label {
            id: caption
            anchors.centerIn: parent; text: parent.label
            color: parent.current ? Theme.text : parent.built ? Theme.dim : Theme.faint
        }
        MouseArea { anchors.fill: parent; enabled: parent.built; onClicked: parent.pressed() }
    }
    // ---- Top bar -------------------------------------------------------------------------
    Rectangle {
        id: topBar
        objectName: "box:top"
        anchors { left: parent.left; right: parent.right; top: parent.top }
        height: 44; color: Theme.panel

        Row {
            anchors { left: parent.left; leftMargin: 14; verticalCenter: parent.verticalCenter }
            spacing: 20
            Row {
                spacing: 8; anchors.verticalCenter: parent.verticalCenter
                Image { source: "icon.svg"; sourceSize: Qt.size(20, 20); anchors.verticalCenter: parent.verticalCenter }
                Label { text: "Photone"; color: Theme.text; font.weight: Font.DemiBold; anchors.verticalCenter: parent.verticalCenter }
            }
            Row {
                spacing: 2; anchors.verticalCenter: parent.verticalCenter
                NavButton { label: "Library"; current: root.place === "library"; onPressed: root.show("library") }
                NavButton { label: "Edit"; current: root.editing; onPressed: root.show("edit") }
                NavButton { label: "Export"; current: root.place === "export"; onPressed: root.show("export") }
            }
        }
        Row {
            anchors { right: parent.right; rightMargin: 14; verticalCenter: parent.verticalCenter }
            spacing: 14
            // The photo, or a test chart in its place with the photo's edit on it: to see what
            // an edit does to every hue and to a row of grays. Said loudly while it is on
            // (the bar over the picture), and never on by itself.
            Rectangle {
                objectName: "chartSwitch"
                visible: session.source !== "" && root.editing
                width: chartSides.width + 4; height: 30; radius: 7; color: Theme.raised
                anchors.verticalCenter: parent.verticalCenter
                Row {
                    id: chartSides
                    anchors.centerIn: parent
                    Segment { objectName: "show-photo"; width: 60; label: "Photo"; on: !session.panels.chart; onChosen: session.setPanel("chart", false) }
                    Segment { objectName: "show-chart"; width: 84; label: "Test chart"; on: session.panels.chart; onChosen: session.setPanel("chart", true) }
                }
            }
            // SDR or HDR: how the photo is drawn. HDR lets highlights go above the white of the
            // interface. Instant: only the photo changes. Dimmed where the window has no HDR
            // surface; choosing it there says why.
            Rectangle {
                objectName: "hdrSwitch"
                width: 2 * 52 + 4; height: 30; radius: 7; color: Theme.raised
                anchors.verticalCenter: parent.verticalCenter
                Row {
                    anchors.centerIn: parent
                    Segment { label: "SDR"; on: !session.hdr; onChosen: session.setHdr(false) }
                    Segment { label: "HDR"; on: session.hdr; available: session.hdrSurface; onChosen: session.setHdr(true) }
                }
            }
        }
        Rectangle { anchors { left: parent.left; right: parent.right; bottom: parent.bottom } height: 1; color: Theme.line }
    }

    // ---- The foot: what the program is and who made it -----------------------------------
    // (The owner: "add back the footer at the very bottom of the window to offset everything
    // up a little", with the version, "Made by Morphox", "View on GitHub". Nothing about the
    // photo is said here: that is the left panel's and the tip's.)
    Rectangle {
        id: foot
        objectName: "box:foot"
        anchors { left: parent.left; right: parent.right; bottom: parent.bottom }
        height: 26; color: Theme.panel
        Rectangle { width: parent.width; height: 1; color: Theme.line }
        Label {
            objectName: "text:version"
            anchors { left: parent.left; leftMargin: 14; verticalCenter: parent.verticalCenter }
            text: "Version " + session.version; font.pixelSize: 12; color: Theme.dim
        }
        TextButton {   // an export, wherever it was started: how far it is, then what came of it
            objectName: "foot-export"
            visible: exporter.running || exporter.total > 0 || exporter.failures.length > 0
            anchors { left: parent.left; leftMargin: 130; verticalCenter: parent.verticalCenter }
            font.pixelSize: 12; says: "Export"
            readonly property int done: exporter.finished - exporter.failures.length
            text: exporter.running ? "Exporting " + Math.min(exporter.total, exporter.finished + 1) + " of " + exporter.total + "…"
                : done + " exported" + (exporter.failures.length ? ", " + exporter.failures.length + " failed" : "")
            onPressed: root.show("export")
        }
        Label {   // what the program is using (the owner's wish), read every two seconds
            objectName: "text:usage"
            anchors.centerIn: parent
            text: usage.text; font.pixelSize: 12; font.features: { "tnum": 1 }; color: Theme.faint
        }
        Row {
            anchors { right: parent.right; rightMargin: 14; verticalCenter: parent.verticalCenter }
            spacing: 14
            // (Links: the author's page on GitHub, and the project's, with GitHub's mark.)
            TextButton {
                objectName: "foot-author"
                anchors.verticalCenter: parent.verticalCenter
                text: "Made by Morphox"; font.pixelSize: 12
                Accessible.role: Accessible.Link
                onPressed: Qt.openUrlExternally(session.homepage.replace(/\/[^\/]+\/?$/, ""))   // (the account the project is under)
            }
            Rectangle { width: 1; height: 12; color: Theme.line; anchors.verticalCenter: parent.verticalCenter }
            TextButton {
                objectName: "foot-github"
                anchors.verticalCenter: parent.verticalCenter
                text: "View on GitHub"; font.pixelSize: 12; rightPadding: 20
                Accessible.role: Accessible.Link
                onPressed: Qt.openUrlExternally(session.homepage)
                Icon {
                    anchors { right: parent.right; verticalCenter: parent.verticalCenter }
                    width: 14; height: 14; units: 24; filled: true; color: parent.color
                    // GitHub's mark (its own drawing, for a link to GitHub).
                    path: "M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"
                }
            }
        }
    }

    // A list in a dark box has no scrollbar: the bar took a strip off every row (the owner: "i
    // hate how ... the content is offset by the scrollbar size"). That there is more below
    // is said by the last rows fading out at the box's foot, until the list is at its end.
    // (`above`: the same at the box's head, for what lies over the first row in sight; the
    // checkpoints' list stands at its end, the newest, so what it hides is above.)
    // Glass over the photo, under whatever floats there (the zoom's bar, the card dock): the
    // photo behind it blurred and darkened, and an outline (the owner: "a blur glass
    // transparent background and an outline", "make sure the text is readable"). It fills
    // its parent, which must be a sibling of the stage.
    component Glass: Item {
        id: glass
        property real radius: 8
        anchors.fill: parent
        ShaderEffectSource {
            id: behind
            visible: false
            sourceItem: stage
            sourceRect: Qt.rect(glass.parent.x - stage.x, glass.parent.y - stage.y, glass.width, glass.height)
        }
        Rectangle { id: shape; anchors.fill: parent; radius: glass.radius; visible: false; layer.enabled: true }
        MultiEffect {
            anchors.fill: parent
            source: behind
            blurEnabled: true; blur: 1.0; blurMax: 48; autoPaddingEnabled: false
            maskEnabled: true; maskSource: shape
        }
        Rectangle {
            anchors.fill: parent; radius: glass.radius
            color: Qt.rgba(0.05, 0.05, 0.06, 0.58); border { width: 1; color: Qt.rgba(1, 1, 1, 0.28) }
        }
    }
    component MoreBelow: Rectangle {
        property Flickable of
        property bool above: false
        property color ground: Theme.bg   // what the rows stand on, and fade into
        readonly property real gone: of ? of.contentY - of.originY : 0
        // How strong it is goes by how far there is still to scroll that way: full with its
        // own height or more to go, less and less over the last of the way, nothing at the
        // end (the owner: "they should fade with scroll distance"). `wanted`: whether at all.
        property bool wanted: true
        readonly property real togo: !of ? 0 : above ? gone : of.contentHeight - of.height - gone
        opacity: wanted ? Math.max(0, Math.min(1, togo / height)) : 0; visible: opacity > 0
        height: 30
        gradient: Gradient {
            GradientStop { position: above ? 1 : 0; color: Qt.rgba(ground.r, ground.g, ground.b, 0) }
            GradientStop { position: above ? 0 : 1; color: ground }
        }
    }

    // The profile (what only a raw file has: how the sensor's data becomes a picture), and
    // under it what it is. On the new screen it stands over all the cards of Adjust (the
    // owner's choice of two renders: there, or in the Exposure card over the histogram); on
    // the old one, in the Tone panel.
    component ProfileRow: Column {
        width: parent ? parent.width : 0
        Item {  // what only a raw file has: how the sensor's data becomes a picture
            objectName: "profile"
            width: parent.width; height: 34; visible: session.develops
            Label {
                // (Bold, in the color of text; yellow while it is changed and in no checkpoint, as
                // the sliders are: the owner's wish.)
                anchors.verticalCenter: parent.verticalCenter; text: "Profile"; font.weight: Font.DemiBold
                color: session.unsavedKeys.includes("profile") ? Theme.yellow : Theme.text
            }
            Rectangle {
                width: profiles.width + 4; height: 24; radius: 6; color: Theme.raised
                anchors { right: parent.right; verticalCenter: parent.verticalCenter }
                Row {
                    id: profiles
                    anchors.centerIn: parent
                    Segment { width: 76; height: 20; label: "Standard"; on: root.edit.profile === "standard"; onChosen: { session.setEdit("profile", "standard"); session.commitEdit() } }
                    Segment { width: 76; height: 20; label: "Flat"; on: root.edit.profile === "flat"; onChosen: { session.setEdit("profile", "flat"); session.commitEdit() } }
                }
            }
        }
        // (The owner: "i dont understand what the profiles ... actually change and
        // where they come from". Said under them, in the fewest words.)
        Label {
            objectName: "profile-words"
            visible: session.develops
            width: parent.width; height: implicitHeight + 8; wrapMode: Text.WordWrap; font.pixelSize: 12; color: Theme.faint
            // (What a profile is, and on a line of its own what this one is, its name in bold:
            // the owner's wish.)
            textFormat: Text.StyledText
            readonly property string name: root.edit.profile === "flat" ? "Flat" : "Standard"
            readonly property string what: root.edit.profile === "flat" ? "softer and darker in the midtones, room for a grade of your own."
                                                                        : "fitted to look like the camera's own JPEG."
            text: "The base contrast before any slider.<br><b>" + name + "</b>: " + what
        }
    }

    // The loupe: a spot of the photo at 100 %, one pixel of the screen for each of its own,
    // where grain, sharpening and noise reduction are what they will be in the file; zoomed
    // out, a pixel of the screen shows several of the photo's. A drag moves the spot, which
    // is the same in every card that has a loupe. `on`: the card is open.
    property point loupeSpot: Qt.point(0.5, 0.5)   // of the photo's frame, 0..1
    component Loupe: Rectangle {
        id: loupe
        property bool on: true
        property string named: "loupe"
        objectName: named
        width: parent ? parent.width : 0; height: 150; radius: 6; color: Theme.bg; border { width: 1; color: Theme.line }
        clip: true
        Pipeline {
            id: loupePhoto
            objectName: loupe.named + "-photo"
            visible: stage.ready && loupe.on
            width: session.fullWidth; height: session.fullHeight
            x: Math.round(loupe.width / 2 - root.loupeSpot.x * width); y: Math.round(loupe.height / 2 - root.loupeSpot.y * height)
            source: stage.picture; sharp: stage.sharpest; painted: stage.painted; lutPicture: stage.lut
            useSharp: stage.sharpThere ? 1 : 0
            Local { over: loupePhoto; layers: session.maskLayers }
        }
        MouseArea {
            anchors.fill: parent; cursorShape: pressed ? Qt.ClosedHandCursor : Qt.OpenHandCursor
            property point from
            property point start
            onPressed: (mouse) => { from = Qt.point(mouse.x, mouse.y); start = root.loupeSpot }
            onPositionChanged: (mouse) => {
                // (Never past the photo's edge: the spot stays half a loupe inside it, where the photo is larger than the loupe.)
                const half = Qt.point(Math.min(0.5, loupe.width / 2 / Math.max(1, session.fullWidth)), Math.min(0.5, loupe.height / 2 / Math.max(1, session.fullHeight)))
                root.loupeSpot = Qt.point(Math.max(half.x, Math.min(1 - half.x, start.x - (mouse.x - from.x) / Math.max(1, session.fullWidth))),
                                          Math.max(half.y, Math.min(1 - half.y, start.y - (mouse.y - from.y) / Math.max(1, session.fullHeight))))
            }
            onReleased: root.forceActiveFocus()
        }
        Rectangle {
            x: 6; y: 6; width: loupeWords.implicitWidth + 12; height: 20; radius: 5; color: Qt.rgba(0.07, 0.07, 0.08, 0.72)
            Label { id: loupeWords; anchors.centerIn: parent; text: "100 %  ·  drag to look elsewhere"; font.pixelSize: Theme.smallSize; color: Theme.text }
        }
        Rectangle { anchors.fill: parent; radius: parent.radius; color: "transparent"; border { width: 1; color: Theme.line } }
        Accessible.role: Accessible.Graphic; Accessible.name: "The photo at 100 percent"
    }
    // ---- Adjustments ---------------------------------------------------------------------
    component EditSlider: ParamSlider {
        property string key   // the setting in the photo's edit (photone/edit.py)
        objectName: "slider:" + key
        width: parent.width
        value: root.edit[key]
        unsaved: session.unsavedKeys.includes(key)
        onMoved: (v) => session.setEdit(key, v)
        onReleased: session.commitEdit()
        // Where a double click puts it: the edit's own rest, or `rest` where that is not
        // what the slider's zero is (Exposure: see there).
        property var rest
        onReset: {
            if (rest === undefined) return session.resetEdit(key)
            session.commitEdit(); session.setEdit(key, rest); session.commitEdit()
        }
        onTyped: root.forceActiveFocus()
    }
    // The side panels' widths: dragged by their inner edges (the owner: "both the left and
    // right panels resizable (with some minimal width and max until 1/3 of the window for
    // safety)"), remembered; a double click on an edge puts it back. Asked 0: the usual width.
    property real leftAsked: Number(session.remembered("view/left", "0")) || 0
    property real rightAsked: Number(session.remembered("view/right", "0")) || 0
    readonly property real leftWide: Math.round(Math.max(220, Math.min(width / 3, leftAsked > 0 ? leftAsked : 276)))
    readonly property real rightWide: Math.round(Math.max(340, Math.min(width / 3, rightAsked > 0 ? rightAsked
                                                                         : Math.max(360, Math.min(520, width * 0.3)))))
    component PaneEdge: MouseArea {
        property int way: 1   // 1: the left panel's right edge; -1: the right column's left edge
        property real from
        property real start
        width: 6; hoverEnabled: true; cursorShape: Qt.SizeHorCursor
        Rectangle { anchors { horizontalCenter: parent.horizontalCenter; top: parent.top; bottom: parent.bottom } width: 2; color: Theme.dim; visible: parent.containsMouse || parent.pressed }
        onPressed: (mouse) => { from = mapToItem(root, mouse.x, 0).x; start = way > 0 ? root.leftWide : root.rightWide }
        onPositionChanged: (mouse) => {
            if (!pressed) return
            const wide = start + way * (mapToItem(root, mouse.x, 0).x - from)
            if (way > 0) root.leftAsked = wide; else root.rightAsked = wide
        }
        function keep() {
            session.remember(way > 0 ? "view/left" : "view/right", String(Math.round(way > 0 ? root.leftAsked : root.rightAsked)))
        }
        onReleased: { if (way > 0) root.leftAsked = root.leftWide; else root.rightAsked = root.rightWide; keep() }
        onDoubleClicked: { if (way > 0) root.leftAsked = 0; else root.rightAsked = 0; keep() }
    }
    // A switch on a row of its own: its name, and on the right a track with a knob. The whole
    // row takes the click; `flipped` asks for the other state, which the owner of `on` sets.
    component OnOff: Item {
        id: onOff
        property string label
        property bool on: false
        signal flipped()
        width: parent ? parent.width : 0; height: 34
        Accessible.role: Accessible.CheckBox; Accessible.name: label; Accessible.checked: on
        Label { anchors.verticalCenter: parent.verticalCenter; text: onOff.label; color: onOff.on ? Theme.text : Theme.dim }
        Rectangle {
            anchors { right: parent.right; verticalCenter: parent.verticalCenter }
            width: 36; height: 20; radius: 10
            color: onOff.on ? Theme.mark : Theme.bg; border { width: 1; color: onOff.on ? Theme.mark : Theme.faint }
            Rectangle {
                x: onOff.on ? parent.width - width - 3 : 3; anchors.verticalCenter: parent.verticalCenter
                width: 14; height: 14; radius: 7; color: onOff.on ? Theme.bg : Theme.dim
                Behavior on x { NumberAnimation { duration: 90 } }
            }
        }
        MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: onOff.flipped() }
    }
    // A setting of what the chosen mask changes inside it (edit.LOCAL): an amount added to the
    // whole image's, nothing at rest.
    readonly property var localEdit: wheel.mask > 0 && wheel.mask <= session.masks.length ? (session.masks[wheel.mask - 1].edit || {}) : ({})
    // (Which of these settings the chosen mask has of its own, and taking them away again.)
    function leaveMask() { wheel.mask = 0 }
    // What a mask is called: the name it was given, or its kind and its place in the list.
    function maskName(number) {
        const mask = session.masks[number - 1]
        if (!mask) return "Mask"
        // Counted among its own kind: Radial 1, Linear 1, Radial 2. (By its place in the whole
        // list it read Radial 1, Linear 2, Radial 3; the owner: "the name numbering doesnt
        // really make sense".)
        const nth = session.masks.slice(0, number).filter(m => m.kind === mask.kind).length
        return mask.name || ({ radial: "Radial ", linear: "Linear ", brush: "Brush " })[mask.kind] + nth
    }
    // A name being typed in a list's row, in the name's place: Enter or leaving the field
    // takes it (docs/PLAN.md §5), Escape does not. (The owner: "rename masks and checkpoints".)
    component NameEntry: Rectangle {
        id: nameEntry
        signal taken(string name)
        function begin(name) { field.text = name; visible = true; field.forceActiveFocus(); field.selectAll() }
        function end(keep) {
            if (!visible) return
            visible = false
            if (keep && field.text.trim() !== "") taken(field.text.trim())
            root.forceActiveFocus()
        }
        visible: false
        height: 22; radius: 4; color: Theme.bg; border { width: 1; color: Theme.text }
        TextInput {
            id: field
            objectName: "name-entry"
            anchors { fill: parent; leftMargin: 6; rightMargin: 6 }
            verticalAlignment: TextInput.AlignVCenter; clip: true; maximumLength: 40
            color: Theme.text; selectionColor: Theme.faint; selectedTextColor: Theme.text
            font.family: Theme.font; font.pixelSize: Theme.fontSize
            Accessible.name: "Name"
            // (Its keys are its own: Enter is not the window's, the arrows not the photos'.)
            Keys.onReturnPressed: (event) => { nameEntry.end(true); event.accepted = true }
            Keys.onEnterPressed: (event) => { nameEntry.end(true); event.accepted = true }
            Keys.onEscapePressed: (event) => { nameEntry.end(false); event.accepted = true }
            onActiveFocusChanged: if (!activeFocus) nameEntry.end(true)
        }
    }
    function localHas(keys) { return keys.some(key => root.localEdit[key] !== undefined) }
    component LocalCard: Panel {
        property var keys: []
        property bool tinted: false   // it works on colors: not there for a black and white photo
        local: true; changed: root.localHas(keys)
        visible: root.mode === "masks" && maskRow.mask !== null && !(tinted && root.mono)
        onReset: session.resetMaskEdit(wheel.mask, keys)
        onCompare: (held) => session.setMaskBypass(wheel.mask, held ? keys : [])
    }
    component LocalSlider: ParamSlider {
        property string key
        objectName: "local:" + key
        width: parent.width
        neutral: 0; signed: true
        value: root.localEdit[key] || 0
        onMoved: (v) => session.setMaskEdit(wheel.mask, key, v)
        onReleased: session.commitEdit()
        onReset: { session.setMaskEdit(wheel.mask, key, 0); session.commitEdit() }
        onTyped: root.forceActiveFocus()
    }
    component PinSlider: ParamSlider {
        property string key   // a setting of the pin being worked on (edit.PIN)
        objectName: "pin:" + key
        width: parent.width
        // (There with no pin too, dimmed: the section keeps its height, see panel:pins.)
        enabled: pinPanel.has; opacity: enabled ? 1 : 0.35
        value: pinPanel.has ? pinPanel.shown[key] : neutral
        onMoved: (v) => wheel.change(wheel.place(), { [key]: v })
        onReleased: session.commitEdit()
        onReset: { wheel.change(wheel.place(), { [key]: neutral }); session.commitEdit() }
        onTyped: root.forceActiveFocus()
    }
    // Going to a checkpoint puts its edit in place of the present one. Where the present one is
    // in no checkpoint, that would lose it: ask first.
    function goTo(point) {
        if (point.state === "here") return
        if (!session.unsaved) return session.goToCheckpoint(point.id)
        confirm.ask("Discard the changes?",
                    "This photo has changes that are in no checkpoint. Going to “" + point.name + "” discards them.",
                    "Discard changes", () => session.goToCheckpoint(point.id))
    }
    // A column left of the photo, which then stands between the two (the owner:
    // "maybe we should also have a left-side panel so that the image we're editing is in the
    // middle and we can put checkpoints at the bottom there", "maybe folder selection could be
    // the main purpose of that left panel"). Where the photos come from and what tells about
    // the open one are here; what changes it is on the right.
    Rectangle {
        id: leftPane
        objectName: "box:left"
        visible: root.editing && session.source !== ""
        anchors { left: parent.left; top: topBar.bottom; bottom: foot.top }
        width: root.leftWide; color: Theme.panel
        PaneEdge { objectName: "left-edge"; anchors { right: parent.right; top: parent.top; bottom: parent.bottom } way: 1 }
        Rectangle { anchors { right: parent.right; top: parent.top; bottom: parent.bottom } width: 1; color: Theme.line }
        readonly property var places: library.places
        // A window too low for all of it: the cards over Checkpoints are folded for as long
        // as it is, the Photo card first and Folders last, only when
        // even its two rows have no room (the owner's rule). What the user had open is not
        // forgotten: with room again, it is open again.
        property real aboutHigh: 250   // the Photo card open, as last measured
        readonly property var fits: {
            const room = points.y - 16, folded = 46
            const need = (p, f) => (p ? aboutHigh : folded) + (f ? 138 : folded)
            let p = root.unfolded.about === true, f = root.unfolded.folders === true
            if (need(p, f) > room) p = false
            if (need(p, f) > room) f = false
            return { about: p, folders: f }
        }
        // Where the photos come from: the library's folders, as a list in a dark box like the
        // masks' and the checkpoints' (more by the wheel), right over Checkpoints
        // (the owner: "a scrollable list just like the others ... right above checkpoints").
        Panel {
            id: foldersCard
            objectName: "panel:folders"
            title: "Folders"; card: "folders"
            visible: points.visible
            // From under the cards above down to Checkpoints, whatever room that is (the
            // owner: "resize dynamically to fill the space above"); never under two rows.
            // (It folds too. It stands under the cards above it like any card, and open it
            // reaches down to Checkpoints; the owner: "anchored to the top like the other
            // cards and size toward the bottom".)
            folds: true; open: leftPane.fits.folders; onFold: root.fold("folders")
            x: 16; width: parent.width - 32; y: leftTop.y + leftTop.height
            readonly property real boxHigh: Math.max(2 * Theme.row + 8, points.y + 16 - 74 - (leftTop.y + leftTop.height))
            Rectangle {
                objectName: "folder-list"
                width: parent.width; height: foldersCard.boxHigh; radius: 6; color: Theme.bg; border { width: 1; color: Theme.line }
                ListView {
                    id: folderList
                    anchors { fill: parent; margins: 4 }
                    clip: true; boundsBehavior: Flickable.StopAtBounds; acceptedButtons: Qt.NoButton
                    model: [{ id: "all", name: "All photos", count: leftPane.places.all || 0 }].concat(leftPane.places.folders || [])
                    MouseArea { parent: folderList.contentItem; width: folderList.width; height: Math.max(folderList.height, folderList.contentHeight); z: -1; acceptedButtons: Qt.NoButton; onWheel: (wheel) => folderBar.turned(wheel) }
                    delegate: Rectangle {
                        required property var modelData
                        width: folderList.width; height: Theme.row; radius: 5
                        color: library.where === modelData.id ? Theme.selected : overPlace.containsMouse ? Theme.raised : "transparent"
                        Accessible.role: Accessible.ListItem; Accessible.name: modelData.name + ", " + modelData.count + " photos"
                        Label {
                            x: 10; anchors.verticalCenter: parent.verticalCenter; width: parent.width - 60; elide: Text.ElideMiddle
                            text: modelData.name; color: library.where === modelData.id ? Theme.text : Theme.dim
                        }
                        Label {
                            anchors { right: parent.right; rightMargin: 10; verticalCenter: parent.verticalCenter }
                            text: modelData.count; font.pixelSize: 12; font.features: { "tnum": 1 }; color: Theme.faint
                        }
                        MouseArea {
                            id: overPlace
                            anchors.fill: parent; hoverEnabled: true; acceptedButtons: Qt.LeftButton | Qt.RightButton
                            // (The folder's photos become the filmstrip; the open photo is the one last looked at there.)
                            onClicked: (mouse) => {
                                if (mouse.button === Qt.RightButton) {
                                    const at = mapToItem(null, mouse.x, mouse.y)
                                    if (modelData.path) menu.openFolder(modelData.path, at.x, at.y)
                                    return
                                }
                                library.show(modelData.id); if (library.count > 0) session.openAt(library.paths(), Math.max(0, library.rowLast()))
                            }
                        }
                    }
                }
                Scrollbar { id: folderBar; of: folderList; step: Theme.row; visible: false }   // (what the wheel scrolls it by)
                MoreBelow { of: folderList; x: 1; width: parent.width - 2; anchors { bottom: parent.bottom; bottomMargin: 1 } radius: 5 }
                MoreBelow { above: true; of: folderList; x: 1; y: 1; width: parent.width - 2; radius: 5 }
            }
        }
        Column {
            id: leftTop
            // (The first card as far from the bar above as from the panel's sides: 6 px.)
            x: 16; y: 14; width: parent.width - 32; spacing: 0
            // What the photo is (the owner: "information about the photo.. metadata and file
            // information in a neat way"): its name and when it was taken; how it was exposed,
            // the four numbers side by side as a camera shows them; with what; how large; where.
            // Only what the file says; nothing is labelled that reads plainly by itself.
            Panel {
                id: aboutCard
                objectName: "panel:about"
                title: "Photo"; card: "photo"
                // (It folds, as the right column's cards do: the owner's wish.
                // Folded, Folders under them has their room.)
                folds: true; open: leftPane.fits.about; onFold: root.fold("about")
                onHeightChanged: if (open && height > 60) leftPane.aboutHigh = height
                readonly property var facts: session.about
                readonly property var exposed: [facts.shutter, facts.aperture, facts.iso, facts.focal].filter(w => w)
                readonly property string picture: session.fullWidth > 0
                    ? session.fullWidth + " × " + session.fullHeight + "  ·  " + (session.fullWidth * session.fullHeight / 1e6).toFixed(1) + " MP" : ""
                Label {
                    objectName: "about:file"
                    width: parent.width; elide: Text.ElideMiddle
                    text: aboutCard.facts.file || ""; color: Theme.text; font.pixelSize: 14; font.weight: Font.DemiBold
                }
                Label {
                    width: parent.width; topPadding: 2; elide: Text.ElideRight
                    text: aboutCard.facts.taken || ""; font.pixelSize: 12; color: Theme.dim
                }
                Item { width: 1; height: 12 }
                Rectangle {   // the exposure: shutter, aperture, ISO, focal length
                    objectName: "about:exposure"
                    visible: aboutCard.exposed.length > 0
                    width: parent.width; height: 44; radius: 7; color: Theme.bg; border { width: 1; color: Theme.line }
                    Row {
                        anchors.fill: parent
                        Repeater {
                            model: aboutCard.exposed
                            Item {
                                required property string modelData
                                required property int index
                                width: parent.width / aboutCard.exposed.length; height: parent.height
                                Rectangle { visible: index > 0; width: 1; height: parent.height - 16; anchors.verticalCenter: parent.verticalCenter; color: Theme.line }
                                Label {
                                    anchors.centerIn: parent; text: modelData; color: Theme.text
                                    font.pixelSize: 13; font.features: { "tnum": 1 }
                                }
                            }
                        }
                    }
                }
                Label {
                    visible: text !== ""
                    width: parent.width; topPadding: 6; horizontalAlignment: Text.AlignRight
                    text: aboutCard.facts.bias ? "Exposure compensation " + aboutCard.facts.bias : ""; font.pixelSize: 11; color: Theme.dim
                }
                Item { width: 1; height: 12; visible: aboutCard.exposed.length > 0 }
                Label {
                    visible: text !== ""
                    width: parent.width; elide: Text.ElideRight
                    text: aboutCard.facts.camera || ""; font.pixelSize: 13; color: Theme.text
                }
                Label {
                    visible: text !== ""
                    width: parent.width; topPadding: 2; elide: Text.ElideRight
                    text: aboutCard.facts.lens || ""; font.pixelSize: 12; color: Theme.dim
                }
                Item { width: 1; height: 12 }
                Label {
                    width: parent.width; elide: Text.ElideRight
                    text: [aboutCard.picture, aboutCard.facts.size].filter(w => w).join("  ·  "); font.pixelSize: 12; color: Theme.dim
                }
                Label {
                    width: parent.width; topPadding: 2; elide: Text.ElideMiddle
                    text: aboutCard.facts.folder || ""; font.pixelSize: 11; color: Theme.faint
                }
            }
        }
    }
    Rectangle {
        id: tools
        objectName: "box:tools"
        anchors { right: parent.right; top: topBar.bottom; bottom: foot.top }
        visible: root.editing
        // Wide enough for the wheel, which is the main thing in it, without taking the photo's place.
        width: root.rightWide; color: Theme.panel
        PaneEdge { objectName: "right-edge"; anchors { left: parent.left; top: parent.top; bottom: parent.bottom } way: -1; z: 3 }
        Rectangle { anchors { left: parent.left; top: parent.top; bottom: parent.bottom } width: 1; color: Theme.line }
        // Everything but the checkpoints: it scrolls (by the wheel) when the window is too low
        // for it. A drag never scrolls: it belongs to the slider, curve or pad under it.
        // (Over the cards: a card's ground reaches a little past the column, onto the bar.)
        // (No scrollbar on show: that there is more below, or above, is said as in the lists,
        // by the cards fading out at that edge; the owner's wish. It is what the wheel scrolls by.)
        Scrollbar { id: toolsBar; of: scroller; step: 150; visible: false }
        MoreBelow { objectName: "tools-more"; of: scroller; ground: Theme.bg; z: 2; height: 120; x: 1; width: parent.width - 1; anchors.bottom: parent.bottom }   // (long and into the dark of the window, to be plain: the owner's wish)
        MoreBelow { above: true; of: scroller; ground: Theme.panel; z: 2; height: 28; x: 1; width: parent.width - 1; y: scrollArea.y }
        // The head of the column stays where it is: undo, redo, Reset all, and the tabs; what
        // is under it scrolls away beneath it and fades out at its foot (the owner: "this whole
        // part of the panel ... should be static (pinned) and the rest of the content slides
        // underneath").
        Column {
            id: toolsHead
            objectName: "tools-head"
            x: 16; y: 12; width: parent.width - 32
            spacing: 0
            Item {
                width: parent.width; height: 32
                Row {
                    spacing: 14
                    // (Signs on the new screen, white when they can be pressed: the owner's wish.)
                    TextButton {
                        objectName: "undo"; says: "Undo"; available: session.canUndo; onPressed: session.undo()
                        text: ""; sign: "M7.5 4.5 L3.5 8.5 L7.5 12.5 M3.5 8.5 H12 A4.25 4.25 0 0 1 12 17 H9"
                        strong: true; width: 20; height: 20
                    }
                    TextButton {
                        objectName: "redo"; says: "Redo"; available: session.canRedo; onPressed: session.redo()
                        text: ""; sign: "M12.5 4.5 L16.5 8.5 L12.5 12.5 M16.5 8.5 H8 A4.25 4.25 0 0 0 8 17 H11"
                        strong: true; width: 20; height: 20
                    }
                }
                TextButton {
                    objectName: "reset-all"
                    anchors.right: parent.right
                    // (Red while there is something to reset, as Delete is: the owner's wish.)
                    text: "Reset all"; danger: true; available: session.edited; onPressed: session.resetAll()
                }
                Rectangle { y: 25; width: parent.width; height: 1; color: Theme.line }
                Item { width: 1; height: 8 }
            }
            Item { width: 1; height: 8 }
            // What the column is showing. Adjust is the photo's sections, in the order
            // a photo is worked on; Masks is a part of the picture with settings of its own;
            // Crop is the frame. (The owner: "we need a Mask tab", "the Lightroom approach".)
            Rectangle {
                objectName: "modes"
                visible: session.source !== ""
                // (As high as the Tone Wheel's row of tones: the owner's wish.)
                width: parent.width; height: 38; radius: 7; color: Theme.raised
                Row {
                    x: 2; y: 2
                    Repeater {
                        id: tabRow
                        model: [["edit", "Adjust"], ["masks", "Masks"], ["crop", "Crop"]].concat(root.presetsOn ? [["presets", "Presets"]] : [])
                        Segment {
                            required property var modelData
                            objectName: "mode:" + modelData[0]
                            width: (develop.width - 4) / tabRow.count; height: 34; label: modelData[1]
                            on: root.mode === modelData[0]; onChosen: root.go(modelData[0])
                        }
                    }
                }
            }
        }
        Item {   // (what scrolls is cut off above and below, but a section's ground reaches to the column's sides)
        id: scrollArea
        anchors { left: parent.left; right: parent.right; top: toolsHead.bottom; bottom: parent.bottom }
        clip: true
        Flickable {
            id: scroller
            objectName: "tools-scroll"
            anchors { fill: parent; leftMargin: 16; rightMargin: 16 }
            // (The last card ends as far above the panel's foot as the cards are from its
            // sides, 6 px: a card's own foot leaves 18 below it open, 10 folded.)
            readonly property real tail: (root.mode === "edit" ? !grainPanel.open : root.mode === "masks" && wheelBlock.visible && !wheelBlock.open) ? 10 : 18
            contentHeight: develop.height - (tail - 6)
            // (Never left scrolled past its end when what it holds gets shorter, and at its top
            // when another tab is chosen: it showed nothing, and nothing there took a click.)
            onContentHeightChanged: contentY = Math.max(0, Math.min(contentY, contentHeight - height))
            Connections { target: root; function onModeChanged() { scroller.contentY = 0 } }
            boundsBehavior: Flickable.StopAtBounds; acceptedButtons: Qt.NoButton
            // (Under everything that scrolls here: the wheel, where nothing else wants it.)
            MouseArea { anchors.fill: parent; z: -1; acceptedButtons: Qt.NoButton; onWheel: (wheel) => { scrolling.restart(); toolsBar.turned(wheel) } }
            Timer { id: scrolling; interval: 400 }   // the column was scrolled a moment ago
        Column {
            id: develop
            width: parent.width
            spacing: 0
            // (Less under the tabs where the profile's row follows, which has room of its own
            // round its words: the two gaps, over it and under it, then look alike.)
            Item { width: 1; height: root.mode === "edit" && session.develops ? 8 : 16 }
            // Color, or black and white (the key M): what the cards under the tone curve are.
            Item {
                objectName: "treatment"
                visible: root.mode === "edit" && session.source !== ""
                width: parent.width; height: 34
                Label {
                    anchors.verticalCenter: parent.verticalCenter; text: "Treatment"; font.weight: Font.DemiBold
                    color: session.unsavedKeys.includes("mono") ? Theme.yellow : Theme.text
                }
                Rectangle {
                    width: treatments.width + 4; height: 24; radius: 6; color: Theme.raised
                    anchors { right: parent.right; verticalCenter: parent.verticalCenter }
                    Row {
                        id: treatments
                        anchors.centerIn: parent
                        Segment { objectName: "treatment:color"; width: 76; height: 20; label: "Color"; on: !root.mono; onChosen: { root.setMono(false); root.forceActiveFocus() } }
                        Segment { objectName: "treatment:mono"; width: 110; height: 20; label: "Black & white"; on: root.mono; onChosen: { root.setMono(true); root.forceActiveFocus() } }
                    }
                }
            }
            ProfileRow { visible: root.mode === "edit" && session.source !== "" }
            Item { width: 1; height: 14; visible: root.mode === "edit" && session.develops }
            // Presets: looks kept by name. Each is shown as the open photo would be with it;
            // a click lays it on the photo, as one step of undo. Under them, keeping the
            // open photo's look as one: its name, and which parts of the edit it carries (what
            // a preset does not carry it leaves alone on the photo it is laid on).
            Panel {
                objectName: "panel:presets"
                id: presetsPanel
                title: "Presets"; card: "presets"; visible: root.mode === "presets" && session.source !== ""
                Label {
                    visible: presets.all.length === 0
                    width: parent.width; height: implicitHeight + 14; wrapMode: Text.WordWrap; font.pixelSize: 12; color: Theme.faint
                    text: "No presets yet. A preset is a look kept by name: edit a photo, then save its look below, and it can be laid on any other photo."
                }
                Flow {
                    width: parent.width; spacing: 8
                    Repeater {
                        model: presets.all
                        Item {
                            id: tile
                            required property var modelData
                            required property int index
                            objectName: "preset:" + index
                            width: Math.floor((presetsPanel.width - 8) / 2)
                            // (As high as the photo is at this width; an upright one no higher than
                            // five quarters of it, and then narrower: never stretched.)
                            readonly property real shape: session.fullWidth / Math.max(1, session.fullHeight)
                            readonly property real high: Math.round(Math.min(width * 1.25, width / shape))
                            height: high + 30
                            Accessible.role: Accessible.Button; Accessible.name: "Use the preset " + modelData.name
                            Rectangle {
                                width: parent.width; height: tile.high; radius: 5; color: Theme.bg; clip: true
                                border { width: 1; color: overTile.containsMouse ? Theme.dim : Theme.line }
                                Pipeline {
                                    anchors.centerIn: parent
                                    height: tile.high - 2; width: Math.round(Math.min(tile.width - 2, height * tile.shape))
                                    visible: stage.ready
                                    source: stage.picture; painted: stage.painted; lutPicture: stage.lut
                                    // (Worked out only while the tab is on show: every change of the edit asks for it anew.)
                                    edit: root.mode === "presets" ? (session.uniforms, session.numbersWith(tile.modelData.look, tile.modelData.parts)) : null
                                    peak: 1; hdrLift: 0; clipping: 0
                                }
                            }
                            MouseArea {
                                id: overTile
                                anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: { session.pasteLook(tile.modelData.look, tile.modelData.parts); root.forceActiveFocus() }
                            }
                            Label {
                                y: tile.high + 6; width: parent.width - (tileButtons.visible ? tileButtons.width + 8 : 0); elide: Text.ElideRight
                                text: tile.modelData.name; color: overTile.containsMouse ? Theme.text : Theme.dim; visible: !tileEntry.visible
                            }
                            Row {
                                id: tileButtons
                                y: tile.high + 6; anchors.right: parent.right; spacing: 10
                                visible: (overTile.containsMouse || tilePencil.containsPointer || tileBin.containsPointer) && !tileEntry.visible
                                TextButton { id: tilePencil; objectName: "preset-rename:" + tile.index; sign: Theme.pencilSign; says: "Rename"; glow: Theme.green; onPressed: tileEntry.begin(tile.modelData.name) }
                                TextButton {
                                    id: tileBin; objectName: "preset-delete:" + tile.index; sign: Theme.binSign; says: "Delete"; glow: Theme.red
                                    onPressed: confirm.ask("Delete the preset “" + tile.modelData.name + "”?", "It cannot be brought back. Photos it was used on stay as they are.",
                                                           "Delete", () => presets.remove(tile.index))
                                }
                            }
                            NameEntry {
                                id: tileEntry
                                y: tile.high + 4; width: parent.width
                                onTaken: (name) => presets.rename(tile.index, name)
                            }
                        }
                    }
                }
            }
            Panel {
                objectName: "panel:preset-save"
                id: presetSave
                title: "Save this look"; card: "presets"; visible: root.mode === "presets" && session.source !== ""
                property var parts: ["light", "curve", "color", "mono", "detail", "lens", "grain"]   // (not the white balance, which is the photo's light: the usual choice)
                readonly property var named: [["light", "Exposure"], ["balance", "White balance"], ["curve", "Tone curve"], ["color", "Color"], ["mono", "Black & white"], ["detail", "Detail"], ["lens", "Lens"], ["grain", "Film grain"]]
                readonly property bool taken: (presets.all, presets.taken(presetName.text))
                function keep() {
                    if (presets.save(presetName.text, session.look, parts)) { presetName.text = ""; root.forceActiveFocus() }
                }
                Label { width: parent.width; height: implicitHeight + 8; wrapMode: Text.WordWrap; font.pixelSize: 12; color: Theme.faint; text: "What the preset carries. The crop and the masks are never part of one." }
                Flow {
                    width: parent.width; spacing: 4; bottomPadding: 10
                    Repeater {
                        model: presetSave.named
                        Rectangle {
                            id: partChip
                            required property var modelData
                            readonly property bool on: presetSave.parts.includes(modelData[0])
                            objectName: "preset-part:" + modelData[0]
                            width: partName.implicitWidth + 20; height: 26; radius: 5
                            color: on ? Theme.selected : Theme.raised; border { width: on ? 1 : 0; color: Theme.dim }
                            Accessible.role: Accessible.CheckBox; Accessible.name: modelData[1]; Accessible.checked: on
                            Label { id: partName; anchors.centerIn: parent; text: partChip.modelData[1]; font.pixelSize: 12; color: partChip.on ? Theme.text : Theme.faint }
                            MouseArea {
                                anchors.fill: parent
                                onClicked: presetSave.parts = partChip.on ? presetSave.parts.filter(part => part !== partChip.modelData[0]) : presetSave.parts.concat([partChip.modelData[0]])
                            }
                        }
                    }
                }
                Item {
                    width: parent.width; height: 32
                    Rectangle {
                        anchors { left: parent.left; right: savePreset.left; rightMargin: 8; verticalCenter: parent.verticalCenter }
                        height: 30; radius: 6; color: Theme.bg; border { width: 1; color: presetSave.taken ? Theme.red : presetName.activeFocus ? Theme.text : Theme.faint }
                        TextInput {
                            id: presetName
                            objectName: "preset-name"
                            anchors { fill: parent; leftMargin: 10; rightMargin: 10 }
                            verticalAlignment: TextInput.AlignVCenter; clip: true; maximumLength: 60
                            color: Theme.text; selectionColor: Theme.faint; selectedTextColor: Theme.text
                            font.family: Theme.font; font.pixelSize: Theme.fontSize
                            Accessible.name: "The preset's name"
                            Keys.onReturnPressed: (event) => { presetSave.keep(); event.accepted = true }
                            Keys.onEnterPressed: (event) => { presetSave.keep(); event.accepted = true }
                            Keys.onEscapePressed: (event) => { root.forceActiveFocus(); event.accepted = true }
                            Label { visible: parent.text === "" && !parent.activeFocus; anchors.verticalCenter: parent.verticalCenter; text: "Name"; color: Theme.faint }
                        }
                    }
                    PushButton {
                        id: savePreset
                        objectName: "preset-keep"
                        anchors { right: parent.right; verticalCenter: parent.verticalCenter }
                        label: "Save preset"; opacity: presetName.text.trim() !== "" && presetSave.parts.length && !presetSave.taken ? 1 : 0.4
                        onPressed: presetSave.keep()
                    }
                }
                Label {
                    objectName: "preset-taken"
                    visible: presetSave.taken
                    width: parent.width; height: implicitHeight + 8; verticalAlignment: Text.AlignBottom; font.pixelSize: 12; color: Theme.red
                    text: "A preset of that name is there already."
                }
            }
            // The masks, as a list and nothing else; whatever can be set of the one chosen is
            // in the cards under it (the owner: "it just needs to be a list of masks and ANY
            // settings (including feather) should be in the other cards below").
            Panel {
                objectName: "panel:masks"
                title: "Masks"; card: "masks"; visible: root.mode === "masks" && session.source !== ""
                // (A box darker than the card, to read as a list; five rows high, more by
                // scrolling: the owner's wish.)
                Rectangle {
                    objectName: "mask-list"
                    width: parent.width; height: 5 * Theme.row + 8; radius: 6; color: Theme.bg; border { width: 1; color: Theme.line }
                    ListView {
                        id: maskList
                        anchors { fill: parent; margins: 4 }
                        clip: true; boundsBehavior: Flickable.StopAtBounds; acceptedButtons: Qt.NoButton
                        model: session.masks.length
                        // (Under the rows: the wheel scrolls the list, and a click on its empty
                        // part lets go of the mask. `wheel` in here is this area's own signal.)
                        MouseArea { parent: maskList.contentItem; width: maskList.width; height: Math.max(maskList.height, maskList.contentHeight); z: -1; onWheel: (wheel) => root.listWheel(maskBar, wheel); onClicked: root.leaveMask() }
                        delegate: Rectangle {
                            required property int index
                            readonly property bool chosen: wheel.mask === index + 1
                            objectName: "mask-item:" + (index + 1)
                            width: maskList.width; height: Theme.row; radius: 5
                            color: chosen ? Theme.selected : overMask.containsMouse ? Theme.raised : "transparent"
                            id: maskItem
                            readonly property bool off: (session.masks[index] || {}).off === true
                            Accessible.role: Accessible.ListItem; Accessible.name: maskName.text; Accessible.selected: chosen
                            Label {
                                id: maskName
                                x: 34; anchors.verticalCenter: parent.verticalCenter; color: maskItem.off ? Theme.faint : parent.chosen ? Theme.text : Theme.dim
                                width: parent.width - x - (renameMask.visible ? renameMask.width + 16 : 8); elide: Text.ElideRight
                                visible: !maskEntry.visible
                                text: (session.masks, root.maskName(index + 1))
                            }
                            // (Not set from inside the MouseArea: there `wheel` is its own signal.)
                            function choose() { wheel.mask = index + 1 }
                            // A click chooses it, a second one names it.
                            MouseArea { id: overMask; anchors.fill: parent; hoverEnabled: true; onClicked: parent.choose(); onDoubleClicked: maskEntry.begin(maskName.text) }
                            // The eye: the mask on or off, without losing anything of it (the
                            // owner: "an eye button for each mask in the list").
                            Item {
                                objectName: "mask-eye:" + (index + 1)
                                x: 4; width: 26; height: parent.height
                                Accessible.role: Accessible.CheckBox; Accessible.name: "Mask is on"; Accessible.checked: !maskItem.off
                                Icon {
                                    anchors.centerIn: parent; width: 16; height: 16
                                    color: maskItem.off ? Theme.faint : overEye.containsMouse ? Theme.text : Theme.dim
                                    path: maskItem.off ? "M2 10 C5 5 15 5 18 10 C15 15 5 15 2 10 Z M3.5 3.5 L16.5 16.5"
                                                       : "M2 10 C5 5 15 5 18 10 C15 15 5 15 2 10 Z M10 7.6 a2.4 2.4 0 1 0 0.01 0 Z"
                                }
                                MouseArea {
                                    id: overEye
                                    anchors.fill: parent; hoverEnabled: true
                                    onClicked: { session.setMask(index + 1, { off: !maskItem.off }); session.commitEdit() }
                                }
                            }
                            // A row's two buttons, as a checkpoint's has them (the owner: "a pencil
                            // and trash can icon", the bin red and the pencil green under the
                            // pointer). Delete takes the mask and its pins, in one step of undo.
                            Row {
                                id: renameMask
                                anchors { right: parent.right; rightMargin: 8; verticalCenter: parent.verticalCenter }
                                spacing: 12
                                visible: (overMask.containsMouse || maskPencil.containsPointer || maskBin.containsPointer) && !maskEntry.visible
                                TextButton { id: maskPencil; objectName: "mask-rename:" + (index + 1); sign: Theme.pencilSign; says: "Rename"; glow: Theme.green; onPressed: maskEntry.begin(maskName.text) }
                                TextButton { id: maskBin; objectName: "mask-delete:" + (index + 1); sign: Theme.binSign; says: "Delete"; glow: Theme.red; onPressed: session.deleteMask(index + 1) }
                            }
                            NameEntry {
                                id: maskEntry
                                x: 30; width: parent.width - x - 6; anchors.verticalCenter: parent.verticalCenter
                                onTaken: (name) => { session.setMask(index + 1, { name: name }); session.commitEdit() }
                            }
                        }
                        // (The one chosen is in sight, also one just added below the fifth.)
                        Connections { target: wheel; function onMaskChanged() { if (wheel.mask > 0) maskList.positionViewAtIndex(wheel.mask - 1, ListView.Contain) } }
                    }
                    // (Not on show: it is what the wheel scrolls the list by.)
                    Scrollbar { id: maskBar; of: maskList; step: Theme.row; visible: false; anchors { right: parent.right; rightMargin: 1; top: maskList.top; bottom: maskList.bottom } }
                    MoreBelow { objectName: "masks-more"; of: maskList; x: 1; width: parent.width - 2; anchors { bottom: parent.bottom; bottomMargin: 1 } radius: 5 }
                    MoreBelow { above: true; of: maskList; x: 1; y: 1; width: parent.width - 2; radius: 5 }
                    Label {
                        visible: session.masks.length === 0
                        anchors.centerIn: parent; text: "No masks yet"; color: Theme.faint; font.pixelSize: 12
                    }
                }
                // Adding one, and what is done to the one chosen as a whole (the owner: "the
                // Invert/Delete buttons should be in the MASKS card"; their buttons are
                // mask-row's, given this place in the new screen).
                Item {
                    id: maskActions
                    width: parent.width; height: 32
                    Row {
                        visible: session.masks.length < 8
                        anchors.verticalCenter: parent.verticalCenter; spacing: 16
                        TextButton { objectName: "add-mask:radial"; text: "+ Radial"; onPressed: { wheel.mask = session.addMask("radial"); root.forceActiveFocus() } }
                        TextButton { objectName: "add-mask:linear"; text: "+ Linear"; onPressed: { wheel.mask = session.addMask("linear"); root.forceActiveFocus() } }
                        TextButton { objectName: "add-mask:brush"; text: "+ Brush"; onPressed: { wheel.mask = session.addMask("brush"); root.forceActiveFocus() } }
                    }
                }
            }
            Panel {
                id: maskCard
                objectName: "panel:mask-settings"
                readonly property var mask: wheel.mask > 0 && wheel.mask <= session.masks.length ? session.masks[wheel.mask - 1] : null
                readonly property bool brush: mask !== null && mask.kind === "brush"
                title: (session.masks, root.maskName(wheel.mask)); card: "mask"
                visible: root.mode === "masks" && mask !== null
                Item { id: maskSettingsSlot; width: parent.width; height: Theme.slider; visible: !maskCard.brush }
                // A painted mask: what the left button does on the photo, and with what
                // brush (the owner: "a paintable mask with a brush and eraser", both with
                // "Feathering and Size"). The brush is the same for every painted mask and
                // kept from one day to the next.
                Item {
                    objectName: "brush-tool"
                    width: parent.width; height: 34; visible: maskCard.brush
                    Label { anchors.verticalCenter: parent.verticalCenter; text: "Paint with" }
                    Rectangle {
                        width: brushTools.width + 4; height: 24; radius: 6; color: Theme.raised
                        anchors { right: parent.right; verticalCenter: parent.verticalCenter }
                        Row {
                            id: brushTools
                            anchors.centerIn: parent
                            Segment { objectName: "brush:paint"; width: 76; height: 20; label: "Brush"; on: !stage.erasing; onChosen: stage.erasing = false }
                            Segment { objectName: "brush:erase"; width: 76; height: 20; label: "Eraser"; on: stage.erasing; onChosen: stage.erasing = true }
                        }
                    }
                }
                ParamSlider {
                    objectName: "brush:size"
                    visible: maskCard.brush; width: parent.width
                    name: "Size"; unit: "%"; from: 0.3; to: 30; neutral: 5; step: 0.1; digits: 1; logarithmic: true; signed: false
                    value: stage.brushSize * 100
                    onMoved: (v) => stage.brushSize = v / 100
                    onReleased: session.remember("brush/size", String(stage.brushSize))
                    onReset: { stage.brushSize = 0.05; session.remember("brush/size", "0.05") }
                    onTyped: root.forceActiveFocus()
                }
                ParamSlider {
                    objectName: "brush:feather"
                    visible: maskCard.brush; width: parent.width
                    name: "Feather"; unit: "%"; from: 0; to: 100; neutral: 50; step: 1; digits: 0; signed: false
                    value: stage.brushFeather * 100
                    onMoved: (v) => stage.brushFeather = v / 100
                    onReleased: session.remember("brush/feather", String(stage.brushFeather))
                    onReset: { stage.brushFeather = 0.5; session.remember("brush/feather", "0.5") }
                    onTyped: root.forceActiveFocus()
                }
                // Inside out: the mask works everywhere but where it is drawn (the owner: "maybe
                // Inverted should be a toggle button on the card of the mask").
                OnOff {
                    objectName: "mask:inverted"
                    label: "Inverted"
                    on: maskRow.mask !== null && maskRow.mask.invert
                    onFlipped: { session.setMask(wheel.mask, { invert: !maskRow.mask.invert }); session.commitEdit() }
                }
                Label {
                    width: parent.width; height: implicitHeight + 12; verticalAlignment: Text.AlignBottom; wrapMode: Text.WordWrap; color: Theme.faint; font.pixelSize: 12
                    text: maskCard.brush ? "Paint on the photo; the mask shows in red while the pointer is there. Ctrl and the wheel size the brush. What is set below changes the picture inside it only."
                                              : "Hold H to see the mask. What is set below changes the picture inside it only."
                }
            }
            // What the chosen mask changes inside it: the cards of Adjust, each setting an
            // amount added to the whole image's (the owner: "a mask should have ALL the cards
            // available as long as it makes sense for them"). The sliders add to the whole
            // image's; a curve or the color EQ, once changed here, holds inside the mask in
            // place of the whole image's. The Tone Wheel under these works inside it too.
            LocalCard {
                objectName: "panel:local-light"
                title: "Exposure"; card: "light"; keys: ["exposure", "contrast", "brightness", "highlights", "midtones", "shadows", "whites", "blacks", "curve"]
                LocalSlider { key: "exposure"; name: "Exposure"; unit: "EV"; from: -5; to: 5; step: 0.01; digits: 2 }
                LocalSlider { key: "contrast"; name: "Contrast"; from: -100; to: 100; step: 1; digits: 0 }
                LocalSlider { key: "brightness"; name: "Brightness"; from: -100; to: 100; step: 1; digits: 0 }
                Item { width: 1; height: 16 }   // between two groups: room, no line (a line reads as a slider's track)
                LocalSlider { key: "highlights"; name: "Highlights"; from: -100; to: 100; step: 1; digits: 0 }
                LocalSlider { key: "midtones"; name: "Midtones"; from: -100; to: 100; step: 1; digits: 0 }
                LocalSlider { key: "shadows"; name: "Shadows"; from: -100; to: 100; step: 1; digits: 0 }
                Item { width: 1; height: 16 }   // between two groups: room, no line (a line reads as a slider's track)
                LocalSlider { key: "whites"; name: "Whites"; from: -100; to: 100; step: 1; digits: 0 }
                LocalSlider { key: "blacks"; name: "Blacks"; from: -100; to: 100; step: 1; digits: 0 }
                Item { width: 1; height: 16 }
                CurveEditor {
                    width: parent.width; height: 220; onDone: root.forceActiveFocus()
                    key: "curve"; local: true; mask: wheel.mask; readonly property string lineText: (session.masks, JSON.stringify(session.localLine(wheel.mask, "curve"))); line: JSON.parse(lineText);   // (by way of its text: CurveEditor.qml says why)
                    tally: counts; behind: "light"
                }
            }
            LocalCard {
                objectName: "panel:local-balance"
                title: "White balance"; card: "balance"; keys: ["temperature", "tint"]
                LocalSlider { key: "temperature"; name: "Temperature"; unit: "K"; from: -4000; to: 4000; step: 10; digits: 0 }
                LocalSlider { key: "tint"; name: "Tint"; from: -150; to: 150; step: 1; digits: 0 }
            }
            LocalCard {
                objectName: "panel:local-saturation"
                title: "Saturation & Vibrance"; card: "saturation"; keys: ["saturation", "vibrance"]; tinted: true
                LocalSlider { key: "saturation"; name: "Saturation"; from: -100; to: 100; step: 1; digits: 0 }
                LocalSlider { key: "vibrance"; name: "Vibrance"; from: -100; to: 100; step: 1; digits: 0 }
            }
            LocalCard {
                objectName: "panel:local-chroma"
                title: "Chroma curve"; card: "chroma"; keys: ["chroma"]; tinted: true
                CurveEditor {
                    width: parent.width; height: 220; onDone: root.forceActiveFocus()
                    key: "chroma"; local: true; mask: wheel.mask; readonly property string lineText: (session.masks, JSON.stringify(session.localLine(wheel.mask, "chroma"))); line: JSON.parse(lineText);
                    tally: counts; behind: "chroma"
                    low: "muted"; high: "vivid"
                }
            }
            LocalCard {
                objectName: "panel:local-eq"
                title: "Color EQ"; card: "eq"; keys: ["eq"]; tinted: true
                HueEditor { width: parent.width; height: 200; onDone: root.forceActiveFocus(); local: true; mask: wheel.mask; tally: counts }
            }
            // The cards of Adjust, in the order a photo is worked on.
            Panel {
                objectName: "panel:tone"
                id: tonePanel
                title: "Exposure"; tool: "tone"; card: "light"; visible: session.source !== "" && root.shows("tone")
                folds: true; open: root.isOpen("tone"); onFold: root.fold("tone")
                // The histogram is here, over what moves it (the owner: "the histogram should
                // probably be a part of the LIGHT card"); it was a card of the left panel.
                Histogram { width: parent.width; height: 110; color: "transparent"; border.width: 0; tally: counts }
                OnOff {
                    objectName: "clipping-switch"
                    label: "Show clipping"
                    on: session.panels.clipping
                    onFlipped: session.setPanel("clipping", !session.panels.clipping)
                }
                Item { width: 1; height: 8 }
                // Its zero is the photo as shot, where every photo starts, and a double click
                // puts it there. (A raw file is shown as its camera's own picture by the tone
                // curve at the card's foot, which starts as the photo's own: edit.curve_of.)
                EditSlider { key: "exposure"; name: "Exposure"; unit: "EV"; from: -5; to: 5; step: 0.01; digits: 2; rest: 0 }
                EditSlider { key: "contrast"; name: "Contrast"; from: -100; to: 100; step: 1; digits: 0 }
                EditSlider { key: "brightness"; name: "Brightness"; from: -100; to: 100; step: 1; digits: 0 }
                Item { width: 1; height: 16 }   // between two groups: room, no line (a line reads as a slider's track)
                EditSlider { key: "highlights"; name: "Highlights"; from: -100; to: 100; step: 1; digits: 0 }
                EditSlider { key: "midtones"; name: "Midtones"; from: -100; to: 100; step: 1; digits: 0 }
                EditSlider { key: "shadows"; name: "Shadows"; from: -100; to: 100; step: 1; digits: 0 }
                Item { width: 1; height: 16 }   // between two groups: room, no line (a line reads as a slider's track)
                EditSlider { key: "whites"; name: "Whites"; from: -100; to: 100; step: 1; digits: 0 }
                EditSlider { key: "blacks"; name: "Blacks"; from: -100; to: 100; step: 1; digits: 0 }
                // The tone curve, at the card's foot (it was a card of its own; the owner:
                // "merge tone curve into Exposure at the bottom").
                Item { width: 1; height: 16 }
                CurveEditor {
                    width: parent.width; height: 250; onDone: root.forceActiveFocus()
                    key: "curve"; line: session.curveLine; tally: counts; behind: "light"
                }
            }
            Panel {
                objectName: "panel:balance"
                id: balancePanel
                title: "White balance"; tool: "balance"; card: "balance"; visible: session.source !== "" && root.shows("balance")
                folds: true; open: root.isOpen("balance"); onFold: root.fold("balance")
                // (Its button out of sight, nothing waits for a pick: another tab, the card folded.)
                readonly property bool shown: visible && open
                onShownChanged: if (!shown && stage.picking === "neutral") stage.picking = ""
                BalancePad {
                    width: parent.width; picking: stage.picking === "neutral"
                    onDone: root.forceActiveFocus()
                    onPick: { stage.picking = picking ? "" : "neutral"; root.forceActiveFocus() }
                }
                // The same two numbers as the pad, in kelvin, where a camera gave them a meaning.
                EditSlider {
                    key: "temperature"; visible: session.develops
                    name: "Temperature"; unit: "K"; from: 2000; to: 15000; logarithmic: true; step: 10; digits: 0
                    neutral: session.shot.temperature
                }
                EditSlider {
                    key: "tint"; visible: session.develops
                    name: "Tint"; from: -150; to: 150; step: 1; digits: 0
                    neutral: session.shot.tint
                }
            }
            // LUTs: looks made elsewhere, as .cube (or .clf) files. Those installed in Photone are listed
            // under "Installed"; one is laid on the photo by dragging it into the photo's list,
            // to the place it is to have there (the owner: "what if i could just drag it out
            // into the list"). Those on the photo are
            // laid on in the order of their list, on the developed picture before the tone
            // curve and everything about colors: each can be switched off (the eye), taken
            // away (the bin), dragged to another place, and has its own Strength.
            Panel {
                objectName: "panel:lut"
                id: lutPanel
                title: "LUTs"; tool: "lut"; card: "lut"; visible: session.source !== "" && root.shows("lut")
                folds: true; open: root.isOpen("lut"); onFold: root.fold("lut")
                readonly property var on: session.lutsOn
                property int chosen: 0
                // An installed LUT being dragged: which, where the pointer is (in the window),
                // and the place in the photo's list it would be let go at (-1: not over it).
                // The list shows what would come of a drag while it goes on: a row being dragged
                // (`lifted`, to the place `aimed`) has the others slide out of its way, and an
                // installed LUT coming in opens a gap where it would land (the owner: "a live
                // animated preview of the LUTs rearranging as i drag it around").
                property int lifted: -1
                property int aimed: -1
                property bool settling: false   // a drag was just let go: the rows take their places at once
                function settle(change) { settling = true; change(); Qt.callLater(() => lutPanel.settling = false) }
                property var carried: null
                property point carriedAt
                readonly property int dropAt: {
                    if (!carried || on.length >= session.lutsMost) return -1
                    const at = lutBox.mapFromItem(root, carriedAt.x, carriedAt.y)
                    if (at.x < 0 || at.y < 0 || at.x > lutBox.width || at.y > lutBox.height) return -1
                    return Math.max(0, Math.min(on.length, Math.round((at.y - 4 + lutScroll.contentY) / Theme.row)))
                }
                // Where the pointer of either drag is, down the list's box: near its top or
                // its foot the list scrolls on, to places that are not in sight.
                readonly property real dragY: lifted >= 0 ? liftedY : carried ? lutBox.mapFromItem(root, carriedAt.x, carriedAt.y).y : -1000
                property real liftedY: 0
                function reveal(row) {   // that row in sight
                    lutScroll.contentY = Math.max(0, Math.min(Math.max(0, on.length * Theme.row - lutScroll.height),
                                                              Math.max((row + 1) * Theme.row - lutScroll.height, Math.min(row * Theme.row, lutScroll.contentY))))
                }
                onChosenChanged: reveal(chosen)
                readonly property var one: chosen >= 0 && chosen < on.length ? on[chosen] : null
                Connections { target: session; function onPhotoChanged() { lutPanel.chosen = 0 } }
                Rectangle {
                    id: lutBox
                    objectName: "lut-list"
                    width: parent.width; height: 4 * Theme.row + 8; radius: 6; color: Theme.bg
                    border { width: 1; color: lutPanel.dropAt >= 0 ? Theme.text : lutPanel.carried ? Theme.dim : Theme.line }
                    Label {
                        visible: lutPanel.on.length === 0
                        anchors.centerIn: parent; color: lutPanel.carried ? Theme.dim : Theme.faint; font.pixelSize: 12
                        text: lutPanel.carried ? "Let go here to lay it on the photo" : "No LUT on this photo. Drag one here from those installed."
                    }
                    Timer {   // a drag held near the box's top or foot: the list scrolls on
                        interval: 16; repeat: true; running: lutPanel.carried !== null || lutPanel.lifted >= 0
                        onTriggered: {
                            const most = Math.max(0, lutScroll.contentHeight - lutScroll.height), y = lutPanel.dragY
                            if (y > -18 && y < 18) lutScroll.contentY = Math.max(0, lutScroll.contentY - 5)
                            else if (y > lutBox.height - 18 && y < lutBox.height + 18) lutScroll.contentY = Math.min(most, lutScroll.contentY + 5)
                        }
                    }
                    Flickable {
                        id: lutScroll
                        objectName: "lut-scroll"
                        anchors { fill: parent; margins: 4 }
                        clip: true; boundsBehavior: Flickable.StopAtBounds; acceptedButtons: Qt.NoButton
                        // (A row more while one is coming in: the gap it would land in.)
                        contentHeight: Math.max(height, (lutPanel.on.length + (lutPanel.dropAt >= 0 ? 1 : 0)) * Theme.row)
                        onContentHeightChanged: contentY = Math.max(0, Math.min(contentY, contentHeight - height))
                        MouseArea { parent: lutScroll.contentItem; width: lutScroll.width; height: lutScroll.contentHeight; z: -1; acceptedButtons: Qt.NoButton; onWheel: (wheel) => root.listWheel(lutBar, wheel) }
                        Rectangle {   // the gap the one coming in would land in
                            objectName: "lut-gap"
                            visible: lutPanel.dropAt >= 0 && lutPanel.on.length > 0
                            y: lutPanel.dropAt * Theme.row + 1; width: parent.width; height: Theme.row - 2; radius: 5
                            color: "transparent"; border { width: 1; color: Theme.dim }
                        }
                        Item {
                            id: lutRows
                            width: lutScroll.width; height: lutScroll.contentHeight
                            Repeater {
                                model: lutPanel.on.length
                                Rectangle {
                                    id: lutRow
                                    required property int index
                                    readonly property var mine: lutPanel.on[index] || ({})
                                    readonly property bool picked: lutPanel.chosen === index
                                    readonly property bool off: mine.off === true
                                    // Dragged up or down, it goes to the place it is let go over: as far
                                    // as the pointer has gone down the list since it was pressed (the
                                    // list may scroll under it meanwhile), and no further than the
                                    // first and the last row's place: never out of the list.
                                    readonly property real pulled: !overLut.held ? 0
                                        : Math.max(-index * Theme.row, Math.min((lutPanel.on.length - 1 - index) * Theme.row,
                                                                               overLut.at - 4 + lutScroll.contentY - overLut.from))
                                    readonly property bool moving: overLut.held && Math.abs(pulled) > 4
                                    readonly property int aim: Math.max(0, Math.min(lutPanel.on.length - 1, Math.round((index * Theme.row + pulled) / Theme.row)))
                                    objectName: "lut-item:" + index
                                    // Where it stands meanwhile: out of the way of a row that is
                                    // being dragged past it, or below the gap of one coming in.
                                    readonly property int place: {
                                        const from = lutPanel.lifted, to = lutPanel.aimed
                                        if (from >= 0 && from !== index) return from < index && index <= to ? index - 1 : to <= index && index < from ? index + 1 : index
                                        return lutPanel.dropAt >= 0 && index >= lutPanel.dropAt ? index + 1 : index
                                    }
                                    onMovingChanged: { lutPanel.aimed = aim; lutPanel.lifted = moving ? index : -1 }
                                    onAimChanged: if (moving) lutPanel.aimed = aim
                                    y: moving ? index * Theme.row + pulled : place * Theme.row; z: moving ? 2 : 0
                                    Behavior on y { enabled: !lutRow.moving && !lutPanel.settling; NumberAnimation { duration: 130; easing.type: Easing.OutCubic } }
                                    width: lutRows.width; height: Theme.row; radius: 5
                                    color: moving ? Theme.raised : picked ? Theme.selected : overLut.containsMouse ? Theme.raised : "transparent"
                                    border { width: moving ? 1 : 0; color: Theme.dim }
                                    Accessible.role: Accessible.ListItem; Accessible.name: (mine.name || "") + (off ? ", off" : ""); Accessible.selected: picked
                                    MouseArea {
                                        id: overLut
                                        anchors.fill: parent; hoverEnabled: true; preventStealing: true
                                        property bool held: false
                                        property real from   // where it was pressed, down the whole list
                                        property real at     // where the pointer is, down the list's box
                                        onPressed: (mouse) => {
                                            at = mapToItem(lutBox, 0, mouse.y).y
                                            from = at - 4 + lutScroll.contentY
                                            held = true
                                            lutPanel.liftedY = at
                                            lutPanel.chosen = lutRow.index
                                        }
                                        onPositionChanged: (mouse) => { if (held) { at = mapToItem(lutBox, 0, mouse.y).y; lutPanel.liftedY = at } }
                                        onReleased: {
                                            // (What is needed of this row and of the card is taken first:
                                            // a change of the photo's list may be the end of this row,
                                            // and nothing here can be named after that. It was: "lutPanel
                                            // is not defined", which the owner saw.)
                                            const to = lutRow.aim, was = lutRow.moving, row = lutRow.index, card = lutPanel, window = root
                                            // (At once, not gliding: the rows already stand where the new order has them.)
                                            card.settle(() => {
                                                held = false
                                                if (was && to !== row) { card.chosen = to; session.moveLut(row, to) }
                                            })
                                            window.forceActiveFocus()
                                        }
                                        onCanceled: held = false
                                    }
                                    Item {
                                        objectName: "lut-eye:" + lutRow.index
                                        x: 4; width: 26; height: parent.height
                                        Accessible.role: Accessible.CheckBox; Accessible.name: "LUT is on"; Accessible.checked: !lutRow.off
                                        Icon {
                                            anchors.centerIn: parent; width: 16; height: 16
                                            color: lutRow.off ? Theme.faint : overLutEye.containsMouse ? Theme.text : Theme.dim
                                            path: lutRow.off ? "M2 10 C5 5 15 5 18 10 C15 15 5 15 2 10 Z M3.5 3.5 L16.5 16.5"
                                                             : "M2 10 C5 5 15 5 18 10 C15 15 5 15 2 10 Z M10 7.6 a2.4 2.4 0 1 0 0.01 0 Z"
                                        }
                                        MouseArea {
                                            id: overLutEye
                                            anchors.fill: parent; hoverEnabled: true
                                            onClicked: { session.commitEdit(); session.setLut(lutRow.index, { off: !lutRow.off }); session.commitEdit() }
                                        }
                                    }
                                    Label {
                                        x: 34; anchors.verticalCenter: parent.verticalCenter; elide: Text.ElideMiddle
                                        width: parent.width - x - lutTail.width - 16
                                        text: (lutRow.mine.name || "") + (lutRow.mine.lost ? "  (not installed any more)" : "")
                                        color: lutRow.mine.lost ? Theme.red : lutRow.off ? Theme.faint : lutRow.picked ? Theme.text : Theme.dim
                                    }
                                    Row {
                                        id: lutTail
                                        anchors { right: parent.right; rightMargin: 8; verticalCenter: parent.verticalCenter } spacing: 10
                                        Label {
                                            anchors.verticalCenter: parent.verticalCenter; visible: !lutBin.visible
                                            text: Math.round(lutRow.mine.strength || 0) + " %"; font.pixelSize: 12; font.features: { "tnum": 1 }; color: Theme.faint
                                        }
                                        TextButton {
                                            id: lutBin
                                            objectName: "lut-delete:" + lutRow.index
                                            anchors.verticalCenter: parent.verticalCenter
                                            visible: (overLut.containsMouse || containsPointer) && !lutRow.moving
                                            sign: Theme.binSign; says: "Take off this photo"; glow: Theme.red
                                            // (The selection first, the row's end last: see onReleased above.)
                                            onPressed: {
                                                const row = lutRow.index
                                                lutPanel.chosen = Math.max(0, Math.min(lutPanel.chosen > row ? lutPanel.chosen - 1 : lutPanel.chosen, lutPanel.on.length - 2))
                                                session.removeLut(row)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Scrollbar { id: lutBar; of: lutScroll; step: Theme.row; visible: false }   // (what the wheel scrolls it by)
                    MoreBelow { objectName: "luts-more"; of: lutScroll; x: 1; width: parent.width - 2; anchors { bottom: parent.bottom; bottomMargin: 1 } radius: 5 }
                    MoreBelow { above: true; of: lutScroll; x: 1; y: 1; width: parent.width - 2; radius: 5 }
                }
                Item { width: 1; height: 8 }
                ParamSlider {
                    objectName: "slider:lut"
                    width: parent.width; enabled: lutPanel.one !== null; opacity: enabled ? 1 : 0.35
                    // (Filled from nothing up to how much there is; a double click is all of it.)
                    name: "Strength"; unit: "%"; from: 0; to: 100; neutral: 0; step: 1; digits: 0; signed: false
                    value: lutPanel.one ? lutPanel.one.strength : 100
                    unsaved: session.unsavedKeys.includes("luts")
                    onMoved: (v) => session.setLut(lutPanel.chosen, { strength: v })
                    onReleased: session.commitEdit()
                    onReset: { session.setLut(lutPanel.chosen, { strength: 100 }); session.commitEdit() }
                    onTyped: root.forceActiveFocus()
                }
                Label { height: 30; verticalAlignment: Text.AlignBottom; bottomPadding: 6; text: "INSTALLED"; font.pixelSize: Theme.smallSize; font.weight: Font.DemiBold; font.letterSpacing: 0.9; color: Theme.dim }
                Rectangle {
                    objectName: "lut-installed"
                    width: parent.width; height: 5 * Theme.row + 8; radius: 6; color: Theme.bg; border { width: 1; color: Theme.line }
                    ListView {
                        id: installedList
                        anchors { fill: parent; margins: 4 }
                        clip: true; boundsBehavior: Flickable.StopAtBounds; acceptedButtons: Qt.NoButton
                        model: session.lutsInstalled
                        MouseArea { parent: installedList.contentItem; width: installedList.width; height: Math.max(installedList.height, installedList.contentHeight); z: -1; acceptedButtons: Qt.NoButton; onWheel: (wheel) => root.listWheel(installedBar, wheel) }
                        delegate: Rectangle {
                            id: shelfRow
                            required property var modelData
                            required property int index
                            objectName: "lut-shelf:" + index
                            width: installedList.width; height: Theme.row; radius: 5
                            color: overShelf.containsMouse ? Theme.raised : "transparent"
                            Accessible.role: Accessible.ListItem; Accessible.name: modelData.name + ", drag onto the photo's list to add"
                            MouseArea {
                                id: overShelf
                                anchors.fill: parent; hoverEnabled: true; cursorShape: pressed ? Qt.ClosedHandCursor : Qt.OpenHandCursor
                                preventStealing: true
                                property point from
                                onPressed: (mouse) => from = Qt.point(mouse.x, mouse.y)
                                onPositionChanged: (mouse) => {
                                    if (!pressed) return
                                    if (!lutPanel.carried && Math.abs(mouse.x - from.x) + Math.abs(mouse.y - from.y) > 4) lutPanel.carried = shelfRow.modelData
                                    lutPanel.carriedAt = mapToItem(root, mouse.x, mouse.y)
                                }
                                onReleased: {
                                    const to = lutPanel.dropAt, which = lutPanel.carried, card = lutPanel, window = root
                                    card.settle(() => {
                                        card.carried = null
                                        if (which && to >= 0) { const at = session.addLut(which.id, to); if (at >= 0) card.chosen = at }
                                    })
                                    window.forceActiveFocus()
                                }
                                onCanceled: lutPanel.carried = null
                            }
                            Label {
                                x: 10; anchors.verticalCenter: parent.verticalCenter; elide: Text.ElideMiddle
                                width: parent.width - x - shelfTail.width - 16
                                text: shelfRow.modelData.name; color: Theme.text
                            }
                            Row {
                                id: shelfTail
                                anchors { right: parent.right; rightMargin: 8; verticalCenter: parent.verticalCenter } spacing: 10
                                Label {
                                    anchors.verticalCenter: parent.verticalCenter; visible: !shelfBin.visible
                                    text: shelfRow.modelData.size + " pt"; font.pixelSize: 12; font.features: { "tnum": 1 }; color: Theme.faint
                                }
                                Label { anchors.verticalCenter: parent.verticalCenter; visible: shelfBin.visible; text: lutPanel.on.length >= session.lutsMost ? "the photo has " + session.lutsMost : "drag to add"; font.pixelSize: 12; color: Theme.dim }
                                TextButton {
                                    id: shelfBin
                                    objectName: "lut-uninstall:" + shelfRow.index
                                    anchors.verticalCenter: parent.verticalCenter
                                    visible: overShelf.containsMouse || containsPointer
                                    sign: Theme.binSign; says: "Uninstall"; glow: Theme.red
                                    onPressed: confirm.ask("Uninstall “" + shelfRow.modelData.name + "”?",
                                                           "Its copy in Photone is deleted. Photos it is laid on keep their entry, which then changes nothing until the LUT is installed again.",
                                                           "Uninstall", () => session.uninstallLut(shelfRow.modelData.id))
                                }
                            }
                        }
                    }
                    Scrollbar { id: installedBar; of: installedList; step: Theme.row; visible: false }
                    MoreBelow { of: installedList; x: 1; width: parent.width - 2; anchors { bottom: parent.bottom; bottomMargin: 1 } radius: 5 }
                    MoreBelow { above: true; of: installedList; x: 1; y: 1; width: parent.width - 2; radius: 5 }
                    Label {
                        visible: session.lutsInstalled.length === 0
                        anchors.centerIn: parent; width: parent.width - 30; horizontalAlignment: Text.AlignHCenter; wrapMode: Text.WordWrap
                        text: "None installed yet. Install .cube or .clf files, or drop one on the photo."; color: Theme.faint; font.pixelSize: 12
                    }
                }
                Rectangle {   // the one being dragged, under the pointer, over everything
                    objectName: "lut-carried"
                    parent: root; z: 20
                    visible: lutPanel.carried !== null
                    x: lutPanel.carriedAt.x + 10; y: lutPanel.carriedAt.y - height / 2
                    width: carriedName.implicitWidth + 24; height: Theme.row; radius: 5
                    color: Theme.raised; border { width: 1; color: lutPanel.dropAt >= 0 ? Theme.text : Theme.dim }
                    Label { id: carriedName; anchors.centerIn: parent; text: lutPanel.carried ? lutPanel.carried.name : ""; color: Theme.text }
                }
                Item {
                    width: parent.width; height: 42
                    PushButton {
                        objectName: "lut-install"
                        anchors { left: parent.left; bottom: parent.bottom }
                        label: "Install LUTs…"; onPressed: lutOpener.open()
                    }
                    PushButton {
                        objectName: "look-table"
                        anchors { right: parent.right; bottom: parent.bottom }
                        label: "Export as one LUT…"; opacity: exporter.running ? 0.4 : 1
                        onPressed: if (!exporter.running) tableSaver.open()
                    }
                }
                Label {
                    width: parent.width; height: implicitHeight + 10; verticalAlignment: Text.AlignBottom; wrapMode: Text.WordWrap; font.pixelSize: 12; color: Theme.faint
                    text: "Export writes one .cube that does all this photo's edit does to colors: these LUTs as they are set, the tone curve, the color tools, black and white. Not masks, crop or grain."
                }
            }
            // The photo's frame: its shape, its straightening, its turns. The frame itself is
            // dragged on the photo, which shows whole while this is open.
            Panel {
                objectName: "panel:crop"
                id: cropPanel
                title: "Crop and straighten"; tool: "crop"; card: "crop"; visible: session.source !== "" && root.shows("crop")
                open: root.isOpen("crop")
                // The frame's shape: the usual ones, or Custom, typed (the owner: "writing a
                // custom ratio doesnt indicate it is being used in any way and there is no way to
                // make it use it other than to press Enter"). A typed shape is used a moment after
                // the typing stops; the Custom chip stays lit while it is. Portrait stands every
                // shape on end.
                property bool custom: false
                function shaped(wide) { return wide > 0 && portrait.on ? 1 / wide : wide }
                Flow {
                    width: parent.width; spacing: 4; bottomPadding: 6
                    // (Two lines: the three that are no fixed shape, then the shapes; the owner's wish.)
                    Component {
                        id: shapeChipMade
                        Rectangle {
                            id: shapeChip
                            required property var modelData
                            readonly property bool mine: modelData[1] === -2
                            // (As shot: the picture's own shape, which stands as it was shot.)
                            readonly property real asked: modelData[1] === -1 ? session.framing.picture : cropPanel.shaped(modelData[1])
                            readonly property bool on: mine ? cropPanel.custom : !cropPanel.custom && Math.abs(stage.cropRatio - Math.max(0, asked)) < 1e-6
                            objectName: "crop-shape:" + modelData[0]
                            width: 58; height: 26; radius: 5
                            color: on ? Theme.selected : Theme.raised
                            border { width: on ? 1 : 0; color: Theme.dim }
                            Accessible.role: Accessible.RadioButton; Accessible.name: modelData[0]; Accessible.checked: on
                            Label { anchors.centerIn: parent; text: shapeChip.modelData[0]; font.pixelSize: 12; color: shapeChip.on ? Theme.text : Theme.dim }
                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    cropPanel.custom = shapeChip.mine
                                    if (!shapeChip.mine) return stage.reshape(shapeChip.asked)
                                    shapeEntry.forceActiveFocus()
                                    shapeEntry.selectAll()
                                    shapeEntry.use()
                                }
                            }
                        }
                    }
                    Repeater { model: [["Free", 0], ["As shot", -1], ["Custom", -2]]; delegate: shapeChipMade }
                    Item { width: parent.width; height: 1 }   // (ends the line: a thing of no height is not laid out)
                    Repeater { model: [["1 : 1", 1], ["5 : 4", 1.25], ["4 : 3", 4 / 3], ["3 : 2", 1.5], ["16 : 9", 16 / 9]]; delegate: shapeChipMade }
                }
                Item {
                    objectName: "crop-custom"
                    visible: cropPanel.custom
                    width: parent.width; height: 36
                    Label { anchors.verticalCenter: parent.verticalCenter; text: "Width : height" }
                    Rectangle {
                        anchors { right: parent.right; verticalCenter: parent.verticalCenter }
                        width: 120; height: 26; radius: 5; color: Theme.bg
                        border { width: 1; color: !shapeEntry.fine ? Theme.red : shapeEntry.activeFocus ? Theme.text : Theme.faint }
                        TextInput {
                            id: shapeEntry
                            objectName: "crop-custom-entry"
                            anchors { fill: parent; leftMargin: 10; rightMargin: 10 }
                            verticalAlignment: TextInput.AlignVCenter; horizontalAlignment: TextInput.AlignRight; clip: true
                            color: Theme.text; selectionColor: Theme.faint; selectedTextColor: Theme.text
                            font.family: Theme.font; font.pixelSize: Theme.fontSize; font.features: { "tnum": 1 }
                            Accessible.name: "Custom shape, width to height"
                            text: "7 : 5"
                            // Width over height of what is typed ("7:5", "2.35"); 0 if it is not a shape.
                            readonly property real asked: {
                                const parts = text.replace(/,/g, ".").split(/[:x×\/ ]+/).filter(part => part !== "").map(Number)
                                const shape = parts.length === 2 ? parts[0] / parts[1] : parts.length === 1 ? parts[0] : 0
                                return isFinite(shape) && shape >= 0.1 && shape <= 10 ? shape : 0
                            }
                            readonly property bool fine: text.trim() === "" || asked > 0
                            function use() {
                                const wanted = cropPanel.shaped(asked)
                                if (asked > 0 && Math.abs(stage.cropRatio - wanted) > 1e-6) stage.reshape(wanted)
                            }
                            onTextChanged: typing.restart()
                            Timer { id: typing; interval: 350; onTriggered: shapeEntry.use() }
                            // (Enter is taken here: let through, it is the window's "done cropping".)
                            Keys.onReturnPressed: (event) => { typing.stop(); use(); root.forceActiveFocus(); event.accepted = true }
                            Keys.onEnterPressed: (event) => { typing.stop(); use(); root.forceActiveFocus(); event.accepted = true }
                            Keys.onEscapePressed: (event) => { root.forceActiveFocus(); event.accepted = true }
                            onActiveFocusChanged: if (!activeFocus) { typing.stop(); use() }   // (left: what is typed is used)
                        }
                    }
                }
                OnOff {
                    id: portrait
                    objectName: "crop-portrait"
                    label: "Portrait"
                    onFlipped: { on = !on; stage.flipCrop() }
                }
                EditSlider { key: "angle"; name: "Straighten"; unit: "°"; from: -15; to: 15; step: 0.1; digits: 1 }
                Item {
                    width: parent.width; height: 28
                    Row {
                        anchors.verticalCenter: parent.verticalCenter; spacing: 14
                        // (Signs on the new screen: an arrow going up and round to that side; and no
                        // line about the keys there, which the tip over the photo says: the owner's wishes.)
                        TextButton {
                            objectName: "turn-left"; says: "Turn left"; onPressed: session.turn(-1)
                            text: ""; strong: true; width: 22; height: 22
                            sign: "M15 17 V10 A5 5 0 0 0 10 5 H4 M7 2 L4 5 L7 8"
                        }
                        TextButton {
                            objectName: "turn-right"; says: "Turn right"; onPressed: session.turn(1)
                            text: ""; strong: true; width: 22; height: 22
                            sign: "M5 17 V10 A5 5 0 0 1 10 5 H16 M13 2 L16 5 L13 8"
                        }
                    }
                }
            }
            // The three that work on every color of the picture, after the pins.
            Panel {
                objectName: "panel:saturation"
                id: saturationPanel
                title: "Saturation & Vibrance"; tool: "saturation"; card: "saturation"; visible: session.source !== "" && root.shows("saturation")
                folds: true; open: root.isOpen("saturation"); onFold: root.fold("saturation")
                EditSlider { key: "saturation"; name: "Saturation"; unit: "%"; from: 0; to: 200; neutral: 100; step: 1; digits: 0 }
                // The muted colors more than the vivid ones, and skin least.
                EditSlider { key: "vibrance"; name: "Vibrance"; from: -100; to: 100; step: 1; digits: 0 }
            }
            // Black and white, in the three cards' place: a film look to start from, which is a
            // set of points of the EQ under it and nothing else, and that EQ: how light each
            // hue becomes. A look is lit while the EQ is exactly it.
            Panel {
                objectName: "panel:mono"
                id: monoPanel
                title: "Black & white"; tool: "mono"; card: "mono"; visible: session.source !== "" && root.shows("mono")
                folds: true; open: root.isOpen("mono"); onFold: root.fold("mono")
                readonly property var looks: [
                    ["none", "None", "every color as light as it is"],
                    ["panchromatic", "Panchromatic", "as ordinary black and white film sees: yellows a little lighter, blues a little darker"],
                    ["red", "Red filter", "dark, dramatic skies; skin and reds light"],
                    ["orange", "Orange filter", "darker skies, lighter skin"],
                    ["yellow", "Yellow filter", "the usual one: skies a little darker, foliage lighter"],
                    ["green", "Green filter", "light foliage, darker reds and skin"],
                    ["orthochromatic", "Orthochromatic", "as early film saw: reds dark, blues and skies light"]]
                readonly property string now: JSON.stringify(session.bw)
                readonly property var lit: looks.find(look => JSON.stringify(session.films[look[0]]) === now)
                Flow {
                    width: parent.width; spacing: 4; bottomPadding: 6
                    Repeater {
                        model: monoPanel.looks
                        Rectangle {
                            id: filmChip
                            required property var modelData
                            readonly property bool on: monoPanel.lit !== undefined && monoPanel.lit[0] === modelData[0]
                            objectName: "film:" + modelData[0]
                            width: filmName.implicitWidth + 20; height: 26; radius: 5
                            color: on ? Theme.selected : Theme.raised
                            border { width: on ? 1 : 0; color: Theme.dim }
                            Accessible.role: Accessible.RadioButton; Accessible.name: modelData[1]; Accessible.checked: on
                            Label { id: filmName; anchors.centerIn: parent; text: filmChip.modelData[1]; font.pixelSize: 12; color: filmChip.on ? Theme.text : Theme.dim }
                            MouseArea {
                                anchors.fill: parent
                                onClicked: { session.commitEdit(); session.setEdit("bw", session.films[filmChip.modelData[0]]); session.commitEdit(); root.forceActiveFocus() }
                            }
                        }
                    }
                }
                Label {
                    objectName: "film-words"
                    width: parent.width; height: implicitHeight + 10; wrapMode: Text.WordWrap; font.pixelSize: 12; color: Theme.faint
                    textFormat: Text.StyledText
                    text: monoPanel.lit ? "<b>" + monoPanel.lit[1] + "</b>: " + monoPanel.lit[2] + "." : "<b>Your own</b>: a look changed in the graph below."
                }
                HueEditor { width: parent.width; height: 230; gray: true; onDone: root.forceActiveFocus(); tally: counts }
            }
            Panel {
                objectName: "panel:chroma"
                id: chromaPanel
                title: "Chroma curve"; tool: "chroma"; card: "chroma"; visible: session.source !== "" && root.shows("chroma")
                folds: true; open: root.isOpen("chroma"); onFold: root.fold("chroma")
                CurveEditor {
                    width: parent.width; height: 250; onDone: root.forceActiveFocus()
                    key: "chroma"; line: session.chromaLine; tally: counts; behind: "chroma"
                    low: "muted"; high: "vivid"
                }
            }
            Panel {
                objectName: "panel:eq"
                id: eqPanel
                title: "Color EQ"; tool: "eq"; card: "eq"; visible: session.source !== "" && root.shows("eq")
                folds: true; open: root.isOpen("eq"); onFold: root.fold("eq")
                HueEditor { width: parent.width; height: 220; onDone: root.forceActiveFocus(); tally: counts }
            }
            // The pin being worked on, as numbers: everything the wheel does by gesture, and
            // what has no gesture (its kind, its span of lightness).
            Panel {
                id: pinPanel
                objectName: "panel:pins"
                // (In the wheel's own card, under the wheel; the owner: "cannot be a separate
                // card, it needs to be inside the color wheel card below the tonewheel".)
                title: "Pin"; visible: session.source !== ""
                plain: true; parent: pinSlot
                // What it shows: the pin being worked on, or the color just marked as the pin
                // it would be (it becomes one the moment anything here is moved). With neither
                // the section stays as it is, dimmed: the wheel under it must not move when a
                // color is marked (the owner: "the entire wheel gets shifted down").
                readonly property var shown: wheel.selected >= 0 ? wheel.pins[wheel.selected]
                    : wheel.fresh ? { mode: "offset", range: 0.08, feather: 70, variance: 0, light: 0, low: session.range.low, high: session.range.high } : null
                readonly property bool has: shown !== null
                Label {
                    objectName: "pin-hint"
                    width: parent.width; height: 40; wrapMode: Text.WordWrap; color: Theme.faint; font.pixelSize: 12
                    text: wheel.selected >= 0 ? "Pin " + (wheel.selected + 1) + " of " + wheel.pins.length + ", works on " + wheel.tonesOf(wheel.pins[wheel.selected]) + (wheel.tonesOf(wheel.pins[wheel.selected]) === "all tones" ? "" : " only") + ". A right click or Delete takes it away."
                        : wheel.fresh ? "A new pin. Drag it on the wheel to shift that color and those near it, or set it here."
                        : wheel.pins.length ? "No pin chosen. Click a pin on the wheel to work on it, or a color to make another."
                                            : "No pins yet. A pin shifts one color and those near it, as far as its range. Click a color in the photo or on the wheel to place one."
                }
                Item {
                    width: parent.width; height: 34; enabled: pinPanel.has; opacity: enabled ? 1 : 0.35
                    Label { anchors.verticalCenter: parent.verticalCenter; text: "Kind" }
                    Rectangle {
                        width: kinds.width + 4; height: 24; radius: 6; color: Theme.raised
                        anchors { right: parent.right; verticalCenter: parent.verticalCenter }
                        Row {
                            id: kinds
                            anchors.centerIn: parent
                            Repeater {
                                model: [["offset", "Offset"], ["rotate", "Rotate"], ["protect", "Protect"]]
                                Segment {
                                    required property var modelData
                                    objectName: "pin-kind:" + modelData[0]
                                    width: 68; height: 20; label: modelData[1]
                                    on: pinPanel.has && pinPanel.shown.mode === modelData[0]
                                    onChosen: { wheel.change(wheel.place(), { mode: modelData[0] }); session.commitEdit() }
                                }
                            }
                        }
                    }
                }
                PinSlider { key: "range"; name: "Range"; from: 0.01; to: 0.5; neutral: 0.08; step: 0.005; digits: 3; logarithmic: true }
                PinSlider { key: "feather"; name: "Feather"; from: 0; to: 100; neutral: 70; step: 1; digits: 0 }
                PinSlider { key: "variance"; name: "Variance"; from: -100; to: 100; step: 1; digits: 0 }
                PinSlider { key: "light"; name: "Lightness"; from: -0.5; to: 0.5; step: 0.01; digits: 2 }
                PinSlider { key: "low"; name: "Only tones lighter than"; from: 0; to: 1; neutral: 0; step: 0.01; digits: 2; signed: false }
                PinSlider { key: "high"; name: "Only tones darker than"; from: 0; to: 1; neutral: 1; step: 0.01; digits: 2 }
            }
            // The wheel: the photo's colors. It takes the room that is left, as a square, and
            // is never smaller than it can be read at: below that the column scrolls instead.
            Item {
                id: wheelBlock
                objectName: "wheel-block"
                visible: session.source !== "" && !root.mono && (root.mode === "edit" || (root.mode === "masks" && wheel.mask > 0))
                readonly property real lead: 30   // its title
                readonly property bool open: root.unfolded.wheel === true
                // (As a Panel is, for the dock beside the column: Main.reveal.)
                readonly property string card: "wheel"
                readonly property string title: "Tone Wheel"
                readonly property bool folds: true
                function fold() { root.fold("wheel") }
                function flash() { wheelGround.flash() }
                readonly property real head: lead + 16
                readonly property real side: Math.min(parent.width, 400)
                readonly property real tabs: 34   // the tone tabs' height (taller than they were: the owner's wish)
                width: parent.width; height: open ? head + tabs + 8 + side + 46 + 22 + (pinPanel.height + 14) : lead + 16
                onOpenChanged: if (open) wheelGround.flash()
                Ground { id: wheelGround; card: "wheel"; wide: wheelBlock.width; high: wheelBlock.height - (wheelBlock.open ? 24 : 16) }
                MouseArea {
                    x: -10; y: -8; width: parent.width + 20; height: 44   // (the card's head)
                    cursorShape: Qt.PointingHandCursor
                    onClicked: root.fold("wheel")
                }
                Icon {
                    x: -2; y: Theme.cardMid - height / 2; width: 14; height: 14; stroke: 2
                    path: "M7.5 4.5 L13 10 L7.5 15.5"; color: Theme.cardInk("wheel", Theme.dim)
                    rotation: wheelBlock.open ? 90 : 0
                    Behavior on rotation { NumberAnimation { duration: 110; easing.type: Easing.OutCubic } }
                }
                Icon {
                    x: 17; y: Theme.cardMid - height / 2; width: Theme.cardSign; height: Theme.cardSign; stroke: 1.8
                    path: Theme.cardIcon("wheel"); color: Theme.cardInk("wheel", wheelBlock.open ? Theme.text : Theme.dim)
                }
                Label {
                    id: wheelTitle
                    x: 18 + Theme.cardSign + 8; y: Theme.cardMid - height / 2; text: "TONE WHEEL"
                    color: Theme.cardInk("wheel", wheelBlock.open ? Theme.text : Theme.dim)
                    font.pixelSize: Theme.titleSize; font.weight: Font.DemiBold; font.letterSpacing: 0.9
                }
                Label {   // which pins the wheel is showing and making (the Masks section chooses)
                    visible: wheelBlock.open && root.mode === "masks"
                    anchors { right: wheelCompare.left; rightMargin: 14 } y: Theme.cardMid - height / 2; font.pixelSize: Theme.smallSize; color: wheel.mask > 0 ? Theme.mark : Theme.dim
                    text: wheel.mask > 0 ? "inside " + (session.masks, root.maskName(wheel.mask)) : "whole image"
                }
                // The head's dot, Reset and A/B, as every card has them (Panel.qml), about the
                // pins the wheel is showing (the whole image's, or the chosen mask's): Reset
                // takes them away in one step of undo, A/B held shows the photo without them.
                // Not the masks themselves, which the old screen's Pin tool took along.
                Rectangle {
                    id: wheelDot
                    visible: wheelReset.visible
                    x: wheelTitle.x + wheelTitle.width + 10; y: Theme.cardMid - 4; width: 8; height: 8; radius: 4; color: Theme.mark
                }
                TextButton {
                    id: wheelReset
                    objectName: "reset:wheel"
                    readonly property var kept: wheel.pins.filter(pin => (pin.mask || 0) !== wheel.mask)
                    visible: kept.length < wheel.pins.length
                    x: wheelDot.x + wheelDot.width + 12; y: Theme.cardMid - height / 2
                    sign: Theme.resetSign; says: "Reset"; strong: true
                    onPressed: { wheel.deselect(); session.commitEdit(); session.setEdit("pins", kept); session.commitEdit() }
                }
                TextButton {
                    id: wheelCompare
                    objectName: "compare:wheel"
                    anchors { right: parent.right; rightMargin: 6 } y: Theme.cardMid - height / 2
                    text: "A/B"; says: "Compare"; strong: true; font.weight: Font.DemiBold
                    font.pixelSize: Theme.titleSize; font.letterSpacing: 0.6
                    available: wheelReset.visible
                    onHeldChanged: session.setMaskBypass(wheel.mask, held ? ["pins"] : [])
                }
                Item {
                    id: maskRow
                    objectName: "mask-row"
                    readonly property var mask: wheel.mask > 0 && wheel.mask <= session.masks.length ? session.masks[wheel.mask - 1] : null
                    // (Of the chosen mask as a shape; it stands in the mask's own card.)
                    parent: maskSettingsSlot
                    y: 0; width: parent.width; height: Theme.slider
                    ParamSlider {
                        objectName: "slider:mask-feather"
                        visible: maskRow.mask !== null && maskRow.mask.kind === "radial"
                        width: parent.width
                        name: "Feather"; unit: " %"; from: 0; to: 100; neutral: 50; step: 1; digits: 0; signed: false
                        value: maskRow.mask ? maskRow.mask.feather * 100 : neutral
                        onMoved: (v) => session.setMask(wheel.mask, { feather: v / 100 })
                        onReleased: session.commitEdit()
                        onReset: { session.setMask(wheel.mask, { feather: 0.5 }); session.commitEdit() }
                        onTyped: root.forceActiveFocus()
                    }
                    Label {
                        visible: maskRow.mask !== null && maskRow.mask.kind === "linear"
                        anchors.verticalCenter: parent.verticalCenter; width: parent.width
                        text: "Fades toward the far line"
                        color: Theme.dim; font.pixelSize: 12; elide: Text.ElideRight
                    }
                }
                // The working range: the wheel shows only colors that light. "Pick" takes the
                // lightness from a click on the photo.
                Rectangle {
                    objectName: "range"
                    visible: wheelBlock.open
                    id: rangeRow
                    y: wheelBlock.head; width: parent.width; height: wheelBlock.tabs; radius: 7; color: Theme.raised
                    Row {
                        Repeater {
                            model: [["full", "All tones"], ["highlights", "Highlights"], ["midtones", "Midtones"], ["shadows", "Shadows"]]
                            Segment {
                                required property var modelData
                                objectName: "range:" + modelData[0]
                                width: wheelBlock.width / 4; height: wheelBlock.tabs; label: modelData[1]
                                on: session.range.name === modelData[0]
                                onChosen: session.setRange(modelData[0])
                            }
                        }
                        // (There was a fifth, Pick: tones as light as a spot clicked in the
                        // photo. The owner: "really not obvious what it does and ... not needed".)
                    }
                }
                Wheel {
                    id: wheel
                    objectName: "wheel"
                    visible: wheelBlock.open
                    // (As wide as its card, as the lightness strip under it is; the owner's wish.)
                    y: wheelBlock.head + wheelBlock.tabs + 8; width: parent.width; height: wheelBlock.side
                    anchors.horizontalCenter: parent.horizontalCenter
                    hover: stage.hover
                    picture: stage.picture; painted: stage.painted; lut: stage.lut
                    onDone: root.forceActiveFocus()
                    // (The column was being scrolled when the wheel came under the pointer:
                    // the scroll goes on; the owner: "it will get swallowed by the tonewheel".)
                    yields: scrolling.running
                }
                Item {   // the pin's settings, under the wheel
                    id: pinSlot
                    visible: wheelBlock.open
                    y: wheelBlock.head + wheelBlock.tabs + 8 + wheelBlock.side + 46 + 14; width: parent.width; height: pinPanel.height
                }
                LightnessStrip {
                    y: wheel.y + wheel.height + 8; width: parent.width
                    objectName: "lightness"
                    visible: wheelBlock.open
                    tally: counts
                    // The pin being worked on: lighter or darker.
                    held: wheel.selected >= 0
                    shift: held ? wheel.pins[wheel.selected].light : 0
                    onMoved: (v) => wheel.change(wheel.selected, { light: v })
                    onReleased: { session.commitEdit(); root.forceActiveFocus() }
                    onReset: { wheel.change(wheel.selected, { light: 0 }); session.commitEdit() }
                    probe: wheel.probe.length === 3 ? wheel.probe[2] : -1
                }
            }
            // Detail: sharpening and noise reduction, which are done first of all (on the
            // picture as it is stored) but are set late, with the photo's tones and colors in
            // place; judged at 100 %, in the loupe or the photo itself.
            Panel {
                objectName: "panel:detail"
                id: detailPanel
                title: "Detail"; tool: "detail"; card: "detail"; visible: session.source !== "" && root.shows("detail")
                folds: true; open: root.isOpen("detail"); onFold: root.fold("detail")
                Loupe { named: "detail-loupe"; on: detailPanel.open }
                Label { height: 30; verticalAlignment: Text.AlignBottom; bottomPadding: 4; text: "SHARPENING"; font.pixelSize: Theme.smallSize; font.weight: Font.DemiBold; font.letterSpacing: 0.9; color: Theme.dim }
                EditSlider { key: "sharpen"; name: "Amount"; from: 0; to: 150; step: 1; digits: 0; signed: false }
                EditSlider { key: "sharpenRadius"; name: "Radius"; unit: "px"; from: 0.5; to: 3; neutral: 1; step: 0.1; digits: 1; signed: false }
                EditSlider { key: "sharpenMask"; name: "Masking"; from: 0; to: 100; step: 1; digits: 0; signed: false }
                Label { height: 34; verticalAlignment: Text.AlignBottom; bottomPadding: 4; text: "NOISE REDUCTION"; font.pixelSize: Theme.smallSize; font.weight: Font.DemiBold; font.letterSpacing: 0.9; color: Theme.dim }
                // A file the neural denoiser wrote has no noise left to take out: only that is
                // said. A raw file has, under the two sliders, the button to the net's window
                // (which writes a new file).
                Label {
                    objectName: "detail-denoised"
                    visible: session.denoised
                    width: parent.width; height: implicitHeight + 8; verticalAlignment: Text.AlignVCenter; wrapMode: Text.WordWrap; color: Theme.dim
                    text: "This file is already denoised by Photone's neural denoiser."
                }
                EditSlider { visible: !session.denoised; key: "denoise"; name: "Luminance"; from: 0; to: 100; step: 1; digits: 0; signed: false }
                EditSlider { visible: !session.denoised; key: "denoiseColor"; name: "Color"; from: 0; to: 100; step: 1; digits: 0; signed: false }
                Label {
                    width: parent.width; height: implicitHeight + 10; verticalAlignment: Text.AlignBottom; wrapMode: Text.WordWrap; font.pixelSize: 12; color: Theme.faint
                    text: "Seen at 100 % and closer. Masking keeps sharpening off what differs too little from its surroundings to be an edge: noise."
                }
                Item { visible: denoiseButton.visible; width: 1; height: 10 }
                PushButton {
                    id: denoiseButton
                    objectName: "detail-denoise"
                    visible: session.develops && !session.denoised
                    width: parent.width; label: "Neural Denoise…"; tint: Theme.green
                    onPressed: root.denoisePhoto(session.path)
                }
            }
            // Lens corrections, from the camera's own figures for the lens that was on it,
            // which a raw file carries (Sony's so far): each on or off, nothing to set.
            Panel {
                objectName: "panel:lens"
                id: lensPanel
                title: "Lens"; tool: "lens"; card: "lens"; visible: session.source !== "" && root.shows("lens")
                folds: true; open: root.isOpen("lens"); onFold: root.fold("lens")
                readonly property var has: session.lensHas
                readonly property bool any: has.distortion || has.color || has.vignetting
                Repeater {
                    model: [["lensDistortion", "distortion", "Distortion"], ["lensColor", "color", "Chromatic aberration"], ["lensVignetting", "vignetting", "Vignetting"]]
                    OnOff {
                        required property var modelData
                        objectName: "lens:" + modelData[1]
                        label: modelData[2]
                        enabled: lensPanel.has[modelData[1]] === true; opacity: enabled ? 1 : 0.35
                        on: root.edit[modelData[0]] === true
                        onFlipped: { session.commitEdit(); session.setEdit(modelData[0], !on); session.commitEdit(); root.forceActiveFocus() }
                    }
                }
                Label {
                    objectName: "lens-words"
                    width: parent.width; height: implicitHeight + 10; verticalAlignment: Text.AlignBottom; wrapMode: Text.WordWrap; font.pixelSize: 12; color: Theme.faint
                    text: lensPanel.any ? "By the camera's own figures for the lens that was on it, which the raw file carries: straight lines straight, no colored fringes toward the corners, corners as light as the middle."
                                        : "This photo's file carries no figures for its lens. Raw files of Sony cameras do."
                }
                // For any file: the seams of color a lens leaves along edges, whatever the
                // camera's figures took off, or all of them where a file has none.
                Label { height: 34; verticalAlignment: Text.AlignBottom; bottomPadding: 4; text: "DEFRINGE"; font.pixelSize: Theme.smallSize; font.weight: Font.DemiBold; font.letterSpacing: 0.9; color: Theme.dim }
                EditSlider { key: "defringe"; name: "Amount"; from: 0; to: 100; step: 1; digits: 0; signed: false }
                Label {
                    width: parent.width; height: implicitHeight + 10; verticalAlignment: Text.AlignBottom; wrapMode: Text.WordWrap; font.pixelSize: 12; color: Theme.faint
                    text: "Takes the seam of color off edges between dark and light: blue, purple, yellow, cyan or green. A thing of such a color keeps it."
                }
            }
            // Film grain, last of all, as it is laid on the picture last: a character to start
            // from (an amount, a size and a roughness), those three and how colored it is,
            // and how strong it is from black to white: a response to start from, and its curve.
            Panel {
                objectName: "panel:grain"
                id: grainPanel
                title: "Film grain"; tool: "grain"; card: "grain"; visible: session.source !== "" && root.shows("grain")
                folds: true; open: root.isOpen("grain"); onFold: root.fold("grain")
                component Choice: Rectangle {   // one of a row of starting points, lit while the edit is exactly it
                    id: choice
                    property string label
                    property bool on: false
                    signal chosen()
                    width: choiceName.implicitWidth + 20; height: 26; radius: 5
                    color: on ? Theme.selected : Theme.raised
                    border { width: on ? 1 : 0; color: Theme.dim }
                    Accessible.role: Accessible.RadioButton; Accessible.name: label; Accessible.checked: on
                    Label { id: choiceName; anchors.centerIn: parent; text: choice.label; font.pixelSize: 12; color: choice.on ? Theme.text : Theme.dim }
                    MouseArea { anchors.fill: parent; onClicked: { choice.chosen(); root.forceActiveFocus() } }
                }
                Loupe { named: "grain-loupe"; on: grainPanel.open }
                Item { width: 1; height: 10 }
                Flow {
                    width: parent.width; spacing: 4; bottomPadding: 6
                    Repeater {
                        model: [["fine", "Fine"], ["classic", "Classic"], ["bold", "Bold"], ["pushed", "Pushed"]]
                        Choice {
                            required property var modelData
                            readonly property var is: session.grains.characters[modelData[0]]
                            objectName: "grain-character:" + modelData[0]
                            label: modelData[1]
                            on: root.edit.grain === is.grain && root.edit.grainSize === is.grainSize && root.edit.grainRough === is.grainRough
                            onChosen: { session.commitEdit(); session.setEdits(is); session.commitEdit() }
                        }
                    }
                }
                EditSlider { key: "grain"; name: "Amount"; from: 0; to: 100; step: 1; digits: 0; signed: false }
                EditSlider { key: "grainSize"; name: "Size"; unit: "×"; from: 0.4; to: 4; neutral: 1; step: 0.05; digits: 2; logarithmic: true; signed: false }
                EditSlider { key: "grainRough"; name: "Roughness"; from: 0; to: 100; neutral: 50; step: 1; digits: 0; signed: false }
                EditSlider { key: "grainColor"; name: "Color"; from: 0; to: 100; step: 1; digits: 0; signed: false; visible: !root.mono }
                Item {
                    width: parent.width; height: 40
                    Label { anchors { bottom: parent.bottom; bottomMargin: 8 } text: "Strongest in"; color: session.unsavedKeys.includes("grainCurve") ? Theme.yellow : Theme.dim }
                    Row {
                        anchors { right: parent.right; bottom: parent.bottom; bottomMargin: 4 } spacing: 4
                        Repeater {
                            model: [["print", "Midtones"], ["negative", "Shadows"], ["even", "All tones"]]
                            Choice {
                                required property var modelData
                                objectName: "grain-response:" + modelData[0]
                                label: modelData[1]
                                on: JSON.stringify(session.grainCurve) === JSON.stringify(session.grains.responses[modelData[0]])
                                onChosen: { session.commitEdit(); session.setEdit("grainCurve", session.grains.responses[modelData[0]]); session.commitEdit() }
                            }
                        }
                    }
                }
                CurveEditor {
                    width: parent.width; height: 170; onDone: root.forceActiveFocus()
                    key: "grainCurve"; line: session.grainLine; tally: counts; behind: "light"; rest: false
                }
                Label {
                    width: parent.width; height: implicitHeight + 14; verticalAlignment: Text.AlignBottom; wrapMode: Text.WordWrap; font.pixelSize: 12; color: Theme.faint
                    text: "How strong the grain is from black to white."
                }
            }
        }

        }
        }

        // Checkpoints: the edit as it was when "Save" was pressed, to come back to. The one the
        // edit stands at is lit; changed since, it keeps a hollow mark and says so. A low
        // section of its own at the foot of the column, under everything (the owner's choice):
        // four rows at most, more by scrolling.
        Scrollbar { id: pointsBar; of: pointsList; step: Theme.row; visible: false }   // (what the wheel scrolls it by)
        Item {
            id: points
            objectName: "checkpoints"
            visible: session.source !== ""
            parent: leftPane
            readonly property int rows: Math.min(6, session.checkpoints.length)
            // (Its card reaches 6 px below it: with this, the card ends as far above the status
            // line as it begins below the card over it. It stood right on the line.)
            anchors { left: parent.left; right: parent.right; bottom: parent.bottom; leftMargin: 16; rightMargin: 16; bottomMargin: 12}
            // The new screen (the owner: "like the masks list... dark rectangle for the scrollable
            // area", Save "at the very bottom under that list", "latest at the bottom"): a dark
            // box five rows high that scrolls, the newest checkpoint last, Save across its foot.
            readonly property real boxHigh: 5 * Theme.row + 8
            // How long ago, in a word or two (the owner: "how long ago, not when"); kept true by
            // a look at the clock twice a minute.
            property real now: Date.now() / 1000
            Timer { interval: 30000; running: points.visible; repeat: true; onTriggered: points.now = Date.now() / 1000 }
            function ago(made) {
                const s = Math.max(0, now - made), m = Math.floor(s / 60), h = Math.floor(m / 60), d = Math.floor(h / 24)
                return m < 1 ? "just now" : m < 60 ? m + " min ago" : h < 24 ? h + " h ago" : d < 30 ? d + (d === 1 ? " day ago" : " days ago") : Math.floor(d / 30) + " mo ago"
            }
            Connections { target: session; function onChanged() { points.now = Date.now() / 1000 } }
            // (New screen: Save's foot is as far from the card's as its sides are, 10 px; the card
            // reaches 6 px below this item.)
            height: 12 + 54 + boxHigh + 10 + 30 + 4
            Item {
                id: pointsHeader
                y: 12; width: parent.width; height: 54
                Ground { y: -4; head: 48; card: "checkpoints"; wide: points.width; high: points.height - 16 }
                Icon {
                    y: 21 - height / 2; width: Theme.cardSign; height: Theme.cardSign; stroke: 1.8
                    path: Theme.cardIcon("checkpoints"); color: Theme.cardInk("checkpoints", Theme.dim)
                }
                Label {
                    x: Theme.cardSign + 8; y: 21 - height / 2; color: Theme.cardInk("checkpoints", Theme.dim)
                    text: "CHECKPOINTS"
                    font.pixelSize: Theme.titleSize; font.weight: Font.DemiBold; font.letterSpacing: 0.9
                }
            }
            Rectangle {
                id: pointsBox
                anchors { top: pointsHeader.bottom; left: parent.left; right: parent.right }
                height: points.boxHigh; radius: 6; color: Theme.bg; border { width: 1; color: Theme.line }
                Label {
                    visible: session.checkpoints.length === 0
                    anchors.centerIn: parent; text: "No checkpoints yet"; color: Theme.faint; font.pixelSize: 12
                }
            }
            MoreBelow { objectName: "checkpoints-more"; z: 1; of: pointsList; x: 1; width: parent.width - 2; anchors { bottom: pointsBox.bottom; bottomMargin: 1 } radius: 5 }
            MoreBelow { above: true; z: 1; of: pointsList; x: 1; width: parent.width - 2; anchors { top: pointsBox.top; topMargin: 1 } radius: 5 }
            PushButton {
                objectName: "checkpoint-save"
                anchors { top: pointsBox.bottom; topMargin: 10; left: parent.left; right: parent.right }
                label: "Save checkpoint"; tint: Theme.green; onPressed: session.saveCheckpoint()
            }
            ListView {
                id: pointsList
                objectName: "checkpoint-list"
                x: 4; y: pointsHeader.y + pointsHeader.height + (4)
                width: parent.width + (-8)
                height: points.boxHigh - 8
                clip: true; boundsBehavior: Flickable.StopAtBounds; acceptedButtons: Qt.NoButton
                onCountChanged: positionViewAtEnd()   // (the newest is the last, and in sight)
                MouseArea { parent: pointsList.contentItem; width: pointsList.width; height: Math.max(pointsList.height, pointsList.contentHeight); z: -1; acceptedButtons: Qt.NoButton; onWheel: (wheel) => pointsBar.turned(wheel) }
                model: session.checkpoints.slice().reverse()
                delegate: Rectangle {
                    id: row
                    required property var modelData
                    objectName: "checkpoint:" + modelData.id
                    width: pointsList.width; height: Theme.row; radius: 6
                    // (The new screen: the checkpoint the edit stands at has its whole row in the accent
                    // color, faintly, with an outline of it; it was a thin strip at the row's
                    // edge, which the owner wanted easier to see.)
                    border { width: modelData.state === "here" ? 1 : 0; color: Qt.rgba(Theme.mark.r, Theme.mark.g, Theme.mark.b, 0.6) }
                    color: modelData.state === "here" ? (Qt.rgba(Theme.mark.r, Theme.mark.g, Theme.mark.b, 0.2)) : over.containsMouse ? (Theme.raised) : "transparent"
                    Accessible.role: Accessible.ListItem
                    Accessible.name: modelData.name + (modelData.state === "here" ? ", current" : modelData.state === "changed" ? ", changed since" : "")
                    Label {
                        // (No column kept free for a mark: the names of all stand on one edge.)
                        x: 12; anchors.verticalCenter: parent.verticalCenter
                        width: parent.width - x - trailing.width - 16; elide: Text.ElideRight
                        text: row.modelData.name
                        color: row.modelData.state === "" ? Theme.dim : Theme.text
                        visible: !pointEntry.visible
                    }
                    MouseArea {
                        id: over
                        anchors.fill: parent; hoverEnabled: true
                        onClicked: root.goTo(row.modelData)
                    }
                    Row {
                        id: trailing
                        anchors { right: parent.right; rightMargin: 10; verticalCenter: parent.verticalCenter }
                        spacing: 10
                        visible: !pointEntry.visible
                        Label {
                            anchors.verticalCenter: parent.verticalCenter
                            font.pixelSize: 12; font.features: { "tnum": 1 }
                            // (While the pointer is on the row its two buttons have this place.)
                            visible: !(renamePoint.visible)
                            color: row.modelData.state === "changed" ? Theme.mark : Theme.faint
                            text: row.modelData.state === "changed" ? "changed" : points.ago(row.modelData.made)
                        }
                        TextButton {
                            id: renamePoint
                            objectName: "checkpoint-rename:" + row.modelData.id
                            anchors.verticalCenter: parent.verticalCenter
                            visible: (over.containsMouse || containsPointer || deletePoint.containsPointer)
                            sign: Theme.pencilSign; says: "Rename"; glow: Theme.green; onPressed: pointEntry.begin(row.modelData.name)
                        }
                        TextButton {
                            id: deletePoint
                            objectName: "checkpoint-delete:" + row.modelData.id
                            anchors.verticalCenter: parent.verticalCenter
                            visible: over.containsMouse || containsPointer || renamePoint.containsPointer
                            text: ""; sign: Theme.binSign; says: "Delete"; glow: Theme.red
                            onPressed: confirm.ask("Delete “" + row.modelData.name + "”?",
                                                   "The checkpoint cannot be brought back. The photo's edit stays as it is.",
                                                   "Delete", () => session.deleteCheckpoint(row.modelData.id))
                        }
                    }
                    NameEntry {
                        id: pointEntry
                        x: 8; width: parent.width - 16; anchors.verticalCenter: parent.verticalCenter
                        onTaken: (name) => session.renameCheckpoint(row.modelData.id, name)
                    }
                }
            }
        }
    }

    // ---- The photo -----------------------------------------------------------------------
    PhotoView {
        id: stage
        objectName: "box:stage"
        visible: root.editing
        missing: root.photoMissing
        anchors { left: leftPane.visible ? leftPane.right : parent.left; right: tools.left; top: chartBar.visible ? chartBar.bottom : topBar.bottom; bottom: filmstrip.visible ? filmstrip.top : foot.top }
        onPicked: (u, v) => { session.pickRange(u, v); picking = "" }
        // Pick neutral: while the button is held and slid, the spot under the pointer is made
        // gray, as one step of undo; it stays on for another try, until Esc or its button.
        onSampled: (u, v, first) => { if (first) session.commitEdit(); session.pickNeutral(u, v) }
        onSampleDone: session.commitEdit()
        // A plain click on the photo marks its color on the wheel, to be pinned.
        onTapped: (u, v) => wheel.mark(session.colorAt(u, v))
        onMaskLeft: wheel.mask = 0
        asking: confirm.visible || denoiseWindow.visible   // (files dropped while a question is open are not opened)
        brushSize: Math.max(0.003, Math.min(0.3, Number(session.remembered("brush/size", "0.05")) || 0.05))
        brushFeather: Math.max(0, Math.min(1, Number(session.remembered("brush/feather", "0.5"))))
        mask: wheel.mask
        standIn: (library.thumbs, library.thumbFor(root.openPath))
        // Pressing the photo takes the keys back from whatever control had them, so that the
        // arrows step through photos again and not through a slider.
        onClicked: root.forceActiveFocus()
    }
    // The photos at hand, in a row under the photo (there once there is more than one). Its
    // top edge is dragged to make it taller or lower, down to nothing; a double click on that
    // edge puts it away and brings it back. How tall it is, is remembered.
    readonly property int openAt: session.index   // (asked at every change; another only with another photo)
    property real stripHigh: Math.max(0, Math.min(240, Number(session.remembered("view/strip", "104")) || 0))
    Filmstrip {
        id: filmstrip
        objectName: "box:strip"
        visible: root.editing && session.count > 1
        anchors { left: leftPane.visible ? leftPane.right : parent.left; right: tools.left; bottom: foot.top }
        height: root.stripHigh < 40 ? 7 : root.stripHigh   // (put away, its edge stays to take hold of)
        current: root.openAt
        onOpened: (index) => { session.goTo(index); root.forceActiveFocus() }
        onMenuAsked: (files, x, y) => menu.open(files, x, y)
        MouseArea {
            objectName: "strip-edge"
            anchors { left: parent.left; right: parent.right; top: parent.top }
            height: 7; cursorShape: Qt.SizeVerCursor
            property real from
            property real start
            property real before: 104   // how tall it was when it was put away
            onPressed: (mouse) => { from = mapToItem(root, 0, mouse.y).y; start = root.stripHigh < 40 ? 0 : root.stripHigh }
            onPositionChanged: (mouse) => {
                if (!pressed) return   // (the pointer merely passing over it)
                const high = start + from - mapToItem(root, 0, mouse.y).y
                root.stripHigh = high < 40 ? 0 : Math.min(240, Math.max(64, high))
            }
            onReleased: session.remember("view/strip", String(Math.round(root.stripHigh)))
            onDoubleClicked: {
                if (root.stripHigh >= 40) { before = root.stripHigh; root.stripHigh = 0 } else root.stripHigh = before
                session.remember("view/strip", String(Math.round(root.stripHigh)))
            }
            Rectangle { anchors.horizontalCenter: parent.horizontalCenter; y: 2; width: 36; height: 3; radius: 1.5; color: parent.containsMouse || parent.pressed ? Theme.dim : Theme.faint }
            hoverEnabled: true
        }
    }
    // The open photo's thumbnail is the photo as it is edited: when the edit has come to rest
    // (and a moment after an edited photo is opened), a small render of it through the same
    // shader is read back and handed to the library, which keeps it in place of the one made
    // from the file. It takes its numbers only then, so a slider being dragged does not reach it.
    Pipeline {
        id: keeper
        objectName: "keeper"
        visible: false; z: 5
        readonly property real share: 512 / Math.max(1, session.fullWidth, session.fullHeight)
        width: Math.max(1, Math.round(session.fullWidth * share)); height: Math.max(1, Math.round(session.fullHeight * share))
        source: stage.picture; sharp: stage.picture; useSharp: 0; painted: stage.painted; lutPicture: stage.lut
        property var numbers: session.inFrame
        property var layers: []
        follows: true; edit: numbers
        peak: 1; hdrLift: 0; linearOut: 0; clipping: 0
        Local { over: keeper; layers: keeper.layers }
        Component.onCompleted: take()   // (assignments: from here on only `keep` sets them)
        function fit() { return stage.ready && !session.original && !session.panels.chart }
        function take() { numbers = session.inFrame; layers = session.maskLayersInFrame }   // the edit as it is now
        function keep() {
            if (!fit() || !Window.window || !Window.window.visible) return
            const path = session.path, steps = session.steps
            take()
            // (Taken by the library, not by a function here: app.Session.grabScope says why.)
            wanted = { path: path, steps: steps }
            if (!library.grabThumb(keeper, path)) wanted = null
        }
        property var wanted: null   // the photo and the edit the render under way is of
        Connections {
            target: library
            function onThumbTaken() {
                const of = keeper.wanted
                keeper.wanted = null
                library.keepTaken(of !== null && session.path === of.path && session.steps === of.steps && keeper.fit())
            }
        }
    }
    Timer { id: keeping; interval: 250; onTriggered: keeper.keep() }
    Connections {
        target: session
        function onSaved() { keeping.restart(); library.setEdited(session.path, session.edited) }
    }
    // (A photo that was edited before thumbnails were kept this way gets its own when opened.)
    Connections {
        target: stage
        function onReadyChanged() { if (stage.ready && session.edited) keeping.restart() }
    }
    // While the test chart is on show, a bar over it says so in the one accent color.
    Rectangle {
        id: chartBar
        objectName: "box:chart"
        visible: session.panels.chart && root.editing
        // (Over the photo only: it lay over the top of the left panel; the owner: "overflows".)
        anchors { left: leftPane.visible ? leftPane.right : parent.left; right: tools.left; top: topBar.bottom }
        height: chartWords.height + 16; color: Theme.mark
        Accessible.role: Accessible.AlertMessage; Accessible.name: chartWords.text
        Label {
            id: chartWords
            objectName: "text:chart"
            anchors { left: parent.left; right: chartBack.left; margins: 14; verticalCenter: parent.verticalCenter }
            wrapMode: Text.WordWrap; color: Theme.bg; font.weight: Font.DemiBold
            text: "TEST CHART  ·  This is not your photo. It shows what the edit of " + session.name + " does to a set of test colors. Anything you change here still changes that photo's edit."
        }
        Rectangle {
            id: chartBack
            objectName: "chart-back"
            anchors { right: parent.right; rightMargin: 14; verticalCenter: parent.verticalCenter }
            width: backWords.implicitWidth + 24; height: 28; radius: 6; color: Theme.bg
            activeFocusOnTab: true
            Accessible.role: Accessible.Button; Accessible.name: backWords.text
            Keys.onSpacePressed: session.setPanel("chart", false)
            Label { id: backWords; anchors.centerIn: parent; text: "Back to my photo"; color: Theme.text }
            MouseArea { anchors.fill: parent; onClicked: session.setPanel("chart", false) }
        }
    }
    Counts { id: counts; objectName: "counts"; mask: wheel.mask; source: stage.picture; painted: stage.painted; lut: stage.lut; colors: wheel.colors; port: scroller }
    Scopes { objectName: "scopes"; z: 5; mask: wheel.mask; source: stage.picture; painted: stage.painted; lut: stage.lut; ready: stage.ready }   // (shots show it for a moment)
    Column {
        anchors.centerIn: stage; spacing: 10
        objectName: "no-photo"
        visible: (session.source === "" || root.photoFailed || root.photoMissing) && root.editing
        Label { text: root.photoMissing ? "This photo’s file is missing" : root.photoFailed ? "This photo could not be opened" : "No photo open"; color: Theme.text; font.pixelSize: 15; anchors.horizontalCenter: parent.horizontalCenter }
        Label { text: root.photoMissing ? session.name + " is no longer on the disk." : "Drop a photo here, or"; anchors.horizontalCenter: parent.horizontalCenter }
        PushButton { visible: !root.photoMissing; label: "Open…"; anchors.horizontalCenter: parent.horizontalCenter; onPressed: opener.open() }
        PushButton { objectName: "remove-missing"; visible: root.photoMissing; label: "Remove…"; anchors.horizontalCenter: parent.horizontalCenter; onPressed: root.removeMissing([session.path]) }
    }
    // Fit, 100 %, and what the zoom is now.
    Item {
        objectName: "screen:zoom"
        visible: stage.ready && root.editing
        anchors { horizontalCenter: stage.horizontalCenter; bottom: stage.bottom; bottomMargin: 12 }
        id: zoomBox
        width: zoomRow.width + 6; height: 30
        Glass {}
        Row {
            id: zoomRow
            anchors.centerIn: parent
            // Two buttons, then the zoom in effect as plain dim text behind a divider, so that
            // it reads as a reading and not as a third button.
            Segment { bright: true; label: "Fit"; on: stage.fitted; onChosen: stage.glideFit() }
            Segment { bright: true; label: "1:1"; on: !stage.fitted && Math.abs(stage.zoom - 1) < 0.0005; onChosen: stage.actual() }
            Item {
                width: 60; height: 26
                Rectangle { x: 5; y: 5; width: 1; height: 16; color: Qt.rgba(1, 1, 1, 0.25) }
                Label {
                    objectName: "text:zoom"
                    anchors { fill: parent; leftMargin: 6 }
                    horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
                    color: Qt.rgba(1, 1, 1, 0.82); font.pixelSize: 12; font.features: { "tnum": 1 }
                    text: Math.round(stage.shown * 100) + " %"
                }
            }
        }
    }
    // The cards of the tab on show (Adjust or Masks), as buttons in their own icons and
    // colors on the zoom bar's glass, against the right column's edge over the photo; a click
    // opens that card as a click on its head does (Main.reveal). The owner's wish. Read from
    // the column itself: every card on show there that has a `card` name has a button.
    Item {
        id: dock
        objectName: "screen:dock"
        readonly property var cards: root.mode === "edit" || root.mode === "masks"
            ? Array.prototype.filter.call(develop.children, c => c.visible && !!c.card) : []
        visible: stage.ready && root.editing && cards.length > 0
        // (Its top level with the column's first card, below the tabs.)
        anchors { right: tools.left; rightMargin: 10 }
        y: Math.max(stage.y + 10, tools.y + scrollArea.y + 8)
        width: dockColumn.width + 8; height: dockColumn.height + 8
        Glass {}
        Column {
            id: dockColumn
            anchors.centerIn: parent
            Repeater {
                model: dock.cards
                Rectangle {
                    required property var modelData
                    objectName: "dock:" + modelData.card
                    width: 34; height: 34; radius: 6
                    color: press.containsMouse ? Qt.rgba(1, 1, 1, 0.14) : "transparent"
                    activeFocusOnTab: true
                    Accessible.role: Accessible.Button; Accessible.name: modelData.title
                    Keys.onSpacePressed: root.reveal(modelData)
                    border { width: activeFocus ? 1 : 0; color: Theme.text }
                    Icon {
                        anchors.centerIn: parent; width: Theme.cardSign; height: Theme.cardSign; stroke: 1.8
                        path: Theme.cardIcon(parent.modelData.card); color: Theme.cardInk(parent.modelData.card, Theme.text)
                    }
                    MouseArea { id: press; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: root.reveal(parent.modelData) }
                }
            }
        }
    }
    // ---- The library, in Edit's place while it is on show ---------------------------------
    Library {
        id: shelf
        objectName: "box:library"
        visible: root.place === "library"
        anchors { left: parent.left; right: parent.right; top: topBar.bottom; bottom: foot.top }
        onOpened: (row) => { session.openAt(library.paths(), row); root.show("edit") }
        onFolderAsked: root.pickFolder()
        onFolderMenuAsked: (folder, x, y) => menu.openFolder(folder, x, y)
        onFolderRemoveAsked: (folder) => root.removeFolder(folder)
        onMenuAsked: (rows, x, y) => { const all = library.paths(); menu.open(rows.map(row => all[row]).filter(path => path !== undefined), x, y) }
    }
    // ---- Export, likewise --------------------------------------------------------------------
    Export {
        id: desk
        objectName: "box:export"
        visible: root.place === "export"
        anchors { left: parent.left; right: parent.right; top: topBar.bottom; bottom: foot.top }
        picture: stage.picture; painted: stage.painted; lut: stage.lut; ready: stage.ready
        onMarkFileAsked: markOpener.open()
        // How and where to is remembered between runs.
        Component.onCompleted: {
            kind = session.remembered("export/kind", "jpeg"); quality = Number(session.remembered("export/quality", "90"))
            longest = Number(session.remembered("export/longest", "3000")); resized = session.remembered("export/resized", "no") === "yes"
            folder = session.remembered("export/folder", "")
        }
        onKindChanged: session.remember("export/kind", kind)
        onQualityChanged: session.remember("export/quality", String(Math.round(quality)))
        onLongestChanged: session.remember("export/longest", String(Math.round(longest)))
        onResizedChanged: session.remember("export/resized", resized ? "yes" : "no")
        onFolderChanged: session.remember("export/folder", folder)
        onFolderAsked: exportFolder.open()
        onEditAsked: (photo) => root.editPhoto(photo)
    }
    FolderDialog {
        id: exportFolder
        title: "Export to this folder"
        onAccepted: {
            desk.folder = exporter.folder_of(selectedFolder); root.forceActiveFocus()
            if (root.waiting.length) exporter.start(root.waiting, desk.options())
            root.waiting = []
        }
        onRejected: { root.waiting = []; root.forceActiveFocus() }
    }
    Bench { objectName: "bench"; z: 6 }   // (shots show it for a moment)
    // Adding a folder begins where the last one was added from (the owner: the dialog
    // "doesn't remember the last picked path, which is annoying"): in the folder that one is in.
    function pickFolder() {
        const last = session.remembered("library/added", "")
        if (last !== "") folderPicker.currentFolder = last
        folderPicker.open()
    }
    // A photo whose file is gone is taken out when the owner says so, and asked first (the
    // owner: "with a confirm dialog").
    function removeMissing(paths) {
        const one = paths.length === 1
        confirm.ask(one ? "Remove “" + paths[0].split(/[\\/]/).pop() + "”?" : "Remove " + paths.length + " missing photos?",
                    (one ? "Its file is" : "Their files are") + " no longer on the disk. " + (one ? "Its edit and marks" : "Their edits and marks")
                    + " are kept, and are there again should the file" + (one ? "" : "s") + " come back.",
                    "Remove", () => library.remove(paths))
    }
    // Taking a folder out of the library asks first (the owner: it "needs a confirmation").
    function removeFolder(folder) {
        confirm.ask("Remove “" + folder.split("/").pop() + "” from the library?",
                    "Its photos stay where they are on the disk. Their edits and marks are kept, and are there again if the folder is added back.",
                    "Remove", () => library.removeFolder(folder))
    }
    FolderDialog {
        id: folderPicker
        title: "Add a folder of photos"
        onAccepted: {
            library.addFolder(selectedFolder)
            session.remember("library/added", String(selectedFolder).replace(/\/[^\/]+\/?$/, ""))
            root.forceActiveFocus()
        }
        onRejected: root.forceActiveFocus()
    }
    FileDialog {
        id: opener
        title: "Open photos"
        fileMode: FileDialog.OpenFiles
        nameFilters: ["Photos (*.arw *.cr2 *.cr3 *.dng *.nef *.orf *.pef *.raf *.rw2 *.jpg *.jpeg *.png *.tif *.tiff)", "All files (*)"]
        onAccepted: { session.open(selectedFiles); root.show("edit") }
        onRejected: root.forceActiveFocus()
    }

    FileDialog {
        id: tableSaver
        title: "Export the look as a LUT"
        fileMode: FileDialog.SaveFile; defaultSuffix: "cube"
        nameFilters: ["LUTs (*.cube)"]
        onAccepted: { exporter.startTable(String(selectedFile)); root.forceActiveFocus() }
        onRejected: root.forceActiveFocus()
    }
    FileDialog {
        id: markOpener
        title: "Choose the watermark's picture"
        nameFilters: ["Pictures (*.png *.jpg *.jpeg *.tif *.tiff *.webp *.svg)", "All files (*)"]
        onAccepted: { exporter.setMark({ file: String(selectedFile), kind: "picture" }); root.forceActiveFocus() }
        onRejected: root.forceActiveFocus()
    }
    FileDialog {
        id: lutOpener
        title: "Install LUTs"
        fileMode: FileDialog.OpenFiles
        nameFilters: ["LUTs (*.cube *.clf)", "All files (*)"]
        onAccepted: { session.installLuts(selectedFiles); root.forceActiveFocus() }
        onRejected: root.forceActiveFocus()
    }

    // ---- Leaving a text field ------------------------------------------------------------
    // Every press anywhere comes here first and goes on to what is under it. Pressed outside
    // the text field that has the keys, the field loses them, and so takes its value (a click
    // on a card or on the empty ground would not take them by itself). (docs/PLAN.md §5.)
    // (A handler that only looks on, on an item of its own over everything. As a MouseArea
    // it also took the pointer's shape from all under it: no hand on a card's head, no
    // arrows on a panel's edge, no text cursor on a number; the owner: "i still dont see a
    // cursor change".)
    Item {
        id: leaving
        anchors.fill: parent; z: 1000
        PointHandler {
            acceptedButtons: Qt.AllButtons
            onActiveChanged: {
                if (!active) return
                const field = root.Window.activeFocusItem, at = point.pressPosition
                if (field && field.cursorPosition !== undefined && !field.contains(field.mapFromItem(leaving, at.x, at.y)))
                    root.forceActiveFocus()
            }
        }
    }

    // ---- A tip, at the top of the photo, while a gesture on it needs saying ---------------
    // (The owner: "nobody looks at the bottom of the window", and asked for "a toast-like GUI
    // ... middle-top of the photo view ... short, concise and easy to understand". So the
    // status line is gone; this comes when the thing it is about begins, and fades.)
    Rectangle {
        id: tip
        objectName: "screen:tip"
        readonly property string says: !root.editing || !stage.ready ? ""
            : stage.picking === "neutral" ? "Click something that should be gray  ·  Esc stops"
            : root.cropping ? "Drag the edges  ·  Drag outside to straighten  ·  Enter when done"
            : wheel.mask > 0 && (session.masks[wheel.mask - 1] || {}).kind === "brush" ? "Paint the mask on the photo  ·  Ctrl + wheel sizes the brush  ·  Hold H to see it"
            : wheel.mask > 0 ? "Drag the handles to shape it  ·  Drag outside to turn it  ·  Hold H to see it"
            : ""
        property bool fresh: false
        onSaysChanged: { fresh = says !== ""; if (fresh) fading.restart() }
        Timer { id: fading; interval: 5000; onTriggered: tip.fresh = false }
        opacity: fresh ? 1 : 0; visible: opacity > 0
        Behavior on opacity { NumberAnimation { duration: 250 } }
        anchors { horizontalCenter: stage.horizontalCenter; top: stage.top; topMargin: 14 }
        width: tipText.implicitWidth + 32; height: 32; radius: 16
        color: Qt.rgba(0.07, 0.07, 0.08, 0.86); border { width: 1; color: Qt.rgba(1, 1, 1, 0.12) }
        Accessible.role: Accessible.StaticText; Accessible.name: says
        Label { id: tipText; anchors.centerIn: parent; text: tip.says; color: Theme.text; font.pixelSize: 13 }
    }

    // ---- What went wrong, in one sentence; click to dismiss -------------------------------
    Rectangle {
        objectName: "screen:notice"
        visible: session.error !== ""
        // Above the zoom control when that is there: two floating things never share a place.
        anchors { horizontalCenter: stage.horizontalCenter; bottom: stage.bottom; bottomMargin: stage.ready ? 54 : 16 }
        width: Math.min(message.implicitWidth + 28, stage.width * 0.8); height: message.height + 20
        radius: 8; color: Theme.raised; border { width: 1; color: Theme.faint }
        Accessible.role: Accessible.AlertMessage; Accessible.name: session.error
        Label {
            id: message
            anchors.centerIn: parent; width: parent.width - 28
            text: session.error; color: Theme.text; wrapMode: Text.WordWrap; horizontalAlignment: Text.AlignHCenter
        }
        MouseArea { anchors.fill: parent; onClicked: session.say("") }
    }

    // ---- What can be done with a photo: a right click on it, in the filmstrip or the library -
    // (the owner: "a right-click context menu on images on the film strip AND in the library").
    // Copy and Paste edits come here with M21.
    // Export these at once, as the Export page has it set up (the owner: "a context menu with
    // options that make sense, mainly Export All (number)"); the footer says how far it is.
    // With no folder chosen yet, that is asked first.
    // Open a photo in Edit by its path (from the log of exports): among the photos the
    // library is showing if it is one of them, else among all of them, else alone.
    function editPhoto(path) {
        let row = library.rowOf(path)
        if (row < 0) { library.show("all"); row = library.rowOf(path) }
        if (row >= 0) session.openAt(library.paths(), row)
        else session.open([path])
        if (session.path === path) show("edit")
    }
    property var waiting: []
    function exportNow(paths) {
        if (exporter.running) return
        if (desk.folder === "") { waiting = paths; return exportFolder.open() }
        exporter.start(paths, desk.options())
    }
    // The open photo is reset as any edit is made (one step of undo, its thumbnail made from
    // what is on screen); the others in the catalog, where the reset becomes a step of theirs.
    function resetThese(paths) {
        const open = paths.includes(session.path)
        if (open) { session.resetAll(); session.flush() }
        library.resetEdits(open && editing ? paths.filter(path => path !== session.path) : paths)
    }
    // A look (everything of an edit but the frame and the masks) goes from one photo to
    // others. The open photo's edit is read from the catalog like any other's, so it is
    // written there first; and it takes a look as any edit is made, as one step of undo.
    function copyLook(path) {
        if (path === session.path) session.flush()
        library.copyLook(path)
    }
    function pasteLook(paths) {
        if (!library.hasLook) return
        const open = paths.includes(session.path)
        if (open) { session.pasteLook(library.look); session.flush() }
        library.pasteLook(open && editing ? paths.filter(path => path !== session.path) : paths)
    }
    Item {
        id: menu
        anchors.fill: parent
        z: 9
        visible: false
        property var photos: []
        property string folder: ""   // set: the menu is a folder's, not photos'
        // Those of its photos whose files are gone: they can be taken out, and with nothing
        // but such photos that is all there is to do.
        readonly property var lost: (library.busy, photos.filter(path => library.missing(path)))   // (not only while it is open: an entry acts once it has closed)
        readonly property bool here: folder !== "" || lost.length < photos.length
        function openFolder(path, x, y) { folder = path; photos = []; place(x, y) }
        function open(paths, x, y) {
            if (!paths.length) return
            photos = paths; folder = ""
            place(x, y)
        }
        property real atX: 0   // where it was asked for; the card keeps inside the window from there
        property real atY: 0
        function place(x, y) {
            atX = x; atY = y
            visible = true
            card.forceActiveFocus()
        }
        function close() { visible = false; root.forceActiveFocus() }
        MouseArea { anchors.fill: parent; acceptedButtons: Qt.AllButtons; onPressed: menu.close(); onWheel: (wheel) => menu.close() }
        Rectangle {
            id: card
            objectName: "modal:menu"
            width: 210; height: entries.height + 8; radius: 8
            // (Bound, not set once: the entries shown, and so the height, change as it opens.)
            x: Math.max(6, Math.min(menu.atX, menu.width - width - 6))
            y: Math.max(6, Math.min(menu.atY, menu.height - height - 6))
            color: Theme.panel; border { width: 1; color: Theme.faint }
            Accessible.role: Accessible.PopupMenu; Accessible.name: menu.folder !== "" ? "Folder" : "Photo"
            Keys.onPressed: (event) => { if (event.key === Qt.Key_Escape) menu.close(); event.accepted = true }   // (its own while it is open)
            MouseArea { anchors.fill: parent; acceptedButtons: Qt.AllButtons }   // (a press on the card itself is not one beside it)
            Column {
                id: entries
                x: 4; y: 4; width: parent.width - 8
                component Entry: Rectangle {
                    property string label
                    signal pressed()
                    width: parent.width; height: 30; radius: 5
                    color: entryArea.containsMouse ? Theme.selected : "transparent"
                    Accessible.role: Accessible.MenuItem; Accessible.name: label
                    Label { x: 10; anchors.verticalCenter: parent.verticalCenter; text: parent.label; color: Theme.text }
                    MouseArea { id: entryArea; anchors.fill: parent; hoverEnabled: true; onClicked: { menu.close(); parent.pressed() } }
                }
                Label {
                    x: 10; height: 24; verticalAlignment: Text.AlignVCenter; font.pixelSize: 11; color: Theme.faint
                    width: parent.width - 20; elide: Text.ElideMiddle
                    text: menu.folder !== "" ? menu.folder.split("/").pop() : menu.photos.length === 1 ? String(menu.photos[0]).split("/").pop() : menu.photos.length + " photos"
                }
                // (The owner: "right-click a folder entry in both the Library AND the card on
                // the Edit page ... a `Show in explorer` option". Offered for a photo too.)
                Entry {
                    objectName: "menu-remove-missing"; visible: menu.lost.length > 0
                    label: menu.lost.length > 1 ? "Remove missing (" + menu.lost.length + ")…" : "Remove missing photo…"
                    onPressed: root.removeMissing(menu.lost)
                }
                Entry {
                    objectName: "menu-reveal"; label: "Show in file manager"; visible: menu.here
                    onPressed: library.reveal(menu.folder !== "" ? menu.folder : String(menu.photos[0]))
                }
                Entry { objectName: "menu-remove-folder"; visible: menu.folder !== ""; label: "Remove from library…"; onPressed: root.removeFolder(menu.folder) }
                Entry {
                    objectName: "menu-denoise"; label: "Neural Denoise…"
                    visible: menu.folder === "" && menu.photos.length === 1 && menu.visible && menu.here && denoiser.fits(String(menu.photos[0]))
                    onPressed: root.denoisePhoto(String(menu.photos[0]))
                }
                Entry {
                    objectName: "menu-export"; visible: menu.folder === "" && menu.here; opacity: exporter.running ? 0.4 : 1
                    label: exporter.running ? "An export is running" : menu.photos.length > 1 ? "Export all (" + menu.photos.length + ")" : "Export"
                    onPressed: root.exportNow(menu.photos)
                }
                Entry {
                    objectName: "menu-copy-look"; label: "Copy look"; visible: menu.folder === "" && menu.photos.length === 1 && menu.here
                    onPressed: root.copyLook(String(menu.photos[0]))
                }
                Entry {
                    objectName: "menu-paste-look"; visible: menu.folder === "" && library.hasLook && menu.here
                    label: menu.photos.length > 1 ? "Paste look on all (" + menu.photos.length + ")" : "Paste look"
                    onPressed: root.pasteLook(menu.photos)
                }
                Entry {
                    objectName: "menu-reset"; label: "Reset edits…"; visible: menu.folder === "" && menu.here
                    onPressed: {
                        const these = menu.photos, one = these.length === 1
                        confirm.ask(one ? "Reset the edits of “" + these[0].split("/").pop() + "”?" : "Reset the edits of " + these.length + " photos?",
                                    (one ? "The photo goes" : "They go") + " back to how " + (one ? "it was" : "they were") + " shot. Checkpoints stay, and with "
                                    + (one ? "the" : "a") + " photo open, Undo brings its edit back.",
                                    "Reset edits", () => root.resetThese(these))
                    }
                }
            }
        }
    }

    // ---- Neural Denoise: its window, over everything but a question -----------------------
    // The photo is opened in Edit first if it is not the open one (from the library's menu):
    // the window shows the open photo, and uses its pictures.
    function denoisePhoto(path) {
        if (!denoiser.fits(path)) return
        if (path !== session.path || !editing) editPhoto(path)
        if (session.path === path) denoiseWindow.open()
    }
    Denoise {
        id: denoiseWindow
        objectName: "denoise-window"
        z: 9.5
        stage: stage
        onClosed: root.forceActiveFocus()
        // The new file: found by the library where its folder is one of the library's, and
        // shown straight after its original.
        onDone: (path) => { library.take(path); session.openBeside(path) }
    }

    // ---- A question that must be answered before anything else ---------------------------
    Item {
        id: confirm
        anchors.fill: parent
        z: 10
        visible: false
        property var then: null
        function ask(title, text, action, then) {
            question.text = title; detail.text = text; yes.label = action; confirm.then = then
            visible = true
            no.forceActiveFocus()   // the safe answer is the one under the keys
        }
        function answer(agreed) {
            visible = false
            root.forceActiveFocus()
            if (agreed && then) then()
            then = null
        }
        // While it is up, the keys are its own: the arrows (and Tab) go from one button to the
        // other, Enter or Space presses the one that has them, Escape is Cancel; nothing reaches
        // what lies under it (the arrows stepped through the photos behind the question).
        Keys.onPressed: (event) => {
            if (event.key === Qt.Key_Escape) answer(false)
            else if ([Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down, Qt.Key_Tab, Qt.Key_Backtab].includes(event.key))
                (yes.activeFocus ? no : yes).forceActiveFocus()
            event.accepted = true
        }
        Keys.onReleased: (event) => event.accepted = true
        Rectangle { anchors.fill: parent; color: Theme.bg; opacity: 0.6 }
        MouseArea { anchors.fill: parent; acceptedButtons: Qt.AllButtons; hoverEnabled: true; onWheel: (wheel) => wheel.accepted = true }
        Rectangle {
            objectName: "modal:confirm"
            anchors.centerIn: parent
            width: 400; height: words.height + 40; radius: 10
            color: Theme.panel; border { width: 1; color: Theme.faint }
            Accessible.role: Accessible.Dialog; Accessible.name: question.text
            Column {
                id: words
                x: 20; y: 20; width: parent.width - 40; spacing: 10
                Label { id: question; width: parent.width; color: Theme.text; font.pixelSize: 15; wrapMode: Text.WordWrap }
                Label { id: detail; width: parent.width; wrapMode: Text.WordWrap }
                Row {
                    anchors.right: parent.right; spacing: 8; topPadding: 6
                    PushButton { id: no; objectName: "confirm-no"; label: "Cancel"; onPressed: confirm.answer(false) }
                    PushButton { id: yes; objectName: "confirm-yes"; onPressed: confirm.answer(true) }
                }
            }
        }
    }
}
