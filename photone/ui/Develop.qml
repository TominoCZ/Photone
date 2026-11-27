import QtQuick

// A raw file's mosaic (decode.texels) developed by the GPU as LibRaw develops it, for the
// photo's full size: decode.figures says how, develop*.frag do it in three passes, the last
// one turned and cut to the photo. `picture` is the result, a full-size picture as the one
// decoded on the CPU was (0.35 s of a Sony's 0.6 s, and 48 MB sent to the GPU where 264 were).
// ponytail: the two passes' pictures stay on the GPU with the result (a byte a pixel of the
// sensor four times over each, 260 MB at 33 MP). The upgrade is to keep only the result (a
// ShaderEffectSource with `live` off) and let them go once it is drawn.
Item {
    id: develop
    property var mosaic: null    // the Image of the mosaic
    property var figures: null   // session.mosaicOf(its address): decode.figures, and `kept`
    readonly property alias picture: done
    readonly property bool on: figures !== null && figures !== undefined && mosaic !== null
    visible: false

    component Pass: ShaderEffect {
        property var figures: develop.figures
        property var mosaic: develop.mosaic
        property var green: mosaic   // (a pass that reads none: any picture)
        property var homo: mosaic
        readonly property var f: develop.on ? figures : null
        function four(a) { return a ? Qt.vector4d(a[0], a[1], a[2], a[3]) : Qt.vector4d(0, 0, 0, 0) }
        function three(m) { return m ? Qt.matrix4x4(m[0], m[1], m[2], 0, m[3], m[4], m[5], 0, m[6], m[7], m[8], 0, 0, 0, 0, 1) : Qt.matrix4x4() }
        property vector4d scale: four(f && f.scale)
        property vector4d black: four(f && f.black)
        property vector4d cell: four(f && f.cell)
        property vector4d sensor: f ? Qt.vector4d(f.sensor[0], f.sensor[1], f.flip, 0) : Qt.vector4d(1, 1, 0, 0)
        property vector4d place: f ? Qt.vector4d(f.kept[0], f.kept[1], f.kept[2], f.kept[3]) : Qt.vector4d(0, 0, 1, 1)
        property matrix4x4 xyzCam: three(f && f.xyz_cam)
        property matrix4x4 outCam: three(f && f.out_cam)
        readonly property size sourceSize: Qt.size(width, height)
        width: Math.max(1, sensor.x); height: Math.max(1, sensor.y)
        visible: false; blending: false
        vertexShader: developVertex
        // (Always on, 1 x 1 while there is nothing to develop: a ShaderEffect takes the
        // picture of an item it reads once, and an item is one only while its layer is on.
        // Turned on with `on`, the passes and the photo read none, green was 0: magenta.)
        layer.enabled: true; layer.smooth: false
        layer.format: ShaderEffectSource.RGBA8; layer.textureSize: Qt.size(width, height)
    }
    Pass { id: greens; fragmentShader: developGreenShader }
    Pass { id: votes; green: greens; fragmentShader: developHomoShader }
    Pass {
        id: done
        objectName: "developed"
        green: greens; homo: votes
        width: Math.max(1, place.z); height: Math.max(1, place.w)
        fragmentShader: developShader
        layer.smooth: true; layer.format: ShaderEffectSource.RGBA16F
    }
}
