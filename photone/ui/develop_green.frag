#version 440
// Develop.qml's first pass, over the mosaic: ahd_interpolate_green_h_and_v, the green at each
// red and blue from its row and from its column (a green's own value both ways), as 4 bytes.
layout(location = 0) in vec2 qt_TexCoord0;
layout(location = 0) out vec4 fragColor;
layout(binding = 1) uniform sampler2D mosaic;
#define MOSAIC
#include "develop.glsl"

int along(ivec2 p, ivec2 by, int m) {
    int a = scaled(p - by), b = scaled(p + by);
    int g = ((a + m + b) * 2 - scaled(p - 2 * by) - scaled(p + 2 * by)) >> 2;
    return clamp(g, min(a, b), max(a, b));
}

void main() {
    ivec2 p = ivec2(qt_TexCoord0 * sensor.xy);
    int m = scaled(p);
    bool green = greenAs(colorAt(p)) == 1;
    int h = green ? m : along(p, ivec2(1, 0), m), v = green ? m : along(p, ivec2(0, 1), m);
    fragColor = vec4(h >> 8, h & 255, v >> 8, v & 255) / 255.0;
}
