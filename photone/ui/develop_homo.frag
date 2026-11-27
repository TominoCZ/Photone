#version 440
// Develop.qml's second pass: ahd_interpolate_build_homogeneity_map. Each way's colors in
// CIELab here and at the four pixels beside; how many of those are as near as the nearer way
// lets them be, for each way (0..4, a byte each).
layout(location = 0) in vec2 qt_TexCoord0;
layout(location = 0) out vec4 fragColor;
layout(binding = 1) uniform sampler2D mosaic;
layout(binding = 2) uniform sampler2D green;
#define MOSAIC
#define GREEN
#include "develop.glsl"

// cielab(): xyz = 0.5 + xyz_cam * rgb (floats, left to right), LibRaw's cube root of each
// (its table: x / 65535 above 0.008856, else 7.787 x + 16 / 116), then integer L, a, b.
float root(float xyz) {
    precise float r = float(clamp(int(xyz), 0, 65535)) / 65535.0;
    return r > 0.008856 ? pow(r, 1.0 / 3.0) : 7.787 * r + 16.0 / 116.0;
}
ivec3 lab(ivec3 rgb) {
    vec3 c = vec3(rgb), f;
    for (int i = 0; i < 3; i++) {
        precise float xyz = 0.5 + xyzCam[0][i] * c.r;
        xyz = xyz + xyzCam[1][i] * c.g;
        xyz = xyz + xyzCam[2][i] * c.b;
        f[i] = root(xyz);
    }
    precise float el = 64.0 * (116.0 * f.y - 16.0), a = 32000.0 * (f.x - f.y), b = 12800.0 * (f.y - f.z);
    return ivec3(int(el), int(a), int(b));
}

void main() {
    ivec2 p = ivec2(qt_TexCoord0 * sensor.xy);
    const ivec2 steps[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));
    ivec3 h, v, here[2], there[8];   // (k * 4 + s: ES 300 has no arrays of arrays)
    colorsAt(p, h, v);
    here[0] = lab(h);
    here[1] = lab(v);
    for (int s = 0; s < 4; s++) {
        colorsAt(p + steps[s], h, v);
        there[s] = lab(h);
        there[4 + s] = lab(v);
    }
    int dl[8];
    uint dab[8];
    for (int k = 0; k < 2; k++) {
        for (int s = 0; s < 4; s++) {
            ivec3 d = here[k] - there[k * 4 + s];
            dl[k * 4 + s] = abs(d.x);
            dab[k * 4 + s] = uint(d.y) * uint(d.y) + uint(d.z) * uint(d.z);   // (wraps, as LibRaw's)
        }
    }
    int leps = min(max(dl[0], dl[1]), max(dl[6], dl[7]));
    uint abeps = min(max(dab[0], dab[1]), max(dab[6], dab[7]));
    int homo[2] = int[2](0, 0);
    for (int k = 0; k < 2; k++)
        for (int s = 0; s < 4; s++)
            homo[k] += dl[k * 4 + s] <= leps && dab[k * 4 + s] <= abeps ? 1 : 0;
    fragColor = vec4(homo[0], homo[1], 0, 255) / 255.0;
}
