#version 440
// Develop.qml's last pass, over the photo: each pixel of it found on the sensor (cut, turned
// back by LibRaw's flip_index), its colors taken along the way that was steadier over the 3 x 3
// around it (the mean of both where neither was), or in LibRaw's 5 px frame the mean of each
// color in the 3 x 3 (border_interpolate); then to Rec.2020, 0..1.
layout(location = 0) in vec2 qt_TexCoord0;
layout(location = 0) out vec4 fragColor;
layout(binding = 1) uniform sampler2D mosaic;
layout(binding = 2) uniform sampler2D green;
layout(binding = 3) uniform sampler2D homo;
#define MOSAIC
#define GREEN
#include "develop.glsl"

void main() {
    ivec2 u = ivec2(qt_TexCoord0 * place.zw) + ivec2(place.xy);
    int flip = int(sensor.z);
    ivec2 p = (flip & 4) != 0 ? u.yx : u, size = ivec2(sensor.xy);
    if ((flip & 2) != 0) p.y = size.y - 1 - p.y;
    if ((flip & 1) != 0) p.x = size.x - 1 - p.x;
    ivec3 rgb;
    if (any(lessThan(p, ivec2(5))) || any(greaterThanEqual(p, size - 5))) {
        ivec3 sum = ivec3(0), n = ivec3(0);
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                ivec2 q = p + ivec2(x, y);
                if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, size))) continue;
                int c = greenAs(colorAt(q));
                sum[c] += scaled(q);
                n[c] += 1;
            }
        }
        rgb = sum / max(n, ivec3(1));
        rgb[greenAs(colorAt(p))] = scaled(p);
    } else {
        ivec2 votes = ivec2(0);
        for (int y = -1; y <= 1; y++)
            for (int x = -1; x <= 1; x++)
                votes += ivec2(texture(homo, (vec2(p + ivec2(x, y)) + 0.5) / sensor.xy).rg * 255.0 + 0.5);
        ivec3 h, v;
        colorsAt(p, h, v);
        rgb = votes.x == votes.y ? (h + v) >> 1 : votes.y > votes.x ? v : h;
    }
    vec3 c = vec3(rgb), out_;
    for (int i = 0; i < 3; i++) {
        precise float acc = outCam[0][i] * c.r;
        acc = acc + outCam[1][i] * c.g;
        acc = acc + outCam[2][i] * c.b;
        out_[i] = float(clamp(int(acc), 0, 65535));
    }
    fragColor = vec4(out_ / 65535.0, 1.0);
}
