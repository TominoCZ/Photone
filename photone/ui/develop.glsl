// A raw file's mosaic developed as LibRaw 0.22 develops it (Develop.qml, decode.figures; the
// port of Neuro's libraw.py, PLAN.md §4.44 there): each color's black off and scaled by the
// white balance, AHD's two greens, its colors along each way, their CIELab, the way that
// keeps the colors steadier, LibRaw's 5 px frame, Rec.2020. Integers where LibRaw has them,
// floats where it has floats; `precise` keeps a multiply and an add from being fused.
layout(std140, binding = 0) uniform buf {  // one for every pass and the vertex stage
    mat4 qt_Matrix;
    float qt_Opacity;
    vec4 scale;    // LibRaw's scale_mul for each of its colors (0 red, 1 green, 2 blue, 3 green)
    vec4 black;    // each color's black
    vec4 cell;     // the color at (0, 0), (1, 0), (0, 1), (1, 1) of the mosaic (x, y)
    vec4 sensor;   // the mosaic's width and height; LibRaw's flip
    vec4 place;    // where the photo lies in the upright picture: x, y, width, height
    mat4 xyzCam;   // the camera's colors to XYZ for the Lab (upper 3 x 3, rows as LibRaw's)
    mat4 outCam;   // and to Rec.2020
};

int colorAt(ivec2 p) { return int(cell[(p.y & 1) * 2 + (p.x & 1)]); }
int greenAs(int c) { return c == 3 ? 1 : c; }  // (LibRaw's second green is green from here on)
bool redRow(int y) { return int(cell[(y & 1) * 2]) == 0 || int(cell[(y & 1) * 2 + 1]) == 0; }

#ifdef MOSAIC
// The sensor's value: two to a texel, each as its low and high byte (decode.texels).
int rawAt(ivec2 p) {
    vec4 t = texture(mosaic, (vec2(p.x >> 1, p.y) + 0.5) / vec2((int(sensor.x) + 1) >> 1, sensor.y));
    vec2 b = (p.x & 1) == 0 ? t.rg : t.ba;
    return int(b.x * 255.0 + 0.5) + 256 * int(b.y * 255.0 + 0.5);
}
// copy_bayer and scale_colors: black off (none below 0), times scale_mul, cut off at 65535.
int scaled(ivec2 p) {
    int c = colorAt(p);
    return clamp(int(float(max(rawAt(p) - int(black[c]), 0)) * scale[c]), 0, 65535);
}
#endif

#ifdef GREEN
// The two greens (along the row, along the column) the first pass wrote.
ivec2 greens(ivec2 p) {
    ivec4 b = ivec4(texture(green, (vec2(p) + 0.5) / sensor.xy) * 255.0 + 0.5);
    return ivec2(b.x * 256 + b.y, b.z * 256 + b.w);
}

// ahd_interpolate_r_and_b_in_rgb, both ways at once: [0] along the rows, [1] the columns.
void colorsAt(ivec2 p, out ivec3 h, out ivec3 v) {
    int m = scaled(p), c = greenAs(colorAt(p));
    int l = scaled(p + ivec2(-1, 0)), r = scaled(p + ivec2(1, 0));
    int u = scaled(p + ivec2(0, -1)), d = scaled(p + ivec2(0, 1));
    int diag = scaled(p + ivec2(-1, -1)) + scaled(p + ivec2(1, -1)) + scaled(p + ivec2(-1, 1)) + scaled(p + ivec2(1, 1));
    ivec2 g = greens(p), gl = greens(p + ivec2(-1, 0)), gr = greens(p + ivec2(1, 0));
    ivec2 gu = greens(p + ivec2(0, -1)), gd = greens(p + ivec2(0, 1));
    ivec2 gdiag = greens(p + ivec2(-1, -1)) + greens(p + ivec2(1, -1)) + greens(p + ivec2(-1, 1)) + greens(p + ivec2(1, 1));
    bool across = redRow(p.y);
    ivec3 way[2];
    for (int k = 0; k < 2; k++) {
        int hz = clamp(m + ((l + r - gl[k] - gr[k]) >> 1), 0, 65535);    // at a green: beside it
        int vt = clamp(m + ((u + d - gu[k] - gd[k]) >> 1), 0, 65535);    // above and below it
        int dg = clamp(g[k] + ((diag - gdiag[k] + 1) >> 2), 0, 65535);   // across the corners
        int red = c == 0 ? m : c == 2 ? dg : across ? hz : vt;
        int blue = c == 2 ? m : c == 0 ? dg : across ? vt : hz;
        way[k] = ivec3(red, g[k], blue);
    }
    h = way[0];
    v = way[1];
}
#endif
