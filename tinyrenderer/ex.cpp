#include "tgaimage.h"
#include "geometry.h"

void line(int x0, int y0, int x1, int y1, TGAImage& image, TGAColor color) {
    int t0, t1, t2, t3;
    bool xsample;
    if (std::abs(x0 - x1) < std::abs(y0 - y1)) {
        xsample = false;
        t0 = y0; t1 = y1;
        t2 = x0; t3 = x1;
    }
    else {
        xsample = true;
        t0 = x0; t1 = x1;
        t2 = y0; t3 = y1;
    }
    if (t0 > t1) {
        std::swap(t0, t1);
        std::swap(t2, t3);
    }
    int dx = t1 - t0;
    int dy = t3 - t2;
    float derror = std::abs(dy / float(dx));
    float error = 0;
    int y = t2;
    for (int x = t0; x <= t1; x++) {
        if (xsample) image.set(x, y, color);
        else image.set(y, x, color);
        error += derror;
        if (error > .5) {
            y += (y1 > y0 ? 1 : -1);
            error -= 1.;
        }
    }
}

void Triangle(int x0, int y0, int x1, int y1, int x2, int y2, TGAImage& image, TGAColor color) {
    int maxx = std::max(x0, std::max(x1, x2)), maxy = std::max(y0, std::max(y1, y2));
    int minx = std::min(x0, std::min(x1, x2)), miny = std::min(y0, std::min(y1, y2));
    int tx, ty;
    for (tx = minx; tx <= maxx; tx++) {
        for (ty = miny; ty <= maxy; ty++) {
            //if (isInsideTriangle(x0, y0, x1, y1, x2, y2, tx, ty)) {
            //    image.set(tx, ty, color);
            //}
        }
    }
}


Vec3f barycentric(Vec3f v0, Vec3f v1, Vec3f v2, Vec3f p) {
    float alpha = ((v1.y - v2.y) * (p.x - v2.x) + (v2.x - v1.x) * (p.y - v2.y)) /
        ((v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y));
    float beta = ((v2.y - v0.y) * (p.x - v2.x) + (v0.x - v2.x) * (p.y - v2.y)) /
        ((v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y));
    float gamma = 1.0f - alpha - beta;

    return Vec3f(alpha, beta, gamma);
}