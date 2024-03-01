#include "tgaimage.h"
#include "geometry.h"

extern Matrix ViewTrans;
extern Matrix Viewport;
extern Matrix Projection;

void viewport(int x, int y, int w, int h);
void projection(float coeff = 0.f); // coeff = -1/c
void viewTrans(Vec3f eye_pos, Vec3f lookAt, Vec3f up);

struct IShader {
    virtual ~IShader();
    virtual Vec4f vertex(int iface, int nthvert) = 0;
    virtual bool fragment(Vec3f bar, TGAColor& color, int mode) = 0;
    virtual bool fragment(Vec3f gl_FragCoord, Vec3f bar, TGAColor& color) = 0;
};

void triangle(Vec4f* pts, IShader& shader, TGAImage& image, TGAImage& zbuffer);
void triangle(Vec4f* pts, IShader& shader, TGAImage& image, float* zbuffer);
void triangle(mat<4, 3, float>& clipc, IShader& shader, TGAImage& image, float* zbuffer);