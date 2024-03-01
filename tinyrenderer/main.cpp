#include <vector>
#include <iostream>

#include "tgaimage.h"
#include "model.h"
#include "geometry.h"
#include "our_gl.h"

const TGAColor white = TGAColor(255, 255, 255, 255);
const TGAColor red = TGAColor(255, 0, 0, 255);
const TGAColor green = TGAColor(0, 255, 0, 0);
const TGAColor blue = TGAColor(0, 0, 255, 255);

Model* model = NULL;
const int width = 800;
const int height = 800;
const int depth = 255;
Vec3f eye(0, 1, 3);
Vec3f lookAt(0, 0, 0);
Vec3f up(0, 1, 0);
Vec3f light_dir(1, 1, 1);

float* shadowbuffer = NULL;
float* zbuffer = NULL;
TGAImage occl(1024, 1024, TGAImage::GRAYSCALE);
TGAImage total(1024, 1024, TGAImage::GRAYSCALE);

TGAImage image(width, height, TGAImage::RGB);

struct GouraudShader : public IShader {
    Vec3f          varying_intensity; // written by vertex shader, read by fragment shader
    mat<2, 3, float> varying_uv;      
    mat<4, 4, float> uniform_M;   //  Projection*ModelView
    mat<4, 4, float> uniform_MIT; // (Projection*ModelView).invert_transpose()
    mat<4, 3, float> varying_tri; // 存储三角形的每个顶点的坐标 在裁剪空间中
    mat<3, 3, float> varying_nrm; // 存储每个顶点的法向量 在世界空间中
    mat<3, 3, float> ndc_tri;     // 存储三角形的每个顶点的坐标 在（Normalized Device Coordinates，NDC）空间

    virtual Vec4f vertex(int iface, int nthvert) {
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert)); // read the vertex from .obj file
        gl_Vertex = Viewport * Projection * ViewTrans * gl_Vertex;     // transform it to screen coordinates
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));
        varying_intensity[nthvert] = std::max(0.f, model->normal(iface, nthvert) * light_dir); // get diffuse lighting intensity
        
        varying_nrm.set_col(nthvert, proj<3>((Projection * ViewTrans).invert_transpose() * embed<4>(model->normal(iface, nthvert), 0.f)));
        varying_tri.set_col(nthvert, gl_Vertex);
        ndc_tri.set_col(nthvert, proj<3>(gl_Vertex / gl_Vertex[3]));

        return gl_Vertex;
    }

    virtual bool fragment(Vec3f bar, TGAColor& color, int mode) {
        Vec2f uv = varying_uv * bar;
        Vec3f bn = (varying_nrm * bar).normalize();
        if (mode == 1) {
            float intensity = varying_intensity * bar;   // interpolate intensity for the current pixel
            color = model->diffuse(uv) * intensity; 
            return false;
        }
        else if (mode == 2) {
            Vec3f n = proj<3>(uniform_MIT * embed<4>(model->normal(uv))).normalize(); //法线从模型空间变换到世界空间
            Vec3f l = proj<3>(uniform_M * embed<4>(light_dir)).normalize(); // 光源方向从世界空间变换到模型空间
            float intensity = std::max(0.f, n * l);
            color = model->diffuse(uv) * intensity;
            return false;
        }
        else if (mode == 3) {
            Vec3f n = proj<3>(uniform_MIT * embed<4>(model->normal(uv))).normalize();
            Vec3f l = proj<3>(uniform_M * embed<4>(light_dir)).normalize();
            Vec3f r = (n * (n * l * 2.f) - l).normalize();   // 反射光线的方向
            float spec = pow(std::max(r.z, 0.0f), model->specular(uv)); // 模型的纹理中获取的镜面反射系数
            float diff = std::max(0.f, n * l); // 漫反射的强度diff
            TGAColor c = model->diffuse(uv);
            color = c;
            // 漫反射颜色 * (漫反射强度 + 镜面反射颜色 * 镜面反射强度)
            for (int i = 0; i < 3; i++) color[i] = std::min<float>(5 + c[i] * (diff + .6 * spec), 255);
            return false;
        }
        else if (mode == 4) {
            // 切线空间法线贴图
            mat<3, 3, float> A;
            A[0] = ndc_tri.col(1) - ndc_tri.col(0);
            A[1] = ndc_tri.col(2) - ndc_tri.col(0);
            A[2] = bn;

            mat<3, 3, float> AI = A.invert();

            Vec3f i = AI * Vec3f(varying_uv[0][1] - varying_uv[0][0], varying_uv[0][2] - varying_uv[0][0], 0);
            Vec3f j = AI * Vec3f(varying_uv[1][1] - varying_uv[1][0], varying_uv[1][2] - varying_uv[1][0], 0);

            mat<3, 3, float> B;
            B.set_col(0, i.normalize());
            B.set_col(1, j.normalize());
            B.set_col(2, bn);

            Vec3f n = (B * model->normal(uv)).normalize();

            float diff = std::max(0.f, n * light_dir);
            color = model->diffuse(uv) * diff;

            return false;
        }
    }
    virtual bool fragment(Vec3f gl_FragCoord, Vec3f bar, TGAColor& color) { return false; }
};

struct DepthShader : public IShader{ // 使用屏幕空间中插值得到的深度值
    mat<3, 3, float> varying_tri;
    DepthShader() : varying_tri() {}

    virtual Vec4f vertex(int iface, int nthvert) {
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert)); // read the vertex from .obj file
        gl_Vertex = Viewport * Projection * ViewTrans * gl_Vertex;          // transform it to screen coordinates
        varying_tri.set_col(nthvert, proj<3>(gl_Vertex / gl_Vertex[3]));
        return gl_Vertex;
    }

    virtual bool fragment(Vec3f bar, TGAColor& color, int mode) {
        Vec3f p = varying_tri * bar;
        color = TGAColor(255, 255, 255) * (p.z / depth);
        return false;
    }
    virtual bool fragment(Vec3f gl_FragCoord, Vec3f bar, TGAColor& color) { return false; }
};

struct ZShader : public IShader { // 使用裁剪空间中的深度值
    mat<4, 3, float> varying_tri;

    virtual Vec4f vertex(int iface, int nthvert) {
        Vec4f gl_Vertex = Projection * ViewTrans * embed<4>(model->vert(iface, nthvert));
        varying_tri.set_col(nthvert, gl_Vertex);
        return gl_Vertex;
    }
    virtual bool fragment(Vec3f bar, TGAColor& color, int mode) { return false; }

    virtual bool fragment(Vec3f gl_FragCoord, Vec3f bar, TGAColor& color) {
        color = TGAColor(255, 255, 255) * ((gl_FragCoord.z + 1.f) / 2.f);
        return false;
    }
};

struct RandomShader :public IShader {
    mat<2, 3, float> varying_uv;
    mat<4, 3, float> varying_tri;

    virtual Vec4f vertex(int iface, int nthvert) {
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));
        Vec4f gl_Vertex = Projection * ViewTrans * embed<4>(model->vert(iface, nthvert));
        varying_tri.set_col(nthvert, gl_Vertex);
        return gl_Vertex;
    }
    virtual bool fragment(Vec3f bar, TGAColor& color, int mode) { return false; }

    virtual bool fragment(Vec3f gl_FragCoord, Vec3f bar, TGAColor& color) {
        Vec2f uv = varying_uv * bar;
        if (std::abs(shadowbuffer[int(gl_FragCoord.x + gl_FragCoord.y * width)] - gl_FragCoord.z < 1e-2)) {
            occl.set(uv.x * 1024, uv.y * 1024, TGAColor(255));
        }
        color = TGAColor(255, 0, 0);
        return false;
    }
};

struct AOShader : public IShader {
    mat<2, 3, float> varying_uv;
    mat<4, 3, float> varying_tri;
    TGAImage aoimage;

    virtual Vec4f vertex(int iface, int nthvert) {
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));
        Vec4f gl_Vertex = Projection * ViewTrans * embed<4>(model->vert(iface, nthvert));
        varying_tri.set_col(nthvert, gl_Vertex);
        return gl_Vertex;
    }

    virtual bool fragment(Vec3f gl_FragCoord, Vec3f bar, TGAColor& color) {
        Vec2f uv = varying_uv * bar;
        int t = aoimage.get(uv.x * 1024, uv.y * 1024)[0];
        TGAColor c = model->diffuse(uv);
        color = c * t;
        return false;
    }

    virtual bool fragment(Vec3f bar, TGAColor& color, int mode) { return false; }
};

struct SecondShader :public IShader {
    mat<4, 4, float> uniform_M;   //  Projection*ModelView
    mat<4, 4, float> uniform_MIT; // (Projection*ModelView).invert_transpose()
    mat<4, 4, float> uniform_Mshadow; // transform framebuffer screen coordinates to shadowbuffer screen coordinates
    mat<2, 3, float> varying_uv;  // triangle uv coordinates, written by the vertex shader, read by the fragment shader
    mat<3, 3, float> varying_tri; // triangle coordinates before Viewport transform, written by VS, read by FS

    SecondShader(Matrix M, Matrix MIT, Matrix MS) : uniform_M(M), uniform_MIT(MIT), uniform_Mshadow(MS), varying_uv(), varying_tri() {}

    virtual Vec4f vertex(int iface, int nthvert) {
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));
        Vec4f gl_Vertex = Viewport * Projection * ViewTrans * embed<4>(model->vert(iface, nthvert));
        varying_tri.set_col(nthvert, proj<3>(gl_Vertex / gl_Vertex[3]));
        return gl_Vertex;
    }

    virtual bool fragment(Vec3f bar, TGAColor& color, int mode) {
        Vec4f sb_p = uniform_Mshadow * embed<4>(varying_tri * bar); // corresponding point in the shadow buffer
        sb_p = sb_p / sb_p[3];
        int idx = int(sb_p[0]) + int(sb_p[1]) * width; // index in the shadowbuffer array
        float shadow = .3 + .7 * (shadowbuffer[idx] < sb_p[2] + 0.5); // +0.5 to avoid z-fighting
        Vec2f uv = varying_uv * bar;
        Vec3f n = proj<3>(uniform_MIT * embed<4>(model->normal(uv))).normalize(); // 法线
        Vec3f l = proj<3>(uniform_M * embed<4>(light_dir)).normalize(); // 光线方向
        Vec3f r = (n * (n * l * 2.f) - l).normalize();   // reflected light
        float spec = pow(std::max(r.z, 0.0f), model->specular(uv));
        float diff = std::max(0.f, n * l);
        TGAColor c = model->diffuse(uv);
        for (int i = 0; i < 3; i++) color[i] = std::min<float>(20 + c[i] * shadow * (1.2 * diff + .6 * spec), 255);
        return false;
    }
    virtual bool fragment(Vec3f gl_FragCoord, Vec3f bar, TGAColor& color) { return false; }
};

float max_elevation_angle(float* zbuffer, Vec2f p, Vec2f dir) {
    float maxangle = 0;
    for (float t = 0.; t < 1000.; t += 1.) {
        Vec2f cur = p + dir * t;
        if (cur.x >= width || cur.y >= height || cur.x < 0 || cur.y < 0) return maxangle;

        float distance = (p - cur).norm();
        if (distance < 1.f) continue;
        float elevation = zbuffer[int(cur.x) + int(cur.y) * width] - zbuffer[int(p.x) + int(p.y) * width];
        maxangle = std::max(maxangle, atanf(elevation / distance));
    }
    return maxangle;
}

Vec3f rand_point_on_unit_sphere() {
    float u = (float)rand() / (float)RAND_MAX;
    float v = (float)rand() / (float)RAND_MAX;
    float theta = 2.f * 3.14 * u;
    float phi = acos(2.f * v - 1.f);
    return Vec3f(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
}

void Depth() {
    DepthShader depthShader;
    TGAImage depth(width, height, TGAImage::RGB);
    viewTrans(light_dir, lookAt, up); // 注意 光源视角
    viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
    projection(0);

    for (int i = 0; i < model->nfaces(); i++) {
        Vec4f screen_coords[3];
        for (int j = 0; j < 3; j++) {
            screen_coords[j] = depthShader.vertex(i, j);
        }
        triangle(screen_coords, depthShader, image, shadowbuffer);
    }
    depth.flip_vertically();
    depth.write_tga_file("depth.tga");

    Matrix M = Viewport * Projection * ViewTrans;

    TGAImage frame(width, height, TGAImage::RGB);
    viewTrans(eye, lookAt, up);
    viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
    projection(-1.f / (eye - lookAt).norm());

    SecondShader secondShader(ViewTrans, (Projection * ViewTrans).invert_transpose(), M * (Viewport * Projection * ViewTrans).invert());
    Vec4f screen_coords[3];
    for (int i = 0; i < model->nfaces(); i++) {
        for (int j = 0; j < 3; j++) {
            screen_coords[j] = secondShader.vertex(i, j);
        }
        triangle(screen_coords, secondShader, frame, zbuffer);
    }
    frame.flip_vertically();
    frame.write_tga_file("shadowoutput.tga");
}

void Random() {
    // Random Render
    const int nrenders = 1;
    for (int iter = 1; iter <= nrenders; iter++) {
        std::cerr << iter << " from " << nrenders << std::endl;
        for (int i = 0; i < 3; i++) up[i] = (float)rand() / (float)RAND_MAX;
        eye = rand_point_on_unit_sphere();
        eye.y = std::abs(eye.y);
        std::cout << "v " << eye << std::endl;

        for (int i = width * height; i--; shadowbuffer[i] = zbuffer[i] = -std::numeric_limits<float>::max());

        TGAImage frame(width, height, TGAImage::RGB);
        viewTrans(eye, lookAt, up);
        viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
        projection(0);//-1.f/(eye-center).norm());

        ZShader zshader;
        for (int i = 0; i < model->nfaces(); i++) {
            for (int j = 0; j < 3; j++) {
                zshader.vertex(i, j);
            }
            triangle(zshader.varying_tri, zshader, frame, shadowbuffer);
        }
        frame.flip_vertically();
        RandomShader shader;
        occl.clear();
        for (int i = 0; i < model->nfaces(); i++) {
            for (int j = 0; j < 3; j++) {
                shader.vertex(i, j);
            }
            triangle(shader.varying_tri, shader, frame, zbuffer);
        }

        //        occl.gaussian_blur(5);
        for (int i = 0; i < 1024; i++) {
            for (int j = 0; j < 1024; j++) {
                float tmp = total.get(i, j)[0];
                total.set(i, j, TGAColor((tmp * (iter - 1) + occl.get(i, j)[0]) / (float)iter + .5f));
            }
        }
    }
    total.flip_vertically();
    total.write_tga_file("occlusion.tga");
    occl.flip_vertically();
    occl.write_tga_file("occl.tga");

    TGAImage frame(width, height, TGAImage::RGB);
    viewTrans(eye, lookAt, up);
    viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
    projection(-1.f / (eye - lookAt).norm());
    for (int i = width * height; i--; zbuffer[i] = -std::numeric_limits<float>::max());
    AOShader aoshader;
    aoshader.aoimage.read_tga_file("occlusion.tga");
    aoshader.aoimage.flip_vertically();
    for (int i = 0; i < model->nfaces(); i++) {
        for (int j = 0; j < 3; j++) {
            aoshader.vertex(i, j);
        }
        triangle(aoshader.varying_tri, aoshader, frame, zbuffer);
    }
    frame.flip_vertically();
    frame.write_tga_file("frame.tga");
}

int main(int argc, char** argv) {
    if (2 == argc) {
        model = new Model(argv[1]);
    }
    else {
        model = new Model("obj/african_head.obj");
    }

    light_dir.normalize();
    zbuffer = new float[width * height];
    shadowbuffer = new float[width * height];
    for (int i = width * height; --i; ) {
        zbuffer[i] = shadowbuffer[i] = -std::numeric_limits<float>::max();
    }

    TGAImage frame(width, height, TGAImage::RGB);
    viewTrans(eye, lookAt, up);
    viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
    projection(-1.f / (eye - lookAt).norm());

    ZShader zshader;
    for (int i = 0; i < model->nfaces(); i++) {
        for (int j = 0; j < 3; j++) {
            zshader.vertex(i, j);
        }
        triangle(zshader.varying_tri, zshader, frame, zbuffer);
    }

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            if (zbuffer[x + y * width] < -1e5) continue;
            float total = 0;
            for (float a = 0; a < 3.14 * 2 - 1e-4; a += 3.14 / 4) {
                total += 3.14 / 2 - max_elevation_angle(zbuffer, Vec2f(x, y), Vec2f(cos(a), sin(a)));
            }
            total /= (3.14 / 2) * 8;
            total = pow(total, 100.f);
            frame.set(x, y, TGAColor(total * 255, total * 255, total * 255));
        }
    }

    frame.flip_vertically();
    frame.write_tga_file("framebuffer.tga");

    delete model;
    return 0;
}