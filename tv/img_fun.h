#ifndef IMG_FUN_H
#define IMG_FUN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
extern int height, width;
extern float l0, dt;

/* imgfun.c */
void set_mask(float *mask, const float *im);
void set_value(float *A, float value, int size);
fftw_complex* computeDenom(double **in, fftw_complex **out, fftw_plan *p, float p0, float p1);
inline int Ic(int i, int j, int c) {
    if(i < 0)
        i += height;
    else if(i >= height)
        i -= height;
    if(j < 0)
        j += width;
    else if(j >= width) 
        j -= width;

    return c*height*width + i*width + j;
}
inline int In(int i, int j) {
    if(i < 0)
        i += height;
    else if(i >= height)
        i -= height;
    if(j < 0)
        j += width;
    else if(j >= width)
        j -= width;

    return i*width + j;
}
inline void* xmalloc(size_t size) {
    void* mem = malloc(size);
    if(mem==NULL) {
        perror("Memory allocation error");
        exit(EXIT_FAILURE);
    }
    return mem;
}
inline float MAX(float x, float y) { 
    return x<=y ? y:x;
}
inline float dyy(const float *x, int i, int j) {
    return x[In(i+1,j)] + x[In(i-1,j)] - 2*x[In(i,j)];
}
inline float dxx(const float* x, int i, int j) {
    return x[In(i,j+1)] - 2*x[In(i,j)] + x[In(i,j-1)];
}
inline float cdx(const float *x, int i, int j) {
    return 0.5*(x[In(i,j+1)] - x[In(i,j-1)]);
}
inline float cdy(const float *x, int i, int j) {
    return 0.5*(x[In(i+1,j)] - x[In(i-1,j)]);
}
inline float cdxy(const float *x, int i, int j){
    return 0.25*(x[In(i+1,j+1)] + x[In(i-1,j-1)] - x[In(i-1,j+1)] - x[In(i+1,j-1)]);
}
inline float frob_dist(const float* A, const float* B) {
    float top[3] = {0,0,0};
    float bot[3] = {0,0,0};
    int i, j, c;

#pragma omp parallel for \
    shared(top, bot, A, B) \
    private(i, j, c)
    for(c = 0; c < 3; c++) {
        for(i = 0; i < height; i++) {
            for(j = 0; j < width; j++) {
                top[c] += (A[Ic(i,j,c)] - B[Ic(i,j,c)]) * (A[Ic(i,j,c)] - B[Ic(i,j,c)]);
                bot[c] += B[Ic(i,j,c)] * B[Ic(i,j,c)];
            }
        }
    }
    return MAX(MAX(top[0]/bot[0], top[1]/bot[1]), top[2]/bot[2]);
}

#ifdef __cplusplus
}
#endif

#endif /* IMG_FUN_H */