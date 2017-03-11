/**
 * @file impl_diff.c
 *
 * @brief Uses simple diffusion and implicit time stepping to inpaint an image
 *
 * The inpainting image is found by evolving the PDE
 *  u_t = u_{xx} + u_{yy} + \lambda_{\Omega\setminus D}(u_0 - u)
 *
 * Uses ideas from fstv.c: http://www.ipol.im/pub/art/2013/40/
 * 
 */

#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <complex.h>
#include <math.h>
#include <fftw3.h>
#include "io_png.h"
#include "img_fun.h"

float* impl_diff(const float* image, const float* mask);

#define DPAD 0.01 
#define THRESHOLD 0.000000064
#define MAXITER 2000

int height, width;
float l0, dt, p0, p1;

fftw_plan p[3], pinv[3];
fftw_complex *out[3];
double *in[3];

int main(int argc, char** argv) {
    // check inputs 
    if(argc < 4) {
        printf("Too few arguments.\n");
        printf("Syntax: expl_diff [image] [mask] [output]\n");
        return EXIT_FAILURE;
    }

    // read in image to be inpainted
    float* im = NULL;
    size_t swidth, sheight;
    im = io_png_read_f32_rgb(argv[1], &swidth, &sheight);
    if(im == NULL) {
        printf("Are you sure %s is a png image?\n", argv[1]);
        return EXIT_FAILURE;
    }
    width = (int) swidth; 
    height = (int) sheight;

    // read and process the inpainting mask
    float* m;
    m = io_png_read_f32_rgb(argv[2], &swidth, &sheight);
    if(m == NULL) {
        printf("Are you sure %s is a png image?\n", argv[2]);
        free(im);
        return EXIT_FAILURE;
    }
    if(width != (int) swidth || height != (int) sheight) {
        printf("Dimensions of %s and %s disagree!\n", argv[1], argv[2]);
        free(im);
        free(m);
        return EXIT_FAILURE;
    }
    set_mask(m, m);

    l0 = 10;
    dt = 2.0;
    p0 = 0.50;
    p1 = 1.0;

    // map rgb values to [0, 1]
    int i; 
    for(i = 0; i < width*height*3; i++) 
        im[i] /= 255;

    // restoration using explicit diffusion
    float* res = impl_diff(im, m); 
    for(i = 0; i < width*height*3; i++)
        res[i] *= 255;
    io_png_write_f32(argv[3], res, swidth, sheight, 3);

    free(im); 
    free(m); 
    free(res);
    return EXIT_SUCCESS;
}
float* impl_diff(const float* image, const float* mask) {
    int i, j, c;
    float *u, *uold;

    u       = (float*) xmalloc(sizeof(float)*width*height*3);
    uold    = (float*) xmalloc(sizeof(float)*width*height*3);

    memcpy(u, image, sizeof(float)*width*height*3);

    // Fourier tools
    for(i = 0; i < 3; i++) {
        in[i] = (double*) xmalloc(sizeof(double)*width*height);
        out[i] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*(width/2+1)*height);
        p[i] = fftw_plan_dft_r2c_2d(height, width, in[i], out[i], FFTW_MEASURE);
        pinv[i] = fftw_plan_dft_c2r_2d(height, width, out[i], in[i], FFTW_MEASURE);
    }
    fftw_complex *D; 
    D = computeDenom(in, out, p, p0, p1);

    // start iterations 
    int it = 0;
    int chan;
do {
    memcpy(uold, u, sizeof(float)*width*height*3);
#pragma omp parallel for \
	shared(u,mask,image,l0) \
	private(c,i,j,chan)     
    for(c = 0; c < 3; c++) {
        chan = c*width*height;
        for(i = 0; i < height; i++) {
            for(j = 0; j < width; j++) {
                in[c][i*width+j] = (1+0.5*dt*l0)*u[chan+i*width+j] + dt*l0*mask[chan+i*width+j]*(image[chan+i*width+j] - u[chan+i*width+j]);
            }
        }

        fftw_execute(p[c]);
        for(i = 0; i < height*(width/2+1); i++)
            out[c][i] /= D[i];
        fftw_execute(pinv[c]);
        for(i = 0; i < height; i++) {
            for(j = 0; j < width; j++) {
                u[chan+i*width+j] = in[c][i*width+j] / (float) (height*width);
            }
        }
    }   
    it++;
} while(it<MAXITER && frob_dist(u, uold) > THRESHOLD);
    
    printf("Iterations: %i\tMax iterations: %i\nFrobenius distance: %e\tThreshold: %e\n", it, MAXITER, frob_dist(u, uold), THRESHOLD);

    free(uold);
    return u;    
}