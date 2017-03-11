/**
 * @file expl_tv.c
 *
 * @brief Uses TV inpainting with explicit time stepping to inpaint an image
 *
 * The inpainting image is found by evolving the PDE
 *  u_t = div(grad(u)/abs(grad(u))) + \lambda_{\Omega\setminus D}(u_0 - u)
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

float* expl_tv(const float* image, const float* mask);

#define DPAD 0.1 
#define THRESHOLD 0.000000064
#define MAXITER 2000

int height, width;
float l0, dt;

int main(int argc, char** argv) {
    // check inputs 
    if(argc < 4) {
        printf("Too few arguments.\n");
        printf("Syntax: expl_tv [image] [mask] [output]\n");
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

    // set parameters
    dt = 0.005; 
    l0 = 200;

    // map rgb values to [0, 1]
    int i; 
    for(i = 0; i < width*height*3; i++) 
        im[i] /= 255;

    // restoration using explicit TV
    float* res = expl_tv(im, m); 
    for(i = 0; i < width*height*3; i++)
        res[i] *= 255;
    io_png_write_f32(argv[3], res, swidth, sheight, 3);

    free(im); 
    free(m); 
    free(res);
    return EXIT_SUCCESS;
}
float* expl_tv(const float* image, const float* mask) {
    int i, j, c;
    float *u, *uold, *v, *w, *absg;

    u       = (float*) xmalloc(sizeof(float)*width*height*3);
    uold    = (float*) xmalloc(sizeof(float)*width*height*3);
    v       = (float*) xmalloc(sizeof(float)*width*height*3);
    w       = (float*) xmalloc(sizeof(float)*width*height*3);
    absg    = (float*) xmalloc(sizeof(float)*width*height*3);

    memcpy(u, image, sizeof(float)*width*height*3);

    // start iterations 
    int it = 0;
    int chan;
do {
    memcpy(uold, u, sizeof(float)*width*height*3);

#pragma omp parallel for \
    shared(u,v,w) \
    private(c,i,j,chan)
    for(c = 0; c < 3; c++) {
        chan = c*width*height;
        for(i = 0; i < height; i++) {
            for(j = 0; j < width; j++) {
                v[chan+i*width+j] = cdx((u+chan), i, j);
                w[chan+i*width+j] = cdy((u+chan), i, j);
            }
        }
    }
    
#pragma omp parallel for \
    shared(v,w) \
    private(c,i,j,chan)
    for(c = 0; c < 3; c++) {
        chan = c*width*height;
        for(i = 0; i < height; i++) {
            for(j = 0; j < width; j++) {
                absg[chan+i*width+j] = sqrt(v[chan+i*width+j]*v[chan+i*width+j]
                + w[chan+i*width+j]*w[chan+i*width+j]
                + DPAD*DPAD);
            }
        }
    }

#pragma omp parallel for \
	shared(u,uold,v,w,absg,mask,image,l0,dt) \
	private(c,i,j,chan)     
    for(c = 0; c < 3; c++) {
        chan = c*width*height;
        for(i = 0; i < height; i++) {
            for(j = 0; j < width; j++) {
                u[chan+i*width+j] = uold[chan+i*width+j] 
                + dt*(dxx((uold+chan), i, j)*(w[chan+i*width+j]*w[chan+i*width+j] + DPAD*DPAD)
                + dyy((uold+chan), i, j)*(v[chan+i*width+j]*v[chan+i*width+j] + DPAD*DPAD)
                - 2*v[chan+i*width+j]*w[chan+i*width+j]*cdxy((uold+chan), i, j)) / (absg[chan+i*width+j]*absg[chan+i*width+j]*absg[chan+i*width+j])
                + dt*l0*mask[chan+i*width+j]*(image[chan+i*width+j] - uold[chan+i*width+j]);
            }
        }
    }   
    it++;
} while(it<MAXITER && frob_dist(u, uold) > THRESHOLD);
    
    printf("Iterations: %i\tMax iterations: %i\nFrobenius distance: %e\tThreshold: %e\n", it, MAXITER, frob_dist(u, uold), THRESHOLD);

    free(uold);
    free(v);
    free(w);
    free(absg);
    return u;    
}