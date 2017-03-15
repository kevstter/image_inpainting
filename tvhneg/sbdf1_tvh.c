/**
 * @file sbdf1_tvh.c
 *
 * @brief Uses TV-H^{-1} inpainting with a linearly stabilized semi-implicit Euler time stepping scheme to inpaint an image
 *
 * The inpainting image is found by evolving the PDE
 *  u_t = -Delta div(grad(u)/abs(grad(u))) + \lambda_{\Omega\setminus D}(u_0 - u)
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

float* sbdf1_tvh(const float* image, const float* mask);

#define DPAD 0.10 
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

    // Some parameters
    dt = 0.16; 
    l0 = 100;
    p0 = 0.50;
    p1 = 0.535/DPAD;

    // map rgb values to [0, 1]
    int i; 
    for(i = 0; i < width*height*3; i++) 
        im[i] /= 255;

    // restoration using explicit TV
    float* res = sbdf1_tvh(im, m); 
    for(i = 0; i < width*height*3; i++)
        res[i] *= 255;
    io_png_write_f32(argv[3], res, swidth, sheight, 3);

    free(im); 
    free(m); 
    free(res);
    return EXIT_SUCCESS;
}
float* sbdf1_tvh(const float* image, const float* mask) {
    int i, j, c;
    float *u, *uold, *v, *w, *absg, *f, *tmpcpy, stall, change = 1.0;

    u       = (float*) xmalloc(sizeof(float)*width*height*3);
    uold    = (float*) xmalloc(sizeof(float)*width*height*3);
    v       = (float*) xmalloc(sizeof(float)*width*height*3);
    w       = (float*) xmalloc(sizeof(float)*width*height*3);
    absg    = (float*) xmalloc(sizeof(float)*width*height*3);
    f       = (float*) xmalloc(sizeof(float)*width*height*3);

    memcpy(u, image, sizeof(float)*width*height*3);
    memcpy(uold, u, sizeof(float)*width*height*3);    

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
    stall = change;
    tmpcpy = uold;
    uold = u; 
    u = tmpcpy;    

// first derivatives
#pragma omp parallel for \
    shared(uold,v,w) \
    private(c,i,j,chan)
    for(c = 0; c < 3; c++) {
        chan = c*width*height;
        for(i = 0; i < height; i++) {
            for(j = 0; j < width; j++) {
                v[chan+i*width+j] = cdx(uold+chan, i, j);
                w[chan+i*width+j] = cdy(uold+chan, i, j);
            }
        }
    }
// curvature term
#pragma omp parallel for \
    shared(absg,v,w,uold) \
    private(c,i,j,chan)
    for(c = 0; c < 3; c++) {
        chan = c*width*height;
        for(i = 0; i < height; i++) {
            for(j = 0; j < width; j++) {
                absg[chan+i*width+j] = sqrt(v[chan+i*width+j]*v[chan+i*width+j]
                + w[chan+i*width+j]*w[chan+i*width+j]
                + DPAD*DPAD);
                
                f[chan+i*width+j] = (dxx(uold+chan, i, j)*(w[chan+i*width+j]*w[chan+i*width+j]+ DPAD*DPAD) 
                + dyy(uold+chan, i, j)*(v[chan+i*width+j]*v[chan+i*width+j] + DPAD*DPAD) 
                - 2*v[chan+i*width+j]*w[chan+i*width+j]*cdxy(uold+chan, i, j)) / (absg[chan+i*width+j]*absg[chan+i*width+j]*absg[chan+i*width+j]);
            }
        }
    }

#pragma omp parallel for \
	shared(uold,v,w,absg,mask,image,l0,dt) \
	private(c,i,j,chan)     
    for(c = 0; c < 3; c++) {
        chan = c*width*height;
        for(i = 0; i < height; i++) {
            for(j = 0; j < width; j++) {
                in[c][i*width+j] = uold[chan+i*width+j] + dt*(
                -dxx(f+chan, i, j) - dyy(f+chan, i, j) 
                + p1*(dxxxx(uold+chan, i, j) + dyyyy(uold+chan, i, j) + 2*dxxdyy(uold+chan, i, j))
                + p0*l0*uold[chan+i*width+j]
                + l0*mask[chan+i*width+j]*(image[chan+i*width+j] - uold[chan+i*width+j]));
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
    change = frob_dist(u, uold);    
    stall = fabs(stall-change)/THRESHOLD;
    // printf("%.3e\t%.3e\n", change, stall);
} while(it<MAXITER && change > THRESHOLD && stall > 2e-2);
    
    if(stall < 2e-2) {
        printf("Warning: Convergence stalled.\n");
    }
    printf("Iterations: %i\tMax iterations: %i\nFrobenius distance: %.2e\tThreshold: %.2e\n", it, MAXITER, frob_dist(u, uold), THRESHOLD);

    free(uold);
    free(v);
    free(w);
    free(absg);
    free(f);
    fftw_free(D);
    fftw_free(out[0]);
    fftw_free(out[1]);
    fftw_free(out[2]);
    free(in[0]);
    free(in[1]);
    free(in[2]);
    return u;    
}