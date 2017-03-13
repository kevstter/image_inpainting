/**
 * @file sbdf2_tv.c
 *
 * @brief Uses TV inpainting with a linearly stabilized SBDF2 time stepping scheme to inpaint an image
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

float* sbdf2_tv(const float* image, const float* mask);

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
    dt = 0.10; 
    l0 = 100;
    p0 = 0.75;
    p1 = 0.8/DPAD;

    // map rgb values to [0, 1]
    int i; 
    for(i = 0; i < width*height*3; i++) 
        im[i] /= 255;

    // restoration using explicit TV
    float* res = sbdf2_tv(im, m); 
    for(i = 0; i < width*height*3; i++)
        res[i] *= 255;
    io_png_write_f32(argv[3], res, swidth, sheight, 3);

    free(im); 
    free(m); 
    free(res);
    return EXIT_SUCCESS;
}
float* sbdf2_tv(const float* image, const float* mask) {
    int i, j, c;
    float *u, *u1, *u2, *v, *w, *absg, *f, *f1, *tmpcpy;

    u       = (float*) xmalloc(sizeof(float)*width*height*3);
    u1      = (float*) xmalloc(sizeof(float)*width*height*3);
    u2      = (float*) xmalloc(sizeof(float)*width*height*3);
    v       = (float*) xmalloc(sizeof(float)*width*height*3);
    w       = (float*) xmalloc(sizeof(float)*width*height*3);
    absg    = (float*) xmalloc(sizeof(float)*width*height*3);
    f       = (float*) xmalloc(sizeof(float)*width*height*3);
    f1      = (float*) xmalloc(sizeof(float)*width*height*3);

    memcpy(u, image, sizeof(float)*width*height*3);
    memcpy(u1, image, sizeof(float)*width*height*3);

    // Fourier tools
    for(i = 0; i < 3; i++) {
        in[i] = (double*) xmalloc(sizeof(double)*width*height);
        out[i] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*(width/2+1)*height);
        p[i] = fftw_plan_dft_r2c_2d(height, width, in[i], out[i], FFTW_MEASURE);
        pinv[i] = fftw_plan_dft_c2r_2d(height, width, out[i], in[i], FFTW_MEASURE);
    }
    fftw_complex *D; 
    D = computeDenom(in, out, p, 1.1*p0, 1.1*p1);


    // start iterations 
    int it = 0;
    int chan;
do {
    // memcpy(u1, u, sizeof(float)*width*height*3);
// compute first derivatives in x and y
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
// compute absolute value of gradient
#pragma omp parallel for \
    shared(absg,v,w) \
    private(c,i,j,chan)
    for(c = 0; c < 3; c++) {
        chan = c*width*height;
        for(i = 0; i < height; i++) {
            for(j = 0; j < width; j++) {
                absg[chan+i*width+j] = sqrt(
                    v[chan+i*width+j]*v[chan+i*width+j]
                + w[chan+i*width+j]*w[chan+i*width+j]
                + DPAD*DPAD);
            }
        }
    }
// first iteration is SBDF1, otherwise CNAB
if(it > 0) {
    tmpcpy = u2;
    u2 = u1; 
    u1 = u; 
    u = tmpcpy;

#pragma omp parallel for \
	shared(u1,v,w,absg,mask,image,l0,dt) \
	private(c,i,j,chan)     
    for(c = 0; c < 3; c++) {
        chan = c*width*height;
        for(i = 0; i < height; i++) {
            for(j = 0; j < width; j++) {
                f[chan+i*width+j] = (dxx((u1+chan), i, j)*(w[chan+i*width+j]*w[chan+i*width+j]+ DPAD*DPAD) 
                + dyy((u1+chan), i, j)*(v[chan+i*width+j]*v[chan+i*width+j] + DPAD*DPAD) 
                - 2*v[chan+i*width+j]*w[chan+i*width+j]*cdxy((u1+chan), i, j)) / (absg[chan+i*width+j]*absg[chan+i*width+j]*absg[chan+i*width+j]) 
                - p1*(dxx((u1+chan), i, j) + dyy((u1+chan), i, j))
                + p0*l0*u1[chan+i*width+j] 
                + l0*mask[chan+i*width+j]*(image[chan+i*width+j] - u1[chan+i*width+j]);
                
                in[c][i*width+j] = 1.3333333*u1[chan+i*width+j] 
                - 0.33333333*u2[chan+i*width+j]
                + (0.6666667)*dt*(2*f[chan+i*width+j] - f1[chan+i*width+j]);
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
    tmpcpy = f1;
    f1 = f; 
    f = tmpcpy;   
}else{
#pragma omp parallel for \
	shared(u,v,w,absg,mask,image,l0,dt) \
	private(c,i,j,chan)     
    for(c = 0; c < 3; c++) {
        chan = c*width*height;
        for(i = 0; i < height; i++) {
            for(j = 0; j < width; j++) {
                f1[chan+i*width+j] = (dxx((u+chan), i, j)*(w[chan+i*width+j]*w[chan+i*width+j]+ DPAD*DPAD) 
                + dyy((u+chan), i, j)*(v[chan+i*width+j]*v[chan+i*width+j] + DPAD*DPAD) 
                - 2*v[chan+i*width+j]*w[chan+i*width+j]*cdxy((u+chan), i, j)) / (absg[chan+i*width+j]*absg[chan+i*width+j]*absg[chan+i*width+j])
                - p1*(dxx((u+chan), i, j) + dyy((u+chan), i, j))
                + 1.1*p0*l0*u[chan+i*width+j]
                + 1.1*l0*mask[chan+i*width+j]*(image[chan+i*width+j] - u[chan+i*width+j]);

                in[c][i*width+j] = u[chan+i*width+j] + dt*f1[chan+i*width+j];
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
    fftw_free(D);
    D = computeDenom(in, out, p, 0.666666667*p0, 0.666666667*p1);    
}
    it++;
    // printf("%e\n", frob_dist(u, u1));
} while(it<MAXITER && frob_dist(u, u1) > THRESHOLD);
    
    printf("Iterations: %i\tMax iterations: %i\nFrobenius distance: %e\tThreshold: %e\n", it, MAXITER, frob_dist(u, u1), THRESHOLD);

    free(u1);
    free(u2);
    free(v);
    free(w);
    free(absg);
    free(f); 
    free(f1); 
    fftw_free(D);
    fftw_free(out[0]);
    fftw_free(out[1]);
    fftw_free(out[2]);
    free(in[0]);
    free(in[1]);
    free(in[2]);
    return u;    
}