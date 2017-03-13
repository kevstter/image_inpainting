/**
 * @file img_fun.c
 * @brief Common functions use in image inpainting
 *
 * Similar to those found in fstv.c; numerical derivatives and computing position on a torus
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <fftw3.h>
#include "img_fun.h"

void set_mask(float* mask, const float* im) { 
    int i, j, c, val;
    for(i = 0; i < height; i++) {
        for(j = 0; j < width; j++) {
            if(im[Ic(i,j,0)]>253 && im[Ic(i,j,1)]>253 && im[Ic(i,j,2)]>253)
                val = 0;
            else
                val = 1;
            
            for(c = 0; c < 3; c++) { 
                mask[Ic(i, j, c)] = val;
            }
        }
    }
    return;
}
void set_value(float* A, float value, int size) {
    int i;
    for(i = 0; i < size; i++) 
        A[i] = value;

    return;
}
fftw_complex* computeDenom(double **in, fftw_complex **out, fftw_plan *p, float p0, float p1) {
    int i;
    fftw_complex *ret;
    ret = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*(width/2+1)*height);

    for(i = 0; i < height*(width/2+1); i++) 
        ret[i] = 1 + p0*dt*l0;

    /* d4x Matrix */
	memset(in[0], 0, sizeof(double)*width*height);
	in[0][In(0,0)] = 6;
	in[0][In(0,1)] = -4;
	in[0][In(0,2)] = 1;
	in[0][In(0, width-2)] = 1;
	in[0][In(0, width-1)] = -4;
	fftw_execute(p[0]);
	for(i = 0; i < height*(width/2+1); i++)
		ret[i] += p1*dt*out[0][i];

	/* d4y Matrix */
	memset(in[1], 0, sizeof(double)*width*height);
	in[1][In(0,0)] = 6;
	in[1][In(1,0)] = -4;
	in[1][In(2,0)] = 1;
	in[1][In(height-2,0)] = 1;
	in[1][In(height-1,0)] = -4;
	fftw_execute(p[1]);
	for(i = 0; i < height*(width/2+1); i++)
		ret[i] += p1*dt*out[1][i];

	/* d2x2y Matrix */
	memset(in[1], 0, sizeof(double)*width*height);
	in[1][In(0,0)] = 4;
	in[1][In(1,0)] = -2;
	in[1][In(0,1)] = -2;
	in[1][In(1,1)] = 1;
	in[1][In(height-1,0)] = -2;
	in[1][In(height-1,1)] = 1;
	in[1][In(0,width-1)] = -2;
	in[1][In(1,width-1)] = 1;
	in[1][In(height-1,width-1)] = 1;
	fftw_execute(p[1]);
	for(i = 0; i < height*(width/2+1); i++)
		ret[i] += 2*p1*dt*out[1][i];
    
    return ret;
}
