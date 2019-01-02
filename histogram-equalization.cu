#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <chrono>
#include <ctime>  
#include <iostream>

void histogram(int *hist_out, unsigned char *img_in, int img_size, int nbr_bin)
{
    int i;
    for (i = 0; i < nbr_bin; i++)
    {
        hist_out[i] = 0;
    }

    for (i = 0; i < img_size; i++)
    {
        hist_out[img_in[i]]++;
    }
}

void histogram_equalization(unsigned char *img_out, unsigned char *img_in,
                            int *hist_in, int img_size, int nbr_bin)
{
    int *lut = (int *)malloc(sizeof(int) * nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while (min == 0)
    {
        min = hist_in[i++];
    }
    d = img_size - min;
    for (i = 0; i < nbr_bin; i++)
    {
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min) * 255 / d + 0.5);
        if (lut[i] < 0)
        {
            lut[i] = 0;
        }
    }

    /* Get the result image */
    for (i = 0; i < img_size; i++)
    {
        if (lut[img_in[i]] > 255)
        {
            img_out[i] = 255;
        }
        else
        {
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
    }
}

__global__ void histogram_gpu(int *hist_out, unsigned char *img_in, int img_size, int nbr_bin)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();
    if (idx<img_size){
        if (img_in[idx]<nbr_bin){
            atomicAdd(hist_out+img_in[idx], 1);
        }
    }
    __syncthreads();
}

std::chrono::duration<double> histogram_equalization_gpu(unsigned char *img_out, unsigned char *img_in,
                            int *hist_in, int img_size, int nbr_bin)
{
    auto start = std::chrono::system_clock::now();
    int *lut = (int *)malloc(sizeof(int) * nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while (min == 0)
    {
        min = hist_in[i++];
    }
    d = img_size - min;
    for (i = 0; i < nbr_bin; i++)
    {
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min) * 255 / d + 0.5);
        if (lut[i] < 0)
        {
            lut[i] = 0;
        }
    }
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end-start;

    
    int* d_lut;
    unsigned char * d_img_out;
    unsigned char * d_img_in;

    cudaMalloc((void **) &d_img_out, img_size*sizeof(unsigned char));
    cudaMalloc((void **) &d_lut, sizeof(int) * nbr_bin);
    cudaMalloc((void **) &d_img_in, img_size*sizeof(unsigned char));

    cudaMemcpy(d_img_in, img_in, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lut, lut, sizeof(int) * nbr_bin, cudaMemcpyHostToDevice);
    
    start = std::chrono::system_clock::now();
    
    int no_blocks = img_size*sizeof(unsigned char)/1024+1;
    get_result_image<<<no_blocks,1024>>>(d_lut, d_img_out, d_img_in, img_size);
    
    end = std::chrono::system_clock::now();
    elapsed_seconds += end-start;
    
    cudaMemcpy(img_out, d_img_out, img_size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    return elapsed_seconds;
}


__global__ void get_result_image(int* lut, unsigned char *img_out, 
                                unsigned char *img_in, int img_size)
{
    /* Get the result image */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (lut[img_in[i]] > 255)
    {
        img_out[i] = 255;
    }
    else
    {
        img_out[i] = (unsigned char)lut[img_in[i]];
    }

}