#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <helper_cuda.h>
#include <iostream>
#define cutilCheckMsg(a) getLastCudaError(a)
#define cutGetMaxGflopsDeviceId() gpuGetMaxGflopsDeviceId()

#define MIN(a,b) (a) < (b) ? (a) : (b)
#include <cuda_runtime.h>
#include "hist-equ.h"

#include <chrono>
#include <ctime>  


void run_cpu_color_test(PPM_IMG img_in);
void run_gpu_color_test(PPM_IMG img_in);
void run_cpu_gray_test(PGM_IMG img_in);
void run_gpu_gray_test(PGM_IMG img_in);


int main(){
    PGM_IMG img_ibuf_g;
    PPM_IMG img_ibuf_c;
    
    printf("Running contrast enhancement for gray-scale images.\n");
    img_ibuf_g = read_pgm("in.pgm");
    //run_cpu_gray_test(img_ibuf_g);
    run_gpu_gray_test(img_ibuf_g);
    free_pgm(img_ibuf_g);
    
    printf("Running contrast enhancement for color images.\n");
    img_ibuf_c = read_ppm("in.ppm");
    //run_cpu_color_test(img_ibuf_c);
    run_gpu_color_test(img_ibuf_c);
    free_ppm(img_ibuf_c);
    
    return 0;
}

void run_gpu_color_test(PPM_IMG img_in)
{
    printf("Starting GPU color test...\n");
    PPM_IMG img_obuf_hsl, img_obuf_yuv;
    
    auto start = std::chrono::system_clock::now();
    img_obuf_hsl = contrast_enhancement_c_hsl_gpu(img_in);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout<<"HSL Test took "<<elapsed_seconds.count()<<" seconds, including memory copy"<<std::endl;
    write_ppm(img_obuf_hsl, "out_hsl_gpu.ppm");

    start = std::chrono::system_clock::now();
    img_obuf_yuv = contrast_enhancement_c_yuv_gpu(img_in);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    std::cout<<"YUV Test took "<<elapsed_seconds.count()<<" seconds, including memory copy"<<std::endl;
    write_ppm(img_obuf_yuv, "out_yuv_gpu.ppm");
    
    free_ppm(img_obuf_hsl);
    free_ppm(img_obuf_yuv);
}

void run_gpu_gray_test(PGM_IMG img_in)
{
    printf("Starting GPU gray test...\n");
    PGM_IMG img_obuf;
    auto start = std::chrono::system_clock::now();
    img_obuf = contrast_enhancement_g_gpu(img_in);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout<<"Test took "<<elapsed_seconds.count()<<" seconds, including memory copy"<<std::endl;
    write_pgm(img_obuf, "out_gpu.pgm");
    free_pgm(img_obuf);
}

void run_cpu_color_test(PPM_IMG img_in)
{
    PPM_IMG img_obuf_hsl, img_obuf_yuv;
    
    printf("Starting CPU color test...\n");
    auto start = std::chrono::system_clock::now();
    img_obuf_hsl = contrast_enhancement_c_hsl(img_in);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout<<"HSL Test took "<<elapsed_seconds.count()<<" seconds"<<std::endl;
    write_ppm(img_obuf_hsl, "out_hsl_cpu.ppm");

    start = std::chrono::system_clock::now();
    img_obuf_yuv = contrast_enhancement_c_yuv(img_in);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    std::cout<<"YUV Test took "<<elapsed_seconds.count()<<" seconds"<<std::endl;
    write_ppm(img_obuf_yuv, "out_yuv_cpu.ppm");
    
    free_ppm(img_obuf_hsl);
    free_ppm(img_obuf_yuv);
}

void run_cpu_gray_test(PGM_IMG img_in)
{
    PGM_IMG img_obuf;
    
    
    printf("Starting CPU gray test... \n");
    
    auto start = std::chrono::system_clock::now();
    img_obuf = contrast_enhancement_g(img_in);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout<<"Test took "<<elapsed_seconds.count()<<" seconds"<<std::endl;
    
    write_pgm(img_obuf, "out_cpu.pgm");
    free_pgm(img_obuf);
}

PPM_IMG read_ppm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    char *ibuf;
    PPM_IMG result;
    int v_max, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    /*Skip the magic number*/
    fscanf(in_file, "%s", sbuf);


    //result = malloc(sizeof(PPM_IMG));
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    ibuf         = (char *)malloc(3 * result.w * result.h * sizeof(char));

    
    fread(ibuf,sizeof(unsigned char), 3 * result.w*result.h, in_file);

    for(i = 0; i < result.w*result.h; i ++){
        result.img_r[i] = ibuf[3*i + 0];
        result.img_g[i] = ibuf[3*i + 1];
        result.img_b[i] = ibuf[3*i + 2];
    }
    
    fclose(in_file);
    free(ibuf);
    
    return result;
}

void write_ppm(PPM_IMG img, const char * path){
    FILE * out_file;
    int i;
    
    char * obuf = (char *)malloc(3 * img.w * img.h * sizeof(char));

    for(i = 0; i < img.w*img.h; i ++){
        obuf[3*i + 0] = img.img_r[i];
        obuf[3*i + 1] = img.img_g[i];
        obuf[3*i + 2] = img.img_b[i];
    }
    out_file = fopen(path, "wb");
    fprintf(out_file, "P6\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(obuf,sizeof(unsigned char), 3*img.w*img.h, out_file);
    fclose(out_file);
    free(obuf);
}

void free_ppm(PPM_IMG img)
{
    free(img.img_r);
    free(img.img_g);
    free(img.img_b);
}

PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

        
    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    free(img.img);
}

