//  @author: Sonu Gupta
//  @purpose: CUDA functions which runs on GPU
//            'Extended version' (Ex) is used'

#ifndef __CUDA_FUNCS_C__
#define __CUDA_FUNCS_C__

#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cuda_functions.h"


// Extended version
//
__global__ void forward_matrixmulEx(int limit, double* input, double* weight, double* out, double* bias, double* dropped, bool isReLu, int depth, int height, int width)
{
    //int id = threadIdx.x;
    int id  = threadIdx.x + blockIdx.x * blockDim.x;
    
    //todo: sanity check here

    if(id < limit)
    {

        double* curr_weight = weight + (id * depth * height * width);

        float res = 0.0f;

        /*
        for(int i = 0; i<height; i++)
        {
            for(int j = 0; j <width; j++)
            {
                res += a[width*i+j] + m[j*width + i];
            }
        }
        */

        for(int i = 0; i< depth * height * width; i++)
            res += input[i] * curr_weight[i];

        out[id] = res;
        out[id] += bias[id];
        if(isReLu)
        {
            out[id] = (out[id] > 0.0) ? out[id] : 0.0; // Max doesnt work here
        }

        out[id] *= dropped[id];
    }
}

__global__ void update_weightmatrixEx(int limit, double* weight, double* weight_deriv, double* bias, double* bias_deriv, int depth, int height, int width, float rate)
{
	// resulting matrix will be awidth * bheight
    //int index = threadIdx.x;
    int index  = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(index < limit)
    {
        //todo: sanity check here
        double* aa = weight + (index * depth * height * width);
        double* bb = weight_deriv + (index * depth * height * width);

        //compute and reset deriivative matrix to zero
        for(int i = 0; i< depth * height * width; i++)
        {
            aa[i] = aa[i] - rate * bb[i];
            bb[i] = 0;
        }

        bias[index] -= rate * bias_deriv[index];
        bias_deriv[index] = 0;
    }
}

__global__  void backprop_weightmatrixEx(int limit, double* input_device, double* downstream_deriv_device, double* current_kept_device, double* upstream_deriv_device, \
    double* weight_device, double* weight_deriv_device, double* bias_deriv_device, double* output, bool is_relu, int depth, int height, int width, double mb_size)
{
//    int index = threadIdx.x;
    int index  = threadIdx.x + blockIdx.x * blockDim.x;

    if(index < limit)
    {

        if(current_kept_device[index] > 0)
        {
            if (!is_relu || output[index] > 0)
            {
                double* t_weight_device = weight_device + (index * depth * height * width);
                double* t_weight_deriv_device = weight_deriv_device + (index * depth * height * width);
                double* t_downstream_deriv_device = downstream_deriv_device + (index * depth * height * width);

                for(int i = 0; i< depth * height * width; i++)
                {
                    t_downstream_deriv_device[i] =  current_kept_device[index] * upstream_deriv_device[index] * t_weight_device[i];
                    t_weight_deriv_device[i] += (current_kept_device[index] * upstream_deriv_device[index] * input_device[i])/mb_size;
                }

                bias_deriv_device[index] += (current_kept_device[index] * upstream_deriv_device[index])/mb_size;
            }
        }
    }
}

// OLD versins: No more used
// These can be called in for loops. Initially I started with it. However, I kept them because I want to try running them with OpenMP.
// todo: Integrate with OpenMP.

__global__ void forward_matrixmul(double* a, double* b, double *c, int width)
{
	// resulting matrix will be awidth * bheight
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;
    
    //todo: sanity check here
    int index = row * width + column;
    c[index] = a[index] * b[index];
}

__global__ void update_weightmatrix(double* a, double* b, int width, float rate)
{
	// resulting matrix will be awidth * bheight
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;
    
    //todo: sanity check here
    int index = row * width + column;

    //compute and reset deriivative matrix to zero
    a[index] = a[index] - rate * b[index];
    b[index] = 0;
}


__global__  void backprop_weightmatrix(double* input_device, double* downstream_deriv_device, double current_kept_device, double upstream_deriv_device, \
    double* weight_device, double* weight_deriv_device, int width, double mb_size)
{
    // resulting matrix will be awidth * bheight
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    
    //todo: sanity check here
    int index = row * width + column;

    //todo: compute common term once
    //
    downstream_deriv_device[index] =  downstream_deriv_device[index] + current_kept_device * upstream_deriv_device * weight_device[index];
    weight_deriv_device[index] = weight_deriv_device[index] + (current_kept_device * upstream_deriv_device * input_device[index])/mb_size;
}


#endif