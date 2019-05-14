//  @author: Sonu Gupta
//  @purpose: CUDA functions which runs on GPU
//            'Extended version' (Ex) is used'

#ifndef __CUDA_FUNCS_H
#define __CUDA_FUNCS_H

//Initial Version
__global__ void forward_matrixmul(double* a, double* b, double *c, int width);
__global__ void update_weightmatrix(double* a, double* b, int width, float rate);
__global__  void backprop_weightmatrix(double* input_device, double* downstream_deriv_device, double current_kept_device, double upstream_deriv_device, \
    double* weight_device, double* weight_deriv_device, int width, double mb_size);

// Extended Version
__global__ void forward_matrixmulEx(int limit, double* input, double* weight, double* out, double* bias, double* dropped, bool isReLu, int depth, int height, int width);
__global__ void update_weightmatrixEx(int limit, double* weight, double* weight_deriv, double* bias, double* bias_deriv, int depth, int height, int width, float rate);
__global__  void backprop_weightmatrixEx(int limit, double* input_device, double* downstream_deriv_device, double* current_kept_device, double* upstream_deriv_device, \
    double* weight_device, double* weight_deriv_device, double* bias_deriv_device, double* output, bool is_relu, int depth, int height, int width, double mb_size);

#endif