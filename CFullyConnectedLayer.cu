//
// Created by sonu
//

#include "common.h"
#include "CinputOutput.h"
#include "CFullyConnectedLayer.h"
#include <cstring>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cuda_functions.h"

template<typename IN_DIMS, size_t NEURONS>
CFullyConnectedLayer<IN_DIMS, NEURONS>::CFullyConnectedLayer(const std::string &n, bool relu, double do_rate, int ssed_seq)
        : m_layer_name(n), m_relu(relu), m_keep_prob(1 - do_rate), m_all_kept(1), m_eng(7389 + ssed_seq)
{
    std::normal_distribution<double> init;
    // For each neuron
    for (auto &n : m_weight) {
        for (auto &p : n) {
            for (auto &r : p) {
                for (auto &c : r) {
                    c = init(m_eng)/sqrt(IN_DIMS::N);
                }
            }
        }
    }
    m_bias = 0;
    m_weight_deriv = 0;
    m_bias_deriv = 0;
}

template<typename IN_DIMS, size_t NEURONS>
void CFullyConnectedLayer<IN_DIMS, NEURONS>::train(int label, double mb_size) {

    std::uniform_real_distribution<double> dist(0, 1);
    std::generate(m_current_kept.begin(), m_current_kept.end(),
                  [&]() { return dist(m_eng) < m_keep_prob ? 1/m_keep_prob : 0; });

    this->forward(this->previous_layer->output, this->m_weight, this->m_bias, this->m_current_kept, this->output);
    this->next_layer->train(label, mb_size);
}

// CUDA Function
template<typename IN_DIMS, size_t N_NEURONS>
void CFullyConnectedLayer<IN_DIMS, N_NEURONS>::backprop(const Output& full_upstream_deriv, const double mb_size) {

    cudaError_t rv;
    auto &upstream_deriv(full_upstream_deriv[0][0]);
    this->downstream_deriv = 0;
    auto &input(this->previous_layer->output);
    double *downstream_deriv_device, *upstream_device, *current_kept_device, *op_device;
    double *input_device, *weight_device, *weight_deriv_device, *bias_deriv_device;

    const size_t downstream_deriv_device_size = sizeof(double) * N_NEURONS * IN_D * IN_H * IN_W;
    const size_t weight_device_size = sizeof(double) * N_NEURONS * IN_D * IN_H * IN_W;
    const size_t weight_deriv_device_size = sizeof(double) * N_NEURONS * IN_D * IN_H * IN_W;
    const size_t input_device_size = sizeof(double) * IN_D * IN_H * IN_W;
    const size_t upstream_device_size = sizeof(double) * N_NEURONS * IN_D * IN_H * IN_W;
    const size_t current_kept_device_size = sizeof(double) * N_NEURONS *IN_D * IN_H * IN_W;
    const size_t bias_deriv_device_size = sizeof(double) * N_NEURONS * IN_D * IN_H * IN_W;
    const size_t op_device_size = sizeof(double) * N_NEURONS;

    // Memory Allocation
    cudaMalloc((void**)&downstream_deriv_device, downstream_deriv_device_size);
    cudaMalloc((void**)&weight_device, weight_device_size);
    cudaMalloc((void**)&weight_deriv_device, weight_deriv_device_size);
    cudaMalloc((void**)&input_device, input_device_size);
    cudaMalloc((void**)&upstream_device, upstream_device_size);
    cudaMalloc((void**)&current_kept_device, current_kept_device_size);
    cudaMalloc((void**)&bias_deriv_device, bias_deriv_device_size);
    cudaMalloc((void**)&op_device, op_device_size);

    // Copy to device memory
    rv = cudaMemcpy(downstream_deriv_device, &this->downstream_deriv, downstream_deriv_device_size, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(weight_device, &this->m_weight, weight_device_size, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(weight_deriv_device, &this->m_weight_deriv, weight_deriv_device_size, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(input_device, &input, input_device_size, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(upstream_device, &upstream_deriv, upstream_device_size, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(current_kept_device, &m_current_kept, current_kept_device_size, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(bias_deriv_device, &m_bias_deriv, bias_deriv_device_size, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(op_device, &this->output, op_device_size, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }

    // Kernel method
    // Kernel method
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
    // maximum occupancy for a full device launch
    int gridSize;    // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, backprop_weightmatrixEx, 0, 0);

    // Round up according to array size
    gridSize = ((N_NEURONS* IN_D * IN_H * IN_W) + blockSize - 1) / blockSize;

    backprop_weightmatrixEx<<<gridSize, blockSize >>>(N_NEURONS, input_device, downstream_deriv_device, current_kept_device, upstream_device, \
                                                weight_device, weight_deriv_device, bias_deriv_device, op_device, m_relu, IN_D, IN_H, IN_W, mb_size); // need to pass width

    // backprop_weightmatrixEx<<<1, N_NEURONS >>>(N_NEURONS, input_device, downstream_deriv_device, current_kept_device, upstream_device, \
    //                                             weight_device, weight_deriv_device, bias_deriv_device, op_device, m_relu, IN_D, IN_H, IN_W, mb_size); // need to pass width

    cudaDeviceSynchronize();

    // Copy to host device
    double* test_down = new double[downstream_deriv_device_size];

    rv = cudaMemcpy(test_down, downstream_deriv_device, downstream_deriv_device_size, cudaMemcpyDeviceToHost);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(&m_weight_deriv, weight_deriv_device, weight_deriv_device_size, cudaMemcpyDeviceToHost);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(&m_bias_deriv, bias_deriv_device, bias_deriv_device_size, cudaMemcpyDeviceToHost);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }

    int width = IN_D*IN_H*IN_W;

    double t[width] = {0};

    for(int i = 0; i<width; i++ )
    {
        for(int j = 0; j< N_NEURONS; j++)
        {
            t[i] += test_down[width*j + i];
        }
    }
    delete[] test_down;

    memcpy(&this->downstream_deriv, t, sizeof(this->downstream_deriv));

    // Memory De- allocation
    cudaFree(downstream_deriv_device);
    cudaFree(weight_device);
    cudaFree(weight_deriv_device);
    cudaFree(input_device);
    cudaFree(upstream_device);
    cudaFree(current_kept_device);
    cudaFree(bias_deriv_device);
    cudaFree(op_device);

    this->previous_layer->backprop(this->downstream_deriv, mb_size);
}

// CUDA Function

template<typename IN_DIMS, size_t N_NEURONS>
void CFullyConnectedLayer<IN_DIMS, N_NEURONS>::update_weights(double rate)
{
    cudaError_t rv;
    double* weight_device, *weight_deriv_device, *bias_device, *bias_deriv_device;
    const size_t wsize = sizeof(double) * N_NEURONS* IN_D * IN_H * IN_W;
    const size_t wdsize = sizeof(double) * N_NEURONS* IN_D * IN_H * IN_W;
    const size_t bsize = sizeof(double) * N_NEURONS;
    const size_t bdsize = sizeof(double) * N_NEURONS;

    // Memory Allocation
    cudaMalloc((void**)&weight_device, wsize);
    cudaMalloc((void**)&weight_deriv_device, wdsize);
    cudaMalloc((void**)&bias_device, bsize);
    cudaMalloc((void**)&bias_deriv_device, bdsize);

    // Copy to device memory
    rv = cudaMemcpy(weight_device, &m_weight, wsize, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(weight_deriv_device, &m_weight_deriv, wdsize, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(bias_device, &m_bias, bsize, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(bias_deriv_device, &m_bias_deriv, bdsize, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }

    // Kernel method
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
    // maximum occupancy for a full device launch
    int gridSize;    // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, update_weightmatrixEx, 0, 0);

    // Round up according to array size
    gridSize = ((N_NEURONS* IN_D * IN_H * IN_W) + blockSize - 1) / blockSize;

    update_weightmatrixEx<<<gridSize, blockSize>>>(N_NEURONS, weight_device, weight_deriv_device, bias_device, bias_deriv_device, IN_D, IN_H, IN_W, rate); // need to pass width
    //update_weightmatrixEx<<<1, N_NEURONS>>>(N_NEURONS, weight_device, weight_deriv_device, bias_device, bias_deriv_device, IN_D, IN_H, IN_W, rate); // need to pass width

    cudaDeviceSynchronize();

    // Copy to host device
    rv = cudaMemcpy(&m_weight, weight_device, wsize, cudaMemcpyDeviceToHost);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(&m_weight_deriv, weight_deriv_device, wdsize, cudaMemcpyDeviceToHost);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(&m_bias, bias_device, bsize, cudaMemcpyDeviceToHost);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(&m_bias_deriv, bias_deriv_device, bdsize, cudaMemcpyDeviceToHost);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }

    // Memory De- allocation
    cudaFree(weight_device);
    cudaFree(weight_deriv_device);
    cudaFree(bias_device);
    cudaFree(bias_deriv_device);

    this->next_layer->update_weights(rate);
}

template<typename IN_DIMS, size_t NEURONS>
double CFullyConnectedLayer<IN_DIMS, NEURONS>::loss (Input& in, int label) {
    Output temp_output;
    this->forward(in, this->m_weight, this->m_bias, this->m_all_kept, temp_output);
    return this->next_layer->loss(temp_output, label);
}

template<typename IN_DIMS, size_t NEURONS>
int CFullyConnectedLayer<IN_DIMS, NEURONS>::predict(Input& in) {
    Output out;
    this->forward(in, this->m_weight, this->m_bias, this->m_all_kept, out);
    return this->next_layer->predict(out);
}

// CUDA Function

template<typename IN_DIMS, size_t N_NEURONS>
void CFullyConnectedLayer<IN_DIMS, N_NEURONS>::forward(Input& input, Array<Input, N_NEURONS> &weight, Array<double, N_NEURONS> bias, Array<double, N_NEURONS> &dropped, Output& output)
{
    cudaError_t rv;
    double* input_device, *weight_device, *out_device, *bias_device, *dropped_device;

    const size_t isize = sizeof(double) * IN_D * IN_H * IN_W;
    const size_t wsize = sizeof(double) * N_NEURONS * IN_D * IN_H * IN_W;
    const size_t outsize = sizeof(double) * N_NEURONS;
    const size_t biassize = sizeof(double) * N_NEURONS;
    const size_t droppedsize = sizeof(double) * N_NEURONS;

    // Memory Allocation
    cudaMalloc((void**)&input_device, isize);
    cudaMalloc((void**)&weight_device, wsize);
    cudaMalloc((void**)&out_device, outsize);
    cudaMalloc((void**)&bias_device, biassize);
    cudaMalloc((void**)&dropped_device, droppedsize);

    cudaMemset(input_device, 0, isize);
    // Copy to device memory;
    rv = cudaMemcpy(input_device, &input, isize, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorName(rv)<<std::endl;
        std::cout<<" Expected size: "<<sizeof(input)<<"\n";
        std::cout<<" Actual size: "<<isize<<"\n";
        exit(0);
    }

    rv = cudaMemcpy(weight_device, &weight, wsize, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }

    rv = cudaMemcpy(bias_device, &bias, biassize, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }

    rv = cudaMemcpy(dropped_device, &dropped, droppedsize, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }

    // Kernel method
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
    // maximum occupancy for a full device launch
    int gridSize;    // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, forward_matrixmulEx, 0, 0);

    // Round up according to array size
    gridSize = ((N_NEURONS* IN_D * IN_H * IN_W) + blockSize - 1) / blockSize;

    forward_matrixmulEx<<<gridSize, blockSize >>>(N_NEURONS, input_device, weight_device, out_device, bias_device, dropped_device, m_relu, IN_D, IN_H, IN_W); // need to pass width
    //forward_matrixmulEx<<<1, N_NEURONS >>>(N_NEURONS, input_device, weight_device, out_device, bias_device, dropped_device, m_relu, IN_D, IN_H, IN_W); // need to pass width

    cudaDeviceSynchronize();

    // Copy to host device
    rv = cudaMemcpy(&output, out_device, outsize, cudaMemcpyDeviceToHost);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }

    // Memory De- allocation
    cudaFree(input_device);
    cudaFree(weight_device);
    cudaFree(out_device);
    cudaFree(bias_device);
    cudaFree(dropped_device);
}