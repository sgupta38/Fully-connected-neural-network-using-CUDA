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
#include <chrono>
#include <iostream>

template<typename IN_DIMS, size_t N_NEURONS>
CFullyConnectedLayer<IN_DIMS, N_NEURONS>::CFullyConnectedLayer(const std::string &n, bool relu, double do_rate, int ssed_seq)
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


   //
   // CUDA Memory allocation. Since, CudaMalloc Its so much of time. We know neuron count and image size.
   // Lets allocate only once instead of doing it many time.
   // This memory will be freed in destructor

   //================================== Backprop ==========================================================
   constexpr size_t bp_downstream_deriv_device_size = sizeof(double) * N_NEURONS * IN_D * IN_H * IN_W;
   constexpr size_t bp_weight_device_size = sizeof(double) * N_NEURONS * IN_D * IN_H * IN_W;
   constexpr size_t bp_weight_deriv_device_size = sizeof(double) * N_NEURONS * IN_D * IN_H * IN_W;
   constexpr size_t bp_input_device_size = sizeof(double) * IN_D * IN_H * IN_W;
   constexpr size_t bp_upstream_device_size = sizeof(double) * N_NEURONS * IN_D * IN_H * IN_W;
   constexpr size_t bp_current_kept_device_size = sizeof(double) * N_NEURONS *IN_D * IN_H * IN_W;
   constexpr size_t bp_bias_deriv_device_size = sizeof(double) * N_NEURONS * IN_D * IN_H * IN_W;
   constexpr size_t bp_op_device_size = sizeof(double) * N_NEURONS;


   // Memory Allocation
   cudaMalloc((void**)&bp_downstream_deriv_device, bp_downstream_deriv_device_size);
   cudaMalloc((void**)&bp_weight_device, bp_weight_device_size);
   cudaMalloc((void**)&bp_weight_deriv_device, bp_weight_deriv_device_size);
   cudaMalloc((void**)&bp_input_device, bp_input_device_size);
   cudaMalloc((void**)&bp_upstream_device, bp_upstream_device_size);
   cudaMalloc((void**)&bp_current_kept_device, bp_current_kept_device_size);
   cudaMalloc((void**)&bp_bias_deriv_device, bp_bias_deriv_device_size);
   cudaMalloc((void**)&bp_op_device, bp_op_device_size);

   //================================== update_weight ==========================================================
   constexpr size_t uw_wsize = sizeof(double) * N_NEURONS* IN_D * IN_H * IN_W;
   constexpr size_t uw_wdsize = sizeof(double) * N_NEURONS* IN_D * IN_H * IN_W;
   constexpr size_t uw_bsize = sizeof(double) * N_NEURONS;
   constexpr size_t uw_bdsize = sizeof(double) * N_NEURONS;

   // Memory Allocation
   cudaMalloc((void**)&uw_weight_device, uw_wsize);
   cudaMalloc((void**)&uw_weight_deriv_device, uw_wdsize);
   cudaMalloc((void**)&uw_bias_device, uw_bsize);
   cudaMalloc((void**)&uw_bias_deriv_device, uw_bdsize);

   //============================= forward===========================================================
   constexpr size_t f_isize = sizeof(double) * IN_D * IN_H * IN_W;
   constexpr size_t f_wsize = sizeof(double) * N_NEURONS * IN_D * IN_H * IN_W;
   constexpr size_t f_outsize = sizeof(double) * N_NEURONS;
   constexpr size_t f_biassize = sizeof(double) * N_NEURONS;
   constexpr size_t f_droppedsize = sizeof(double) * N_NEURONS;

   // Memory Allocation
   cudaMalloc((void**)&f_input_device, f_isize);
   cudaMalloc((void**)&f_weight_device, f_wsize);
   cudaMalloc((void**)&f_out_device, f_outsize);
   cudaMalloc((void**)&f_bias_device, f_biassize);
   cudaMalloc((void**)&f_dropped_device, f_droppedsize);
}

template<typename IN_DIMS, size_t N_NEURONS>
CFullyConnectedLayer<IN_DIMS, N_NEURONS>::~CFullyConnectedLayer()
{
   cudaFree(bp_downstream_deriv_device);
   cudaFree(bp_weight_device);
   cudaFree(bp_weight_deriv_device);
   cudaFree(bp_input_device);
   cudaFree(bp_upstream_device);
   cudaFree(bp_current_kept_device);
   cudaFree(bp_bias_deriv_device);
   cudaFree(bp_op_device);
   cudaFree(uw_weight_device);
   cudaFree(uw_weight_deriv_device);
   cudaFree(uw_bias_device);
   cudaFree(uw_bias_deriv_device);
   cudaFree(f_input_device);
   cudaFree(f_weight_device);
   cudaFree(f_out_device);
   cudaFree(f_bias_device);
   cudaFree(f_dropped_device);
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

    constexpr size_t bp_downstream_deriv_device_size = sizeof(double) * N_NEURONS * IN_D * IN_H * IN_W;
    constexpr size_t bp_weight_device_size = sizeof(double) * N_NEURONS * IN_D * IN_H * IN_W;
    constexpr size_t bp_weight_deriv_device_size = sizeof(double) * N_NEURONS * IN_D * IN_H * IN_W;
    constexpr size_t bp_input_device_size = sizeof(double) * IN_D * IN_H * IN_W;
    constexpr size_t bp_upstream_device_size = sizeof(double) * N_NEURONS * IN_D * IN_H * IN_W;
    constexpr size_t bp_current_kept_device_size = sizeof(double) * N_NEURONS *IN_D * IN_H * IN_W;
    constexpr size_t bp_bias_deriv_device_size = sizeof(double) * N_NEURONS * IN_D * IN_H * IN_W;
    constexpr size_t bp_op_device_size = sizeof(double) * N_NEURONS;

// #ifdef LOCAL_ALLOCATION
//      double *bp_downstream_deriv_device, *bp_upstream_device, *bp_current_kept_device, *bp_op_device;
//      double *bp_input_device, *bp_weight_device, *bp_weight_deriv_device, *bp_bias_deriv_device;
//     // Memory Allocation
//     cudaMalloc((void**)&bp_downstream_deriv_device, bp_downstream_deriv_device_size);
//     cudaMalloc((void**)&bp_weight_device, bp_weight_device_size);
//     cudaMalloc((void**)&bp_weight_deriv_device, bp_weight_deriv_device_size);
//     cudaMalloc((void**)&bp_input_device, bp_input_device_size);
//     cudaMalloc((void**)&bp_upstream_device, bp_upstream_device_size);
//     cudaMalloc((void**)&bp_current_kept_device, bp_current_kept_device_size);
//     cudaMalloc((void**)&bp_bias_deriv_device, bp_bias_deriv_device_size);
//     cudaMalloc((void**)&bp_op_device, bp_op_device_size);
// #endif

    // Copy to device memory
    rv = cudaMemcpy(bp_downstream_deriv_device, &this->downstream_deriv, bp_downstream_deriv_device_size, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(bp_weight_device, &this->m_weight, bp_weight_device_size, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(bp_weight_deriv_device, &this->m_weight_deriv, bp_weight_deriv_device_size, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(bp_input_device, &input, bp_input_device_size, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(bp_upstream_device, &upstream_deriv, bp_upstream_device_size, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(bp_current_kept_device, &m_current_kept, bp_current_kept_device_size, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(bp_bias_deriv_device, &m_bias_deriv, bp_bias_deriv_device_size, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(bp_op_device, &this->output, bp_op_device_size, cudaMemcpyHostToDevice);
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

    backprop_weightmatrixEx<<<gridSize, blockSize >>>(N_NEURONS, bp_input_device, bp_downstream_deriv_device, bp_current_kept_device, bp_upstream_device, \
                                                bp_weight_device, bp_weight_deriv_device, bp_bias_deriv_device, bp_op_device, m_relu, IN_D, IN_H, IN_W, mb_size); // need to pass width

    // backprop_weightmatrixEx<<<1, N_NEURONS >>>(N_NEURONS, input_device, downstream_deriv_device, current_kept_device, upstream_device, \
    //                                             weight_device, weight_deriv_device, bias_deriv_device, op_device, m_relu, IN_D, IN_H, IN_W, mb_size); // need to pass width

   // cudaDeviceSynchronize();

    // Copy to host device
    double* test_down = new double[bp_downstream_deriv_device_size];

    rv = cudaMemcpy(test_down, bp_downstream_deriv_device, bp_downstream_deriv_device_size, cudaMemcpyDeviceToHost);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(&m_weight_deriv, bp_weight_deriv_device, bp_weight_deriv_device_size, cudaMemcpyDeviceToHost);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(&m_bias_deriv, bp_bias_deriv_device, bp_bias_deriv_device_size, cudaMemcpyDeviceToHost);
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

// #ifdef LOCAL_ALLOCATION
//     // Memory De- allocation
//     cudaFree(bp_downstream_deriv_device);
//     cudaFree(bp_weight_device);
//     cudaFree(bp_weight_deriv_device);
//     cudaFree(bp_input_device);
//     cudaFree(bp_upstream_device);
//     cudaFree(bp_current_kept_device);
//     cudaFree(bp_bias_deriv_device);
//     cudaFree(bp_op_device);
// #endif

    this->previous_layer->backprop(this->downstream_deriv, mb_size);
}

// CUDA Function

template<typename IN_DIMS, size_t N_NEURONS>
void CFullyConnectedLayer<IN_DIMS, N_NEURONS>::update_weights(double rate)
{
    cudaError_t rv;
    constexpr size_t uw_wsize = sizeof(double) * N_NEURONS* IN_D * IN_H * IN_W;
    constexpr size_t uw_wdsize = sizeof(double) * N_NEURONS* IN_D * IN_H * IN_W;
    constexpr size_t uw_bsize = sizeof(double) * N_NEURONS;
    constexpr size_t uw_bdsize = sizeof(double) * N_NEURONS;

    // Memory Allocation
// #ifdef LOCAL_ALLOCATION
//     double* uw_weight_device, *uw_weight_deriv_device, *uw_bias_device, *uw_bias_deriv_device;
//     cudaMalloc((void**)&uw_weight_device, uw_wsize);
//     cudaMalloc((void**)&uw_weight_deriv_device, uw_wdsize);
//     cudaMalloc((void**)&uw_bias_device, uw_bsize);
//     cudaMalloc((void**)&uw_bias_deriv_device, uw_bdsize);
// #endif
    // Copy to device memory
    rv = cudaMemcpy(uw_weight_device, &m_weight, uw_wsize, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(uw_weight_deriv_device, &m_weight_deriv, uw_wdsize, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(uw_bias_device, &m_bias, uw_bsize, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(uw_bias_deriv_device, &m_bias_deriv, uw_bdsize, cudaMemcpyHostToDevice);
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

    update_weightmatrixEx<<<gridSize, blockSize>>>(N_NEURONS, uw_weight_device, uw_weight_deriv_device, uw_bias_device, uw_bias_deriv_device, IN_D, IN_H, IN_W, rate); // need to pass width
    //update_weightmatrixEx<<<1, N_NEURONS>>>(N_NEURONS, weight_device, weight_deriv_device, bias_device, bias_deriv_device, IN_D, IN_H, IN_W, rate); // need to pass width

   // cudaDeviceSynchronize();

    // Copy to host device
    rv = cudaMemcpy(&m_weight, uw_weight_device, uw_wsize, cudaMemcpyDeviceToHost);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(&m_weight_deriv, uw_weight_deriv_device, uw_wdsize, cudaMemcpyDeviceToHost);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(&m_bias, uw_bias_device, uw_bsize, cudaMemcpyDeviceToHost);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
    rv = cudaMemcpy(&m_bias_deriv, uw_bias_deriv_device, uw_bdsize, cudaMemcpyDeviceToHost);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }
// #ifdef LOCAL_ALLOCATION

//     // Memory De- allocation
//     cudaFree(uw_weight_device);
//     cudaFree(uw_weight_deriv_device);
//     cudaFree(uw_bias_device);
//     cudaFree(uw_bias_deriv_device);
// #endif

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

    constexpr size_t f_isize = sizeof(double) * IN_D * IN_H * IN_W;
    constexpr size_t f_wsize = sizeof(double) * N_NEURONS * IN_D * IN_H * IN_W;
    constexpr size_t f_outsize = sizeof(double) * N_NEURONS;
    constexpr size_t f_biassize = sizeof(double) * N_NEURONS;
    constexpr size_t f_droppedsize = sizeof(double) * N_NEURONS;

    auto start = std::chrono::system_clock::now();

    // Memory Allocation

// #ifdef LOCAL_ALLOCATION
//     double *f_input_device, *f_weight_device, *f_out_device, *f_bias_device ,*f_dropped_device;

//     cudaMalloc((void**)&f_input_device, f_isize);
//     cudaMalloc((void**)&f_weight_device, f_wsize);
//     cudaMalloc((void**)&f_out_device, f_outsize);
//     cudaMalloc((void**)&f_bias_device, f_biassize);
//     cudaMalloc((void**)&f_dropped_device, f_droppedsize);
// #endif

    // Copy to device memory;
    rv = cudaMemcpy(f_input_device, &input, f_isize, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorName(rv)<<std::endl;
        std::cout<<" Expected size: "<<sizeof(input)<<"\n";
        std::cout<<" Actual size: "<<f_isize<<"\n";
        exit(0);
    }

    rv = cudaMemcpy(f_weight_device, &weight, f_wsize, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }

    rv = cudaMemcpy(f_bias_device, &bias, f_biassize, cudaMemcpyHostToDevice);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }

    rv = cudaMemcpy(f_dropped_device, &dropped, f_droppedsize, cudaMemcpyHostToDevice);
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

    forward_matrixmulEx<<<gridSize, blockSize >>>(N_NEURONS, f_input_device, f_weight_device, f_out_device, f_bias_device, f_dropped_device, m_relu, IN_D, IN_H, IN_W); // need to pass width
    //forward_matrixmulEx<<<1, N_NEURONS >>>(N_NEURONS, input_device, weight_device, out_device, bias_device, dropped_device, m_relu, IN_D, IN_H, IN_W); // need to pass width

    //cudaDeviceSynchronize();

    // Copy to host device
    rv = cudaMemcpy(&output, f_out_device, f_outsize, cudaMemcpyDeviceToHost);
    if(rv != cudaSuccess)
    {
        std::cout<<" Error at "<<__LINE__<<" "<<cudaGetErrorString(rv)<<std::endl;
        exit(0);
    }

// #ifdef LOCAL_ALLOCATION
//     // Memory De- allocation
//     cudaFree(f_input_device);
//     cudaFree(f_weight_device);
//     cudaFree(f_out_device);
//     cudaFree(f_bias_device);
//     cudaFree(f_dropped_device);
// #endif
}