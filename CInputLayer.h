//
// Created by sonu on 5/6/19.
// This is first class where image data will be passed

#ifndef NEURAL_NETWORK_CINPUTLAYER_H
#define NEURAL_NETWORK_CINPUTLAYER_H

#include "common.h"
#include "CinputOutput.h"

template <typename OUT_DIMS>
class CInputLayer: public COutputLayer<OUT_DIMS>
{
public:
    using typename COutputLayer<OUT_DIMS>::Output;
    constexpr static size_t OUT_D = COutputLayer<OUT_DIMS>::output::D;
    constexpr static size_t OUT_H = COutputLayer<OUT_DIMS>::output::H;
    constexpr static size_t OUT_W = COutputLayer<OUT_DIMS>::output::W;

    //member functions
    void train(float (&image)[OUT_H][OUT_W], int label, myfloat mb_size);
    virtual void backprop(Output& , myfloat ) override {} // No backprop for first layer
    void update_weights(myfloat rate);
    int predict(float (&image)[OUT_H][OUT_W]);
};


#endif //NEURAL_NETWORK_CINPUTLAYER_H
