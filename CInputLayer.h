//
// Created by sonu
// This is first class where image data will be passed

#ifndef NEURAL_NETWORK_CINPUTLAYER_H
#define NEURAL_NETWORK_CINPUTLAYER_H

#include "common.h"
#include "CinputOutput.h"

template <typename OUT_DIMS>
class CInputLayer: public CBaseOutputLayer<OUT_DIMS>
{
public:
    int width, height;

    using OutputIF = CBaseOutputLayer<OUT_DIMS>;
    using typename OutputIF::Output;
    constexpr static size_t OUT_D = OutputIF::OutputDims::D;
    constexpr static size_t OUT_H = OutputIF::OutputDims::H;
    constexpr static size_t OUT_W = OutputIF::OutputDims::W;

    //member functions
    void train(float (&image)[OUT_H][OUT_W], int label, double mb_size)
    {
        this->output[0] = image; // This will act as i/p later on
        this->next_layer->train(label, mb_size);
    }

    virtual void backprop(const Output& , const double ) override {} // No backprop for first layer

    void update_weights(double rate)
    {
        this->next_layer->update_weights(rate);
    }

    int predict(float (&image)[OUT_H][OUT_W])
    {
        Output output;
        output[0] = image;
        return this->next_layer->predict(output);
    }
};

#endif //NEURAL_NETWORK_CINPUTLAYER_H
