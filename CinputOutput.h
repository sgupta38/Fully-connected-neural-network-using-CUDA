//
// Created by sonu
//

#ifndef NEURAL_NETWORK_CINPUTOUTPUT_H
#define NEURAL_NETWORK_CINPUTOUTPUT_H

#include "common.h"
#include "Carray.h"

template<typename T> class CBaseInputLayer;
template<typename T> class CBaseOutputLayer;

template<size_t D, size_t H, size_t W>
class CBaseInputLayer<Dims<D, H, W>>
{
public:
    using InputDims = Dims<D, H, W>;
    using Input = typename ArrayDims<Precision, InputDims>::type;
    CBaseOutputLayer<Dims<D,H,W>> *previous_layer;
    Input downstream_deriv;

    CBaseInputLayer(): previous_layer(nullptr)
    {
        // todo: downstream signalling_NAN?
    }

    virtual ~CBaseInputLayer()
    {

    }

    //Functions which needs to be implemented by every required class
    virtual void train(int label, double minibatch_size) = 0;
    virtual void update_weights(double rate) = 0;
    virtual double loss(Input& in, int label) = 0;
    virtual int predict(Input& in) = 0;
};

template<size_t D, size_t H, size_t W>
class CBaseOutputLayer<Dims<D, H, W>>
{
public:
    using OutputDims = Dims<D, H, W>;
    CBaseInputLayer<Dims<D,H,W>> *next_layer;
    using Output = typename ArrayDims<Precision , Dims<D,H,W> >::type;
    Output output;

    CBaseOutputLayer(): next_layer(nullptr)
    {
        // todo: downstream signalling_NAN?
    }

    virtual ~CBaseOutputLayer()
    {

    }

    //Functions which needs to be implemented by every required class
    virtual void backprop(const Output &deriv, const double mb_size) = 0;
};
#endif //NEURAL_NETWORK_CINPUTOUTPUT_H
