//
// Created by sonu on 5/5/19.
//

#ifndef NEURAL_NETWORK_CINPUTOUTPUT_H
#define NEURAL_NETWORK_CINPUTOUTPUT_H

#include "common.h"

// Forward declaration
template<typename T> class CInputLayer;
template<typename T> class COutputLayer;

template<size_t D, size_t H, size_t W>
class CInputLayer<Dims<D, H, W>>
{
public:
    COutputLayer<Dims<D,H,W>> *previous_layer;
    using Input = typename ArrayDims<Precision , Dims<D,H,W> >::type;
    Input downstream_deriv;

    CInputLayer(): previous_layer(nullptr)
    {
        // todo: downstream signalling_NAN?
    }

    virtual ~CInputLayer()
    {

    }

    //Functions which needs to be implemented by every required class
    virtual void train(int label, myfloat minibatch_size) = 0;
    virtual void update_weights(myfloat rate) = 0;
    virtual myfloat loss(Input& in, int label) = 0;
    virtual void predict(Input& in) = 0;
};

template<size_t D, size_t H, size_t W>
class COutputLayer<Dims<D, H, W>>
{
public:
    CInputLayer<Dims<D,H,W>> *next_layer;
    using Output = typename ArrayDims<Precision , Dims<D,H,W> >::type;
    Output output;

    COutputLayer(): next_layer(nullptr)
    {
        // todo: downstream signalling_NAN?
    }

    virtual ~COutputLayer()
    {

    }

    //Functions which needs to be implemented by every required class
    virtual void backprop(Output& deriv, myfloat minibatch_size) = 0;
};

#endif //NEURAL_NETWORK_CINPUTOUTPUT_H
