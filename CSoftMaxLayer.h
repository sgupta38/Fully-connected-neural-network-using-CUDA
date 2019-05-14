//
// Created by sonu on 5/13/19.
//

#ifndef NEURAL_NETWORK_CSOFTMAXLAYER_H
#define NEURAL_NETWORK_CSOFTMAXLAYER_H

#include "CinputOutput.h"
#include "Carray.h"

template <size_t N>
class SoftmaxLayer : public CBaseInputLayer<Dims<1, 1, N>>, public CBaseOutputLayer<Dims<1, 1, N>> {

using InputIF = CBaseInputLayer<Dims<1, 1, N>>;
using OutputIF = CBaseOutputLayer<Dims<1, 1, N>>;
using typename InputIF::Input;
using typename OutputIF::Output;

public:

// This layer has no loss function, so will always call it's forward layer.
// If it has no forward layer, that's a bug.
virtual void train(const int label, const double mb_size)  override
{
    forward(this->previous_layer->output, this->output);
    this->next_layer->train(label, mb_size);
}

virtual void backprop(const typename OutputIF::Output &full_upstream_deriv, const double mb_size) override;
virtual void update_weights(const double rate) override{
   // No weights in this layer.
    this->next_layer->update_weights(rate);
}
virtual double loss(Input &in, int label) override {
    Output temp_output;
    this->forward(in, temp_output);
    return this->next_layer->loss(temp_output, label);
}

virtual int predict(Input &in) override
{
    auto pos = std::max_element(std::begin(in[0][0]), std::end(in[0][0]));
    return std::distance(std::begin(in[0][0]), pos);
}

private:

static void forward(const Input &input, Output &output);
};
#endif //NEURAL_NETWORK_CSOFTMAXLAYER_H
