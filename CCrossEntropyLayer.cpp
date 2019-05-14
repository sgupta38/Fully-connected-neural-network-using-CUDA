//
// Created by sonu
//

#include "common.h"
#include "CinputOutput.h"
#include "CCrossEntropyLayer.h"

template<size_t N>
void CCrossEntropyLayer<N>::train(const int label, const double mb_size) {
    // Note that there is no actual need to calculate the loss at this point.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    double loss = -log(this->previous_layer->output[0][0][label]);
#pragma GCC diagnostic pop
    this->downstream_deriv = 0;
    this->downstream_deriv[0][0][label] = -1/(this->previous_layer->output[0][0][label]);
    this->previous_layer->backprop(this->downstream_deriv, mb_size);
}


template<size_t N>
void CCrossEntropyLayer<N>::update_weights(const double) {
    // No weights in this layer, and this layer has no output.
}

template<size_t N>
double CCrossEntropyLayer<N>::loss( Input &in, int label) {
    return -std::log(in[0][0][label]);
}

template<size_t N>
int CCrossEntropyLayer<N>::predict(Input &) {
    assert(false);
    return -1;
}
