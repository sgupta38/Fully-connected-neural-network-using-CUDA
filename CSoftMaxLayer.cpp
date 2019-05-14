//
// Created by sonu
//

#include "common.h"
#include "CinputOutput.h"
#include "CSoftMaxLayer.h"

template <size_t N>
void
SoftmaxLayer<N>::backprop(const typename OutputIF::Output &full_upstream_deriv, const double mb_size) {

    // Note that we assume that ultimately we are computing the derivative of a scalar with respect to
    // each element of the softmax, so we simply add the derivatives.
    //
    auto &upstream_deriv(full_upstream_deriv[0][0]);
    this->downstream_deriv = 0;
    auto &downstream_deriv(this->downstream_deriv[0][0]);
    auto &output(this->output[0][0]);
    for (size_t j = 0; j < N; j++) {
        downstream_deriv[j] = 0;
        for (size_t i = 0; i < N; i++) {
            if (i == j) {
                downstream_deriv[j] += upstream_deriv[i]*(output[i]*(1 - output[j]));
            } else {
                downstream_deriv[j] += upstream_deriv[i]*(-output[j]*output[i]);
            }
        }
    }
    this->previous_layer->backprop(this->downstream_deriv, mb_size);
}

template <size_t N>
void
SoftmaxLayer<N>::forward(const Input &input, Output &output) {
    // Assume just a 1-D vector.  Note that this is a bit confusing,
    // because in C++, we think of this as just a single row, but
    // mathematically, we like to think of it as a column vector.
    auto &out(output[0][0]);
    auto &in(input[0][0]);
    // D is constant to improve numeric stability.
    const double D = *std::max_element(std::begin(in), std::end(in));
    double sum = 0;
    for (size_t i = 0; i < N; i++) {
        out[i] = exp(in[i] - D);
        sum += out[i];
    }
    for (size_t i = 0; i < N; i++) {
        out[i] = out[i]/sum;
    }
}
