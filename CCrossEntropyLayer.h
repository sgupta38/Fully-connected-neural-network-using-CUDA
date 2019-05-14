//
// Created by sonu on 5/13/19.
//

#ifndef NEURAL_NETWORK_CCROSSENTROPYLAYER_H
#define NEURAL_NETWORK_CCROSSENTROPYLAYER_H

#include "CinputOutput.h"
#include "Carray.h"

template <size_t N>
class CCrossEntropyLayer : public CBaseInputLayer<Dims<1, 1, N>> {
    using InputIF = CBaseInputLayer<Dims<1, 1, N>>;
    using typename InputIF::Input;
public:
    virtual void train(const int label, const double mb_size);
    virtual void update_weights(const double);
    virtual double loss( Input &in, int label) override;
    virtual int predict(Input &) override;
};

#endif //NEURAL_NETWORK_CCROSSENTROPYLAYER_H
