//
// Created by sonu
//

#include "common.h"
#include "CinputOutput.h"

template <size_t N>
class CCrossEntropyLayer : public CBaseInputLayer<Dims<1, 1, N>> {
    using InputIF = CBaseInputLayer<Dims<1, 1, N>>;
    using typename InputIF::Input;
public:
    virtual void train(const int label, const double mb_size) override;
    virtual void update_weights(const float) override;
    virtual double loss(const Input &in, const int label) override;
    virtual int predict(const Input &) override;
};