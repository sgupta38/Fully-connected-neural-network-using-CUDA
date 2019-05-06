//
// Created by sonu
//

#include "common.h"
#include "CinputOutput.h"

template<size_t N>
class CSoftMaxLayer: public CBaseInputLayer<Dims<1,1,N>>, public CBaseOutputLayer<Dims<1,1,N>>
{
    using InputIF = CBaseInputLayer<Dims<1, 1, N>>;
    using OutputIF = CBaseOutputLayer<Dims<1, 1, N>>;
    using typename InputIF::Input;
    using typename OutputIF::Output;
public:
    virtual void train(const int label, const double mb_size) override;
    virtual void backprop(const typename OutputIF::Output &full_upstream_deriv, const double mb_size) override;
    virtual void update_weights(const float rate) override;
    virtual double loss(const Input &in, const int label) override;
    virtual int predict(const Input &in) override;
private:
    static void forward(const Input &input, Output &output);
};