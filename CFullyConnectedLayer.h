//
// Created by sonu on 5/13/19.
//

#ifndef NEURAL_NETWORK_CFULLYCONNECTEDLAYER_H
#define NEURAL_NETWORK_CFULLYCONNECTEDLAYER_H

#include "common.h"
#include "CinputOutput.h"
#include "CInputLayer.h"
#include "Carray.h"
#include "cuda_functions.h"

template <typename IN_DIMS, size_t NEURONS>
class CFullyConnectedLayer: public CBaseInputLayer<IN_DIMS>, public CBaseOutputLayer<Dims<1,1, NEURONS >>
{
    using InputIF = CBaseInputLayer<IN_DIMS>;
    using OutputIF = CBaseOutputLayer<Dims<1, 1, NEURONS>>;
    using typename InputIF::Input;
    using typename OutputIF::Output;
    constexpr static size_t IN_D = InputIF::InputDims::D;
    constexpr static size_t IN_H = InputIF::InputDims::H;
    constexpr static size_t IN_W = InputIF::InputDims::W;
    constexpr static size_t OUT_D = OutputIF::OutputDims::D;
    constexpr static size_t OUT_H = OutputIF::OutputDims::H;
    constexpr static size_t OUT_W = OutputIF::OutputDims::W;
    public:
    const std::string m_layer_name;
    bool m_relu;
    Array<Input , NEURONS> m_weight;
    Array<Input , NEURONS> m_weight_deriv;
    Array<double, NEURONS> m_bias;
    Array<double, NEURONS> m_bias_deriv;
    double m_keep_prob;
    Array<double , NEURONS> m_current_kept;
    Array<double , NEURONS> m_all_kept;
    std::default_random_engine m_eng;

    CFullyConnectedLayer(const std::string& n, bool relu, double do_rate, int ssed_seq);

    void train(int label, double mb_size);
    virtual void backprop(const Output& full_upstream_deriv, const double mb_size);
    void update_weights(double rate);
    virtual double loss (Input& in, int label) override;
    virtual int predict(Input& in) override;

    private:
    void forward(Input& input, Array<Input, NEURONS> &weight, Array<double, NEURONS> bias, Array<double, NEURONS> &dropped, Output& output);
};
#endif //NEURAL_NETWORK_CFULLYCONNECTEDLAYER_H
