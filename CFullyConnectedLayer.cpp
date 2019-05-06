//
// Created by sonu
//

#include "common.h"
#include "CinputOutput.h"

template <typename IN_DIMS, size_t NEURONS>
class CFullyConnectedLayer: public CBaseInputLayer<IN_DIMS>, public CBaseOutputLayer<Dims<1,1, NEURONS >>
{
    using IF = CBaseInputLayer<IN_DIMS>;
    using OF = CBaseOutputLayer<Dims<1,1,NEURONS>>;

    using typename IF::Input;
    using typename OF::Output;

    constexpr static size_t IN_D = IF::Input::D;
    constexpr static size_t IN_H = IF::Input::H;
    constexpr static size_t IN_W = IF::Input::W;

    constexpr static size_t OUT_D = OF::Output::D;
    constexpr static size_t OUT_H = OF::Output::H;
    constexpr static size_t OUT_W = OF::Output::W;

public:
    std::string m_layer_name;
    bool m_relu;
    Array<Input , NEURONS> m_weight;
    Array<Input , NEURONS> m_weight_deriv;
    Array<double, NEURONS> m_bias;
    Array<double, NEURONS> m_bias_deriv;
    double m_keep_prob;
    Array<double , NEURONS> m_current_kept;
    Array<double , NEURONS> m_all_kept;
    std::default_random_engine eng;

    CFullyConnectedLayer(std::string& n, bool relu, double do_rate, int ssed_seq);
    virtual void train(int label, double mb_size) override;
    virtual void backprop(Output& full_upstream_deriv, double mb_size) override;
    virtual void update_weight(double rate) override;
    virtual double loss (Input& in, int label) override;
    virtual int predict(Input& in) override;

private:
    void forward(Input& input, Array<Input, NEURONS> &weight, Array<double, NEURONS> bias, Array<double, NEURONS> &dropped, Output& output);
};