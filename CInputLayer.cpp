//
// Created by sonu on 5/6/19.
//

#include "CInputLayer.h"

template<typename OUT_DIMS>
void CInputLayer<OUT_DIMS>::update_weights(myfloat rate) {
    this->next_layer->update_weights(rate);
}

template<typename OUT_DIMS>
void CInputLayer<OUT_DIMS>::train(float (&image)[OUT_H][OUT_W], int label, myfloat mb_size) {
    this->output[0] = image; // This will act as i/p later on
    this->next_layer->train(label, mb_size);
}

template<typename OUT_DIMS>
int CInputLayer<OUT_DIMS>::predict(float (&image)[OUT_H][OUT_W]) {
    Output output;
    output[0] = image;
    return this->next_layer->predict(output);
}
