//
// Created by sonu on 5/5/19.
//

#ifndef NEURAL_NETWORK_COMMON_H
#define NEURAL_NETWORK_COMMON_H

#include <cmath>
#include <cassert>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include <assert.h>

inline double
derivative_error(double n, double d) {
    return std::abs(n - d)/std::max(std::abs(n), std::abs(d));
}

// This holds a sequence of dimensions together in a single type.
template <size_t DP, size_t HP, size_t WP>
struct Dims {
    constexpr static size_t D = DP;
    constexpr static size_t H = HP;
    constexpr static size_t W = WP;
    constexpr static size_t N = D*H*W;
};

template <typename T, size_t D, size_t H, size_t W>
std::ostream &operator<<(std::ostream &os, const T (&a)[D][H][W]) {
    for (size_t h = 0; h < D; h++) {
        if (h > 0) {
            os << "----------" << std::endl;
        }
        for (size_t i = 0; i < H; i++) {
            for (size_t j = 0; j < W; j++) {
                if (j > 0) {
                    os << " ";
                }
                os << std::fixed << std::setprecision(7) << a[h][i][j];
            }
            os << "\n";
        }
    }
    return os;
}

// Adding this because I want to test my accuracy over float and double.
// This will act as a single point of contact.
using myfloat = float;
using std::size_t;
using Precision = float;
#include "Carray.h"

#endif //NEURAL_NETWORK_COMMON_H
