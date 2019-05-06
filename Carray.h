//
// Created by sonu on 5/5/19.
//

#ifndef NEURAL_NETWORK_CARRAY_H
#define NEURAL_NETWORK_CARRAY_H

#include "common.h"

/*
 * Array class:  This is a wrapper around native arrays to get range-checking.
 * It is similar to std::array, but more convenient for multi-dimensional arrays.
 */

// Forward declaration for output operators.
template <typename T, size_t D, size_t... Ds> class Array;

// Output operators for up to 4-D.
template <typename T, size_t D0>
std::ostream &
operator<<(std::ostream &os, const Array<T, D0> &a) {
    for (size_t i = 0; i < D0; i++) {
        if (i > 0) {
            os << " ";
        }
        os << std::fixed << std::setprecision(7) << a[i];
    }
    os << std::endl;
    return os;
}

template <typename T, size_t D1, size_t D0>
std::ostream &
operator<<(std::ostream &os, const Array<T, D1, D0> &a) {
    for (size_t i = 0; i < D1; i++) {
        os << std::fixed << std::setprecision(7) << a[i];
    }
    return os;
}

template <typename T, size_t D2, size_t D1, size_t D0>
std::ostream &
operator<<(std::ostream &os, const Array<T, D2, D1, D0> &a) {
    for (size_t h = 0; h < D2; h++) {
        os << "Matrix " << h << ":" << std::endl;
        os << a[h];
    }
    return os;
}

template <typename T, size_t D3, size_t D2, size_t D1, size_t D0>
std::ostream &
operator<<(std::ostream &os, const Array<T, D3, D2, D1, D0> &a) {
    for (size_t g = 0; g < D3; g++) {
        os << "Tensor " << g << ":" << std::endl;
        os << a[g];
    }
    return os;
}

// General definition of template.
template <typename T, size_t D, size_t... Ds>
class Array {
    friend std::ostream &operator<<<>(std::ostream &, const Array &);
public:
    Array() = default;
    template <typename U>
    Array(const U &v) {
        *this = v;
    }
    Array<T, Ds...> &operator[](const size_t i) {
        assert(i < D);
        return array[i];
    }
    const Array<T, Ds...> &operator[](const size_t i) const {
        assert(i < D);
        return array[i];
    }
    template <typename... Ts>
    T &operator()(const size_t i, const Ts... rest) {
        return (*this)[i](rest...);
    }
    template <typename... Ts>
    const T &operator()(const size_t i, const Ts... rest) const {
        return (*this)[i](rest...);
    }
    template <typename U>
    Array &operator=(const U &v) {
        std::fill(std::begin(array), std::end(array), v);
        return *this;
    }
    template <typename U>
    Array &operator=(const U (&a)[D]) {
        std::copy(std::begin(a), std::end(a), std::begin(array));
        return *this;
    }
    Array<T, Ds...> *begin() { return &array[0]; }
    Array<T, Ds...> *end() { return &array[D]; }
    const Array<T, Ds...> *begin() const { return &array[0]; }
    const Array<T, Ds...> *end() const { return &array[D]; }
private:
    Array<T, Ds...> array[D];
};

// Base case.
template <typename T, size_t D>
class Array<T, D> {
    friend std::ostream &operator<<<>(std::ostream &, const Array &);
public:
    Array() = default;
    template <typename U>
    Array(const U &v) {
        *this = v;
    }
    T &operator[](const size_t i) {
#ifndef NDEBUG
        if (i >= D) {
            std::cerr << "Index " << i << " beyond end of array of size " << D << "." << std::endl;
            assert(false);
            abort();
        }
#endif
        return array[i];
    }
    const T&operator[](const size_t i) const {
#ifndef NDEBUG
        if (i >= D) {
            std::cerr << "Index " << i << " beyond end of array of size " << D << "." << std::endl;
            assert(false);
            abort();
        }
#endif
        return array[i];
    }
    T &operator()(const size_t i) {
        return (*this)[i];
    }
    const T &operator()(const size_t i) const {
        return (*this)[i];
    }
    template <typename U>
    Array &operator=(const Array<U, D> &a) {
        std::copy(std::begin(a), std::end(a), std::begin(array));
        return *this;
    }
    template <typename U>
    Array &operator=(const U (&a)[D]) {
        std::copy(std::begin(a), std::end(a), std::begin(array));
        return *this;
    }
    template <typename U>
    Array &operator=(const U &v) {
        std::fill(std::begin(array), std::end(array), v);
        return *this;
    }
    T *begin() { return &array[0]; }
    T *end() { return &array[D]; }
    const T *begin() const { return &array[0]; }
    const T *end() const { return &array[D]; }
private:
    T array[D];
};

// Conversion.
template <typename T1, typename T2> struct ArrayDims;
template <typename T, size_t... Ds>
struct ArrayDims<T, Dims<Ds...>> {
using type = Array<T, Ds...>;
};

#endif //NEURAL_NETWORK_CARRAY_H
