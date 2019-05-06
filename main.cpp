#include "common.h"
#include "Carray.h"
#include "FileParser.h"
#include "CInputLayer.h"
#include <iostream>

int main()
{
    CFileParser parser;
    static float training_images[60000][28][28];
    parser.read_mnist_images("mnist/train-images.idx3-ubyte", training_images);
    parser.output_pgm("img0.pgm", training_images[0]);
    parser.output_pgm("img59999.pgm", training_images[59999]);

    static unsigned char training_labels[60000];
    parser.read_mnist_labels("mnist/train-labels.idx1-ubyte", training_labels);
    assert(training_labels[0] == 5);
    assert(training_labels[59999] == 8);

    static float test_images[10000][28][28];
    parser.read_mnist_images("mnist/t10k-images.idx3-ubyte", test_images);
    static unsigned char test_labels[10000];
    parser.read_mnist_labels("mnist/t10k-labels.idx1-ubyte", test_labels);

    static CInputLayer<Dims<1, 28, 28>> il;
}