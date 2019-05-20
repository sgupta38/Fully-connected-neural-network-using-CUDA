//  @author: Sonu Gupta
//  @purpose: Main file which creates all NN layers and calls CUDA functions.
//
//  @citation: Professor's code is referenced for serial execution. forward, backprop and update_Weight runs on GPU

#include "common.h"
#include "Carray.h"
#include "FileParser.h"
#include "CinputOutput.h"
#include "CInputLayer.h"
#include "CSoftMaxLayer.h"
#include "CCrossEntropyLayer.h"
#include "CCrossEntropyLayer.cpp"
#include "CSoftMaxLayer.cpp"
#include "cuda_functions.h"
#include "CFullyConnectedLayer.cu"

#define TRAINING_IMAGES "mnist/train-images.idx3-ubyte"
#define TRAINING_LABELS "mnist/train-labels.idx1-ubyte"
#define TEST_IMAGES "mnist/t10k-images.idx3-ubyte"
#define TEST_LABELS "mnist/t10k-labels.idx1-ubyte"

int main()
{
    // Parsing the image files here.
    CFileParser parser;
    static float training_images[60000][28][28];
    parser.read_mnist_images(TRAINING_IMAGES, training_images); // train image

    static unsigned char training_labels[60000];
    parser.read_mnist_labels(TRAINING_LABELS, training_labels); // train label
    assert(training_labels[0] == 5);
    assert(training_labels[59999] == 8);

    static float test_images[10000][28][28];
    parser.read_mnist_images(TEST_IMAGES, test_images); // test image
    static unsigned char test_labels[10000];
    parser.read_mnist_labels(TEST_LABELS, test_labels); // test label

    // Layer declarations

    static CInputLayer<Dims<1, 28, 28>> il;

    //This is hidden layer 1
    static CFullyConnectedLayer<Dims<1, 28, 28>, 1024> dl1("hd1", true, .4, 1);

    //this is hidden layer 2
    static CFullyConnectedLayer<Dims<1, 1, 1024>, 10> dl2("hd2", false, 0, 2);

    // Followed by softmax
    static SoftmaxLayer<10> sm;

    //Followed by CRoss entropy
    static CCrossEntropyLayer<10> ce;

    //
    // Connecting Layers and neurons
    //
    il.next_layer = &dl1; dl1.previous_layer = &il;
    dl1.next_layer = &dl2; dl2.previous_layer = &dl1;
    dl2.next_layer = &sm; sm.previous_layer = &dl2;
    sm.next_layer = &ce; ce.previous_layer = &sm;

    //Keeping values as it is
    std::default_random_engine eng(9815);
    std::uniform_int_distribution<size_t> pick_test(0, 9999);

    //epochs start here

    for (int e = 0; e < 6; e++) {
        std::vector<int> training(60000);
        std::iota(training.begin(), training.end(), 0);
        assert(*--training.end() == 59999);
        std::shuffle(training.begin(), training.end(), eng);

        for (int r = 0; r < 600; r++) {
            if (r%100 == 0) {

                int correct = 0;
                
                for (size_t i = 0; i < 10000; i++) {
                    size_t ind = pick_test(eng);
                    if (il.predict(test_images[ind]) == test_labels[ind]) {
                        correct++;
                    }
                }

                fprintf(stderr, "Current Epoch is := %d: Round %d: accuracy is :=%f\n", e, r, correct/10000.0);
            }

            for (size_t i = 0; i < 100; i++) {
                il.train(training_images[training.at(100*r + i)], training_labels[training.at(100*r + i)], 100);
            }

            il.update_weights(.002);
        }
    }
}
