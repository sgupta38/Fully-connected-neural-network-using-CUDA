# neural-network-in-cuda-sgupta38

# Fully Connected Neural Network

This project achieves a 90% accuracy for given images. Expensive functions such as 'forward', 'backprop' and 'update_weights' are executed on GPU device. 'mnist' images has been used for training and testing purpose.

Following are some of the impotant files.

- **main.cu**  --> File which has entry point function and this is where we configure our layers & adjust our 'configurations' such as Number of neurons, dropout rate etc.
- **cuda_functions.cu** --> GPU API's for forwarding, backpropogation and updating weights.
- **CFullyConnectedLayer.cu**  --> Hidden layer which calls CUDA API's

## How to RUN?

	> ./fnn

## Architecture:


##### Layers

## Input Layer

The input layer is a 28×28×1 grayscale image.

## Fully Connected Hidden Layer

This is just a fully connected layer with ReLU activation. It consists of 1024 neurons with dropout rate of 0.4. This means that during training, any given neuron has a 0.4 probability of “dropping out”, which means that it is set to 0, regardless of the inputs.


## Output Layer

This is the final classification layer. It is a fully connected layer consisting of 10 neurons, one for each class. It will compute a softmax


