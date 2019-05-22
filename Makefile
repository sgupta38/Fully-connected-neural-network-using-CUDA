all:
	#nvcc -g -G main.cu cuda_functions.cu -std=c++14
	#nvcc -O4 main.cu cuda_functions.cu CFullyConnectedLayer.cu CSoftMaxLayer.cpp CCrossEntropyLayer.cpp FileParser.cpp -std=c++14
	nvcc -O3 -o fnn main.cu cuda_functions.cu CFullyConnectedLayer.cu CSoftMaxLayer.cpp CCrossEntropyLayer.cpp FileParser.cpp -std=c++14
	#nvcc -g -G -o fnn main.cu cuda_functions.cu CFullyConnectedLayer.cu CSoftMaxLayer.cpp CCrossEntropyLayer.cpp FileParser.cpp -std=c++14
clean:
	rm *.out *.pgm
