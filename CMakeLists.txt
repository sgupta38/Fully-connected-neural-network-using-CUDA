cmake_minimum_required(VERSION 3.14)
project(neural_network)

set(CMAKE_CXX_STANDARD 14)

add_executable(neural_network
        main.cpp
        common.h
        FileParser.cpp
        FileParser.h
        Carray.h CinputOutput.h CInputLayer.cpp CInputLayer.h)
