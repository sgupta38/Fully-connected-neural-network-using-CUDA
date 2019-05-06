//
// Created by sonu on 5/5/19.
//

#ifndef NEURAL_NETWORK_FILEPARSER_H
#define NEURAL_NETWORK_FILEPARSER_H

#include "common.h"

class CFileParser
{
public:
    void swap(int &i);
    int read_int(int fd);
    void output_pgm(const std::string &fn, const float (&img)[28][28]);
    template <int N> void read_mnist_images(const std::string &fn, float (&imgs)[N][28][28])
    {
        {

            int rv;

            int fd;
            fd = open(fn.c_str(), O_RDONLY);
            assert(fd >= 0);

            int magic = read_int(fd);
            assert(magic == 0x803);

            int n_images = read_int(fd);
            assert(n_images == N);

            int n_rows = read_int(fd);
            assert(n_rows == 28);

            int n_cols = read_int(fd);
            assert(n_cols == 28);

            for (int i = 0; i < N; i++) {
                unsigned char tmp[28][28];
                rv = read(fd, tmp, 28*28); assert(rv == 28*28);
                for (int r = 0; r < 28; r++) {
                    for (int c = 0; c < 28; c++) {
                        // Make go from -1 to 1.
                        imgs[i][r][c] = double(tmp[r][c])/127.5 - 1;
                    }
                }
            }

            rv = close(fd); assert(rv == 0);
        }

    }
    template <int N> void read_mnist_labels(const std::string &fn, unsigned char (&labels)[N])
    {
        {

            int rv;

            int fd;
            fd = open(fn.c_str(), O_RDONLY);
            assert(fd >= 0);

            int magic = read_int(fd);
            assert(magic == 0x801);

            int n_labels = read_int(fd);
            assert(n_labels == N);

            rv = read(fd, labels, N); assert(rv == N);
            for (int i = 0; i < N; i++) {
                assert(labels[i] >= 0 && labels[i] <= 9);
            }

            rv = close(fd); assert(rv == 0);
        }

    }
};


#endif //NEURAL_NETWORK_FILEPARSER_H
