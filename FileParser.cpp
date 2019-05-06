//
// Created by sonu on 5/5/19.
//

#include "FileParser.h"
void CFileParser::swap(int &i) {
    i = (0xff&(i >> 24)) |
        (0xff00&(i >> 8)) |
        (0xff0000&(i << 8)) |
        (0xff000000&(i << 24));
}

int CFileParser::read_int(int fd) {
    int rv;
    int i;
    rv = read(fd, &i, 4); assert(rv == 4);
    swap(i);
    return i;
}

void CFileParser::output_pgm(const std::string &fn, const float (&img)[28][28]) {
    std::ofstream ofs(fn, std::fstream::out|std::fstream::trunc);

    ofs << "P2\n";
    ofs << "28 28\n";
    ofs << "255\n";
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            if (j > 0) {
                ofs << " ";
            }
            ofs << 255 - int(std::round(127.5*(img[i][j] + 1)));
        }
        ofs << "\n";
    }
}
