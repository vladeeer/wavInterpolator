#include <iostream>
#include <cstdint>

#include "waver.h"
// clang++ --std=c++17 -o wavInterpolator wavInterpolator.cpp waver.cpp
int main(int argc, char *argv[])
{
    Wav wav;
    if (!wav.read(argv[1]))
    {
        std::cout << "Failed to read wav file";
        return 1;
    }
    wav.print();

    return 0;
}
