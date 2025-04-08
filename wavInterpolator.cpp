#include "wavReader.h"

int main(int argc, char* argv[])
{
	std::ifstream file(argv[1], std::ios::binary);
    if (!file.is_open())
    {
        std::cout << "Failed to open file:" <<  argv[1] << '\n';
        return 1;
    }

	return 0;
}
