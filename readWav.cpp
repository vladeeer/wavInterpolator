#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <vector>

struct WavHeader
{
    char     fileTypeBlockID[4];
    uint32_t fileSize;
    char     fileFormatID[4];

    char     formatBlockID[4];
    uint32_t blockSize;
    uint16_t audioFormat;
    uint16_t nbrChannels;
    uint32_t frequency;
    uint32_t bytePerSec;
    uint16_t bytePerBlock;
    uint16_t bitsPerSample;
};

int main(int argc, char* argv[])
{
	ifile.clear();
	ifile.seekg(0);

	std::ifstream file(argv[1], std::ios::binary);
	if (!file.is_open())
    {
        std::cout << "Failed to open file:" <<  argv[1] << '\n';
        return 1;
    }

	WavHeader header;
	file.read((char*)&header, sizeof(WavHeader));

	// Ignore "LIST" chunk
	char chunkName[4];
	file.read(chunkName, 4);
	if (std::string(chunkName, 4) == "LIST")
	{
	
		uint32_t listSize;
		file.read((char*)&listSize, 4);
		file.ignore(listSize);
	}

	char     dataBlockID[4];
	uint32_t dataSize;
	file.read(dataBlockID, 4);
	file.read((char*)&dataSize, 4);

	std::cout
		<< argv[1] << " header:"<< '\n'
		<< "fileTypeBlockID " << std::string(header.fileTypeBlockID, 4) << '\n'
    	<< "fileSize        " << header.fileSize << '\n'
        << "fileFormatID    " << std::string(header.fileFormatID, 4) << '\n'
		<< "---------------------------------" << '\n'
		<< "formatBlockID   " << std::string(header.formatBlockID, 4) << '\n'
		<< "blockSize       " << header.blockSize << '\n'
		<< "audioFormat     " << header.audioFormat << '\n'
		<< "nbrChannels     " << header.nbrChannels << '\n'
		<< "frequency       " << header.frequency << '\n'
		<< "bytePerSec      " << header.bytePerSec << '\n'
		<< "bytePerBlock    " << header.bytePerBlock << '\n'
		<< "bitsPerSample   " << header.bitsPerSample << '\n'
		<< "---------------------------------" << '\n'
		<< "dataBlockID     " << std::string(dataBlockID, 4) << '\n'
		<< "dataSize        " << dataSize << '\n';
	
	if (header.audioFormat != 1)
	{
		std::cout << "Only integer uncompressed format is supported!";
		return 1;
	}

	if (header.bitsPerSample != 16)
	{
		std::cout << "Only two byte samples are supported!";
        return 1;
	}
	
	// Read one channel to array
	std::vector<uint16_t> data(dataSize);
	uint16_t* dataPtr = data.data();
	for (uint32_t i; i < dataSize / 2; i++)
	{
		file.read((char*)dataPtr, 2);
		file.ignore(2); // ignore 2nd channel
		dataPtr++;
	}

	std::cout << "Done reading file" << '\n';

	file.close();
    return 0;
}
