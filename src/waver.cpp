#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "waver.h"

Wav::Wav() {}

Wav::Wav(const std::string &name)
{
	this->read(name);
}

Wav::~Wav() {}

bool Wav::read(const std::string &name)
{
	std::ifstream file(name, std::ios::binary);
	if (!file.is_open())
	{
		std::cout << "Failed to open file:" << name << '\n';
		return false;
	}

	// Read file & fmr chunks
	file.read((char *)&(this->header), 36);

	// Ignore "LIST" chunk if present
	char chunkName[4];
	auto pos = file.tellg();
	file.read(chunkName, 4);
	if (std::string(chunkName, 4) == "LIST")
	{

		uint32_t listSize;
		file.read((char *)&listSize, 4);
		file.ignore(listSize);
	}
	else
	{
		file.seekg(pos);
	}

	file.read(this->header.dataBlockID, 4);
	file.read((char *)&(this->header.dataSize), 4);

	if (this->header.audioFormat != 1)
	{
		std::cout << "Only integer uncompressed format is supported!";
		return false;
	}

	if (this->header.bitsPerSample != 16)
	{
		std::cout << "Only two byte samples are supported!";
		return false;
	}

	if (this->header.nbrChannels != 2)
	{
		std::cout << "Only stereo supported!";
		return false;
	}

	this->samples[0].resize(this->header.dataSize / 4); // 2 * bytesPerSample
	this->samples[1].resize(this->header.dataSize / 4);
	uint16_t *lPtr = this->samples[0].data();
	uint16_t *rPtr = this->samples[1].data();
	for (uint32_t i; i < this->header.dataSize / 4; i++)
	{
		file.read((char *)lPtr, 2);
		file.read((char *)rPtr, 2);
		lPtr++;
		rPtr++;
	}

	file.close();
	return true;
}

void Wav::print() const
{
	std::cout
		 << "Wav Header:" << '\n'
		 << "---------------------------------" << '\n'
		 << "fileTypeBlockID " << std::string(this->header.fileTypeBlockID, 4) << '\n'
		 << "fileSize        " << this->header.fileSize << '\n'
		 << "fileFormatID    " << std::string(this->header.fileFormatID, 4) << '\n'
		 << "---------------------------------" << '\n'
		 << "formatBlockID   " << std::string(this->header.formatBlockID, 4) << '\n'
		 << "blockSize       " << this->header.blockSize << '\n'
		 << "audioFormat     " << this->header.audioFormat << '\n'
		 << "nbrChannels     " << this->header.nbrChannels << '\n'
		 << "frequency       " << this->header.frequency << '\n'
		 << "bytePerSec      " << this->header.bytePerSec << '\n'
		 << "bytePerBlock    " << this->header.bytePerBlock << '\n'
		 << "bitsPerSample   " << this->header.bitsPerSample << '\n'
		 << "---------------------------------" << '\n'
		 << "dataBlockID     " << std::string(this->header.dataBlockID, 4) << '\n'
		 << "dataSize        " << this->header.dataSize << '\n'
		 << "---------------------------------" << '\n';
}
