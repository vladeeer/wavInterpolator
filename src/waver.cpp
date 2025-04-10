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

	// Read "file" & "fmt" chunks
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

	this->samples[0].resize(this->header.dataSize / 4); // numChannels * bytesPerSample
	this->samples[1].resize(this->header.dataSize / 4);
	float *lData = this->samples[0].data();
	float *rData = this->samples[1].data();
	int16_t buff;
	for (uint32_t i; i < this->header.dataSize / 4; i++)
	{
		file.read((char *)&buff, 2);
		lData[i] = static_cast<float>(buff);
		file.read((char *)&buff, 2);
		rData[i] = static_cast<float>(buff);
		// std::cout << this->samples[0][i] << ' ' << this->samples[1][i] << '\n';
	}

	file.close();

	std::cout << "Read wav file: " << name << '\n';
	return true;
}

bool Wav::write(const std::string &name)
{
	std::ofstream file(name, std::ios::binary);
	if (!file.is_open())
	{
		std::cout << "Failed to open file:" << name << '\n';
		return false;
	}

	file.write((char *)&(this->header), sizeof(WavHeader));

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

	float *lData = this->samples[0].data();
	float *rData = this->samples[1].data();
	int16_t buff;
	for (uint32_t i; i < this->header.dataSize / 4; i++)
	{
		buff = static_cast<int16_t>(lData[i]);
		file.write((char *)&buff, 2);
		buff = static_cast<int16_t>(rData[i]);
		file.write((char *)&buff, 2);
	}

	file.close();

	std::cout << "Written to wav file: " << name << '\n';
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
		 << "bytesPerSec     " << this->header.bytesPerSec << '\n'
		 << "bytesPerBlock   " << this->header.bytesPerBlock << '\n'
		 << "bitsPerSample   " << this->header.bitsPerSample << '\n'
		 << "---------------------------------" << '\n'
		 << "dataBlockID     " << std::string(this->header.dataBlockID, 4) << '\n'
		 << "dataSize        " << this->header.dataSize << '\n'
		 << "---------------------------------" << '\n';
}
