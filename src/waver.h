#include <string>
#include <cstdint>
#include <vector>

struct WavHeader
{
   char fileTypeBlockID[4]; // 4
   uint32_t fileSize;       // 4
   char fileFormatID[4];    // 4 (12 Total)

   char formatBlockID[4];  // 4
   uint32_t blockSize;     // 4
   uint16_t audioFormat;   // 2
   uint16_t nbrChannels;   // 2
   uint32_t frequency;     // 4
   uint32_t bytePerSec;    // 4
   uint16_t bytePerBlock;  // 2
   uint16_t bitsPerSample; // 2 (24 Total)

   char dataBlockID[4]; // 4
   uint32_t dataSize;   // 4 (8 Total)
};

class Wav
{
public:
   Wav();
   Wav(const std::string &name);
   ~Wav();

public:
   bool read(const std::string &name);
   void print() const;

public:
   WavHeader header;
   std::vector<uint16_t> samples[2];
};