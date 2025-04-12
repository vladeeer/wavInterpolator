#include <fstream>
#include <iostream>
#include <string>
#include <cstdint>
#include <cmath>

#include "waver.h"

bool compareFiles(const std::string &resultPath, const std::string &expectedPath)
{
   Wav result;
   Wav expected;

   result.read(resultPath);
   expected.read(expectedPath);

   if (result.header.dataSize != expected.header.dataSize)
   {
      fprintf(stderr, "Data size missmatch. Result: %u. Expected: %u.\n",
              result.header.dataSize, expected.header.dataSize);
      return false;
   }

   if (result.header.frequency != expected.header.frequency)
   {
      fprintf(stderr, "Frequency missmatch. Result: %u. Expected: %u.\n",
              result.header.dataSize, expected.header.dataSize);
      return false;
   }

   const unsigned int numChannelSamples = expected.header.dataSize / (expected.header.nbrChannels * sizeof(int16_t));
   const float thr = 1;
   for (uint32_t i; i < numChannelSamples; i++)
   {
      if (std::abs(result.samples[0][i] - expected.samples[0][i]) > thr)
      {
         fprintf(stderr, "Sample missmatch. Channel: left. Total channel samples: %u. Sample num: %u. Result: %f. Expected: %f.\n",
                 numChannelSamples, i, result.samples[0][i], expected.samples[0][i]);
         return false;
      }
      if (std::abs(result.samples[1][i] - expected.samples[1][i]) > thr)
      {
         fprintf(stderr, "Sample missmatch. Channel: right. Total channel samples: %u. Sample num: %u. Result: %f. Expected: %f.\n",
                 numChannelSamples, i, result.samples[1][i], expected.samples[1][i]);
         return false;
      }
   }

   return true;
}

int main(int argc, char **argv)
{
   int result = std::system("../wavInterpolator test_data/input.wav 4 test_data/out.wav"); /////

   if (result != 0)
   {
      std::cerr << "Running the program failed with exit code: " << result << "\n";
      return 1;
   }

   if (compareFiles("test_data/expected_output.wav", "test_data/out.wav"))
   {
      std::cout << "Test passed: Output matches expected output.\n";
      return 0;
   }
   else
   {
      std::cerr << "Test failed: Output does not match expected output.\n";
      return 1;
   }
}