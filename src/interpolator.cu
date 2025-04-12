#include <cstdint>
#include <cmath>

__global__ void filterDataKernel(const float *inSamples,
                                 const uint32_t numFilteredSamples,
                                 const float *taps,
                                 const uint32_t nTaps,
                                 float *outSamples)
{
   uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
   float convolutionSum = 0.0f;
   if (i < numFilteredSamples)
   {
      for (uint32_t j = 0; j < nTaps; j++)
      {
         // printf("numFilteredSamples: %u, i: %u, j: %u\n", numFilteredSamples, i, j);
         convolutionSum += taps[j] * inSamples[i + nTaps - j - 1];
      }
   }

   outSamples[i] = convolutionSum;
}

__global__ void addZeroesKernel(const float *inSamples,
                                const uint32_t numInSamples,
                                const uint32_t interpolationFactor,
                                float *outSamples)
{
   uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i < numInSamples)
   {
      outSamples[i * interpolationFactor] = inSamples[i] * interpolationFactor * 0.7; // Compensate for filter loss
      for (uint32_t idx = 1; idx < interpolationFactor; idx++)
      {
         outSamples[i * interpolationFactor + idx] = 0;
      }
   }
}
