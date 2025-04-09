#include <cstdint>

__global__ void addZeroes(const uint16_t *inSamples, const uint32_t numInSamples,
                          uint16_t *outSamples, uint32_t k)
{
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i < numInSamples)
   {
      outSamples[i * k] = inSamples[i];
      for (uint32_t idx = 1; idx < k; idx++)
      {
         outSamples[i * k + idx] = 0;
      }
   }
}
