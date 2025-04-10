#include <iostream>
#include <cstdint>
#include <string>
#include <utility>
#include <cuda_runtime.h>

#include "waver.h"
#include "interpolator.cu"

// Kaiser lowpass filters
constexpr uint32_t nTaps = 65;
constexpr float tapsArr[nTaps * 3] = {
#include "taps.h"
};

bool addZeroes(float *&d_lReadSwapBuff, float *&d_rReadSwapBuff,
               const uint32_t numChannelSamples, const uint32_t interpolationFactor,
               float *&d_lWriteSwapBuff, float *&d_rWriteSwapBuff)
{
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    const uint32_t threadsPerBlock = 512;
    const uint32_t blocksPerGrid = (numChannelSamples + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernels \"addZeroesKernel\" launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    addZeroesKernel<<<blocksPerGrid, threadsPerBlock, 0, s1>>>(d_lReadSwapBuff, numChannelSamples, interpolationFactor, d_lWriteSwapBuff);
    addZeroesKernel<<<blocksPerGrid, threadsPerBlock, 0, s2>>>(d_rReadSwapBuff, numChannelSamples, interpolationFactor, d_rWriteSwapBuff);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Failed to run cuda kernel (error code %s)!\n",
               cudaGetErrorString(err));
        return false;
    }

    std::swap(d_lReadSwapBuff, d_lWriteSwapBuff);
    std::swap(d_rReadSwapBuff, d_rWriteSwapBuff);

    return true;
}

bool filterData(float *&d_lReadSwapBuff, float *&d_rReadSwapBuff,
                const uint32_t numFilteredSamples, const uint32_t interpolationFactor,
                const float *d_tapsArr, const uint32_t nTaps,
                float *&d_lWriteSwapBuff, float *&d_rWriteSwapBuff)
{
    const float *d_taps;
    switch (interpolationFactor)
    {
    case 2:
        d_taps = d_tapsArr;
        break;
    case 3:
        d_taps = d_tapsArr + nTaps;
        break;
    case 5:
        d_taps = d_tapsArr + nTaps * 2;
        break;
    default:
        std::cout << "Invalid interpolationFactor" << '\n';
        return false;
    }

    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    const uint32_t threadsPerBlock = 512;
    const uint32_t blocksPerGrid = (numFilteredSamples + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernels \"filterDataKernel\" launch with %d blocks of %d threads and interpolation factor %d\n",
           blocksPerGrid, threadsPerBlock, interpolationFactor);
    filterDataKernel<<<blocksPerGrid, threadsPerBlock, 0, s1>>>(d_lReadSwapBuff, numFilteredSamples, d_taps, nTaps, d_lWriteSwapBuff);
    filterDataKernel<<<blocksPerGrid, threadsPerBlock, 0, s2>>>(d_rReadSwapBuff, numFilteredSamples, d_taps, nTaps, d_rWriteSwapBuff);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Failed to run cuda kernel (error code %s)!\n",
               cudaGetErrorString(err));
        return 1;
    }

    std::swap(d_lReadSwapBuff, d_lWriteSwapBuff);
    std::swap(d_rReadSwapBuff, d_rWriteSwapBuff);

    return true;
}

int main(int argc, char *argv[])
{
    Wav wav;
    if (!wav.read(argv[1]))
    {
        std::cout << "Failed to read wav file" << '\n';
        return 1;
    }

    wav.print();

    const uint32_t interpolationFactor = std::stoi(argv[2]);
    uint32_t k = interpolationFactor;
    uint32_t num5interpolations = 0;
    uint32_t num3interpolations = 0;
    uint32_t num2interpolations = 0;
    while (k % 5 == 0)
    {
        k /= 5;
        num5interpolations++;
    }
    while (k % 3 == 0)
    {
        k /= 3;
        num3interpolations++;
    }
    while (k % 2 == 0)
    {
        k /= 2;
        num2interpolations++;
    }
    if (k != 1)
    {
        std::cout << "Interpolation factor must be a multiple of 2, 3, 5." << '\n'
                  << "Example: 60 = 5 * 3 * 2 * 2" << '\n';
        return 1;
    }

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    std::cout << "Device: " << props.name << '\n';

    const uint32_t channelDataSize = 2 * wav.header.dataSize / wav.header.nbrChannels;
    const uint32_t numChannelSamples = channelDataSize / 4;
    const uint32_t numSamplesWithZeroes = numChannelSamples * interpolationFactor;
    const uint32_t numFilteredSamples = numSamplesWithZeroes - nTaps + 1;

    float *d_lReadSwapBuff;
    float *d_rReadSwapBuff;
    float *d_lWriteSwapBuff;
    float *d_rWriteSwapBuff;
    float *d_tapsArr;

    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void **)&d_lWriteSwapBuff, numSamplesWithZeroes * sizeof(float));
    if (err != cudaSuccess)
    {
        printf("Failed to allocate device memory for left channel write swap buffer (error code %s)!\n",
               cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc((void **)&d_rWriteSwapBuff, numSamplesWithZeroes * sizeof(float));
    if (err != cudaSuccess)
    {
        printf("Failed to allocate device memory for right channel write swap buffer (error code %s)!\n",
               cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc((void **)&d_lReadSwapBuff, numSamplesWithZeroes * sizeof(float));
    if (err != cudaSuccess)
    {
        printf("Failed to allocate device memory for left channel read swap buffer (error code %s)!\n",
               cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc((void **)&d_rReadSwapBuff, numSamplesWithZeroes * sizeof(float));
    if (err != cudaSuccess)
    {
        printf("Failed to allocate device memory for right channel read swap buffer (error code %s)!\n",
               cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc((void **)&d_tapsArr, sizeof(tapsArr));
    if (err != cudaSuccess)
    {
        printf("Failed to allocate device memory for taps (error code %s)!\n",
               cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(d_lReadSwapBuff, wav.samples[0].data(), channelDataSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("Failed to copy left channel data to device (error code %s)!\n",
               cudaGetErrorString(err));
        return 1;
    }
    err = cudaMemcpy(d_rReadSwapBuff, wav.samples[1].data(), channelDataSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("Failed to copy right channel data to device (error code %s)!\n",
               cudaGetErrorString(err));
        return 1;
    }
    err = cudaMemcpy(d_tapsArr, tapsArr, sizeof(tapsArr), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("Failed to copy taps to device (error code %s)!\n",
               cudaGetErrorString(err));
        return 1;
    }

    bool res;
    uint32_t numSamples = numChannelSamples;

    while (num2interpolations)
    {
        res = addZeroes(d_lReadSwapBuff, d_rReadSwapBuff, numSamples, 2, d_lWriteSwapBuff, d_rWriteSwapBuff);
        if (!res)
            return 1;

        res = filterData(d_lReadSwapBuff, d_rReadSwapBuff,
                         numFilteredSamples, 2,
                         d_tapsArr, nTaps,
                         d_lWriteSwapBuff, d_rWriteSwapBuff);
        if (!res)
            return 1;

        numSamples *= 2;
        num2interpolations--;
    }
    while (num3interpolations)
    {
        res = addZeroes(d_lReadSwapBuff, d_rReadSwapBuff, numSamples, 3, d_lWriteSwapBuff, d_rWriteSwapBuff);
        if (!res)
            return 1;

        res = filterData(d_lReadSwapBuff, d_rReadSwapBuff,
                         numFilteredSamples, 3,
                         d_tapsArr, nTaps,
                         d_lWriteSwapBuff, d_rWriteSwapBuff);
        if (!res)
            return 1;

        numSamples *= 3;
        num3interpolations--;
    }
    while (num5interpolations)
    {
        res = addZeroes(d_lReadSwapBuff, d_rReadSwapBuff, numSamples, 5, d_lWriteSwapBuff, d_rWriteSwapBuff);
        if (!res)
            return 1;

        res = filterData(d_lReadSwapBuff, d_rReadSwapBuff,
                         numFilteredSamples, 5,
                         d_tapsArr, nTaps,
                         d_lWriteSwapBuff, d_rWriteSwapBuff);
        if (!res)
            return 1;

        numSamples *= 5;
        num5interpolations--;
    }

    wav.header.frequency *= interpolationFactor;
    wav.header.bytesPerSec *= interpolationFactor;
    wav.header.dataSize = numFilteredSamples * sizeof(uint16_t) * wav.header.nbrChannels;
    wav.header.fileSize = 36 + wav.header.dataSize;
    wav.samples[0].resize(numFilteredSamples);
    wav.samples[1].resize(numFilteredSamples);

    err = cudaMemcpy(wav.samples[0].data(), d_lReadSwapBuff, numFilteredSamples * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("Failed to copy left channel samples from device (error code %s)!\n",
               cudaGetErrorString(err));
        return 1;
    }
    err = cudaMemcpy(wav.samples[1].data(), d_rReadSwapBuff, numFilteredSamples * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("Failed to copy right channel samples from device (error code %s)!\n",
               cudaGetErrorString(err));
        return 1;
    }

    cudaFree(d_lWriteSwapBuff);
    cudaFree(d_rWriteSwapBuff);
    cudaFree(d_lReadSwapBuff);
    cudaFree(d_rReadSwapBuff);
    cudaFree(d_tapsArr);

    wav.print();

    if (!wav.write("out.wav"))
    {
        std::cout << "Failed to write wav file" << '\n';
        return 1;
    }

    return 0;
}
