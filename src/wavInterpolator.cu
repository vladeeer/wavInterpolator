#include <iostream>
#include <cstdint>
#include <string>
#include <cuda_runtime.h>

#include "waver.h"
#include "interpolator.cu"
// nvcc --std=c++17 -arch=sm_60 -Wno-deprecated-gpu-targets -o wavInterpolator wavInterpolator.cu waver.cpp
int main(int argc, char *argv[])
{
    Wav wav;
    if (!wav.read(argv[1]))
    {
        std::cout << "Failed to read wav file" << '\n';
        return 1;
    }

    wav.print();

    uint32_t k = std::stoi(argv[2]);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    std::cout << "Device: " << props.name << '\n';

    cudaError_t err = cudaSuccess;
    uint32_t channelDataSize = wav.header.dataSize / wav.header.nbrChannels;
    uint32_t numChannelSamples = channelDataSize / 2;
    uint32_t outChannelDataSize = channelDataSize * k;

    uint16_t *d_lData;
    uint16_t *d_rData;
    uint16_t *d_lResData;
    uint16_t *d_rResData;
    err = cudaMalloc((void **)&d_lData, channelDataSize);
    if (err != cudaSuccess)
    {
        printf("Failed to allocate device memory for left channel data (error code %s)!\n",
               cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc((void **)&d_rData, channelDataSize);
    if (err != cudaSuccess)
    {
        printf("Failed to allocate device memory for right channel data (error code %s)!\n",
               cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc((void **)&d_lResData, outChannelDataSize);
    if (err != cudaSuccess)
    {
        printf("Failed to allocate device memory for left channel result (error code %s)!\n",
               cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc((void **)&d_rResData, outChannelDataSize);
    if (err != cudaSuccess)
    {
        printf("Failed to allocate device memory for right channel result (error code %s)!\n",
               cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(d_lData, wav.samples[0].data(), channelDataSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("Failed to copy left channel data to device (error code %s)!\n",
               cudaGetErrorString(err));
        return 1;
    }
    err = cudaMemcpy(d_rData, wav.samples[1].data(), channelDataSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("Failed to copy right channel data to device (error code %s)!\n",
               cudaGetErrorString(err));
        return 1;
    }

    uint32_t threadsPerBlock = 512;
    int blocksPerGrid = (numChannelSamples + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernels launch with %d blocks of %d threads\n", blocksPerGrid,
           threadsPerBlock);
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    addZeroes<<<blocksPerGrid, threadsPerBlock, 0, s1>>>(d_lData, numChannelSamples, d_lResData, k);
    addZeroes<<<blocksPerGrid, threadsPerBlock, 0, s2>>>(d_rData, numChannelSamples, d_rResData, k);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Failed to run cuda kernel (error code %s)!\n",
               cudaGetErrorString(err));
        return 1;
    }

    wav.header.fileSize += wav.header.dataSize * (k - 1);
    wav.header.frequency *= k;
    wav.header.dataSize *= k;
    wav.header.bytePerSec *= k;
    wav.samples[0].resize(wav.header.dataSize / 4);
    wav.samples[1].resize(wav.header.dataSize / 4);

    err = cudaMemcpy(wav.samples[0].data(), d_lResData, outChannelDataSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("Failed to copy left channel samples from device (error code %s)!\n",
               cudaGetErrorString(err));
        return 1;
    }
    err = cudaMemcpy(wav.samples[1].data(), d_rResData, outChannelDataSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("Failed to copy right channel samples from device (error code %s)!\n",
               cudaGetErrorString(err));
        return 1;
    }

    cudaFree(d_lData);
    cudaFree(d_rData);
    cudaFree(d_lResData);
    cudaFree(d_rResData);

    wav.print();

    if (!wav.write("../out.wav"))
    {
        std::cout << "Failed to write wav file" << '\n';
        return 1;
    }

    return 0;
}
