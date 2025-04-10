# wavInterpolator
Compiled with:
```
nvcc --std=c++17 -arch=sm_60 -Wno-deprecated-gpu-targets -o wavInterpolator src/wavInterpolator.cu src/waver.cpp
```

Run with
```
./wavInterpolator file factor
```
file - path to RIFF WAVE file (.wav format)
factor - integer interpolation factor, must be a multiple of 2, 3 or 5. For example: 72