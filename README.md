# wavInterpolator
Clone and build (Ubuntu):
```
user@host:~$ git clone git@github.com:vladeeer/wavInterpolator.git
user@host:~$ cd wavInterpolator
user@host:~/wavInterpolator$ mkdir build
user@host:~/wavInterpolator$ cd build
user@host:~/wavInterpolator/build$ cmake ..
user@host:~/wavInterpolator/build$ make
```
<br />Run with
```
user@host:~/wavInterpolator/build$ ./wavInterpolator inputFile.wav factor outputFile.wav
```
inputFile.wav - path to input RIFF WAVE file (.wav format)<br />
factor - integer interpolation factor, must be a multiple of 2, 3 or 5. For example: 72<br />
outputFile.wav - path to output RIFF WAVE file (.wav format)<br />
<br />
Run example:
```
user@host:~/wavInterpolator/build$ ./wavInterpolator ../exampleWavs/Hellcat11025.wav 60 hugeCat.wav
```

