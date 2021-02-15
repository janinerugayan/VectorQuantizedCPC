#!/bin/bash

set -e

# Install VectorQuantizedCPC
if [ ! -d ../VectorQuantizedCPC ]; then
    git clone https://github.com/kamperh/VectorQuantizedCPC \
        ../VectorQuantizedCPC
fi

# Install VectorQuantizedVAE
if [ ! -d ../VectorQuantizedVAE ]; then
    git clone https://github.com/kamperh/ZeroSpeech ../VectorQuantizedVAE
fi


if [ ! -d ../src ]; then
    mkdir ../src/
fi
cd ../src/

# Install speech_dtw
if [ ! -d speech_dtw ]; then
    git clone https://github.com/kamperh/speech_dtw.git
    cd speech_dtw
    make
    make test
    cd -
fi