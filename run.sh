#!/bin/bash

set -e

mkdir -p build
cd build

cmake ..

cmake --build .

./HeterogenousSystem

convert cpu.ppm cpu.png
convert thread.ppm thread.png
convert cuda.ppm cuda.png
convert uma.ppm uma.png

mkdir -p ../img

mv cpu.png ../img
mv thread.png ../img
mv cuda.png ../img
mv uma.png ../img
