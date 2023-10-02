# CUDA Dynamic Time Warping (DTW)

This repo contains a very simple implementation of the Dynamic Time Warping (DTW) algorithm using CUDA. DTW is a similarity measure used to compare two temporal sequences, allowing for temporal shifts and distortions between them. By leveraging the parallel processing power of CUDA-enabled GPUs, this implementation offers accelerated computation of the DTW distance between two sequences.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [File Structure](#file-structure)
- [Setup and Usage](#setup-and-usage)
- [Testing](#testing)
- [License](#license)
- [Contributing](#contributing)

## Overview

This implementation of DTW computes the minimal distance between two sequences of doubles. The sequences are generated using a uniform random number generator.
The CUDA kernel for the DTW algorithm operates in parallel on a GPU. The kernel computes the DTW distance in blocks, and each block operates independently from the others.

## Prerequisites
- NVIDIA GPU (with CUDA capability)
- CUDA Toolkit installed
- Modern C++ compiler

## File Structure

```
project/
│
├── src/
│   ├── DTW.cpp       # Main DTW algorithm
│   ├── DTW.h         
│   ├── main.cpp      
│   └── Makefile      # Compilation instructions
│
└── test/
    ├── TestDTW.cpp   # Tests for DTW
    └── Makefile      # Compilation instructions for tests
```

## Setup and Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/mbm6448/CUDA_DTW.git
   ```

2. Navigate into the project's `src` directory:
   ```sh
   cd CUDA_DTW/src
   ```

3. Compile the project using the Makefile:
   ```sh
   make
   ```

4. Run the main program:
   ```sh
   ./dtwProgram
   ```

5. Observe the DTW distance output.

## Testing

1. Navigate into the project's `test` directory:
   ```sh
   cd ../test
   ```

2. Compile the tests using the Makefile:
   ```sh
   make
   ```

3. Run the test binary:
   ```sh
   ./testDTW
   ```
.

## Contributing

Contributions are welcome! Please fork the repository and open a pull request with your changes or open an issue with suggestions and feedback.

