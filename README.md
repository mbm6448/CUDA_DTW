# Dynamic Time Warping (DTW) Acceleration with CUDA

This repo contains an advanced implementation of the Dynamic Time Warping (DTW) algorithm using CUDA. DTW is a similarity measure used to compare two temporal sequences, allowing for temporal shifts and distortions between them. Leveraging the parallel processing capabilities of CUDA-enabled GPUs, this implementation provides accelerated computation of the DTW distance between two sequences.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [File Structure](#file-structure)
- [Setup and Usage](#setup-and-usage)
- [Testing](#testing)
- [License](#license)
- [Contributing](#contributing)

## Overview

The project implements DTW to compute the minimal distance between two sequences of doubles. The sequences can be generated using a uniform random number generator.
A CUDA kernel accelerates the DTW algorithm, enabling parallel operations on a GPU. The kernel computes the DTW distance in blocks, with each block operating independently.

## Prerequisites

- NVIDIA GPU (with CUDA capability)
- CUDA Toolkit
- Modern C++ compiler
- Google Test (for unit testing)

## File Structure

```
CUDA_DTW/
│
├── include/          # Header files
│   └── DTW.h
│
├── src/              # Source files
│   ├── DTW.cu        # Main DTW algorithm
│   └── main.cu
│
├── test/             # Test files
│   └── TestDTW.cu    # Tests for DTW
│
├── Makefile          # Compilation instructions
│
└── LICENSE
```

## Setup and Usage

1. Clone the repository:

   ```sh
   git clone https://github.com/mbm6448/CUDA_DTW.git
   ```

2. Navigate into the project's root directory:

   ```sh
   cd CUDA_DTW
   ```

3. Build the main program using the Makefile:

   ```sh
   make all
   ```

4. Run the main binary from the `bin/` directory:
   ```sh
   ./bin/main
   ```

## Testing

1. Ensure Google Test is set up in your environment.
2. Build the test binary using the Makefile:

   ```sh
   make test
   ```

3. Execute the test binary from the `bin/` directory:
   ```sh
   ./bin/test_run
   ```

## Contributing

Contributions are most welcome! Please fork the repository and open a pull request with your changes, or open an issue with suggestions and feedback.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for more details.
