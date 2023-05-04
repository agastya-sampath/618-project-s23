# PIPP: Parallel Image Processing Pipeline

- **Authors**: [Agastya Sampath](https://github.com/agastya-sampath) and [Shijie Bian](https://github.com/BrandonBian)

- **Links**:
  - [Project Proposal](https://github.com/agastya-sampath/618-project-s23/blob/main/docs/project-proposal.pdf)
  - [Milestone Report](https://github.com/agastya-sampath/618-project-s23/blob/main/docs/project-milestone.pdf)
  - [Final Report](https://github.com/agastya-sampath/618-project-s23/blob/main/docs/project-report.pdf)

## Summary
In this project, we aimed to explore the benefits of parallel programming in the context of image processing. We implemented several image processing techniques, including image denoising, color enhancement, and color conversion, and created a comprehensive image processing library (PIPL - Parallel Image Processing Library) using these techniques. To improve the performance of the library, we utilized parallelizing tools such as OpenMP and CUDA on GPU to distribute the workload and increase processing speed. We conducted detailed profiling of the library's performance under various tasks and parallelization scenarios with different computing resources, hoping to provide further insight into how parallel programming can greatly benefit tasks related to image processing. 

## Dependencies
This library requires the use of CImg (https://cimg.eu/download.html). Set it up in the project directory under CImg/.

For the OpenMP version, the OpenMP library must be installed on the machine

For the CUDA version, NVCC is used. Please ensure that is set up on the machine. 

Thrust library might be needed for some auxiliary functions in our source code.

## Usage
#### Compilation Commands
```bash
make -f serialMakefile
```
```bash
make -f openmpMakefile
```
```bash
make -f cudaMakefile
```
#### Run Commands
```bash
./pipl-serial
```
```bash
./pipl-openmp
```
```bash
./pipl-cuda
```

## Test Machine
GHC 49 and GHC 74 machines from the GHC cluster at Carnegie Mellon University were used for all our testing.
