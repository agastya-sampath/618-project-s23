# PIPP: Parallel Image Processing Pipeline

- **Authors**: [Agastya Sampath](https://github.com/agastya-sampath) and [Shijie Bian](https://github.com/BrandonBian)

- **Links**:
  - [Project Proposal](TODO)
  - [Milestone Report](TODO)
  - [Final Report](TODO)

## Summary
We plan to explore parallel implementations of several image processing techniques in the areas of image denoising, color correction and data augmentation. We plan to create an image processing pipeline using these techniques and compare the techniques/applications in terms of their parallelizability.

## Background
As a team, we believe that image processing is a critical area of study due to the ubiquity of images in our daily lives. From social media to medical imaging, images are essential in a variety of fields, and ensuring their quality is of utmost importance. However, image processing can be a computationally intensive task, especially when dealing with large images. As such, we recognize the importance of leveraging parallelism to accelerate image processing techniques. This project provides us with a unique opportunity to contribute to the development of a parallel image processing library, where we can apply and analyze various parallel programming techniques, such as OpenMP, CUDA, and MPI. The non-trivial nature of this project lies in the fact that different processing techniques may benefit from parallelism differently, and thus require careful profiling and analysis to ensure optimal performance

One important aspect of the problem that can benefit from parallelism is the pixel-level operations, such as denoising and color augmentations. For example, in the case of using Ising and Gibbs sampling to do denoising, we can apply parallelism across pixels to accelerate the process. Similarly, for scaling operations, parallelism can be applied to distribute the workload among multiple threads or processes to reduce the overall processing time. As mentioned, different processing techniques may benefit from different parallelism approaches and units (i.e., units other than pixels). For example, in image scaling, we probably can divide the image into smaller blocks and perform parallel scaling on each block. In color augmentations, we think that we can perform parallel operations on individual color channels. In image segmentation, we should be able to use parallel algorithms to partition the image into segments. Therefore, the major goal of this project is to identify the specific aspects of the multifaceted problem we ae trying to solve that can benefit from parallelism, and leverage the correct tools to speed up the image processing library. 

## The Challenge
The image processing problem is challenging because different processing techniques may benefit from parallelism differently, which requires careful analysis and profiling to determine the optimal parallelization strategy. Moreover, some image processing algorithms have dependencies between pixels or between neighboring regions, which can make it difficult to parallelize them efficiently. The workload may have varying memory access characteristics, such as locality or high communication-to-computation ratio, which can impact the effectiveness of parallelization. Additionally, some image processing techniques involve divergent execution, where different parts of the code follow different execution paths, which can also make parallelization more challenging. The constraints of the system, such as the available resources and communication bandwidth, may also make mapping the workload to the system challenging. By tackling these challenges, we hope to learn about the trade-offs and best practices in parallelizing image processing algorithms, and ultimately improve the performance of our library.

## Resources
Our resources for the project will include the code and templates from assignments 1 to 4, including CUDA, OpenMP, and MPI, as well as access to the PSC computer cluster. We will also use basic algorithms for image processing, starting with the Ising and Gibbs sampling methods from the [MC_Ising GitHub repository](https://github.com/g0bel1n/MC_Ising), and then adding MPI and parallelism. We also intend to reference basic sequential and serialized implementations of [common image processing algorithms](https://github.com/iocentos/ImageProcessing/blob/master/processing/serialImageProcessing.cpp) and to start by reproducing results, analizing performance, and add our parallelism profiling and optimizations. We have not identified any other specific resources needed at this time, but we will keep an open mind for any additional tools or resources that may be beneficial to the project.


## Goals and Deliverables
