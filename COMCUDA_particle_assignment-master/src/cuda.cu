#include "cuda.cuh"
#include "helper.h"
#include <device_launch_parameters.h>
#include "cuda_runtime.h"   
#include <cstring>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

///
/// Algorithm storage
///
// Number of particles in d_particles
unsigned int cuda_particles_count;
// Device pointer to a list of particles
Particle* d_particles;
// Device pointer to a histogram of the number of particles contributing to each pixel
unsigned int* d_pixel_contribs;
// Device pointer to an index of unique offsets for each pixels contributing colours
unsigned int* d_pixel_index;
// Device pointer to storage for each pixels contributing colours
unsigned char* d_pixel_contrib_colours;
// Device pointer to storage for each pixels contributing colours' depth
float* d_pixel_contrib_depth;
// The number of contributors d_pixel_contrib_colours and d_pixel_contrib_depth have been allocated for
unsigned int cuda_pixel_contrib_count;
// Host storage of the output image dimensions
int cuda_output_image_width;
int cuda_output_image_height;
// Device storage of the output image dimensions
__constant__ int D_OUTPUT_IMAGE_WIDTH;
__constant__ int D_OUTPUT_IMAGE_HEIGHT;
// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device
unsigned char* d_output_image_data;
CImage cuda_output_image;


void cuda_begin(const Particle* init_particles, const unsigned int init_particles_count,
    const unsigned int out_image_width, const unsigned int out_image_height) {
    // These are basic CUDA memory allocations that match the CPU implementation
    // Depending on your optimisation, you may wish to rewrite these (and update cuda_end())

    // Allocate a opy of the initial particles, to be used during computation
    cuda_particles_count = init_particles_count;
    CUDA_CALL(cudaMalloc(&d_particles, init_particles_count * sizeof(Particle)));
    CUDA_CALL(cudaMemcpy(d_particles, init_particles, init_particles_count * sizeof(Particle), cudaMemcpyHostToDevice));

    // Allocate a histogram to track how many particles contribute to each pixel
    CUDA_CALL(cudaMalloc(&d_pixel_contribs, out_image_width * out_image_height * sizeof(unsigned int)));
    // Allocate an index to track where data for each pixel's contributing colour starts/ends
    CUDA_CALL(cudaMalloc(&d_pixel_index, (out_image_width * out_image_height + 1) * sizeof(unsigned int)));
    // Init a buffer to store colours contributing to each pixel into (allocated in stage 2)
    d_pixel_contrib_colours = 0;
    // Init a buffer to store depth of colours contributing to each pixel into (allocated in stage 2)
    d_pixel_contrib_depth = 0;
    // This tracks the number of contributes the two above buffers are allocated for, init 0
    cuda_pixel_contrib_count = 0;

    // Allocate output image
    cuda_output_image_width = (int)out_image_width;
    cuda_output_image_height = (int)out_image_height;
    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_WIDTH, &cuda_output_image_width, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_HEIGHT, &cuda_output_image_height, sizeof(int)));
    const int CHANNELS = 3;  // RGB
    CUDA_CALL(cudaMalloc(&d_output_image_data, cuda_output_image_width * cuda_output_image_height * CHANNELS * sizeof(unsigned char)));
}

__global__ void stage1(Particle* d_particles, unsigned int* d_pixel_contribs) {

    
    // Update each particle & calculate how many particles contribute to each image
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Compute bounding box [inclusive-inclusive]
    int x_min = (int)roundf(d_particles[i].location[0] - d_particles[i].radius);
    int y_min = (int)roundf(d_particles[i].location[1] - d_particles[i].radius);
    int x_max = (int)roundf(d_particles[i].location[0] + d_particles[i].radius);
    int y_max = (int)roundf(d_particles[i].location[1] + d_particles[i].radius);
    // Clamp bounding box to image bounds
    x_min = x_min < 0 ? 0 : x_min;
    y_min = y_min < 0 ? 0 : y_min;
    x_max = x_max >= D_OUTPUT_IMAGE_WIDTH ? D_OUTPUT_IMAGE_WIDTH - 1 : x_max;
    y_max = y_max >= D_OUTPUT_IMAGE_HEIGHT ? D_OUTPUT_IMAGE_HEIGHT - 1 : y_max;
    // For each pixel in the bounding box, check that it falls within the radius
    for (int x = x_min; x <= x_max; ++x) {
        for (int y = y_min; y <= y_max; ++y) {
            const float x_ab = (float)x + 0.5f - d_particles[i].location[0];
            const float y_ab = (float)y + 0.5f - d_particles[i].location[1];
            const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
            if (pixel_distance <= d_particles[i].radius) {
                const unsigned int pixel_offset = y * D_OUTPUT_IMAGE_WIDTH + x;
                atomicAdd(&d_pixel_contribs[pixel_offset], 1);

            }
        }
    }
}
void cuda_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // You will need to copy the data back to host before passing to these functions
    // skip_pixel_contribs(particles, particles_count, return_pixel_contribs, out_image_width, out_image_height);
    const int block_size = 256;
    const int num_grids = (cuda_particles_count + block_size - 1) / block_size;
    CUDA_CALL(cudaMemset(d_pixel_contribs, 0, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int)));
    stage1 << <num_grids, block_size >> > (d_particles, d_pixel_contribs);


#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    Particle* particles;
    unsigned int* pixel_contribs;
    particles = (Particle*)malloc(cuda_particles_count * sizeof(Particle));
    CUDA_CALL(cudaMemcpy(particles, d_particles, cuda_particles_count * sizeof(Particle), cudaMemcpyDeviceToHost));
    pixel_contribs = (unsigned int*)malloc(cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int));
    CUDA_CALL(cudaMemcpy(pixel_contribs, d_pixel_contribs, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    validate_pixel_contribs(particles, cuda_particles_count, pixel_contribs, cuda_output_image_width, cuda_output_image_height);
#endif
}


__global__ void store_colors_depths_kernel(Particle* particles, unsigned int particles_count, CImage output_image, unsigned int* pixel_index, unsigned int* pixel_contribs, unsigned char* pixel_contrib_colours, float* pixel_contrib_depth) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particles_count) return;

    // Compute bounding box [inclusive-inclusive]
    int x_min = (int)roundf(particles[i].location[0] - particles[i].radius);
    int y_min = (int)roundf(particles[i].location[1] - particles[i].radius);
    int x_max = (int)roundf(particles[i].location[0] + particles[i].radius);
    int y_max = (int)roundf(particles[i].location[1] + particles[i].radius);

    // Clamp bounding box to image bounds
    x_min = x_min < 0 ? 0 : x_min;
    y_min = y_min < 0 ? 0 : y_min;
    x_max = x_max >= output_image.width ? output_image.width - 1 : x_max;
    y_max = y_max >= output_image.height ? output_image.height - 1 : y_max;
    
    for (int x = x_min; x <= x_max; ++x) {
        for (int y = y_min; y <= y_max; ++y) {
            const float x_ab = (float)x + 0.5f - particles[i].location[0];
            const float y_ab = (float)y + 0.5f - particles[i].location[1];
            const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
            if (pixel_distance <= particles[i].radius) {
                const unsigned int pixel_offset = y * output_image.width + x;

                // Use atomicAdd to increment the value in pixel_contribs safely and get the previous value
                unsigned int storage_offset = atomicAdd(pixel_contribs + pixel_offset, 1);
                storage_offset += pixel_index[pixel_offset];

                // Copy data to pixel_contrib buffers
                printf("hi");
                memcpy(pixel_contrib_colours + (4 * storage_offset), particles[i].color, 4 * sizeof(unsigned char));
                memcpy(pixel_contrib_depth + storage_offset, &particles[i].location[2], sizeof(float));
            }
        }
    }
    //__syncthreads();
    
}

// Bitonic sort kernel for sorting pairs
__device__ void cuda_sort_pairs(float* keys, unsigned char* colors, int size) {
    int tid = threadIdx.x;

    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;

            if (ixj > tid) {
                if ((tid & k) == 0) {
                    if (keys[tid] > keys[ixj]) {
                        float temp_key = keys[tid];
                        keys[tid] = keys[ixj];
                        keys[ixj] = temp_key;

                        for (int c = 0; c < 4; ++c) {
                            unsigned char temp_color = colors[4 * tid + c];
                            colors[4 * tid + c] = colors[4 * ixj + c];
                            colors[4 * ixj + c] = temp_color;
                        }
                    }
                }
                else {
                    if (keys[tid] < keys[ixj]) {
                        float temp_key = keys[tid];
                        keys[tid] = keys[ixj];
                        keys[ixj] = temp_key;

                        for (int c = 0; c < 4; ++c) {
                            unsigned char temp_color = colors[4 * tid + c];
                            colors[4 * tid + c] = colors[4 * ixj + c];
                            colors[4 * ixj + c] = temp_color;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

// CUDA kernel for pair sorting
__global__ void pair_sort_kernel(int image_width, int image_height, unsigned int* pixel_index, unsigned char* pixel_contrib_colours, float* pixel_contrib_depth) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= image_width * image_height) return;

    int left = pixel_index[i];
    int right = pixel_index[i + 1] - 1;

    int size = right - left + 1;

    if (size > 1) {
        float* keys = pixel_contrib_depth + left;
        unsigned char* colors = pixel_contrib_colours + 4 * left;

        cuda_sort_pairs(keys, colors, size);
    }
}



void cuda_stage2() {
    // Optionally during development call the skip function/s with the correct inputs to skip this stage
    // You will need to copy the data back to host before passing to these functions
    // skip_pixel_index(pixel_contribs, return_pixel_index, out_image_width, out_image_height);
    // skip_sorted_pairs(particles, particles_count, pixel_index, out_image_width, out_image_height, return_pixel_contrib_colours, return_pixel_contrib_depth);
    // Exclusive prefix sum across the histogram to create an index
    // 
    thrust::device_ptr<unsigned int> d_pixel_contribs_ptr(d_pixel_contribs);
    thrust::device_ptr<unsigned int> d_pixel_index_ptr(d_pixel_index);
    thrust::exclusive_scan(d_pixel_contribs_ptr, d_pixel_contribs_ptr + (cuda_output_image_width * cuda_output_image_height), d_pixel_index_ptr);
    unsigned int total_contribs;
    CUDA_CALL(cudaMemcpy(&total_contribs, d_pixel_index + cuda_output_image.width * cuda_output_image.height, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    if (total_contribs > cuda_pixel_contrib_count) {
        // (Re)Allocate colour storage
        if (d_pixel_contrib_colours) CUDA_CALL(cudaFree(d_pixel_contrib_colours));
        if (d_pixel_contrib_depth) CUDA_CALL(cudaFree(d_pixel_contrib_depth));

        CUDA_CALL(cudaMalloc((void**)&d_pixel_contrib_colours, total_contribs * 4 * sizeof(unsigned char)));
        CUDA_CALL(cudaMalloc((void**)&d_pixel_contrib_depth, total_contribs * sizeof(float)));

        cuda_pixel_contrib_count = total_contribs;
    }

    // Reset the pixel contributions histogram
    CUDA_CALL(cudaMemset(d_pixel_contribs, 0, cuda_output_image.width * cuda_output_image.height * sizeof(unsigned int)));

    // Launch the CUDA kernel for storing colors and depths
    const int blockSize = 256;
    const int gridSize = (cuda_particles_count + blockSize - 1) / blockSize;

    //print gridSize
    store_colors_depths_kernel << <gridSize, blockSize >> > (d_particles, cuda_particles_count, cuda_output_image, d_pixel_index, d_pixel_contribs, d_pixel_contrib_colours, d_pixel_contrib_depth);
    //CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    printf("grid2");
    //const int gridSize2 = (cuda_output_image.width * cuda_output_image.height + blockSize - 1) / blockSize;
    pair_sort_kernel << <gridSize, blockSize >> > (cuda_output_image.width, cuda_output_image.height, d_pixel_index, d_pixel_contrib_colours, d_pixel_contrib_depth);
    //CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    
#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // Note: Only validate_equalised_histogram() MUST be uncommented, the others are optional
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)

    /*unsigned int* pixel_contribs;
    unsigned int* pixel_index;

    pixel_contribs = (unsigned int*)malloc(cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int));
    CUDA_CALL(cudaMemcpy(pixel_contribs, d_pixel_contribs, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    pixel_index = (unsigned int*)malloc((cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int));
    CUDA_CALL(cudaMemcpy(pixel_index, d_pixel_index, (cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    validate_pixel_index(pixel_contribs, pixel_index, cuda_output_image_width, cuda_output_image_height);

    Particle* particles;
    unsigned char* pixel_contrib_colours;
    float* pixel_contrib_depth;

    particles = (Particle*)malloc(cuda_particles_count * sizeof(Particle));
    CUDA_CALL(cudaMemcpy(particles, d_particles, cuda_particles_count * sizeof(Particle), cudaMemcpyDeviceToHost));

    const unsigned int TOTAL_CONTRIBS = pixel_index[cuda_output_image_width * cuda_output_image_height];
    pixel_contrib_colours = (unsigned char*)malloc(TOTAL_CONTRIBS * 4 * sizeof(unsigned char));
    CUDA_CALL(cudaMemcpy(pixel_contrib_colours, d_pixel_contrib_colours, TOTAL_CONTRIBS * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    pixel_contrib_depth = (float*)malloc(TOTAL_CONTRIBS * sizeof(float));
    CUDA_CALL(cudaMemcpy(pixel_contrib_depth, d_pixel_contrib_depth, TOTAL_CONTRIBS * sizeof(float), cudaMemcpyDeviceToHost));

    validate_sorted_pairs(particles, cuda_particles_count, pixel_index, cuda_output_image_width, cuda_output_image_height, pixel_contrib_colours, pixel_contrib_depth);*/




    //validate_pixel_index(pixel_contribs, d_pixel_index, cuda_output_image_width, cuda_output_image_height);
    //validate_sorted_pairs(particles, cuda_particles_count, d_pixel_index, cuda_output_image_width, cuda_output_image_height, pixel_contrib_colours, pixel_contrib_depth);
    //validate_equalised_histogram(pixel_index, output_image_width, cpu_output_image_height);
    
#endif    
}

//__global__ void stage3(unsigned char* d_output_image_data, unsigned char* d_pixel_contrib_colours, unsigned int* d_pixel_index) {
//
//    // Order dependent blending into output image
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    const int idx_start = d_pixel_index[i];
//    const int idx_end = d_pixel_index[i + 1];
//    const float inv_255 = 1.0f / 255.0f;
//
//    // Load a chunk of d_pixel_contrib_colours into shared memory
//    __shared__ unsigned char shared_pixel_contrib_colours[256 * 4];
//    for (int k = threadIdx.x * 4; k < threadIdx.x * 4 + 4; ++k) {
//        for (int j = idx_start; j < idx_end; ++j) {
//            shared_pixel_contrib_colours[(j - idx_start) * 4 + k - threadIdx.x * 4] = d_pixel_contrib_colours[j * 4 + k];
//        }
//    }
//    __syncthreads();
//
//    // Blend each of the red/green/blue colours according to the below blend formula
//    // dest = src * opacity + dest * (1 - opacity);
//    float r = d_output_image_data[(i * 3) + 0];
//    float g = d_output_image_data[(i * 3) + 1];
//    float b = d_output_image_data[(i * 3) + 2];
//    for (int j = 0; j < idx_end - idx_start; ++j) {
//        const float opacity = (float)shared_pixel_contrib_colours[j * 4 + 3] * inv_255;
//        r = shared_pixel_contrib_colours[j * 4 + 0] * opacity + r * (1 - opacity);
//        g = shared_pixel_contrib_colours[j * 4 + 1] * opacity + g * (1 - opacity);
//        b = shared_pixel_contrib_colours[j * 4 + 2] * opacity + b * (1 - opacity);
//    }
//    d_output_image_data[(i * 3) + 0] = (unsigned char)r;
//    d_output_image_data[(i * 3) + 1] = (unsigned char)g;
//    d_output_image_data[(i * 3) + 2] = (unsigned char)b;
//}

__global__ void stage3(unsigned char* d_output_image_data, unsigned char* d_pixel_contrib_colours, unsigned int* d_pixel_index) {

    // Order dependent blending into output image
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int j = d_pixel_index[i]; j < d_pixel_index[i + 1]; ++j) {
        // Blend each of the red/green/blue colours according to the below blend formula
        // dest = src * opacity + dest * (1 - opacity);
        const float opacity = (float)d_pixel_contrib_colours[j * 4 + 3] / (float)255;
        d_output_image_data[(i * 3) + 0] = (unsigned char)((float)d_pixel_contrib_colours[j * 4 + 0] * opacity + (float)d_output_image_data[(i * 3) + 0] * (1 - opacity));
        d_output_image_data[(i * 3) + 1] = (unsigned char)((float)d_pixel_contrib_colours[j * 4 + 1] * opacity + (float)d_output_image_data[(i * 3) + 1] * (1 - opacity));
        d_output_image_data[(i * 3) + 2] = (unsigned char)((float)d_pixel_contrib_colours[j * 4 + 2] * opacity + (float)d_output_image_data[(i * 3) + 2] * (1 - opacity));
        // cpu_pixel_contrib_colours is RGBA
        // cpu_output_image.data is RGB (final output image does not have an alpha channel!)
    }
}

void cuda_stage3() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // You will need to copy the data back to host before passing to these functions
    // skip_blend(pixel_index, pixel_contrib_colours, return_output_image);
    const int CHANNEL = 3;
    const int block_size = 256;
    const int num_grid = (cuda_particles_count + block_size - 1) / block_size;
    CUDA_CALL(cudaMemset(d_output_image_data, 255, cuda_output_image_width * cuda_output_image_height * CHANNEL * sizeof(unsigned char)));
    stage3 << <num_grid, block_size >> > (d_output_image_data, d_pixel_contrib_colours, d_pixel_index);


#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    CImage output_image;
    unsigned int* pixel_index;
    unsigned char* pixel_contrib_colours;

    pixel_index = (unsigned int*)malloc((cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int));
    CUDA_CALL(cudaMemcpy(pixel_index, d_pixel_index, (cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    const unsigned int TOTAL_CONTRIBS = pixel_index[cuda_output_image_width * cuda_output_image_height];

    pixel_contrib_colours = (unsigned char*)malloc(TOTAL_CONTRIBS * 4 * sizeof(unsigned char));
    CUDA_CALL(cudaMemcpy(pixel_contrib_colours, d_pixel_contrib_colours, TOTAL_CONTRIBS * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    output_image.data = (unsigned char*)malloc(cuda_output_image_width * cuda_output_image_height * CHANNEL * sizeof(unsigned char));
    output_image.width = cuda_output_image_width;
    output_image.height = cuda_output_image_height;
    output_image.channels = CHANNEL;

    CUDA_CALL(cudaMemcpy(output_image.data, d_output_image_data, cuda_output_image_width * cuda_output_image_height * CHANNEL * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    validate_blend(pixel_index, pixel_contrib_colours, &output_image);
#endif    
}

void cuda_end(CImage *output_image) {
    // This function matches the provided cuda_begin(), you may change it if desired

    // Store return value
    const int CHANNELS = 3;
    output_image->width = cuda_output_image_width;
    output_image->height = cuda_output_image_height;
    output_image->channels = CHANNELS;
    CUDA_CALL(cudaMemcpy(output_image->data, d_output_image_data, cuda_output_image_width * cuda_output_image_height * CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    // Release allocations
    CUDA_CALL(cudaFree(d_pixel_contrib_depth));
    CUDA_CALL(cudaFree(d_pixel_contrib_colours));
    CUDA_CALL(cudaFree(d_output_image_data));
    CUDA_CALL(cudaFree(d_pixel_index));
    CUDA_CALL(cudaFree(d_pixel_contribs));
    CUDA_CALL(cudaFree(d_particles));
    // Return ptrs to nullptr
    d_pixel_contrib_depth = 0;
    d_pixel_contrib_colours = 0;
    d_output_image_data = 0;
    d_pixel_index = 0;
    d_pixel_contribs = 0;
    d_particles = 0;
}

