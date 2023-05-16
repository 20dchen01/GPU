#include "cuda.cuh"
#include "helper.h"
#include "cuda_runtime.h"
#include <cstring>
#include <cmath>
#include <device_launch_parameters.h>
#include <cub/device/device_scan.cuh>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#define BLOCK_SIZE 1024
///
/// Algorithm storage
///
/// 
/// 
__device__ void cpu_sort_pairs(float* keys_start, unsigned char* colours_start, int first, int last);

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


__global__ void stage1(Particle* d_particles, unsigned int* d_pixel_contribs, unsigned int cuda_particles_count) {
    // Declare shared memory 
    __shared__ Particle shared_particles[BLOCK_SIZE];


    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cuda_particles_count; i += blockDim.x * gridDim.x) {
        // Load data into shared memory
        shared_particles[threadIdx.x] = d_particles[i];
        __syncthreads();

        int x_min = (int)roundf(shared_particles[threadIdx.x].location[0] - shared_particles[threadIdx.x].radius);
        int y_min = (int)roundf(shared_particles[threadIdx.x].location[1] - shared_particles[threadIdx.x].radius);
        int x_max = (int)roundf(shared_particles[threadIdx.x].location[0] + shared_particles[threadIdx.x].radius);
        int y_max = (int)roundf(shared_particles[threadIdx.x].location[1] + shared_particles[threadIdx.x].radius);

        // Clamp bounding box to image bounds
        x_min = x_min < 0 ? 0 : x_min;
        y_min = y_min < 0 ? 0 : y_min;
        x_max = x_max >= D_OUTPUT_IMAGE_WIDTH ? D_OUTPUT_IMAGE_WIDTH - 1 : x_max;
        y_max = y_max >= D_OUTPUT_IMAGE_HEIGHT ? D_OUTPUT_IMAGE_HEIGHT - 1 : y_max;

        // For each pixel in the bounding box, check if it falls within the radius
        for (int x = x_min; x <= x_max; ++x) {
            for (int y = y_min; y <= y_max; ++y) {
                const float x_ab = (float)x + 0.5f - shared_particles[threadIdx.x].location[0];
                const float y_ab = (float)y + 0.5f - shared_particles[threadIdx.x].location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);

                if (pixel_distance <= shared_particles[threadIdx.x].radius) {
                    const unsigned int pixel_offset = y * D_OUTPUT_IMAGE_WIDTH + x;
                    atomicAdd(&d_pixel_contribs[pixel_offset], 1);
                }
            }
        }

        __syncthreads();
    }
}

void cuda_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // You will need to copy the data back to host before passing to these functions
    // skip_pixel_contribs(particles, particles_count, return_pixel_contribs, out_image_width, out_image_height);

    int gridSize = (cuda_particles_count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CALL(cudaMemset(d_pixel_contribs, 0, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int)));
    stage1 << <gridSize, BLOCK_SIZE >> > (d_particles, d_pixel_contribs, cuda_particles_count);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    Particle* particles;
    unsigned int* pixel_contribs;
    particles = (Particle*)malloc(cuda_particles_count * sizeof(Particle));
    CUDA_CALL(cudaMemcpy(particles, d_particles, cuda_particles_count * sizeof(Particle), cudaMemcpyDeviceToHost));
    pixel_contribs = (unsigned int*)malloc(cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int));
    CUDA_CALL(cudaMemcpy(pixel_contribs, d_pixel_contribs, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    validate_pixel_contribs(particles, cuda_particles_count, pixel_contribs, cuda_output_image_width, cuda_output_image_height);
    free(particles);
    free(pixel_contribs);
#endif
}

__global__ void stage2(Particle* d_particles, unsigned int* d_pixel_contribs, unsigned int cuda_particles_count, unsigned char* d_pixel_contrib_colours, float* d_pixel_contrib_depth,
    unsigned int* d_pixel_index, unsigned int cuda_pixel_contrib_count) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cuda_particles_count; i += blockDim.x * gridDim.x) {
    if (i < cuda_particles_count) {
        Particle particle = d_particles[i];

        int x_min = (int)roundf(particle.location[0] - particle.radius);
        int y_min = (int)roundf(particle.location[1] - particle.radius);
        int x_max = (int)roundf(particle.location[0] + particle.radius);
        int y_max = (int)roundf(particle.location[1] + particle.radius);
        // Clamp bounding box to image bounds
        x_min = max(x_min, 0);
        y_min = max(y_min, 0);
        x_max = min(x_max, D_OUTPUT_IMAGE_WIDTH - 1);
        y_max = min(y_max, D_OUTPUT_IMAGE_HEIGHT - 1);

        // Store data for every pixel within the bounding box that falls within the radius
        for (int x = x_min; x <= x_max; ++x) {
            for (int y = y_min; y <= y_max; ++y) {
                const float x_ab = (float)x + 0.5f - particle.location[0];
                const float y_ab = (float)y + 0.5f - particle.location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                if (pixel_distance <= particle.radius) {
                    const unsigned int pixel_offset = y * D_OUTPUT_IMAGE_WIDTH + x;

                    unsigned int storage_offset = atomicAdd(&d_pixel_contribs[pixel_offset], 1);
                    unsigned char* pixel_contrib_colours_ptr = d_pixel_contrib_colours + (4 * (d_pixel_index[pixel_offset] + storage_offset));
                    float* pixel_contrib_depth_ptr = d_pixel_contrib_depth + (d_pixel_index[pixel_offset] + storage_offset);

                    pixel_contrib_colours_ptr[0] = particle.color[0];
                    pixel_contrib_colours_ptr[1] = particle.color[1];
                    pixel_contrib_colours_ptr[2] = particle.color[2];
                    pixel_contrib_colours_ptr[3] = particle.color[3];

                    *pixel_contrib_depth_ptr = particle.location[2];
                }
            }
        }
    }
}


__global__ void stageSort(unsigned int* d_pixel_index, unsigned char* d_pixel_contrib_colours, float* d_pixel_contrib_depth, int n) {
    //Pair sort the colours contributing to each pixel based on ascending depth
    // Pair sort the colours which contribute to a single pigment
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        cpu_sort_pairs(
            d_pixel_contrib_depth,
            d_pixel_contrib_colours,
            d_pixel_index[i],
            d_pixel_index[i + 1] - 1
        );
    }
}


void cuda_stage2() {
    // Optionally during development call the skip function/s with the correct inputs to skip this stage
    // You will need to copy the data back to host before passing to these functions
    // skip_pixel_index(pixel_contribs, return_pixel_index, out_image_width, out_image_height);
    // skip_sorted_pairs(particles, particles_count, pixel_index, out_image_width, out_image_height, return_pixel_contrib_colours, return_pixel_contrib_depth);

    unsigned int* pixel_index;
    pixel_index = (unsigned int*)malloc((cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int));
    //sum with thrust
    /*thrust::device_ptr<unsigned int> dev_in_ptr(d_in);
    thrust::device_ptr<unsigned int> dev_out_ptr(d_out);
    thrust::exclusive_scan(dev_in_ptr, dev_in_ptr + num_items, dev_out_ptr); */
    //sum with cub
    void* dev_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(dev_temp_storage, temp_storage_bytes, d_pixel_contribs, d_pixel_index, cuda_output_image_width * cuda_output_image_height + 1);
    cudaMalloc(&dev_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(dev_temp_storage, temp_storage_bytes, d_pixel_contribs, d_pixel_index, cuda_output_image_width * cuda_output_image_height + 1);
    
    
    cudaMemcpy(pixel_index, d_pixel_index, (cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);


    // Recover the total from the index
    const unsigned int TOTAL_CONTRIBS = pixel_index[cuda_output_image_width * cuda_output_image_height];
    if (TOTAL_CONTRIBS > cuda_pixel_contrib_count) {
        // (Re)Allocate colour storage
        if (d_pixel_contrib_colours) CUDA_CALL(cudaFree(d_pixel_contrib_colours));
        if (d_pixel_contrib_depth) CUDA_CALL(cudaFree(d_pixel_contrib_depth));
        CUDA_CALL(cudaMalloc(&d_pixel_contrib_colours, TOTAL_CONTRIBS * 4 * sizeof(unsigned char)));
        //d_pixel_contrib_colours = (unsigned char*)malloc(TOTAL_CONTRIBS * 4 * sizeof(unsigned char));
        CUDA_CALL(cudaMalloc(&d_pixel_contrib_depth, TOTAL_CONTRIBS * sizeof(float)));
        //d_pixel_contrib_depth = (float*)malloc(TOTAL_CONTRIBS * sizeof(float));
        cuda_pixel_contrib_count = TOTAL_CONTRIBS;
    }


    // Reset the pixel contributions histogram
    int gridSize = (cuda_particles_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CALL(cudaMemset(d_pixel_contribs, 0, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int)));
    stage2 << <gridSize, BLOCK_SIZE >> > (d_particles, d_pixel_contribs, cuda_particles_count, d_pixel_contrib_colours, d_pixel_contrib_depth, d_pixel_index, cuda_pixel_contrib_count);

    int gridSizeSort = (cuda_output_image_width * cuda_output_image_height + BLOCK_SIZE - 1) / BLOCK_SIZE;

    stageSort << <gridSizeSort, BLOCK_SIZE >> > (d_pixel_index, d_pixel_contrib_colours, d_pixel_contrib_depth, cuda_output_image_width * cuda_output_image_height);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // Note: Only validate_equalised_histogram() MUST be uncommented, the others are optional
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)

    Particle* particles;
    particles = (Particle*)malloc(cuda_particles_count * sizeof(Particle));
    CUDA_CALL(cudaMemcpy(particles, d_particles, cuda_particles_count * sizeof(Particle), cudaMemcpyDeviceToHost));

    unsigned char* pixel_contrib_colours;
    pixel_contrib_colours = (unsigned char*)malloc(TOTAL_CONTRIBS * 4 * sizeof(unsigned char));
    CUDA_CALL(cudaMemcpy(pixel_contrib_colours, d_pixel_contrib_colours, TOTAL_CONTRIBS * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    float* pixel_contrib_depth;
    pixel_contrib_depth = (float*)malloc(TOTAL_CONTRIBS * sizeof(float));
    CUDA_CALL(cudaMemcpy(pixel_contrib_depth, d_pixel_contrib_depth, TOTAL_CONTRIBS * sizeof(float), cudaMemcpyDeviceToHost));

    unsigned int* pixel_contribs;
    pixel_contribs = (unsigned int*)malloc(cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int));
    CUDA_CALL(cudaMemcpy(pixel_contribs, d_pixel_contribs, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    validate_pixel_index(pixel_contribs, pixel_index, cuda_output_image_width, cuda_output_image_height);
    validate_sorted_pairs(particles, cuda_particles_count, pixel_index, cuda_output_image_width, cuda_output_image_height, pixel_contrib_colours, pixel_contrib_depth);

    free(particles);
    free(pixel_contrib_colours);
    free(pixel_contrib_depth);
    free(pixel_contribs);

#endif    
}


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
    // Memset output image data to 255 (white)
    const int CHANNELS = 3;  // RGB
    int gridSize = (cuda_output_image_width * cuda_output_image_height + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CALL(cudaMemset(d_output_image_data, 255, cuda_output_image_width * cuda_output_image_height * CHANNELS * sizeof(unsigned char)));
    stage3 << <gridSize, BLOCK_SIZE >> > (d_output_image_data, d_pixel_contrib_colours, d_pixel_index);


#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    //validate_blend(pixel_index, pixel_contrib_colours, &output_image);
    CImage output_image;
    unsigned int* pixel_index;
    unsigned char* pixel_contrib_colours;

    pixel_index = (unsigned int*)malloc((cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int));
    CUDA_CALL(cudaMemcpy(pixel_index, d_pixel_index, (cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    const unsigned int TOTAL_CONTRIBS = pixel_index[cuda_output_image_width * cuda_output_image_height];


    pixel_contrib_colours = (unsigned char*)malloc(TOTAL_CONTRIBS * 4 * sizeof(unsigned char));
    CUDA_CALL(cudaMemcpy(pixel_contrib_colours, d_pixel_contrib_colours, TOTAL_CONTRIBS * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    output_image.data = (unsigned char*)malloc(cuda_output_image_width * cuda_output_image_height * CHANNELS * sizeof(unsigned char));
    output_image.width = cuda_output_image_width;
    output_image.height = cuda_output_image_height;
    output_image.channels = CHANNELS;


    CUDA_CALL(cudaMemcpy(output_image.data, d_output_image_data, cuda_output_image_width * cuda_output_image_height * CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    validate_blend(pixel_index, pixel_contrib_colours, &output_image);

#endif    

}
void cuda_end(CImage* output_image) {
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

__device__ void cpu_sort_pairs(float* keys_start, unsigned char* colours_start, const int first, const int last) {
    // Based on https://www.tutorialspoint.com/explain-the-quick-sort-technique-in-c-language
    int i, j, pivot;
    float depth_t;
    unsigned char color_t[4];
    if (first < last) {
        pivot = first;
        i = first;
        j = last;
        while (i < j) {
            while (keys_start[i] <= keys_start[pivot] && i < last)
                i++;
            while (keys_start[j] > keys_start[pivot])
                j--;
            if (i < j) {
                // Swap key
                depth_t = keys_start[i];
                keys_start[i] = keys_start[j];
                keys_start[j] = depth_t;
                // Swap color
                memcpy(color_t, colours_start + (4 * i), 4 * sizeof(unsigned char));
                memcpy(colours_start + (4 * i), colours_start + (4 * j), 4 * sizeof(unsigned char));
                memcpy(colours_start + (4 * j), color_t, 4 * sizeof(unsigned char));
            }
        }
        // Swap key
        depth_t = keys_start[pivot];
        keys_start[pivot] = keys_start[j];
        keys_start[j] = depth_t;
        // Swap color
        memcpy(color_t, colours_start + (4 * pivot), 4 * sizeof(unsigned char));
        memcpy(colours_start + (4 * pivot), colours_start + (4 * j), 4 * sizeof(unsigned char));
        memcpy(colours_start + (4 * j), color_t, 4 * sizeof(unsigned char));
        // Recurse
        cpu_sort_pairs(keys_start, colours_start, first, j - 1);
        cpu_sort_pairs(keys_start, colours_start, j + 1, last);
    }
}
