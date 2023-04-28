#include "cuda.cuh"
#include "helper.h"
#include <device_launch_parameters.h>
#include "cuda_runtime.h"   
#include <cstring>
#include <cmath>

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
    printf("test0");
    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_WIDTH, &cuda_output_image_width, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_HEIGHT, &cuda_output_image_height, sizeof(int)));
    const int CHANNELS = 3;  // RGB
    CUDA_CALL(cudaMalloc(&d_output_image_data, cuda_output_image_width * cuda_output_image_height * CHANNELS * sizeof(unsigned char)));
}
__global__ void test(Particle* d_particles, unsigned int* d_pixel_contribs) {
    //memset(d_pixel_contribs, 0, D_OUTPUT_IMAGE_WIDTH * D_OUTPUT_IMAGE_HEIGHT * sizeof(unsigned int));
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
    const int num_blocks = (cuda_particles_count + block_size - 1) / block_size;
    CUDA_CALL(cudaMemset(d_pixel_contribs, 0, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int)));
    test << <num_blocks, block_size >> > (d_particles, d_pixel_contribs);




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

void cuda_stage2() {
    // Optionally during development call the skip function/s with the correct inputs to skip this stage
    // You will need to copy the data back to host before passing to these functions
    // skip_pixel_index(pixel_contribs, return_pixel_index, out_image_width, out_image_height);
    // skip_sorted_pairs(particles, particles_count, pixel_index, out_image_width, out_image_height, return_pixel_contrib_colours, return_pixel_contrib_depth);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // Note: Only validate_equalised_histogram() MUST be uncommented, the others are optional
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    // validate_pixel_index(pixel_contribs, pixel_index, output_image_width, cpu_output_image_height);
    // validate_sorted_pairs(particles, particles_count, pixel_index, output_image_width, cpu_output_image_height, pixel_contrib_colours, pixel_contrib_depth);
#endif    
}
void cuda_stage3() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // You will need to copy the data back to host before passing to these functions
    // skip_blend(pixel_index, pixel_contrib_colours, return_output_image);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    // validate_blend(pixel_index, pixel_contrib_colours, &output_image);
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