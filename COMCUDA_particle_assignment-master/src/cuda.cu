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
    // Load particle information into shared memory
    __shared__ Particle shared_particles[256];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_particles = min(blockDim.x, cuda_particles_count - blockIdx.x * blockDim.x);
    for (int j = threadIdx.x; j < num_particles; j += blockDim.x) {
        shared_particles[j] = d_particles[blockIdx.x * blockDim.x + j];
    }
    __syncthreads();

    // Compute lookup table for distance calculations
    __shared__ float dist_table[256 * 256];
    for (int j = threadIdx.x; j < 256; j += blockDim.x) {
        for (int k = 0; k < 256; ++k) {
            const float x_ab = (float)k + 0.5f - shared_particles[j].location[0];
            const float y_ab = (float)j + 0.5f - shared_particles[j].location[1];
            const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
            dist_table[j * 256 + k] = pixel_distance;
        }
    }
    __syncthreads();

    // Update pixel contributions
    for (int j = 0; j < num_particles; ++j) {
        const int x_min = max(0, (int)roundf(shared_particles[j].location[0] - shared_particles[j].radius));
        const int y_min = max(0, (int)roundf(shared_particles[j].location[1] - shared_particles[j].radius));
        const int x_max = min(cuda_output_image_width - 1, (int)roundf(shared_particles[j].location[0] + shared_particles[j].radius));
        const int y_max = min(cuda_output_image_height - 1, (int)roundf(shared_particles[j].location[1] + shared_particles[j].radius));
        const int pixel_offset_start = y_min * cuda_output_image_width + x_min;
        const int pixel_offset_end = (y_max + 1) * cuda_output_image_width - 1;
        const float radius = shared_particles[j].radius;
        for (int k = pixel_offset_start + threadIdx.x; k <= pixel_offset_end; k += blockDim.x) {
            const int x = k % cuda_output_image_width;
            const int y = k / cuda_output_image_width;
            const float pixel_distance = dist_table[(int)(y - shared_particles[j].location[1] + radius) * 256 + (int)(x - shared_particles[j].location[0] + radius)];
            if (pixel_distance <= radius) {
                atomicAdd(&d_pixel_contribs[k], 1);
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
    stage1 << <num_blocks, block_size >> > (d_particles, d_pixel_contribs);




#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    //Particle* particles;
    //unsigned int* pixel_contribs;
    //particles = (Particle*)malloc(cuda_particles_count * sizeof(Particle));
    //CUDA_CALL(cudaMemcpy(particles, d_particles, cuda_particles_count * sizeof(Particle), cudaMemcpyDeviceToHost));
    //pixel_contribs = (unsigned int*)malloc(cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int));
    
    //CUDA_CALL(cudaMemcpy(pixel_contribs, d_pixel_contribs, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int), cudaMemcpyDeviceToHost));


    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    //validate_pixel_contribs(particles, cuda_particles_count, pixel_contribs, cuda_output_image_width, cuda_output_image_height);
#endif
}

void cuda_stage2() {
    // Optionally during development call the skip function/s with the correct inputs to skip this stage
    // You will need to copy the data back to host before passing to these functions
    // skip_pixel_index(pixel_contribs, return_pixel_index, out_image_width, out_image_height);
    // skip_sorted_pairs(particles, particles_count, pixel_index, out_image_width, out_image_height, return_pixel_contrib_colours, return_pixel_contrib_depth);
    // Exclusive prefix sum across the histogram to create an index
    //d_pixel_index[0] = 0;
    //for (int i = 0; i < cuda_output_image_width * cuda_output_image_height; ++i) {
    //    d_pixel_index[i + 1] = d_pixel_index[i] + d_pixel_contribs[i];
    //}
    //// recover the total from the index
    //const unsigned int total_contribs = d_pixel_index[cuda_output_image_width * cuda_output_image_height];
    //if (total_contribs > cuda_pixel_contrib_count) {
    //    // (re)allocate colour storage
    //    if (d_pixel_contrib_colours) free(d_pixel_contrib_colours);
    //    if (d_pixel_contrib_depth) free(d_pixel_contrib_depth);
    //    d_pixel_contrib_colours = (unsigned char*)malloc(total_contribs * 4 * sizeof(unsigned char));
    //    d_pixel_contrib_depth = (float*)malloc(total_contribs * sizeof(float));
    //    cuda_pixel_contrib_count = total_contribs;
    //}

    //// reset the pixel contributions histogram
    //memset(d_pixel_contribs, 0, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int));
    //// store colours according to index
    //// for each particle, store a copy of the colour/depth in d_pixel_contribs for each contributed pixel
    //for (unsigned int i = 0; i < cuda_particles_count; ++i) {
    //    // compute bounding box [inclusive-inclusive]
    //    int x_min = (int)roundf(d_particles[i].location[0] - d_particles[i].radius);
    //    int y_min = (int)roundf(d_particles[i].location[1] - d_particles[i].radius);
    //    int x_max = (int)roundf(d_particles[i].location[0] + d_particles[i].radius);
    //    int y_max = (int)roundf(d_particles[i].location[1] + d_particles[i].radius);
    //    // clamp bounding box to image bounds
    //    x_min = x_min < 0 ? 0 : x_min;
    //    y_min = y_min < 0 ? 0 : y_min;
    //    x_max = x_max >= cuda_output_image_width ? cuda_output_image_width - 1 : x_max;
    //    y_max = y_max >= cuda_output_image_height ? cuda_output_image_height - 1 : y_max;
    //    // store data for every pixel within the bounding box that falls within the radius
    //    for (int x = x_min; x <= x_max; ++x) {
    //        for (int y = y_min; y <= y_max; ++y) {
    //            const float x_ab = (float)x + 0.5f - d_particles[i].location[0];
    //            const float y_ab = (float)y + 0.5f - d_particles[i].location[1];
    //            const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
    //            if (pixel_distance <= d_particles[i].radius) {
    //                const unsigned int pixel_offset = y * cuda_output_image_width + x;
    //                // offset into d_pixel_contrib buffers is index + histogram
    //                // increment d_pixel_contribs, so next contributor stores to correct offset
    //                const unsigned int storage_offset = d_pixel_index[pixel_offset] + (d_pixel_contribs[pixel_offset]++);
    //                // copy data to d_pixel_contrib buffers
    //                memcpy(d_pixel_contrib_colours + (4 * storage_offset), d_particles[i].color, 4 * sizeof(unsigned char));
    //                memcpy(d_pixel_contrib_depth + storage_offset, &d_particles[i].location[2], sizeof(float));
    //            }
    //        }
    //    }
    //}

    //// pair sort the colours contributing to each pixel based on ascending depth
    //for (int i = 0; i < cuda_output_image_width * cuda_output_image_height; ++i) {
    //    // pair sort the colours which contribute to a single pigment
    //    cuda_sort_pairs(
    //        d_pixel_contrib_depth,
    //        d_pixel_contrib_colours,
    //        d_pixel_index[i],
    //        d_pixel_index[i + 1] - 1
    //    );
    //}
#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // Note: Only validate_equalised_histogram() MUST be uncommented, the others are optional
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    // validate_pixel_index(pixel_contribs, pixel_index, output_image_width, cpu_output_image_height);
    // validate_sorted_pairs(particles, particles_count, pixel_index, output_image_width, cpu_output_image_height, pixel_contrib_colours, pixel_contrib_depth);
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

void cuda_sort_pairs(float* keys_start, unsigned char* colours_start, const int first, const int last) {
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
        cuda_sort_pairs(keys_start, colours_start, first, j - 1);
        cuda_sort_pairs(keys_start, colours_start, j + 1, last);
    }
}