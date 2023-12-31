/* COMP.CE.350 Parallelization Excercise 2023
   Copyright (c) 2023 Topi Leppanen topi.leppanen@tuni.fi
                      Jan Solanti

VERSION 23.0 - Created
*/

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// OpenCL includes
#include <CL/cl.h>
#include "opencl_util.h"

#include "util.h"

// Is used to find out frame times
int previousFinishTime = 0;
unsigned int frameNumber = 0;
unsigned int seed = 0;

int8_t first_iter = 1;

typedef struct {
    cl_context context;
    cl_program program;
    cl_command_queue cmdQueue;
    cl_mem buf_input;
    cl_mem buf_sobel_x;
    cl_mem buf_sobel_y;
    cl_mem buf_phase;
    cl_mem buf_magnitude;
    cl_mem buf_output;
    cl_kernel kernel_sobel;
    cl_kernel kernel_PnM;
    cl_kernel kernel_nonMax;
} OpenCLObjects;

typedef struct {
    uint16_t x;
    uint16_t y;
} coord_t;

const coord_t neighbour_offsets[8] = {
    {-1, -1}, {0, -1},  {+1, -1}, {-1, 0},
    {+1, 0},  {-1, +1}, {0, +1},  {+1, +1},
};

// ## You may add your own variables here ##

// Used to store the opencl object handles
OpenCLObjects ocl_obj;
OpenCLObjects *ocl = &ocl_obj;

void
edgeTracing(uint8_t *restrict image, size_t width, size_t height) {
    // Uses a stack-based approach to incrementally spread the YES
    // pixels to every (8) neighbouring MAYBE pixel.
    //
    // Modifies the pixels in-place.
    //
    // Since the same pixel is never added to the stack twice,
    // the maximum stack size is quaranteed to never be above
    // the image size and stack overflow should be impossible
    // as long as stack size is 2*2*image_size (2 16-bit coordinates per
    // pixel).
    coord_t *tracing_stack = malloc(width * height * sizeof(coord_t));
    coord_t *tracing_stack_pointer = tracing_stack;

    // LOOP 4.1
    for (uint16_t y = 0; y < height; y++) {
        // LOOP 4.2
        for (uint16_t x = 0; x < width; x++) {
            // Collect all YES pixels into the stack
            if (image[y * width + x] == 255) {
                coord_t yes_pixel = {x, y};
                *tracing_stack_pointer = yes_pixel;
                tracing_stack_pointer++;  // increments by sizeof(coord_t)
            }
        }
    }

    // Empty the tracing stack one-by-one
    // LOOP 4.3
    while (tracing_stack_pointer != tracing_stack) {
        tracing_stack_pointer--;
        coord_t known_edge = *tracing_stack_pointer;
        // LOOP 4.4
        for (int k = 0; k < 8; k++) {
            coord_t dir_offs = neighbour_offsets[k];
            coord_t neighbour = {
                known_edge.x + dir_offs.x, known_edge.y + dir_offs.y};

            // Clamp to edge to prevent the algorithm from leaving the image.
            // Not using the idx()-function, since we want to preserve the x
            // and y on their own, since the pixel might be added to the stack
            // in the end.
            if (neighbour.x < 0) neighbour.x = 0;
            if (neighbour.x >= width) neighbour.x = width - 1;
            if (neighbour.y < 0) neighbour.y = 0;
            if (neighbour.y >= height) neighbour.y = height - 1;

            // Only MAYBE neighbours are potential edges
            if (image[neighbour.y * width + neighbour.x] == 127) {
                // Convert MAYBE to YES
                image[neighbour.y * width + neighbour.x] = 255;

                // Add the newly added pixel to stack, so changes will
                // propagate
                *tracing_stack_pointer = neighbour;
                tracing_stack_pointer++;
            }
        }
    }
    // Clear all remaining MAYBE pixels to NO, these were not reachable from
    // any YES pixels
    // LOOP 4.5
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        // LOOP 4.6
        for (int x = 0; x < width; x++) {
            if (image[y * width + x] == 127) {
                image[y * width + x] = 0;
            }
        }
    }
}

void
cannyEdgeDetection(
    uint8_t *restrict input, size_t width, size_t height,
    uint16_t threshold_lower, uint16_t threshold_upper,
    uint8_t *restrict output, double *restrict runtimes) {

    uint64_t times[5];

    // OPENCL EXECUTION
    cl_int status;
    double execute_time;
    cl_event timing_event;
    size_t globalWorkSize[2];

    globalWorkSize[0] = width;
    globalWorkSize[1] = height;

    size_t datasize = sizeof(uint8_t) * width * height;

    // Write input array A to the device buffer bufferA
    status = clEnqueueWriteBuffer(ocl->cmdQueue, ocl->buf_input, CL_TRUE, 
        0, datasize, input, 0, NULL, &timing_event);
    if(status != CL_SUCCESS) printf("Write input to buffer status: %s\n", clErrorString(status));

    if (first_iter) {
        // Report buffer write time on first iteration
        execute_time = getStartEndTime(timing_event) / 1000000.0;
        printf("Time taken for writing input buffer to device: %f ms\n", execute_time);
    }

    // RUN KERNELS
    ///////////////////////// Sobel /////////////////////////
    status = clEnqueueNDRangeKernel(ocl->cmdQueue, ocl->kernel_sobel, 2, NULL,
        globalWorkSize, NULL, 0, NULL, &timing_event);
    if(status != CL_SUCCESS) printf("sobel kernel status: %s\n", clErrorString(status));

    // Wait for work items to finish
    clFinish(ocl->cmdQueue);

    times[0] = getStartEndTime(timing_event);


    ///////////////////////// Phase and magnitude /////////////////////////
    status = clEnqueueNDRangeKernel(ocl->cmdQueue, ocl->kernel_PnM, 2, NULL,
        globalWorkSize, NULL, 0, NULL, &timing_event);
    if(status != CL_SUCCESS) printf("phaseAndMagnitude kernel status: %s\n", clErrorString(status));

    // Wait for work items to finish
    clFinish(ocl->cmdQueue);

    times[1] = getStartEndTime(timing_event);


    ///////////////////////// NONMAX /////////////////////////
    status = clEnqueueNDRangeKernel(ocl->cmdQueue, ocl->kernel_nonMax, 2, NULL,
        globalWorkSize, NULL, 0, NULL, &timing_event);
    if(status != CL_SUCCESS) printf("nonMax kernel status: %s\n", clErrorString(status));

    // Wait for work items to finish
    clFinish(ocl->cmdQueue);

    times[2] = getStartEndTime(timing_event);

    
    // Read the device output buffer to the host output array
    status = clEnqueueReadBuffer(ocl->cmdQueue, ocl->buf_output, CL_TRUE, 0,
        datasize, output, 0, NULL, &timing_event);
    if(status != CL_SUCCESS) printf("Read output from buffer status: %s\n", clErrorString(status));
    
    if (first_iter) {
        execute_time = getStartEndTime(timing_event) / 1000000.0;
        printf("Time taken for reading output buffer from device: %f ms\n", execute_time);
        first_iter = 0;
    }

    // OPENCL DONE

    times[3] = gettimemono_ns();
    edgeTracing(output, width, height);  // modifies output in-place
    times[4] = gettimemono_ns();

    for (int i = 0; i < 4; i++) {
        runtimes[i] = times[i] / 1000000.0; // Convert ns to ms
        // special case for edgeTracing as it is not read from opencl API
        if (i == 3) {
            runtimes[i] = (times[4] - times[3]) / 1000000.0;
        }
    }
}

// Needed only in Part 2 for OpenCL initialization
void
init(size_t width, size_t height, uint16_t threshold_lower,
    uint16_t threshold_upper) {

    size_t image_size = width * height;

    // Use this to check the output of each API call
    cl_int status;

    // Retrieve the number of platforms
    cl_uint numPlatforms = 0;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
 
    if(status != CL_SUCCESS) printf("clGetPlatformIDs: %s\n", clErrorString(status));

    // Allocate enough space for each platform
    cl_platform_id *platforms = NULL;
    platforms = (cl_platform_id*)malloc(
        numPlatforms*sizeof(cl_platform_id));
 
    // Fill in the platforms
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);

    if(status != CL_SUCCESS) printf("clGetPlatformIDs: %s\n", clErrorString(status));

    // Retrieve the number of devices
    cl_uint numDevices = 0;
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, 
        NULL, &numDevices);

    if(status != CL_SUCCESS) printf("clGetDeviceIDs: %s\n", clErrorString(status));

    // Allocate enough space for each device
    cl_device_id *devices;
    devices = (cl_device_id*)malloc(
        numDevices*sizeof(cl_device_id));

    // Fill in the devices 
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL,        
        numDevices, devices, NULL);

    if(status != CL_SUCCESS) printf("clGetDeviceIDs: %s\n", clErrorString(status));

    size_t time_res;
    // Get time resolution for profiling
    clGetDeviceInfo(devices[0], CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(time_res), &time_res, NULL);

    printf("Time resolution of device in ns: %lu \n", time_res);

    // Create a context and associate it with the devices
    ocl->context= clCreateContext(NULL, numDevices, devices, NULL, 
        NULL, &status);

    if(status != CL_SUCCESS) printf("clCreateContext: %s\n", clErrorString(status));

    // Create a command queue with profiling enabled and associate it with the device
    ocl->cmdQueue = clCreateCommandQueue(ocl->context, devices[0], CL_QUEUE_PROFILING_ENABLE,
        &status);

    if(status != CL_SUCCESS) printf("clCreateCommandQueue: %s\n", clErrorString(status));

    ocl->buf_input = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, image_size * sizeof(uint8_t), NULL, &status);
    if(status != CL_SUCCESS) printf("clCreateBuffer: %s\n", clErrorString(status));
    
    ocl->buf_sobel_x = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, image_size * sizeof(int16_t), NULL, &status);
    if(status != CL_SUCCESS) printf("clCreateBuffer: %s\n", clErrorString(status));

    ocl->buf_sobel_y = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, image_size * sizeof(int16_t), NULL, &status);
    if(status != CL_SUCCESS) printf("clCreateBuffer: %s\n", clErrorString(status));

    ocl->buf_phase = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, image_size * sizeof(uint8_t), NULL, &status);
    if(status != CL_SUCCESS) printf("clCreateBuffer: %s\n", clErrorString(status));
    
    ocl->buf_magnitude = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, image_size * sizeof(int16_t), NULL, &status);
    if(status != CL_SUCCESS) printf("clCreateBuffer: %s\n", clErrorString(status));

    ocl->buf_output = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, image_size * sizeof(uint8_t), NULL, &status);
    if(status != CL_SUCCESS) printf("clCreateBuffer: %s\n", clErrorString(status));

    const char* source = read_source("canny.cl");

    // Create a program with source code
    ocl->program = clCreateProgramWithSource(ocl->context, 1, 
        (const char**)&source, NULL, &status);

    if(status != CL_SUCCESS) printf("clCreateProgramWith Source: %s\n", clErrorString(status));

    // Build (compile) the program for the device
    status = clBuildProgram(ocl->program, numDevices, devices, 
        NULL, NULL, NULL);

    if(status != CL_SUCCESS) printf("clBuildProgram: %s\n", clErrorString(status));

    char result[10000];
    size_t size;
    status = clGetProgramBuildInfo(ocl->program, devices[0], CL_PROGRAM_BUILD_LOG, sizeof(result), &result, &size);
    if(status != CL_SUCCESS) printf("ProgramBuildLog status: %s\n", clErrorString(status));
    //printf("ProgramBuildLog: %s\n", result);

    // Create the vector addition kernel
    ocl->kernel_sobel = clCreateKernel(ocl->program, "sobel", &status);
    if(status != CL_SUCCESS) printf("clCreateKernel: %s\n", clErrorString(status));

    ocl->kernel_PnM = clCreateKernel(ocl->program, "PnM", &status);
    if(status != CL_SUCCESS) printf("clCreateKernel: %s\n", clErrorString(status));

    ocl->kernel_nonMax = clCreateKernel(ocl->program, "nonMax", &status);
    if(status != CL_SUCCESS) printf("clCreateKernel: %s\n", clErrorString(status));

    // Associate the input and output buffers with the kernel
    status = clSetKernelArg(ocl->kernel_sobel, 0, sizeof(cl_mem), &ocl->buf_input);
    if(status != CL_SUCCESS) printf("SetArgs: %s\n", clErrorString(status));
    status = clSetKernelArg(ocl->kernel_sobel, 1, sizeof(cl_mem), &ocl->buf_sobel_x);
    if(status != CL_SUCCESS) printf("SetArgs: %s\n", clErrorString(status));
    status = clSetKernelArg(ocl->kernel_sobel, 2, sizeof(cl_mem), &ocl->buf_sobel_y);
    if(status != CL_SUCCESS) printf("SetArgs: %s\n", clErrorString(status));
    status = clSetKernelArg(ocl->kernel_sobel, 3, sizeof(int), &width);
    if(status != CL_SUCCESS) printf("SetArgs: %s\n", clErrorString(status));
    status = clSetKernelArg(ocl->kernel_sobel, 4, sizeof(int), &height);
    if(status != CL_SUCCESS) printf("SetArgs: %s\n", clErrorString(status));

    status = clSetKernelArg(ocl->kernel_PnM, 0, sizeof(cl_mem), &ocl->buf_sobel_x);
    if(status != CL_SUCCESS) printf("SetArgs: %s\n", clErrorString(status));
    status = clSetKernelArg(ocl->kernel_PnM, 1, sizeof(cl_mem), &ocl->buf_sobel_y);
    if(status != CL_SUCCESS) printf("SetArgs: %s\n", clErrorString(status));
    status = clSetKernelArg(ocl->kernel_PnM, 2, sizeof(cl_mem), &ocl->buf_phase);
    if(status != CL_SUCCESS) printf("SetArgs: %s\n", clErrorString(status));
    status = clSetKernelArg(ocl->kernel_PnM, 3, sizeof(cl_mem), &ocl->buf_magnitude);
    if(status != CL_SUCCESS) printf("SetArgs: %s\n", clErrorString(status));
    status = clSetKernelArg(ocl->kernel_PnM, 4, sizeof(int), &width);
    if(status != CL_SUCCESS) printf("SetArgs: %s\n", clErrorString(status));
    status = clSetKernelArg(ocl->kernel_PnM, 5, sizeof(int), &height);
    if(status != CL_SUCCESS) printf("SetArgs: %s\n", clErrorString(status));
    
    status = clSetKernelArg(ocl->kernel_nonMax, 0, sizeof(cl_mem), &ocl->buf_phase);
    if(status != CL_SUCCESS) printf("SetArgs: %s\n", clErrorString(status));
    status = clSetKernelArg(ocl->kernel_nonMax, 1, sizeof(cl_mem), &ocl->buf_magnitude);
    if(status != CL_SUCCESS) printf("SetArgs: %s\n", clErrorString(status));
    status = clSetKernelArg(ocl->kernel_nonMax, 2, sizeof(cl_mem), &ocl->buf_output);
    if(status != CL_SUCCESS) printf("SetArgs: %s\n", clErrorString(status));
    status = clSetKernelArg(ocl->kernel_nonMax, 3, sizeof(int), &width);
    if(status != CL_SUCCESS) printf("SetArgs: %s\n", clErrorString(status));
    status = clSetKernelArg(ocl->kernel_nonMax, 4, sizeof(int), &height);
    if(status != CL_SUCCESS) printf("SetArgs: %s\n", clErrorString(status));
    status = clSetKernelArg(ocl->kernel_nonMax, 5, sizeof(int), &threshold_lower);
    if(status != CL_SUCCESS) printf("SetArgs: %s\n", clErrorString(status));
    status = clSetKernelArg(ocl->kernel_nonMax, 6, sizeof(int), &threshold_upper);
    if(status != CL_SUCCESS) printf("SetArgs: %s\n", clErrorString(status));

}

void destroy() {

    // Free OpenCL resources
    clReleaseKernel(ocl->kernel_nonMax);
    clReleaseKernel(ocl->kernel_PnM);
    clReleaseKernel(ocl->kernel_sobel);

    clReleaseMemObject(ocl->buf_input);
    clReleaseMemObject(ocl->buf_sobel_x);
    clReleaseMemObject(ocl->buf_sobel_y);
    clReleaseMemObject(ocl->buf_phase);
    clReleaseMemObject(ocl->buf_magnitude);
    clReleaseMemObject(ocl->buf_output);

    clReleaseProgram(ocl->program);
    clReleaseCommandQueue(ocl->cmdQueue);
    clReleaseContext(ocl->context);

}

////////////////////////////////////////////////
// ¤¤ DO NOT EDIT ANYTHING AFTER THIS LINE ¤¤ //
////////////////////////////////////////////////

enum PROCESSING_MODE { DEFAULT, BIG_MODE, SMALL_MODE, VIDEO_MODE };
// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
int
main(int argc, char **argv) {
    enum PROCESSING_MODE mode = DEFAULT;
    if (argc > 1) {
        char *mode_c = argv[1];
        if (strlen(mode_c) == 2) {
            if (strncmp(mode_c, "-B", 2) == 0) {
                mode = BIG_MODE;
            } else if (strncmp(mode_c, "-b", 2) == 0) {
                mode = SMALL_MODE;
            } else if (strncmp(mode_c, "-v", 2) == 0) {
                mode = VIDEO_MODE;
            } else {
                printf(
                    "Invalid usage! Please set either -b, -B, -v or "
                    "nothing\n");
                return -1;
            }
        } else {
            printf("Invalid usage! Please set either -b, -B, -v nothing\n");
            return -1;
        }
    }
    int benchmarking_iterations = 1;
    if (argc > 2) {
        benchmarking_iterations = atoi(argv[2]);
    }

    char *input_image_path = "";
    char *output_image_path = "";
    uint16_t threshold_lower = 0;
    uint16_t threshold_upper = 0;
    switch (mode) {
        case BIG_MODE:
            input_image_path = "hameensilta.pgm";
            output_image_path = "hameensilta_output.pgm";
            // Arbitrarily selected to produce a nice-looking image
            // DO NOT CHANGE THESE WHEN BENCHMARKING
            threshold_lower = 120;
            threshold_upper = 300;
            printf(
                "Enabling %d benchmarking iterations with the large %s "
                "image\n",
                benchmarking_iterations, input_image_path);
            break;
        case SMALL_MODE:
            input_image_path = "x.pgm";
            output_image_path = "x_output.pgm";
            threshold_lower = 750;
            threshold_upper = 800;
            printf(
                "Enabling %d benchmarking iterations with the small %s "
                "image\n",
                benchmarking_iterations, input_image_path);
            break;
        case VIDEO_MODE:
            if (system("which ffmpeg > /dev/null 2>&1") ||
                system("which ffplay > /dev/null 2>&1")) {
                printf(
                    "Video mode is disabled because ffmpeg is not found\n");
                return -1;
            }
            benchmarking_iterations = 0;
            input_image_path = "people.mp4";
            threshold_lower = 120;
            threshold_upper = 300;
            printf(
                "Playing video %s with FFMPEG. Error check disabled.\n",
                input_image_path);
            break;
        case DEFAULT:
        default:
            input_image_path = "x.pgm";
            output_image_path = "x_output.pgm";
            // Carefully selected to produce a discontinuous edge without edge
            // tracing
            threshold_lower = 750;
            threshold_upper = 800;
            printf("Running with %s image\n", input_image_path);
            break;
    }

    uint8_t *input_image = NULL;
    size_t width = 0;
    size_t height = 0;
    if (mode == VIDEO_MODE) {
        width = 3840;
        height = 2160;
        init(width, height, threshold_lower, threshold_upper);

        uint8_t *output_image = malloc(width * height);
        assert(output_image);

        int count;
        uint8_t *frame = malloc(width * height * 3);
        assert(frame);
        char pipein_cmd[1024];
        snprintf(
            pipein_cmd, 1024,
            "ffmpeg -i %s -f image2pipe -vcodec rawvideo -an -s %zux%zu "
            "-pix_fmt gray - 2> /dev/null",
            input_image_path, width, height);
        FILE *pipein = popen(pipein_cmd, "r");
        char pipeout_cmd[1024];
        snprintf(
            pipeout_cmd, 1024,
            "ffplay -f rawvideo -pixel_format gray -video_size %zux%zu "
            "-an - 2> /dev/null",
            width, height);
        FILE *pipeout = popen(pipeout_cmd, "w");
        double runtimes[4];
        while (1) {
            count = fread(frame, 1, height * width, pipein);
            if (count != height * width) break;

            cannyEdgeDetection(
                frame, width, height, threshold_lower, threshold_upper,
                output_image, runtimes);

            double total_time =
                runtimes[0] + runtimes[1] + runtimes[2] + runtimes[3];
            printf("FPS: %0.1f\n", 1000 / total_time);
            fwrite(output_image, 1, height * width, pipeout);
        }
        fflush(pipein);
        pclose(pipein);
        fflush(pipeout);
        pclose(pipeout);
    } else {
        if ((input_image = read_pgm(input_image_path, &width, &height))) {
            printf(
                "Input image read succesfully. Size %zux%zu\n", width,
                height);
        } else {
            printf("Read failed\n");
            return -1;
        }
        init(width, height, threshold_lower, threshold_upper);

        uint8_t *output_image = malloc(width * height);
        assert(output_image);

        int all_the_runs_were_succesful = 1;
        double avg_runtimes[4] = {0.0, 0.0, 0.0, 0.0};
        double avg_total = 0.0;
        for (int iter = 0; iter < benchmarking_iterations; iter++) {
            double iter_runtimes[4];
            cannyEdgeDetection(
                input_image, width, height, threshold_lower, threshold_upper,
                output_image, iter_runtimes);

            for (int n = 0; n < 4; n++) {
                avg_runtimes[n] += iter_runtimes[n] / benchmarking_iterations;
                avg_total += iter_runtimes[n] / benchmarking_iterations;
            }

            uint8_t *output_image_ref = malloc(width * height);
            assert(output_image_ref);
            cannyEdgeDetection_ref(
                input_image, width, height, threshold_lower, threshold_upper,
                output_image_ref);

            uint8_t *fused_comparison = malloc(width * height);
            assert(fused_comparison);
            int failed = validate_result(
                output_image, output_image_ref, width, height,
                fused_comparison);
            if (failed) {
                all_the_runs_were_succesful = 0;
                printf(
                    "Error checking failed for benchmark iteration %d!\n"
                    "Writing your output to %s. The image that should've "
                    "been generated is written to ref.pgm\n"
                    "Generating fused.pgm for debugging purpose. Light-grey "
                    "pixels should've been white and "
                    "dark-grey pixels black. Corrupted pixels are colored "
                    "middle-grey\n",
                    iter, output_image_path);

                write_pgm("ref.pgm", output_image_ref, width, height);
                write_pgm("fused.pgm", fused_comparison, width, height);
            }
        }

        printf("Sobel3x3 time          : %0.3f ms\n", avg_runtimes[0]);
        printf("phaseAndMagnitude time : %0.3f ms\n", avg_runtimes[1]);
        printf("nonMaxSuppression time : %0.3f ms\n", avg_runtimes[2]);
        printf("edgeTracing time       : %0.3f ms\n", avg_runtimes[3]);
        printf("Total time             : %0.3f ms\n", avg_total);
        write_pgm(output_image_path, output_image, width, height);
        printf("Wrote output to %s\n", output_image_path);
        if (all_the_runs_were_succesful) {
            printf("Error checks passed!\n");
        } else {
            printf("There were failing runs\n");
        }
    }
    destroy();
    return 0;
}
