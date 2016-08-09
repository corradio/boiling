#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

// TODO: Support multiple GPUs

void writeCsvLine(FILE* fp, float* array, const uint length) {
    for (int i=0; i < length; i++) {
        if (i==0)
            fprintf(fp, "%f", array[i]);
        else
            fprintf(fp, ";%f", array[i]);
        if (i == length - 1)
            fprintf(fp, "\n");
    }
}


int main(int argc, char** argv)
{
    // Arguments
    bool resume = false;
    
    // Simulation parameters
    unsigned int maxParticles = 2048; // must be a nice power of 2
    unsigned int numParticles = 2048;
    
    float tEnd = 10.0f;
    float dt = 0.00001f;
    float dtSampling = 0.01f;
    float gravity = -10.0f;
    float particleRadius = 0.01;
    float restitutionCoefficient = 0.99;
    float bottomTemperature = 15.0;
    
    float t = 0.0f;
    float data[maxParticles * 4];              // original data set given to device
    float dissipation[maxParticles];
    float energyInput[maxParticles];
    
    // Load Kernel
    FILE *fp;
    const char kernelpath[] = "./kernel.cl";
    size_t kernel_source_size;
    char *kernel_source_str;
    
    /* Load kernel source code */
    fp = fopen(kernelpath, "r");
    if (!fp) {
        printf("Error: Failed to find kernel path!\n");
        return EXIT_FAILURE;
    }
    kernel_source_str = (char *)malloc(MAX_SOURCE_SIZE);
    kernel_source_size = fread(kernel_source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel_integrate;         // compute kernel
    cl_kernel kernel_collide;           // compute kernel
    
    cl_mem input;                       // device memory used for the input array
    cl_mem output;
    cl_mem dissipation_mem;
    cl_mem energy_mem;
    
    size_t global_work_size;            // global domain size for our calculation
    size_t local_work_size;             // local domain size for our calculation
    
    int err;                            // error code returned from api calls
    
    dissipation[0] = 0.0f;
    for (int i=0; i < maxParticles; i++)
        energyInput[i] = 0.0f;
    
    if (resume)
    {
        // Load last line of csv
        fp = fopen("/Users/olc/dev/boiling/opencl/output.csv", "r+");
        if (!fp) {
            printf("Error: Failed to find output.csv!\n");
            return EXIT_FAILURE;
        }
        char* line = (char*)malloc(MAX_SOURCE_SIZE);
        while (fgets(line, MAX_SOURCE_SIZE, fp)) {}
        // Parse
        int i = 0;
        char* tok;
        for (tok = strtok(line, ";");
             tok && *tok;
             tok = strtok(NULL, ";\n"))
        {
            data[i] = atof(tok);
            i++;
        }
        if (!i) {
            printf("Error: Did not read any input to resume from!\n");
            return EXIT_FAILURE;
        }
    }
    else
    {
        // Create data
        int numRows = 64; // some power of 2
        int numCols = maxParticles / numRows;
        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                int n = i * numCols + j;
                data[n*4]     = -1.0 + 2.0 / numCols * (0.5 + j);
                data[n*4 + 1] = -1.0 + 2.0 / numRows * (0.5 + i);
                data[n*4 + 2] = (rand() * 2.0 - (float)RAND_MAX) / (float)RAND_MAX;
                data[n*4 + 3] = (rand() * 2.0 - (float)RAND_MAX) / (float)RAND_MAX;
            }
        }
        
        // Prepare output pipe
        fp = fopen("/Users/olc/dev/boiling/opencl/output.csv", "w");
        if (!fp) {
            printf("Error: Failed to create output!\n");
            return EXIT_FAILURE;
        }
    }
    
    // Connect to a compute device
    //
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
    
    // Create a compute context
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }
    
    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }
    
    // Create the compute program from the source buffer
    //
    program = clCreateProgramWithSource(context, 1, (const char **) &kernel_source_str, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }
    
    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
    
    // Create the compute kernels in the program we wish to run
    //
    kernel_integrate = clCreateKernel(program, "integrate", &err);
    kernel_collide = clCreateKernel(program, "collide", &err);
    if (!kernel_integrate || !kernel_collide || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernels!\n");
        exit(1);
    }
    
    // Allocate state memory
    //
    input  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 4 * maxParticles, NULL, NULL);
    output  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 4 * maxParticles, NULL, NULL);
    dissipation_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * maxParticles, NULL, NULL);
    energy_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * maxParticles, NULL, NULL);
    if (!input) {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }
    
    // Set the arguments to our compute kernel
    //
    err = 0;
    err  = clSetKernelArg(kernel_integrate, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel_integrate, 1, sizeof(cl_mem), &energy_mem);
    err |= clSetKernelArg(kernel_integrate, 2, sizeof(unsigned int), &numParticles);
    err |= clSetKernelArg(kernel_integrate, 3, sizeof(float), &dt);
    err |= clSetKernelArg(kernel_integrate, 4, sizeof(float), &gravity);
    err |= clSetKernelArg(kernel_integrate, 5, sizeof(float), &particleRadius);
    err |= clSetKernelArg(kernel_integrate, 6, sizeof(float), &bottomTemperature);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    
    err = 0;
    err  = clSetKernelArg(kernel_collide, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel_collide, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel_collide, 2, sizeof(cl_mem), &dissipation_mem);
    err |= clSetKernelArg(kernel_collide, 3, sizeof(unsigned int), &numParticles);
    err |= clSetKernelArg(kernel_collide, 4, sizeof(float), &particleRadius);
    err |= clSetKernelArg(kernel_collide, 5, sizeof(float), &restitutionCoefficient);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    
    // Write our data set into the input array in device memory
    //
    err  = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * 4 * maxParticles, data, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, output, CL_TRUE, 0, sizeof(float) * 4 * maxParticles, data, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, dissipation_mem, CL_TRUE, 0, sizeof(float) * maxParticles, dissipation, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, energy_mem, CL_TRUE, 0, sizeof(float) * maxParticles, energyInput, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    
    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel_integrate, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_work_size), &local_work_size, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }
    global_work_size = maxParticles;
    
    FILE* fp_dissipation = fopen("/Users/olc/dev/boiling/opencl/dissipation.csv", "w");
    if (!fp_dissipation) {
        printf("Error: Failed to find dissipation.csv!\n");
        return EXIT_FAILURE;
    }
    
    // Write initial state
    if (!resume)
    {
        writeCsvLine(fp, data, numParticles * 4);
        fprintf(fp_dissipation, "%f;%f\n", 0.0f, 0.0f);
    }
    
    
    float lastSampleTime = 0;
    int timeIndex = 0;
    while (t < tEnd) {
        timeIndex++;
        t = dt * timeIndex;
        // Execute the kernel over the entire range of our 1d input data set
        // using the maximum number of work group items for this device
        //
        err  = clEnqueueNDRangeKernel(commands, kernel_integrate, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        err |= clEnqueueCopyBuffer(commands, input, output, 0, 0, sizeof(float) * 4 * maxParticles, 0, NULL, NULL);
        err |= clEnqueueNDRangeKernel(commands, kernel_collide, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        if (err)
        {
            printf("Error: Failed to execute kernels. Error %d!\n", err);
            return EXIT_FAILURE;
        }
        
        err = clEnqueueCopyBuffer(commands, output, input, 0, 0, sizeof(float) * 4 * maxParticles, 0, NULL, NULL);
        if (err)
        {
            printf("Error: Failed to execute clEnqueueCopyBuffer. Error %d!\n", err);
            return EXIT_FAILURE;
        }
        
        // A tolerance relative to dt is added to prevent numerical rounding error problems
        if (t - lastSampleTime >= dtSampling - dt * 0.01)
        {
            lastSampleTime = t;
            printf("Sampling at t = %f..\n", t);
            
            // Read back the results from the device
            //
            clFinish(commands);
            err  = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(float) * 4 * maxParticles, data, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(commands, dissipation_mem, CL_TRUE, 0, sizeof(float) * maxParticles, dissipation, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(commands, energy_mem, CL_TRUE, 0, sizeof(float) * maxParticles, energyInput, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to read output array! %d\n", err);
                exit(1);
            }
            
            writeCsvLine(fp, data, numParticles * 4);
            
            double totalEnergyInput = 0.0f;
            double totalDissipation = 0.0f;
            for (int i = 0; i < numParticles; i++)
            {
                totalEnergyInput += energyInput[i];
                totalDissipation += dissipation[i];
            }
            fprintf(fp_dissipation, "%f;%f\n", totalDissipation, totalEnergyInput);
        }
    }
    
    fclose(fp);
    fclose(fp_dissipation);
    
    // Shutdown and cleanup
    //
    clReleaseMemObject(input);
    clReleaseProgram(program);
    clReleaseKernel(kernel_integrate);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
 
    return 0;
}
