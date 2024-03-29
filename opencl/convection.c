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

void writeCsvLine(FILE* fp, double* array, const uint length) {
    for (int i=0; i < length; i++) {
        if (i==0)
            fprintf(fp, "%.15e", array[i]);
        else
            fprintf(fp, ";%.15e", array[i]);
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

    double tEnd = 20.0;
    double dt = 0.0001;
    double dtSampling = 0.001;
    double gravity = -10.0f;
    double particleRadius = 0.01;
    double restitutionCoefficient = 0.999;
    double bottomTemperature = 15.0;
    double topTemperature = 1.0;


    double t = 0.0f;

    // Data structures for each time t
    double data[maxParticles * 4];         // length: n x 4 (x, y, vx, vy)
    double energyFluxLower[maxParticles];  // length: n
    double energyFluxUpper[maxParticles];  // length: n
    double dissipation[maxParticles];      // length: n

    FILE *fp;
    FILE* fp_dissipation;
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
    cl_mem energy_flux_upper_mem;
    cl_mem energy_flux_lower_mem;
    cl_mem dissipation_mem;

    size_t global_work_size;            // global domain size for our calculation
    size_t local_work_size;             // local domain size for our calculation

    int err;                            // error code returned from api calls

    for (int i=0; i < maxParticles; i++) {
        energyFluxLower[i] = 0.0;
        energyFluxUpper[i] = 0.0;
        dissipation[i] = 0.0;
    }

    if (resume)
    {
        // Load last line of csv
        fp = fopen("/Users/oliviercorradi/github/boiling/opencl/output.csv", "r+");
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

        // Same for collisions
        fp_dissipation = fopen("/Users/oliviercorradi/github/boiling/opencl/dissipation.csv", "a");
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
                data[n*4 + 2] = (rand() * 2.0 - (double)RAND_MAX) / (double)RAND_MAX;
                data[n*4 + 3] = (rand() * 2.0 - (double)RAND_MAX) / (double)RAND_MAX;
            }
        }

        // Prepare output pipe
        fp = fopen("/Users/oliviercorradi/github/boiling/opencl/output.csv", "w");
        fp_dissipation = fopen("/Users/oliviercorradi/github/boiling/opencl/dissipation.csv", "w");
        if (!fp || !fp_dissipation) {
            printf("Error: Failed to create output.csv or dissipation.csv!\n");
            return EXIT_FAILURE;
        }
    }

    // Connect to a compute device
    //
    bool should_use_gpu = false;
    err = clGetDeviceIDs(NULL, should_use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }

    // Get device info
    //
    unsigned long retSize;
    err = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, 0, NULL, &retSize);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get device info!\n");
        return EXIT_FAILURE;
    }
    char extensions[retSize];
    err = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, retSize, extensions, &retSize);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get device info!\n");
        return EXIT_FAILURE;
    }
    printf("CL_DEVICE_EXTENSIONS: %s\n", extensions);

    cl_uint ui;
    size_t size;
    size_t sizes[3] = { 0, 0, 0 };
    clGetDeviceInfo( device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(ui), &ui, NULL );
    fprintf( stderr, "Device Maximum Compute Units = %d\n", ui );

    clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(ui), &ui, NULL );
    fprintf( stderr, "Device Maximum Work Item Dimensions = %d\n", ui );

    clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(sizes), sizes, NULL );
    fprintf( stderr, "Device Maximum Work Item Sizes = %d x %d x %d\n", sizes[0], sizes[1], sizes[2] );

    clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size), &size, NULL );
    fprintf( stderr, "Device Maximum Work Group Size = %d\n", size );

    clGetDeviceInfo( device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(ui), &ui, NULL );
    fprintf( stderr, "Device Maximum Clock Frequency = %d MHz\n", ui );

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
        printf("Error: Failed to create a command queue!\n");
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
    input  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * 4 * maxParticles, NULL, NULL);
    dissipation_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * maxParticles, NULL, NULL);
    energy_flux_lower_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * maxParticles, NULL, NULL);
    energy_flux_upper_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * maxParticles, NULL, NULL);
    if (!input) {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }

    // Set the arguments to our compute kernel
    //
    err = 0;
    err  = clSetKernelArg(kernel_integrate, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel_integrate, 1, sizeof(cl_mem), &energy_flux_lower_mem);
    err |= clSetKernelArg(kernel_integrate, 2, sizeof(cl_mem), &energy_flux_upper_mem);
    err |= clSetKernelArg(kernel_integrate, 3, sizeof(unsigned int), &numParticles);
    err |= clSetKernelArg(kernel_integrate, 4, sizeof(double), &dt);
    err |= clSetKernelArg(kernel_integrate, 5, sizeof(double), &gravity);
    err |= clSetKernelArg(kernel_integrate, 6, sizeof(double), &particleRadius);
    err |= clSetKernelArg(kernel_integrate, 7, sizeof(double), &bottomTemperature);
    err |= clSetKernelArg(kernel_integrate, 8, sizeof(double), &topTemperature);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    err = 0;
    err  = clSetKernelArg(kernel_collide, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel_collide, 1, sizeof(cl_mem), &dissipation_mem);
    err |= clSetKernelArg(kernel_collide, 2, sizeof(unsigned int), &numParticles);
    err |= clSetKernelArg(kernel_collide, 3, sizeof(double), &particleRadius);
    err |= clSetKernelArg(kernel_collide, 4, sizeof(double), &restitutionCoefficient);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // Write initial conditions into the input array in device memory
    //
    err  = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(double) * 4 * maxParticles, data, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, energy_flux_lower_mem, CL_TRUE, 0, sizeof(double) * maxParticles, energyFluxLower, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, energy_flux_upper_mem, CL_TRUE, 0, sizeof(double) * maxParticles, energyFluxUpper, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, dissipation_mem, CL_TRUE, 0, sizeof(double) * maxParticles, dissipation, 0, NULL, NULL);
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

    // Write initial state
    if (!resume)
    {
        writeCsvLine(fp, data, numParticles * 4);
        fprintf(fp_dissipation, "%.15f;%.15f;%.15f\n", 0.0, 0.0, 0.0);
    }


    double lastSampleTime = 0;
    int timeIndex = 0;
    const secondBottomTemperature = bottomTemperature * 0.9;
    while (t < tEnd) {
        timeIndex++;
        t = dt * timeIndex;

        // Experiment: reduce bottom temperature by half
        // err |= clSetKernelArg(kernel_integrate, 7, sizeof(double), (t > 0.5 * tEnd) ? &secondBottomTemperature : &bottomTemperature);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to set kernel arguments! %d\n", err);
            exit(1);
        }

        // Execute the kernel over the entire range of our 1d input data set
        // using the maximum number of work group items for this device
        //
        err  = clEnqueueNDRangeKernel(commands, kernel_integrate, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        err |= clEnqueueTask(commands, kernel_collide, 0, NULL, NULL);
        if (err)
        {
            printf("Error: Failed to execute kernels. Error %d!\n", err);
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
            err  = clEnqueueReadBuffer(commands, input, CL_TRUE, 0, sizeof(double) * 4 * maxParticles, data, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(commands, energy_flux_lower_mem, CL_TRUE, 0, sizeof(double) * maxParticles, energyFluxLower, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(commands, energy_flux_upper_mem, CL_TRUE, 0, sizeof(double) * maxParticles, energyFluxUpper, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(commands, dissipation_mem, CL_TRUE, 0, sizeof(double) * maxParticles, dissipation, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to read output array! %d\n", err);
                exit(1);
            }

            // Write state to csv
            writeCsvLine(fp, data, numParticles * 4);

            // Write energy fluxes and dissipation to csv
            double totalEnergyFluxLower = 0.0;
            double totalEnergyFluxUpper = 0.0;
            double totalDissipation = 0.0;
            for (int i = 0; i < numParticles; i++)
            {
                totalEnergyFluxLower += energyFluxLower[i];
                totalEnergyFluxUpper += energyFluxUpper[i];
                totalDissipation += dissipation[i];
                // Reset the values for next iteration
                energyFluxLower[i] = 0.0;
                energyFluxUpper[i] = 0.0;
                dissipation[i] = 0.0;
            }
            err |= clEnqueueWriteBuffer(commands, dissipation_mem, CL_TRUE, 0, sizeof(double) * maxParticles, dissipation, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(commands, energy_flux_lower_mem, CL_TRUE, 0, sizeof(double) * maxParticles, energyFluxLower, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(commands, energy_flux_upper_mem, CL_TRUE, 0, sizeof(double) * maxParticles, energyFluxUpper, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to write to source array!\n");
                exit(1);
            }
            fprintf(fp_dissipation, "%.15f;%.15f;%.15f\n", totalDissipation, totalEnergyFluxLower, totalEnergyFluxUpper);
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
