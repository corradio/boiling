#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void integrate(
    __global double *globalState, // input-output,
    __global double *cumulativeEnergyInput, // input-output
    __global double *cumulativeDissipation, // input-output
    const unsigned int numParticles,
    const double dt,
    const double gravity,
    const double particleRadius,
    const double bottomTemperature,
    const double topTemperature)
{
    const uint i = get_global_id(0);
    if (i >= numParticles)
        return;
    const uint index = i * 4; // To account for proper indexing

    // Euler for free falling body
    globalState[index    ] += globalState[index + 2] * dt;
    globalState[index + 1] += globalState[index + 3] * dt + gravity * 0.5 * dt * dt;
    globalState[index + 3] += gravity * dt;

    // Boundary conditions
    if (globalState[index] <= -1.0 + particleRadius && globalState[index + 2] <= 0.0) // left
    {
        globalState[index + 2] *= -1;
    }
    else if (globalState[index] >= 1.0 - particleRadius && globalState[index + 2] >= 0.0) // right
    {
        globalState[index + 2] *= -1;
    }
    else if (globalState[index + 1] <= -1.0 + particleRadius && globalState[index + 3] <= 0.0) // bottom
    {
        globalState[index + 3] *= -1;

        double v_sq_norm = globalState[index + 2] * globalState[index + 2] + globalState[index + 3] * globalState[index + 3];
        double v_inorm = rsqrt((float)v_sq_norm);

        double vHotX = globalState[index + 2] * v_inorm * bottomTemperature;
        double vHotY = globalState[index + 3] * v_inorm * bottomTemperature;
        globalState[index + 2] = vHotX;
        globalState[index + 3] = vHotY;

        cumulativeEnergyInput[i] += 0.5 * (vHotX * vHotX + vHotY * vHotY - v_sq_norm);
    }
    else if (globalState[index + 1] >= 1.0 - particleRadius && globalState[index + 3] >= 0.0) // top
    {
        globalState[index + 3] *= -1;

        double v_sq_norm = globalState[index + 2] * globalState[index + 2] + globalState[index + 3] * globalState[index + 3];
        double v_inorm = rsqrt((float)v_sq_norm);

        double vColdX = globalState[index + 2] * v_inorm * topTemperature;
        double vColdY = globalState[index + 3] * v_inorm * topTemperature;
        globalState[index + 2] = vColdX;
        globalState[index + 3] = vColdY;

        cumulativeDissipation[i] += 0.5 * (v_sq_norm - vColdX * vColdX - vColdY * vColdY);
    }
}

__kernel void collide(
    __global double *globalState, // input-output
    __global double *cumulativeDissipation, // input-output
    const unsigned int numParticles,
    const double particleRadius,
    const double restitutionCoefficient)
{
    uint index_i, index_j;
    const double particleRadius_sq = particleRadius * particleRadius;

    for (int i = 0; i < numParticles; i++) { 
        index_i = i * 4;
        for (int j = 0; j < numParticles; j++)
        {
            if (i >= j) continue;

            index_j = j * 4;

            // Calculate distance
            double dx = globalState[index_j    ] - globalState[index_i    ];
            double dy = globalState[index_j + 1] - globalState[index_i + 1];
            double dist_sq = dx * dx + dy * dy;
            if (dist_sq <= 4.0 * particleRadius_sq)
            {
                double inorm = rsqrt((float)dist_sq);
                double nx = dx * inorm;
                double ny = dy * inorm;
                double relvdotn = nx * (globalState[index_j + 2] - globalState[index_i + 2]) + ny * (globalState[index_j + 3] - globalState[index_i + 3]);
                if (relvdotn >= 0.0) // this condition requires correcting both particles at the same time
                    // Particles are moving away from each other, don't correct
                    return;

                double Q = 0.5 * (1.0 + restitutionCoefficient) * relvdotn;
                double colx = Q * nx;
                double coly = Q * ny;

                double v_sq_norm_i = globalState[index_i + 2] * globalState[index_i + 2] + globalState[index_i + 3] * globalState[index_i + 3];
                double v_sq_norm_j = globalState[index_j + 2] * globalState[index_j + 2] + globalState[index_j + 3] * globalState[index_j + 3];

                globalState[index_i + 2] += colx;
                globalState[index_i + 3] += coly;
                // ** Added
                globalState[index_j + 2] -= colx;
                globalState[index_j + 3] -= coly;

                cumulativeDissipation[i] += 0.5 * (v_sq_norm_i - globalState[index_i + 2] * globalState[index_i + 2] - globalState[index_i + 3] * globalState[index_i + 3]);
                cumulativeDissipation[j] += 0.5 * (v_sq_norm_j - globalState[index_j + 2] * globalState[index_j + 2] - globalState[index_j + 3] * globalState[index_j + 3]);
            }
        }
    }
}
