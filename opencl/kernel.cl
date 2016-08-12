#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void integrate(
    __global float4 *globalState, // input-output,
    __global float *cumulativeEnergyInput, // input-output
    __global float *cumulativeDissipation, // input-output
    const unsigned int numParticles,
    float dt,
    const float gravity,
    const float particleRadius,
    const float bottomTemperature,
    const float topTemperature)
{
    const uint i = get_global_id(0);
    if (i >= numParticles)
        return;

    /*
    // Euler integration
    globalState[i].x += globalState[i].z * dt; // z is vx
    globalState[i].y += globalState[i].w * dt + 0.5f * gravity * dt * dt; // w is vy
    globalState[i].w += gravity * dt; // only acceleration in y
    */

    // Symplectic Euler
    globalState[i].w += gravity * dt; // only acceleration in y
    globalState[i].x += globalState[i].z * dt;
    globalState[i].y += globalState[i].w * dt;

    // Boundary conditions
    if (globalState[i].x <= -1.0f + particleRadius) // left
    {
        globalState[i].z *= -1;
    }
    else if (globalState[i].x >= 1.0f - particleRadius) // right
    {
        globalState[i].z *= -1;
    }
    else if (globalState[i].y <= -1.0f + particleRadius) // bottom
    {
        globalState[i].w *= -1;

        // TODO: Add randomness to direction and length
        float2 v = (float2)(globalState[i].z, globalState[i].w);
        float2 vHot = normalize(v) * bottomTemperature;
        vHot.y +=  bottomTemperature/100.0f; // To avoid getting stuck at bottom
        float delta = 0.5f * (dot(vHot, vHot) - dot(v, v));
        cumulativeEnergyInput[i] += delta;
        globalState[i].z = vHot.x;
        globalState[i].w = vHot.y;
    }
    else if (globalState[i].y >= 1.0f - particleRadius) // top
    {
        globalState[i].w *= -1;

        float2 v = (float2)(globalState[i].z, globalState[i].w);
        float2 vCold = normalize(v) * topTemperature;
        vCold.y -= topTemperature/100.0f; // To avoid getting stuck at top
        cumulativeDissipation[i] += 0.5f * (dot(v, v) - dot(vCold, vCold));
        globalState[i].z = vCold.x;
        globalState[i].w = vCold.y;
    }
}

__kernel void collide(
    __global float4 *globalStateIn, // input
    __global float4 *globalStateOut, // output
    __global float *cumulativeDissipation, // input-output
    const unsigned int numParticles,
    const float particleRadius,
    const float restitutionCoefficient)
{
    const uint i = get_global_id(0);
    if (i >= numParticles)
        return;
    for (int j = 0; j < numParticles; j++)
    {
        if (i == j) continue;

        // Calculate distance
        float4 rel = globalStateIn[j] - globalStateOut[i];
        
        float2 r = (float2)(rel.x, rel.y);
        float dist = length(r);

        if (dist <= 2.0f * particleRadius)
        {
            float2 n = (float2)(r.x / dist, r.y / dist);
            float relvdotn = dot(n, (float2)(rel.z, rel.w));
            if (relvdotn > 0.0f)
                // Particles are moving away from each other, don't correct
                return;
            float2 d = 0.5f * (1.0f + restitutionCoefficient) * relvdotn * n;

            float2 vi = (float2)(globalStateOut[i].z, globalStateOut[i].w);
            float2 vinew = vi + d;
            cumulativeDissipation[i] += 0.5f * (dot(vi, vi) - dot(vinew, vinew));

            globalStateOut[i].z += d.x;
            globalStateOut[i].w += d.y;
        }
    }
}
