__kernel void integrate(
    __global float4 *globalState, // input-output
    const unsigned int numParticles,
    float dt,
    float gravity,
    float particleRadius)
{
    const uint index = get_global_id(0);
    if(index >= numParticles)
        return;

    float4 state = globalState[index];

    // TODO: Vectorise (if it speeds up?)
    state.x += state.z * dt; // z is vx
    state.y += state.w * dt + 0.5 * gravity * dt * dt; // w is vy
    // state.z = state.z;
    state.w += gravity * dt;

    // Boundary conditions
    if(state.x < -1.0f + particleRadius) // left
    {
        //state.x = -1.0f + particleRadius;
        state.z *= -1;
    }
    if(state.x > 1.0f - particleRadius) // right
    {
        //state.x = 1.0f - particleRadius;
        state.z *= -1;
    }

    if(state.y < -1.0f + particleRadius) // bottom
    {
        //state.y = -1.0f + particleRadius;
        state.w *= -1;
        // TODO: Add randomness
        float2 v = (float2)(state.z, state.w);
        float2 vHot = v / length(v) * 15.0f;
        state.z = vHot.x;
        state.w = vHot.y;
    }
    if(state.y > 1.0f - particleRadius) // top
    {
        //state.y = 1.0f - particleRadius;
        state.w *= -1;
    }

    globalState[index] = state;
}

__kernel void collide(
    __global float4 *globalState, // input-output
    const unsigned int numParticles,
    float particleRadius,
    float restitutionCoefficient)
{
    const uint i = get_global_id(0);
    if(i >= numParticles)
        return;
    for (int j = 0; j < numParticles; j++)
    {
        if (i >= j) continue;

        // Calculate distance
        float4 rel = globalState[j] - globalState[i];
        float dist = length(rel);
        if (dist <= 2.0 * particleRadius) // Collision
        {
            float2 n = (float2)(rel.x / dist, rel.y / dist);
            float2 relVel = (float2)(rel.z, rel.w);
            float2 q = 0.5f * (1.0f + restitutionCoefficient) * dot(n, relVel) * n;
            globalState[i].z += q.x;
            globalState[i].w += q.y;
            globalState[j].z -= q.x;
            globalState[j].w -= q.y;
        }
    }
}
