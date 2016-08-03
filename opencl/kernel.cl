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

    state.x += state.z * dt; // z is vx
    state.y += state.w * dt + 0.5 * gravity * dt * dt; // w is vy
    // state.z = state.z;
    state.w += gravity * dt;

    // Boundary conditions
    if(state.x < -1.0f + particleRadius){
        state.x = -1.0f + particleRadius;
        state.z *= -1;
    }
    if(state.x > 1.0f - particleRadius){
        state.x = 1.0f - particleRadius;
        state.z *= -1;
    }

    if(state.y < -1.0f + particleRadius){
        state.y = -1.0f + particleRadius;
        state.w *= -1;
    }
    if(state.y > 1.0f - particleRadius){
        state.y = 1.0f - particleRadius;
        state.w *= -1;
    }

    globalState[index] = state;
}
