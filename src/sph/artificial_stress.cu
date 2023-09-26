#include "../../include/sph/artificial_stress.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"

#if ARTIFICIAL_STRESS
__global__ void SPH::Kernel::calculateArtificialStress(Material *materials, Particles *particles, int numParticles) {
    int i, inc;
    int d, e;
    int matId;
    real sigma[DIM][DIM];
    real epsilon;

    inc = blockDim.x * gridDim.x;
    for( i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc){
        matId = particles->materialId[i];
        epsilon = materials[matId].artificialStress.epsilon_stress;
        // get sigma
        #pragma unroll
        for (d = 0; d < DIM; d++) {
            #pragma unroll
            for (e = 0; e < DIM; e++) {
               sigma[d][e] = particles->sigma[CudaUtils::stressIndex(i,d,e)];
            }
        }
        // calculate artificial Stress
        #pragma unroll
        for (d = 0; d < DIM; d++) {
            #pragma unroll
            for (e = 0; e < DIM; e++) {
                if(sigma[d][e] > 0){
                    particles->R[CudaUtils::stressIndex(i,d,e)] = -epsilon * sigma[d][e];
                }
                else{
                    particles->R[CudaUtils::stressIndex(i,d,e)] = 0.0;
                }
            }
        }


    }
}

real SPH::Kernel::Launch::calculateArtificialStress(Material *materials,Particles *particles, int numParticles) {
    ExecutionPolicy executionPolicy;
    return cuda::launch(true, executionPolicy, ::SPH::Kernel::calculateArtificialStress, materials, particles,
                        numParticles);
    }
#endif