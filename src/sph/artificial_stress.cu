#include "../../include/sph/artificial_stress.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"

#if ARTIFICIAL_STRESS
__device__ void SPH::calcArtificialStress(Material *materials, real sigma[DIM][DIM], real R[DIM][DIM], int matId) {
    real epsilon = materials[matId].artificialStress.epsilon_stress;
    int d, e;
#pragma unroll
    for (d = 0; d < DIM; d++) {
#pragma unroll
        for (e = 0; e < DIM; e++) {
            if(sigma[d][e] > 0){
                R[d][e] = -epsilon * sigma[d][e];
            }
            else{
               R[d][e] = 0.0;
            }
        }
    }
}
#endif