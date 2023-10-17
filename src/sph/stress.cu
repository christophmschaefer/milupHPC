#include "../../include/sph/stress.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"

#if SOLID
        __device__ void SPH::calcStress(Particles *particles, real sigma[DIM][DIM], int p) {
            int d;
            int e;
            sigma[0][0] = particles->Sxx[p];
#if DIM > 1
            sigma[0][1] = sigma[1][0] = particles->Sxy[p];
            sigma[1][1] = particles->Syy[p];
#if DIM == 3
            sigma[1][2] = sigma[2][1] = particles->Syz[p];
            sigma[2][0] = sigma[0][2] = particles->Sxz[p];
            sigma[2][2] = -(particles->Sxx[p]+particles->Syy[p]);
#endif // DIM == 3
#endif // DIM > 1
#pragma unroll
            for (d = 0; d < DIM; d++) {
#pragma unroll
                for (e = 0; e < DIM; e++) {
                    if (d == e) {
                        sigma[d][e] -= particles->p[p];
                    }
                }
            }
        }
#endif // SOLID