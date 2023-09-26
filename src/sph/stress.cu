#include "../../include/sph/stress.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"

#if SOLID
        __global__ void SPH::Kernel::calculateStress(Particles *particles, int numParticles){
            int i, inc;
            real p;
            real Sxx;
#if DIM > 1
            real Sxy;
#if DIM == 3
            real Syy, Sxz, Syz;
#endif
#endif
            real sigma[DIM][DIM];

            int d;
            int e;

            inc = blockDim.x * gridDim.x;
            for( i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc){

                Sxx = particles->Sxx[i];
#if DIM > 1
                Sxy = particles->Sxy[i];
#if DIM == 3
                Syy = particles->Syy[i];
                Sxz = particles->Sxz[i],
                Syz = particles->Syz[i];
#endif
#endif
                p = particles->p[i];

                // Calculate stress
#if DIM == 1
                sigma[0] = Sxx;
#elif DIM == 2
                //sigma = {Sxx, Sxy, Sxy, -Sxx};
                sigma[0][0] = Sxx;
                sigma[1][0] = sigma[0][1] = Sxy;
                sigma[1][1] = -Sxx;
#else
                sigma[0][0] = Sxx;
                sigma[1][0] = sigma[0][1] = Sxy;
                sigma[1][1] = Syy;
                sigma[1][2] = sigma[2][1] = Syz;
                sigma[2][0] = sigma[0][2] = Sxz;
                sigma[2][2] = -(Sxx+Syy);
#endif // DIM
#pragma unroll
                for( d = 0; d < DIM; d++){
#pragma unroll
                    for(e = 0; e < DIM; e++){
                        if(d == e) {
                            sigma[d][e] -= p;
                        }
                    }
                }

                // remember stress
#pragma unroll
                for (d = 0; d < DIM; d++) {
#pragma unroll
                    for (e = 0; e < DIM; e++) {
                        particles->sigma[CudaUtils::stressIndex(i,d,e)] = sigma[d][e];
                    }
                }

            } // particle loop
        }
real SPH::Kernel::Launch::calculateStress(Particles *particles, int numParticles) {
    ExecutionPolicy executionPolicy;
    return cuda::launch(true, executionPolicy, ::SPH::Kernel::calculateStress, particles, numParticles);
}
#endif // SOLID