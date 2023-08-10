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
            real sigma[DIM*DIM];

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

                // Caluclate stress
#if DIM == 1
                sigma[0] = Sxx;
#elif DIM == 2
                //sigma = {Sxx, Sxy, Sxy, -Sxx};
                sigma[0] = Sxx;
                sigma[1] = Sxy;
                sigma[2] = Sxy;
                sigma[3] = -Sxx;
#else
                sigma[0] = Sxx;
                sigma[1] = Sxy;
                sigma[2] = Sxz;
                sigma[3] = Sxy;
                sigma[4] = Syy;
                sigma[5] = Syz;
                sigma[6] = Sxz;
                sigma[7] = Syz;
                sigma[8] = -(Sxx+Syy);
#endif // DIM

                for( d = 0; d < DIM; d++){
                    for(e = 0; e < DIM; e++){
                        if(d == e) {
                            sigma[d*DIM+e] -= p;
                        }
                    }
                }

                // remember stress
                for (d = 0; d < DIM; d++) {
                    for (e = 0; e < DIM; e++) {
                        particles->sigma[CudaUtils::stressIndex(i,d,e)] = sigma[d*DIM+e];
                    }
                }

            } // particle loop
        }
real SPH::Kernel::Launch::calculateStress(Particles *particles, int NumParticles) {
    ExecutionPolicy executionPolicy;
    return cuda::launch(true, executionPolicy, ::SPH::Kernel::calculateStress, particles, NumParticles);
}
#endif // SOLID