/**
 * @file stress.cuh
 * @brief Stress.
 *
 * More detailed information.
 *
 * @author Michael Staneker, Anne Vera Jeschke
 * @bug no known bugs
 */
#ifndef MILUPHPC_STRESS_CUH
#define MILUPHPC_STRESS_CUH

#include "../particles.cuh"
#include "../parameter.h"
#include "cuda_utils/cuda_utilities.cuh"
#include "cuda_utils/cuda_runtime.h"
#include "../cuda_utils/linalg.cuh"

namespace SPH {
    namespace Kernel {
        /**
         * @brief Calculate and Set the stress
         *
         * calculate and set stresstensor \f$\sigma^{ij}\f$ for each particle from the deviatoric stress \f$S^{ij}\f$ and the pressure \f$p\f$
         * \f{align}{
                \sigma^{ij = -p\delta^{ij} + S^{ij}
         * \f}
         *
         * @param particles Particles class instance
         * @param numParticles amount of particles
         */
        __global__ void calculateStress(Particles *particles, int NumParticles);

        namespace Launch {
            /**
             * @brief Wrapper for ::SPH::Kernel::calculateStress().
             *
             * @param particles Particles class instance
             * @param numParticles amount of particles
             * @return Wall time for kernel execution
             */
            real calculateStress(Particles *particles, int NumParticles);
        }
    }
}


#endif //MILUPHPC_STRESS_CUH
