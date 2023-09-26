/**
 * @file artificial_stress.cuh
 * @brief Artificial Stress.
 *
 * Calculate the artificial stress tensor for each particle
 *
 * @author Anne Vera Jeschke
 * @bug no known bugs
 */
#ifndef MILUPHPC_ARTIFICIAL_STRESS_CUH
#define MILUPHPC_ARTIFICIAL_STRESS_CUH

#include "../particles.cuh"
#include "../parameter.h"
#include "cuda_utils/cuda_utilities.cuh"
#include "cuda_utils/cuda_runtime.h"
#include "../cuda_utils/linalg.cuh"
#include "../materials/material.cuh"

namespace SPH {
    namespace Kernel {
        /**
         * @brief Calculate and set the artificial stress
         *
         * calculate and set the artificial stress tensor \f$R^{ij}\f$ for each particle
         * \f{align}{
                R^{ij} = \begin{cases}
                -\epsilon \sigma^{ij}_a & \text{for } \sigma^{ij} > 0 \\
                0 & \text{otherwise} \\
                \end{cases}
         * \f}
         *
         * @param particles Particles class instance
         * @param numParticles amount of particles
         */
        __global__ void calculateArtificialStress(Material *materials, Particles *particles, int numParticles);

        namespace Launch {
            /**
             * @brief Wrapper for ::SPH::Kernel::artificialStress().
             *
             * @param particles Particles class instance
             * @param numParticles amount of particles
             * @return Wall time for kernel execution
             */
            real calculateArtificialStress(Material *materials, Particles *particles, int numParticles);
        }
    }
}


#endif //MILUPHPC_STRESS_CUH