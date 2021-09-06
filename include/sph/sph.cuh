#ifndef MILUPHPC_SPH_CUH
#define MILUPHPC_SPH_CUH

#include "../particles.cuh"
#include "../subdomain_key_tree/subdomain.cuh"
#include "../parameter.h"
#include "../helper.cuh"

#include <assert.h>


namespace SPH {

    namespace Kernel {
        __global__ void
        fixedRadiusNN(Tree *tree, Particles *particles, integer *interactions, integer numParticlesLocal,
                      integer numParticles, integer numNodes);

        __global__ void
        fixedRadiusNN_Test(Tree *tree, Particles *particles, integer *interactions, integer numParticlesLocal,
                                    integer numParticles, integer numNodes);

        __global__ void particles2Send(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                       DomainList *domainList, DomainList *lowestDomainList, integer maxLevel,
                                       integer *toSend, integer *sendCount, integer *alreadyInserted,
                                       integer insertOffset,
                                       integer numParticlesLocal, integer numParticles, integer numNodes, real radius,
                                       Curve::Type curveType = Curve::lebesgue);

        __global__ void info(Tree *tree, Particles *particles, Helper *helper,
                             integer numParticlesLocal, integer numParticles, integer numNodes);

        namespace Launch {

            real fixedRadiusNN(Tree *tree, Particles *particles, integer *interactions, integer numParticlesLocal,
                          integer numParticles, integer numNodes);

            real fixedRadiusNN_Test(Tree *tree, Particles *particles, integer *interactions, integer numParticlesLocal,
                               integer numParticles, integer numNodes);

            real particles2Send(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                           DomainList *domainList, DomainList *lowestDomainList, integer maxLevel,
                                           integer *toSend, integer *sendCount, integer *alreadyInserted,
                                           integer insertOffset,
                                           integer numParticlesLocal, integer numParticles, integer numNodes, real radius,
                                           Curve::Type curveType = Curve::lebesgue);

            real info(Tree *tree, Particles *particles, Helper *helper,
                                 integer numParticlesLocal, integer numParticles, integer numNodes);
        }
    }

}

#endif //MILUPHPC_SPH_CUH
