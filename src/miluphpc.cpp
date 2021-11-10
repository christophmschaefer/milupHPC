#include "../include/miluphpc.h"

Miluphpc::Miluphpc(SimulationParameters simulationParameters) {

    std::ifstream f(simulationParameters.filename.c_str());
    if (!f.good()) {
        Logger(INFO) << "File " << simulationParameters.filename.c_str() << " NOT found!";
        Logger(INFO) << "Please provide an appropriate *.h5 file for the initial particle distribution!";
        MPI_Finalize();
        exit(0);
    }

    Logger(INFO) << "File " << simulationParameters.filename.c_str() << " found!";
    numParticlesFromFile(simulationParameters.filename);
    numNodes = 5 * numParticles + 20000;

    Logger(INFO) << "initialized: numParticlesLocal: " << numParticlesLocal;
    Logger(INFO) << "initialized: numParticles:      " << numParticles;
    Logger(INFO) << "initialized: numNodes:          " << numNodes;

    this->simulationParameters = simulationParameters;

    //curveType = Curve::lebesgue; //curveType = Curve::hilbert;
    curveType = Curve::Type(simulationParameters.curveType);

    cuda::malloc(d_mutex, 1);
    helperHandler = new HelperHandler(numNodes);
    buffer = new HelperHandler(numNodes);
    particleHandler = new ParticleHandler(numParticles, numNodes);
    subDomainKeyTreeHandler = new SubDomainKeyTreeHandler();
    treeHandler = new TreeHandler(numParticles, numNodes);
    domainListHandler = new DomainListHandler(DOMAIN_LIST_SIZE);
    lowestDomainListHandler = new DomainListHandler(DOMAIN_LIST_SIZE);

    // testing
    cuda::malloc(d_particles2SendIndices, numNodes); // numParticles
    cuda::malloc(d_pseudoParticles2SendIndices, numNodes);
    cuda::malloc(d_pseudoParticles2SendLevels, numNodes);
    cuda::malloc(d_pseudoParticles2ReceiveLevels, numNodes);

    cuda::malloc(d_particles2SendCount, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);
    cuda::malloc(d_pseudoParticles2SendCount, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);

    cuda::malloc(d_particles2removeBuffer, numParticles);
    cuda::malloc(d_particles2removeVal, 1);

    cuda::malloc(d_idIntegerBuffer, numParticles);
    cuda::malloc(d_idIntegerCopyBuffer, numParticles);
    // end: testing

    materialHandler = new MaterialHandler("config/material.cfg");
    for (int i=0; i<materialHandler->numMaterials; i++) {
        materialHandler->h_materials[i].info();
    }
    materialHandler->copy(To::device);

    prepareSimulation();

    //kernelHandler = SPH::KernelHandler(Smoothing::spiky);

    kernelHandler = SPH::KernelHandler(Smoothing::cubic_spline);
    simulationTimeHandler = new SimulationTimeHandler(simulationParameters.timestep, 100., 1.e12);

}

Miluphpc::~Miluphpc() {

    delete helperHandler;
    delete buffer;
    delete particleHandler;
    delete subDomainKeyTreeHandler;
    delete treeHandler;
    delete materialHandler;
    delete simulationTimeHandler;

    cuda::free(d_mutex);
    // testing
    cuda::free(d_particles2SendIndices);
    cuda::free(d_pseudoParticles2SendIndices);
    cuda::free(d_pseudoParticles2SendLevels);
    cuda::free(d_pseudoParticles2ReceiveLevels);

    cuda::free(d_particles2SendCount);
    cuda::free(d_pseudoParticles2SendCount);

    cuda::free(d_particles2removeBuffer);
    cuda::free(d_particles2removeVal);

    cuda::free(d_idIntegerBuffer);
    cuda::free(d_idIntegerCopyBuffer);
    // end: testing
}

void Miluphpc::numParticlesFromFile(const std::string& filename) {

    Logger(INFO) << "numParticlesFromFile..";

    boost::mpi::communicator comm;

    HighFive::File file(filename.c_str(), HighFive::File::ReadOnly);

    // containers to be filled
    std::vector<real> m;
    std::vector<std::vector<real>> x; //, v;

    // read datasets from file
    HighFive::DataSet mass = file.getDataSet("/m");
    HighFive::DataSet pos = file.getDataSet("/x");
    //HighFive::DataSet vel = file.getDataSet("/v");

    mass.read(m);
    pos.read(x);
    //vel.read(v);

    numParticles = m.size(); // TODO: reduce numParticles for memory

    integer ppp = m.size()/comm.size();
    integer ppp_remnant = m.size() % comm.size();

//#if DEBUGGING
    Logger(INFO) << "ppp = " << ppp;
    Logger(INFO) << "ppp remnant = " << ppp_remnant;
//#endif

    if (ppp_remnant == 0) {
        numParticlesLocal = ppp;
    }
    else {
        if (comm.rank() < (comm.size()-1)) {
            numParticlesLocal = ppp;
        }
        else {
            numParticlesLocal = ppp + ppp_remnant;
        }

    }
}

void Miluphpc::distributionFromFile(const std::string& filename) {

    HighFive::File file(filename.c_str(), HighFive::File::ReadOnly);

    // containers to be filled
    std::vector<real> m;
    std::vector<std::vector<real>> x, v;
    std::vector<integer> materialId;

    // read datasets from file
    HighFive::DataSet mass = file.getDataSet("/m");
    HighFive::DataSet pos = file.getDataSet("/x");
    HighFive::DataSet vel = file.getDataSet("/v");
#if SPH_SIM
    HighFive::DataSet matId = file.getDataSet("/materialId");
#endif

    // read data
    mass.read(m);
    pos.read(x);
    vel.read(v);
#if SPH_SIM
    matId.read(materialId);
#endif

    integer ppp = m.size()/subDomainKeyTreeHandler->h_numProcesses;
    integer ppp_remnant = m.size() % subDomainKeyTreeHandler->h_numProcesses;

    int startIndex = subDomainKeyTreeHandler->h_subDomainKeyTree->rank * ppp;
    int endIndex = (subDomainKeyTreeHandler->h_rank + 1) * ppp;
    if (subDomainKeyTreeHandler->h_rank == (subDomainKeyTreeHandler->h_numProcesses - 1)) {
        endIndex += ppp_remnant;
    }

    for (int j = startIndex; j < endIndex; j++) {
        int i = j - subDomainKeyTreeHandler->h_rank * ppp;

        particleHandler->h_particles->uid[i] = j;
        particleHandler->h_particles->mass[i] = m[j];

        particleHandler->h_particles->x[i] = x[j][0];
        particleHandler->h_particles->vx[i] = v[j][0];
#if DIM > 1
        particleHandler->h_particles->y[i] = x[j][1];
        particleHandler->h_particles->vy[i] = v[j][1];
#if DIM == 3
        particleHandler->h_particles->z[i] = x[j][2];
        particleHandler->h_particles->vz[i] = v[j][2];
#endif
#endif
#if SPH_SIM
        particleHandler->h_particles->materialId[i] = materialId[j];
        //particleHandler->h_particles->sml[i] = simulationParameters.sml;
        particleHandler->h_particles->sml[i] = materialHandler->h_materials[materialId[j]].sml;

        if (particleHandler->h_particles->sml[i] == 0. || particleHandler->h_particles->mass[i] == 0.) {
            Logger(INFO) << "i: " << i << "  sml: " << particleHandler->h_particles->sml[i]  << "   mass: " << particleHandler->h_particles->mass[i];
            exit(0);
        }

#if DIM == 3
        if (i < 5) {
            Logger(INFO) << "reading x = (" << particleHandler->h_particles->x[i] << ", " << particleHandler->h_particles->y[i] << ", " <<
                                particleHandler->h_particles->z[i] << ") " <<
                                "y = (" << particleHandler->h_particles->vx[i] << ", " << particleHandler->h_particles->vy[i] << ", " <<
                                        particleHandler->h_particles->vz[i] << ") " <<
                                "m = " << particleHandler->h_particles->mass[i] <<
                                " sml = " << particleHandler->h_particles->sml[i] <<
                                " matId = " << materialId[j];
        }
#endif
#endif
    }
}

//TODO: block/warp/stack size for computeBoundingBox and computeForces
void Miluphpc::prepareSimulation() {

    Logger(INFO) << "Preparing simulation ...";

    Logger(INFO) << "initialize particle distribution ...";
    distributionFromFile(simulationParameters.filename);

    // TODO: extend copy distribution to include sml, density, ...
    cuda::copy(particleHandler->h_sml, particleHandler->d_sml, numParticlesLocal, To::device);
    cuda::copy(particleHandler->h_materialId, particleHandler->d_materialId, numParticlesLocal, To::device);
    particleHandler->copyDistribution(To::device, true, true);

    removeParticles();

    Logger(INFO) << "compute bounding box ...";
    TreeNS::Kernel::Launch::computeBoundingBox(treeHandler->d_tree, particleHandler->d_particles, d_mutex,
                                               numParticlesLocal, 256, false);
    //debug
    treeHandler->copy(To::host);
    Logger(INFO) << "x: " << std::abs(*treeHandler->h_maxX) << ", " << std::abs(*treeHandler->h_minX);
    Logger(INFO) << "y: " << std::abs(*treeHandler->h_maxY) << ", " << std::abs(*treeHandler->h_minY);
    Logger(INFO) << "z: " << std::abs(*treeHandler->h_maxZ) << ", " << std::abs(*treeHandler->h_minZ);
    //end: debug

    treeHandler->globalizeBoundingBox(Execution::device);
    treeHandler->copy(To::host);

    if (simulationParameters.loadBalancing) {
        dynamicLoadBalancing();
    }
    else {
        fixedLoadBalancing();
    }

    subDomainKeyTreeHandler->copy(To::device);

#if SPH_SIM
    SPH::Kernel::Launch::initializeSoundSpeed(particleHandler->d_particles, materialHandler->d_materials, numParticlesLocal);
#endif

    boost::mpi::communicator comm;
    sumParticles = numParticlesLocal;
    all_reduce(comm, boost::mpi::inplace_t<integer*>(&sumParticles), 1, std::plus<integer>());

}

real Miluphpc::rhs(int step, bool selfGravity) {

    // TESTING
    //Logger(INFO) << "reduction: max:";
    //HelperNS::reduceAndGlobalize(particleHandler->d_sml, helperHandler->d_realVal, numParticlesLocal, Reduction::max);
    //Logger(INFO) << "reduction: min:";
    //HelperNS::reduceAndGlobalize(particleHandler->d_sml, helperHandler->d_realVal, numParticlesLocal, Reduction::min);
    //Logger(INFO) << "reduction: sum:";
    //HelperNS::reduceAndGlobalize(particleHandler->d_sml, helperHandler->d_realVal, numParticlesLocal, Reduction::sum);
    // end: TESTING

    Logger(INFO) << "Miluphpc::rhs()";

    Timer timer;
    real time;
    real elapsed;
    real *profilerTime = &elapsed; //&time;
    real totalTime = 0;

    // TODO: move loadBalancing outside of rhs()
    /*Logger(INFO) << "rhs::loadBalancing()";
    timer.reset();
    if (simulationParameters.loadBalancing && step != 0 && step % simulationParameters.loadBalancingInterval == 0) {
        dynamicLoadBalancing();
    }
    elapsed = timer.elapsed();
    totalTime += elapsed;
    Logger(TIME) << "rhs::loadBalancing(): " << elapsed << " ms";
    profiler.value2file(ProfilerIds::Time::loadBalancing, elapsed);*/

    Logger(INFO) << "rhs::reset()";
    timer.reset();
    time = reset();
    elapsed = timer.elapsed();
    totalTime += time;
    Logger(TIME) << "rhs::reset(): " << time << " ms";
    profiler.value2file(ProfilerIds::Time::reset, *profilerTime);

    //Logger(INFO) << "checking for nans before bounding box...";
    //ParticlesNS::Kernel::Launch::check4nans(particleHandler->d_particles, numParticlesLocal);

    Logger(INFO) << "rhs::boundingBox()";
    timer.reset();
    time = boundingBox();
    elapsed = timer.elapsed();
    totalTime += time;
    Logger(TIME) << "rhs::boundingBox(): " << time << " ms";
    profiler.value2file(ProfilerIds::Time::boundingBox, *profilerTime);

    Logger(INFO) << "before: numParticlesLocal: " << numParticlesLocal;
    Logger(INFO) << "before: numParticles:      " << numParticles;
    Logger(INFO) << "before: numNodes:          " << numNodes;

    Logger(INFO) << "checking for nans before assigning particles...";
    //ParticlesNS::Kernel::Launch::check4nans(particleHandler->d_particles, numParticlesLocal);

    if (subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses > 1) {
        Logger(INFO) << "rhs::assignParticles()";
        timer.reset();
        time = assignParticles();
        elapsed = timer.elapsed();
        totalTime += time;
        Logger(TIME) << "rhs::assignParticles(): " << time << " ms";
        profiler.value2file(ProfilerIds::Time::assignParticles, *profilerTime);
    }

    Logger(INFO) << "checking for nans after assigning particles...";
    //ParticlesNS::Kernel::Launch::check4nans(particleHandler->d_particles, numParticlesLocal);

    //Logger(INFO) << "after: numParticlesLocal: " << numParticlesLocal;
    //Logger(INFO) << "after: numParticles:      " << numParticles;
    //Logger(INFO) << "after: numNodes:          " << numNodes;

    boost::mpi::communicator comm;
    sumParticles = numParticlesLocal;
    all_reduce(comm, boost::mpi::inplace_t<integer*>(&sumParticles), 1, std::plus<integer>());

    Logger(INFO) << "numParticlesLocal = " << numParticlesLocal;

    subDomainKeyTreeHandler->copy(To::host, true, false);
    for (int i=0; i<=subDomainKeyTreeHandler->h_numProcesses; i++) {
        Logger(INFO) << "rangeValues[" << i << "] = " << subDomainKeyTreeHandler->h_subDomainKeyTree->range[i];
    }

    Logger(INFO) << "rhs::tree()";
    timer.reset();
    time = tree();
    elapsed = timer.elapsed();
    totalTime += time;
    Logger(TIME) << "rhs::tree(): " << time << " ms";
    profiler.value2file(ProfilerIds::Time::tree, *profilerTime);

    Logger(INFO) << "rhs::pseudoParticles()";
    timer.reset();
    time = pseudoParticles();
    elapsed = timer.elapsed();
    totalTime += time;
    Logger(TIME) << "rhs::pseudoParticles(): " << time << " ms";
    profiler.value2file(ProfilerIds::Time::pseudoParticle, *profilerTime);

    if (selfGravity) {
        Logger(INFO) << "rhs::gravity()";
        timer.reset();
        time = gravity();
        timer.elapsed();
        totalTime += time;
        Logger(TIME) << "rhs::gravity(): " << time << " ms";
        profiler.value2file(ProfilerIds::Time::gravity, *profilerTime);
    }

    //Logger(INFO) << "checking for nans before SPH...";
    //ParticlesNS::Kernel::Launch::check4nans(particleHandler->d_particles, numParticlesLocal);

#if SPH_SIM
    Logger(INFO) << "rhs: sph()";
    timer.reset();
    time = sph();
    elapsed = timer.elapsed();
    Logger(TIME) << "Miluphpc::sph(): " << time << " ms";
    profiler.value2file(ProfilerIds::Time::sph, *profilerTime);
    totalTime += time;
#endif

    //Logger(INFO) << "checking for nans after SPH...";
    //ParticlesNS::Kernel::Launch::check4nans(particleHandler->d_particles, numParticlesLocal);

    //DomainListNS::Kernel::Launch::info(particleHandler->d_particles, domainListHandler->d_domainList);

    // TODO: move to somewhere else (outside of rhs())
    //angularMomentum();
    //energy();

    return totalTime;

}

real Miluphpc::angularMomentum() {
    real time;

    boost::mpi::communicator comm;

    const unsigned int blockSizeReduction = 256;
    real *d_outputData;
    cuda::malloc(d_outputData, blockSizeReduction * DIM);
    cuda::set(d_outputData, (real)0., blockSizeReduction * DIM);
    time = Physics::Kernel::Launch::calculateAngularMomentumBlockwise<blockSizeReduction>(particleHandler->d_particles, d_outputData, numParticlesLocal);
    real *d_intermediateAngularMomentum;
    cuda::malloc(d_intermediateAngularMomentum, DIM);
    cuda::set(d_intermediateAngularMomentum, (real)0., DIM);
    time += Physics::Kernel::Launch::sumAngularMomentum<blockSizeReduction>(d_outputData, d_intermediateAngularMomentum);

    real *h_intermediateResult = new real[DIM];
    //cuda::copy(h_intermediateResult, d_intermediateAngularMomentum, DIM, To::host);
    //Logger(INFO) << "angular momentum before MPI: (" << h_intermediateResult[0] << ", " << h_intermediateResult[1] << ", " << h_intermediateResult[2] << ")";

    all_reduce(comm, boost::mpi::inplace_t<real*>(d_intermediateAngularMomentum), DIM, std::plus<real>());

    cuda::copy(h_intermediateResult, d_intermediateAngularMomentum, DIM, To::host);

    Logger(INFO) << "angular momentum: (" << h_intermediateResult[0] << ", " << h_intermediateResult[1] << ", " << h_intermediateResult[2] << ")";

    real angularMomentum;
#if DIM == 1
    angularMomentum = abs(h_intermediateResult[0]);
#elif DIM == 2
    angularMomentum = sqrt(h_intermediateResult[0] * h_intermediateResult[0] + h_intermediateResult[1] * h_intermediateResult[1]);
#else
    angularMomentum = sqrt(h_intermediateResult[0] * h_intermediateResult[0] + h_intermediateResult[1] * h_intermediateResult[1] +
                           h_intermediateResult[2] * h_intermediateResult[2]);
#endif

    Logger(INFO) << "angular momentum: " << angularMomentum;

    delete [] h_intermediateResult;
    cuda::free(d_outputData);
    cuda::free(d_intermediateAngularMomentum);
    Logger(TIME) << "angular momentum: " << time << " ms";
    return time;
}

real Miluphpc::energy() {

    real time = 0;

    time = Physics::Kernel::Launch::kineticEnergy(particleHandler->d_particles, numParticlesLocal);

    boost::mpi::communicator comm;

    const unsigned int blockSizeReduction = 256;
    real *d_outputData;
    cuda::malloc(d_outputData, blockSizeReduction);
    cuda::set(d_outputData, (real)0., blockSizeReduction);
    time += CudaUtils::Kernel::Launch::reduceBlockwise<real, blockSizeReduction>(particleHandler->d_u, d_outputData,
                                                                                 numParticlesLocal);
    real *d_intermediateResult;
    cuda::malloc(d_intermediateResult, 1);
    cuda::set(d_intermediateResult, (real)0., 1);
    time += CudaUtils::Kernel::Launch::blockReduction<real, blockSizeReduction>(d_outputData, d_intermediateResult);

    real h_intermediateResult;
    //cuda::copy(&h_intermediateResult, d_intermediateResult, 1, To::host);
    //Logger(INFO) << "local energy: " << h_intermediateResult;

    all_reduce(comm, boost::mpi::inplace_t<real*>(d_intermediateResult), 1, std::plus<real>());

    cuda::copy(&h_intermediateResult, d_intermediateResult, 1, To::host);
    real energy = h_intermediateResult;

    cuda::free(d_outputData);
    cuda::free(d_intermediateResult);

    Logger(INFO) << "energy: " << energy;
    Logger(TIME) << "energy: " << time << " ms";

    //cuda::copy(particleHandler->h_u, particleHandler->d_u, numParticlesLocal, To::host);
    //cuda::copy(particleHandler->h_mass, particleHandler->d_mass, numParticlesLocal, To::host);
    //for (int i=0; i<numParticlesLocal; i++) {
    //    if (i % 100 == 0 || particleHandler->h_mass[i] > 0.0001) {
    //        Logger(INFO) << "u[" << i << "] = " << particleHandler->h_u[i] << "( mass = " << particleHandler->h_mass[i] << ")";
    //    }
    //}


    return time;

}

real Miluphpc::reset() {
    real time;
    // START: resetting arrays, variables, buffers, ...
    Logger(INFO) << "resetting (device) arrays ...";
    time = Kernel::Launch::resetArrays(treeHandler->d_tree, particleHandler->d_particles, d_mutex, numParticles,
                                       numNodes, true);

    cuda::set(particleHandler->d_u, (real)0., numParticlesLocal);

    cuda::set(particleHandler->d_ax, (real)0., numParticles);
#if DIM > 1
    cuda::set(particleHandler->d_ay, (real)0., numParticles);
#if DIM == 3
    cuda::set(particleHandler->d_az, (real)0., numParticles);
#endif
#endif

    helperHandler->reset();
    buffer->reset();
    domainListHandler->reset();
    lowestDomainListHandler->reset();
    subDomainKeyTreeHandler->reset();

#if SPH_SIM
    //TODO: reset noi and nnl
    cuda::set(particleHandler->d_noi, 0, numParticles);
    cuda::set(particleHandler->d_nnl, -1, MAX_NUM_INTERACTIONS * numParticles);
#endif

    Logger(TIME) << "resetArrays: " << time << " ms";
    // END: resetting arrays, variables, buffers, ...
    return time;
}

real Miluphpc::boundingBox() {

    //ParticlesNS::Kernel::Launch::check4nans(particleHandler->d_particles, numParticlesLocal);

    real time;
    Logger(INFO) << "computing bounding box ...";
    time = TreeNS::Kernel::Launch::computeBoundingBox(treeHandler->d_tree, particleHandler->d_particles, d_mutex,
                                                      numParticlesLocal, 256, true);

    //*treeHandler->h_minX = HelperNS::reduceAndGlobalize(particleHandler->d_x, treeHandler->d_minX, numParticlesLocal, Reduction::min);
    //*treeHandler->h_maxX = HelperNS::reduceAndGlobalize(particleHandler->d_x, treeHandler->d_maxX, numParticlesLocal, Reduction::max);
    //*treeHandler->h_minY = HelperNS::reduceAndGlobalize(particleHandler->d_y, treeHandler->d_minY, numParticlesLocal, Reduction::min);
    //*treeHandler->h_maxY = HelperNS::reduceAndGlobalize(particleHandler->d_y, treeHandler->d_maxY, numParticlesLocal, Reduction::max);
    //*treeHandler->h_minZ = HelperNS::reduceAndGlobalize(particleHandler->d_z, treeHandler->d_minZ, numParticlesLocal, Reduction::min);
    //*treeHandler->h_maxZ = HelperNS::reduceAndGlobalize(particleHandler->d_z, treeHandler->d_maxZ, numParticlesLocal, Reduction::max);
    //treeHandler->copy(To::device);

    //debug
    treeHandler->copy(To::host);

    *treeHandler->h_minX *= 1.1;
    *treeHandler->h_maxX *= 1.1;
    *treeHandler->h_minY *= 1.1;
    *treeHandler->h_maxY *= 1.1;
    *treeHandler->h_minZ *= 1.1;
    *treeHandler->h_maxZ *= 1.1;

    treeHandler->copy(To::device);

    Logger(INFO) << "x: max = " << *treeHandler->h_maxX << ", min = " << *treeHandler->h_minX;
    Logger(INFO) << "y: max = " << *treeHandler->h_maxY << ", min = " << *treeHandler->h_minY;
    Logger(INFO) << "z: max = " << *treeHandler->h_maxZ << ", min = " << *treeHandler->h_minZ;
    //end: debug

    treeHandler->globalizeBoundingBox(Execution::device);
    treeHandler->copy(To::host);

#if DIM == 1
    Logger(INFO) << "Bounding box: x = (" << *treeHandler->h_minX << ", " << *treeHandler->h_maxX << ")";
#elif DIM == 2
    Logger(INFO) << "Bounding box: x = (" << *treeHandler->h_minX << ", " << *treeHandler->h_maxX << ")" << "y = ("
                 << *treeHandler->h_minY << ", " << *treeHandler->h_maxY << ")";
#else
    Logger(INFO) << "Bounding box: x = (" << std::setprecision(9) << *treeHandler->h_minX << ", " << *treeHandler->h_maxX << ")"
                    << "y = (" << *treeHandler->h_minY << ", " << *treeHandler->h_maxY << ")"
                    << "z = " << *treeHandler->h_minZ << ", " << *treeHandler->h_maxZ << ")";
#endif

    Logger(TIME) << "computeBoundingBox: " << time << " ms";
    return time;
}

real Miluphpc::assignParticles() {
    real time;
    time = SubDomainKeyTreeNS::Kernel::Launch::particlesPerProcess(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                                   treeHandler->d_tree, particleHandler->d_particles,
                                                                   numParticlesLocal, numNodes, curveType);


    int *d_particlesProcess = helperHandler->d_integerBuffer;
    int *d_particlesProcessSorted = buffer->d_integerBuffer;
    real *d_tempEntry = helperHandler->d_realBuffer;
    real *d_copyBuffer = buffer->d_realBuffer;
    idInteger *d_idIntTempEntry = d_idIntegerBuffer;
    idInteger *d_idIntCopyBuffer = d_idIntegerCopyBuffer;

    time += SubDomainKeyTreeNS::Kernel::Launch::markParticlesProcess(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                                     treeHandler->d_tree, particleHandler->d_particles,
                                                                     numParticlesLocal, numNodes,
                                                                     d_particlesProcess, curveType);

    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_x, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_vx, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_ax, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_g_ax, d_tempEntry);
#if DIM > 1
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_y, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_vy, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_ay, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_g_ay, d_tempEntry);
#if DIM == 3
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_z, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_vz, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_az, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_g_az, d_tempEntry);
#endif
#endif
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_mass, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_uid, d_idIntTempEntry);

#if SPH_SIM
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_sml, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_e, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_rho, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_cs, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_p, d_tempEntry);
    time += arrangeParticleEntries(d_particlesProcess, d_particlesProcessSorted, particleHandler->d_materialId, d_idIntTempEntry);
#endif

    //TODO: assignParticles: for all entries (sorting/arranging particles locally)...

    subDomainKeyTreeHandler->copy(To::host, true, true);

    Timer timer;
    integer *sendLengths;
    sendLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    sendLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;
    integer *receiveLengths;
    receiveLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    receiveLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;

    for (int proc=0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            sendLengths[proc] = subDomainKeyTreeHandler->h_procParticleCounter[proc];
        }
    }
    mpi::messageLengths(subDomainKeyTreeHandler->h_subDomainKeyTree, sendLengths, receiveLengths);

    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_x, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_vx, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_ax, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_g_ax, d_tempEntry, d_copyBuffer);
#if DIM > 1
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_y, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_vy, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_ay, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_g_ay, d_tempEntry, d_copyBuffer);
#if DIM == 3
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_z, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_vz, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_az, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_g_az, d_tempEntry, d_copyBuffer);
#endif
#endif
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_uid, d_idIntTempEntry, d_idIntCopyBuffer);

#if SPH_SIM
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_sml, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_e, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_rho, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_cs, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_p, d_tempEntry, d_copyBuffer);
    sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_materialId, d_idIntTempEntry, d_idIntCopyBuffer);
#endif
    numParticlesLocal = sendParticlesEntry(sendLengths, receiveLengths, particleHandler->d_mass, d_tempEntry, d_copyBuffer);

    delete [] sendLengths;
    delete [] receiveLengths;

    //real timeSendingParticles = timer.elapsed();
    time += timer.elapsed();

    int resetLength = numParticles-numParticlesLocal;
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_x[numParticlesLocal], (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_vx[numParticlesLocal], (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_ax[numParticlesLocal], (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_g_ax[numParticlesLocal], (real)0, resetLength);
#if DIM > 1
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_y[numParticlesLocal], (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_vy[numParticlesLocal], (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_ay[numParticlesLocal], (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_g_ay[numParticlesLocal], (real)0, resetLength);
#if DIM == 3
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_z[numParticlesLocal], (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_vz[numParticlesLocal], (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_az[numParticlesLocal], (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_g_az[numParticlesLocal], (real)0, resetLength);
#endif
#endif
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_mass[numParticlesLocal], (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_uid[numParticlesLocal], (idInteger)0, resetLength);

#if SPH_SIM
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_sml[numParticlesLocal], (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_e[numParticlesLocal], (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_rho[numParticlesLocal], (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_cs[numParticlesLocal], (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_p[numParticlesLocal], (real)0, resetLength);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_materialId[numParticlesLocal], (integer)0, resetLength);
#endif

    //TODO: assignParticles: for all entries (exchanging Particles via MPI)...

    return time;
}

template <typename T>
real Miluphpc::arrangeParticleEntries(T *entry, T *temp) {
    real time;
    time = HelperNS::sortArray(entry, temp, helperHandler->d_integerBuffer, buffer->d_integerBuffer, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(entry, temp, numParticlesLocal);
    return time;
}

template <typename U, typename T>
real Miluphpc::arrangeParticleEntries(U *sortArray, U *sortedArray, T *entry, T *temp) {
    real time;
    time = HelperNS::sortArray(entry, temp, sortArray, sortedArray, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(entry, temp, numParticlesLocal);
    return time;
}

real Miluphpc::tree() {

    real time = parallel_tree();

    return time;
}

real Miluphpc::parallel_tree() {
    real time;
    real totalTime = 0.;

    // START: creating domain list
    Logger(INFO) << "building domain list ...";
    time = DomainListNS::Kernel::Launch::createDomainList(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                          domainListHandler->d_domainList, MAX_LEVEL,
                                                          curveType);
    totalTime += time;
    Logger(TIME) << "createDomainList: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::Tree::createDomain, time);

    integer domainListLength;
    cuda::copy(&domainListLength, domainListHandler->d_domainListIndex, 1, To::host);
    Logger(INFO) << "domainListLength = " << domainListLength;
    // END: creating domain list

    // START: tree construction (including common coarse tree)
    integer treeIndexBeforeBuildingTree;

    cuda::copy(&treeIndexBeforeBuildingTree, treeHandler->d_index, 1, To::host);
    Logger(INFO) << "treeIndexBeforeBuildingTree: " << treeIndexBeforeBuildingTree;

    Logger(INFO) << "checking for nans before building tree...";
    //ParticlesNS::Kernel::Launch::check4nans(particleHandler->d_particles, numParticlesLocal);

    Logger(INFO) << "building tree ...";
    time = TreeNS::Kernel::Launch::buildTree(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal,
                                             numParticles, true);

    totalTime += time;
    Logger(TIME) << "buildTree: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::Tree::tree, time);

    integer treeIndex;
    cuda::copy(&treeIndex, treeHandler->d_index, 1, To::host);

#if DEBUGGING
    Logger(INFO) << "numParticlesLocal: " << numParticlesLocal;
    Logger(INFO) << "numParticles: " << numParticles;
    Logger(INFO) << "numNodes: " << numNodes;
    Logger(INFO) << "treeIndex: " << treeIndex;
    integer numParticlesSum = numParticlesLocal;
    boost::mpi::communicator comm;
    all_reduce(comm, boost::mpi::inplace_t<integer*>(&numParticlesSum), 1, std::plus<integer>());
    Logger(INFO) << "numParticlesSum: " << numParticlesSum;
    //ParticlesNS::Kernel::Launch::info(particleHandler->d_particles, numParticlesLocal, numParticles, treeIndex);
#endif

    Logger(INFO) << "building domain tree ...";
    cuda::set(domainListHandler->d_domainListCounter, 0, 1);

    // serial version
    //time = SubDomainKeyTreeNS::Kernel::Launch::buildDomainTree(treeHandler->d_tree, particleHandler->d_particles,
    //                                                           domainListHandler->d_domainList, numParticlesLocal,
    //                                                           numNodes);

    time = 0;
    for (int level = 0; level <= MAX_LEVEL; level++) {
        time += SubDomainKeyTreeNS::Kernel::Launch::buildDomainTree(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                                    treeHandler->d_tree,
                                                                    particleHandler->d_particles,
                                                                    domainListHandler->d_domainList,
                                                                    numParticlesLocal,
                                                                    numNodes, level);
    }
    int domainListCounterAfterwards;
    cuda::copy(&domainListCounterAfterwards, domainListHandler->d_domainListCounter, 1, To::host);
    Logger(INFO) << "domain list counter afterwards : " << domainListCounterAfterwards;
    cuda::set(domainListHandler->d_domainListCounter, 0, 1);
    // END: tree construction (including common coarse tree)
    totalTime += time;
    Logger(TIME) << "build(Domain)Tree: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::Tree::buildDomain, time);

    //DomainListNS::Kernel::Launch::info(particleHandler->d_particles, domainListHandler->d_domainList);

    return totalTime;
}

real Miluphpc::pseudoParticles() {

    real time = parallel_pseudoParticles();

    return time;
}

real Miluphpc::parallel_pseudoParticles() {

    real time = 0;
    time += DomainListNS::Kernel::Launch::lowestDomainList(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                                           domainListHandler->d_domainList,
                                                           lowestDomainListHandler->d_domainList, numParticles, numNodes);


    //ParticlesNS::Kernel::Launch::check4nans(particleHandler->d_particles, numParticlesLocal);
    
    real timeCOM = 0;
    for (int level=MAX_LEVEL; level>0; --level) {
        timeCOM += TreeNS::Kernel::Launch::calculateCentersOfMass(treeHandler->d_tree, particleHandler->d_particles, numParticles,
                                                               level, true);
    }

    Logger(TIME) << "calculate COM: " << timeCOM << " ms";
    time += timeCOM;

    Gravity::Kernel::Launch::zeroDomainListNodes(particleHandler->d_particles, domainListHandler->d_domainList,
                                                 lowestDomainListHandler->d_domainList);

    // old version
    //time += Gravity::Kernel::Launch::compLocalPseudoParticles(treeHandler->d_tree, particleHandler->d_particles,
    //                                                          domainListHandler->d_domainList, numParticles);

    //DomainListNS::Kernel::Launch::info(particleHandler->d_particles, domainListHandler->d_domainList);

    integer domainListIndex;
    integer lowestDomainListIndex;

    cuda::copy(&domainListIndex, domainListHandler->d_domainListIndex, 1, To::host);
    cuda::copy(&lowestDomainListIndex, lowestDomainListHandler->d_domainListIndex, 1, To::host);

    Logger(INFO) << "domainListIndex: " << domainListIndex << " | lowestDomainListIndex: " << lowestDomainListIndex;

    boost::mpi::communicator comm;

    //TODO: current approach reasonable
    // or template functions and explicitly hand over buffer(s) (and not instance of buffer class)

    // x ----------------------------------------------------------------------------------------------

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);
    time += Gravity::Kernel::Launch::prepareLowestDomainExchange(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                                 helperHandler->d_helper, Entry::x);

    time += HelperNS::sortArray(helperHandler->d_realBuffer, &helperHandler->d_realBuffer[DOMAIN_LIST_SIZE],
                                lowestDomainListHandler->d_domainListKeys, lowestDomainListHandler->d_sortedDomainListKeys,
                                domainListIndex);

    // share among processes
    //TODO: domainListIndex or lowestDomainListIndex?
    all_reduce(comm, boost::mpi::inplace_t<real*>(&helperHandler->d_realBuffer[DOMAIN_LIST_SIZE]), domainListIndex,
               std::plus<real>());

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);

    time += Gravity::Kernel::Launch::updateLowestDomainListNodes(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                                 helperHandler->d_helper, Entry::x);

#if DIM > 1
    // y ----------------------------------------------------------------------------------------------

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);
    time += Gravity::Kernel::Launch::prepareLowestDomainExchange(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                                 helperHandler->d_helper, Entry::y);

    time += HelperNS::sortArray(helperHandler->d_realBuffer, &helperHandler->d_realBuffer[DOMAIN_LIST_SIZE],
                                lowestDomainListHandler->d_domainListKeys, lowestDomainListHandler->d_sortedDomainListKeys,
                                domainListIndex);

    all_reduce(comm, boost::mpi::inplace_t<real*>(&helperHandler->d_realBuffer[DOMAIN_LIST_SIZE]), domainListIndex,
               std::plus<real>());

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);

    time += Gravity::Kernel::Launch::updateLowestDomainListNodes(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                                 helperHandler->d_helper, Entry::y);

#if DIM == 3
    // z ----------------------------------------------------------------------------------------------

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);
    time += Gravity::Kernel::Launch::prepareLowestDomainExchange(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                                 helperHandler->d_helper, Entry::z);

    time += HelperNS::sortArray(helperHandler->d_realBuffer, &helperHandler->d_realBuffer[DOMAIN_LIST_SIZE],
                                lowestDomainListHandler->d_domainListKeys, lowestDomainListHandler->d_sortedDomainListKeys,
                                domainListIndex);

    all_reduce(comm, boost::mpi::inplace_t<real*>(&helperHandler->d_realBuffer[DOMAIN_LIST_SIZE]), domainListIndex,
               std::plus<real>());

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);

    time += Gravity::Kernel::Launch::updateLowestDomainListNodes(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                                 helperHandler->d_helper, Entry::z);

#endif
#endif
    // m ----------------------------------------------------------------------------------------------

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);
    time += Gravity::Kernel::Launch::prepareLowestDomainExchange(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                                 helperHandler->d_helper, Entry::mass);

    time += HelperNS::sortArray(helperHandler->d_realBuffer, &helperHandler->d_realBuffer[DOMAIN_LIST_SIZE],
                                lowestDomainListHandler->d_domainListKeys, lowestDomainListHandler->d_sortedDomainListKeys,
                                domainListIndex);

    all_reduce(comm, boost::mpi::inplace_t<real*>(&helperHandler->d_realBuffer[DOMAIN_LIST_SIZE]), domainListIndex,
               std::plus<real>());

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0);

    time += Gravity::Kernel::Launch::updateLowestDomainListNodes(particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                                                 helperHandler->d_helper, Entry::mass);

    // ------------------------------------------------------------------------------------------------

    time += Gravity::Kernel::Launch::compLowestDomainListNodes(treeHandler->d_tree, particleHandler->d_particles,
                                                               lowestDomainListHandler->d_domainList);
    //end: for all entries!

    // per level computation of domain list pseudo-particles to ensure the correct order (avoid race condition)
    for (int domainLevel = MAX_LEVEL; domainLevel>= 0; domainLevel--) {
        time += Gravity::Kernel::Launch::compDomainListPseudoParticlesPerLevel(treeHandler->d_tree, particleHandler->d_particles,
                                                                               domainListHandler->d_domainList,
                                                                               lowestDomainListHandler->d_domainList,
                                                                               numParticles, domainLevel);
    }

    return time;
}

real Miluphpc::gravity() {

    real time = parallel_gravity();

    return time;
}

real Miluphpc::parallel_gravity() {

    real time;
    real totalTime = 0;

    totalTime += HelperNS::Kernel::Launch::resetArray(helperHandler->d_realBuffer, (real)0, numParticles);

    cuda::set(domainListHandler->d_domainListCounter, 0);

    Logger(INFO) << "compTheta()";
    time = Gravity::Kernel::Launch::compTheta(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                              particleHandler->d_particles, domainListHandler->d_domainList,
                                              helperHandler->d_helper, curveType);
    totalTime += time;
    Logger(TIME) << "compTheta(): " << time << " ms";
    profiler.value2file(ProfilerIds::Time::Gravity::compTheta, time);

    integer relevantIndicesCounter;
    cuda::copy(&relevantIndicesCounter, domainListHandler->d_domainListCounter, 1, To::host);

    Logger(INFO) << "relevantIndicesCounter: " << relevantIndicesCounter;

    integer *h_relevantDomainListProcess;
    h_relevantDomainListProcess = new integer[relevantIndicesCounter]; //TODO: delete [] h_relevantDomainListProcess;
    cuda::copy(h_relevantDomainListProcess, domainListHandler->d_relevantDomainListProcess, relevantIndicesCounter, To::host);

    for (int i=0; i<relevantIndicesCounter; i++) {
        Logger(INFO) << "relevantDomainListProcess[" << i << "] = " << h_relevantDomainListProcess[i];
    }

    treeHandler->copy(To::host);

#if CUBIC_DOMAINS
    real diam = std::abs(*treeHandler->h_maxX) + std::abs(*treeHandler->h_minX);
    Logger(INFO) << "diam: " << diam;
#else // !CUBIC DOMAINS
    real diam_x = std::abs(*treeHandler->h_maxX) + std::abs(*treeHandler->h_minX);
#if DIM > 1
    real diam_y = std::abs(*treeHandler->h_maxY) + std::abs(*treeHandler->h_minY);
#if DIM == 3
    real diam_z = std::abs(*treeHandler->h_maxZ) + std::abs(*treeHandler->h_minZ);
#endif
#endif
#if DIM == 1
    real diam = diam_x;
    Logger(INFO) << "diam: " << diam << "  (x = " << diam_x << ")";
#elif DIM == 2
    real diam = std::max({diam_x, diam_y});
    Logger(INFO) << "diam: " << diam << "  (x = " << diam_x << ", y = " << diam_y << ")";
#else
    real diam = std::max({diam_x, diam_y, diam_z});
    Logger(INFO) << "diam: " << diam << "  (x = " << diam_x << ", y = " << diam_y << ", z = " << diam_z << ")";
    if (diam > 1e250) {
        Logger(INFO) << "x: " << std::abs(*treeHandler->h_maxX) << ", " << std::abs(*treeHandler->h_minX);
        Logger(INFO) << "y: " << std::abs(*treeHandler->h_maxY) << ", " << std::abs(*treeHandler->h_minY);
        Logger(INFO) << "z: " << std::abs(*treeHandler->h_maxZ) << ", " << std::abs(*treeHandler->h_minZ);
        MPI_Finalize();
        exit(0);
    }
#endif
#endif // CUBIC_DOMAINS

    // TODO: convert theta to a run-time constant (should not be a preprocessor directive)
    real theta_ = theta; //0.5f;

    // TODO: create buffer concept and (re)use buffer for gravity internode communicaton
    cuda::set(domainListHandler->d_domainListCounter, 0);
    integer *d_markedSendIndices = buffer->d_integerBuffer;
    real *d_collectedEntries = buffer->d_realBuffer;

    cuda::set(d_particles2SendIndices, -1, numParticles);
    cuda::set(d_pseudoParticles2SendIndices, -1, numParticles);
    cuda::set(d_pseudoParticles2SendLevels, -1, numParticles);
    cuda::set(d_pseudoParticles2ReceiveLevels, -1, numParticles);

    cuda::set(d_particles2SendCount, 0, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);
    cuda::set(d_pseudoParticles2SendCount, 0, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);

    integer particlesOffset = 0;
    integer pseudoParticlesOffset = 0;

    integer particlesOffsetBuffer;
    integer pseudoParticlesOffsetBuffer;

    integer *h_particles2SendCount = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    integer *h_pseudoParticles2SendCount = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];

    //DomainListNS::Kernel::Launch::info(particleHandler->d_particles, domainListHandler->d_domainList);

    time = 0;
    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            cuda::set(d_markedSendIndices, -1, numNodes);
            for (int level = 0; level < MAX_LEVEL; level++) {
                time += Gravity::Kernel::Launch::intermediateSymbolicForce(subDomainKeyTreeHandler->d_subDomainKeyTree,
                        treeHandler->d_tree, particleHandler->d_particles, domainListHandler->d_domainList,
                        d_markedSendIndices, diam, theta_, numParticlesLocal, numParticles,0, level,
                        curveType);
                for (int relevantIndex = 0; relevantIndex < relevantIndicesCounter; relevantIndex++) {
                    if (h_relevantDomainListProcess[relevantIndex] == proc) {
                        //Logger(INFO) << "h_relevantDomainListProcess[" << relevantIndex << "] = "
                        //             << h_relevantDomainListProcess[relevantIndex];
                        time += Gravity::Kernel::Launch::symbolicForce(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                                       treeHandler->d_tree, particleHandler->d_particles,
                                                                       domainListHandler->d_domainList,
                                                                       d_markedSendIndices, diam, theta_,
                                                                       numParticlesLocal, numParticles,
                                                                       relevantIndex, level, curveType);
                    }
                }
            }
            time += Gravity::Kernel::Launch::collectSendIndices(treeHandler->d_tree, particleHandler->d_particles,
                                                                d_markedSendIndices,
                                                                &d_particles2SendIndices[particlesOffset],
                                                                &d_pseudoParticles2SendIndices[pseudoParticlesOffset],
                                                                &d_pseudoParticles2SendLevels[pseudoParticlesOffset],
                                                                &d_particles2SendCount[proc],
                                                                &d_pseudoParticles2SendCount[proc],
                                                                numParticles, numNodes, curveType);

            cuda::copy(&particlesOffsetBuffer, &d_particles2SendCount[proc], 1, To::host);
            cuda::copy(&pseudoParticlesOffsetBuffer, &d_pseudoParticles2SendCount[proc], 1, To::host);

            Logger(INFO) << "particles2SendCount[" << proc << "] = " << particlesOffsetBuffer;
            Logger(INFO) << "pseudoParticles2SendCount[" << proc << "] = " << pseudoParticlesOffsetBuffer;

            particlesOffset += particlesOffsetBuffer;
            pseudoParticlesOffset += pseudoParticlesOffsetBuffer;
        }
    }

/*#if DEBUGGING
    Gravity::Kernel::Launch::testSendIndices(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                             particleHandler->d_particles, d_pseudoParticles2SendIndices,
                                             d_markedSendIndices,
                                             d_pseudoParticles2SendLevels, curveType, pseudoParticlesOffset);
#endif*/

    totalTime += time;
    Logger(TIME) << "symbolicForce: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::Gravity::symbolicForce, time);

    Timer timer;

    cuda::copy(h_particles2SendCount, d_particles2SendCount, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses, To::host);
    cuda::copy(h_pseudoParticles2SendCount, d_pseudoParticles2SendCount, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses, To::host);

    integer *particleSendLengths;
    particleSendLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    particleSendLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;
    integer *particleReceiveLengths;
    particleReceiveLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    particleReceiveLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;

    integer *pseudoParticleSendLengths;
    pseudoParticleSendLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    pseudoParticleSendLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;
    integer *pseudoParticleReceiveLengths;
    pseudoParticleReceiveLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    pseudoParticleReceiveLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;

    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        //Logger(INFO) << "h_particles2SendCount[" << proc << "] = " << h_particles2SendCount[proc];
        //Logger(INFO) << "h_pseudoParticles2SendCount[" << proc << "] = " << h_pseudoParticles2SendCount[proc];
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            particleSendLengths[proc] = h_particles2SendCount[proc];
            pseudoParticleSendLengths[proc] = h_pseudoParticles2SendCount[proc];
            Logger(INFO) << "particleSendLengths[" << proc << "] = " << particleSendLengths[proc];
            Logger(INFO) << "pseudoParticleSendLengths[" << proc << "] = " << pseudoParticleSendLengths[proc];
        }
    }

    mpi::messageLengths(subDomainKeyTreeHandler->h_subDomainKeyTree, particleSendLengths, particleReceiveLengths);

    integer particleTotalReceiveLength = 0;
    integer particleTotalSendLength = 0;
    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            particleTotalReceiveLength += particleReceiveLengths[proc];
            particleTotalSendLength += particleSendLengths[proc];
        }
    }

    Logger(INFO) << "particleTotalReceiveLength: " << particleTotalReceiveLength;
    Logger(INFO) << "particleTotalSendLength: " << particleTotalSendLength;

    mpi::messageLengths(subDomainKeyTreeHandler->h_subDomainKeyTree, pseudoParticleSendLengths, pseudoParticleReceiveLengths);

    integer pseudoParticleTotalReceiveLength = 0;
    integer pseudoParticleTotalSendLength = 0;
    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            pseudoParticleTotalReceiveLength += pseudoParticleReceiveLengths[proc];
            pseudoParticleTotalSendLength += pseudoParticleSendLengths[proc];
            Logger(INFO) << "particleReceiveLengths[" << proc << "] = " << particleReceiveLengths[proc];
            Logger(INFO) << "pseudoParticleReceiveLengths[" << proc << "] = " << pseudoParticleReceiveLengths[proc];
        }
    }

    //Gravity::Kernel::Launch::testSendIndices(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
    //                                         particleHandler->d_particles, d_pseudoParticles2SendIndices,
    //                                         d_markedSendIndices,
    //                                         d_pseudoParticles2SendLevels, curveType, pseudoParticleTotalSendLength);

    // debug
    //particleHandler->copyDistribution(To::host, false, false, true);
    //int *h_pseudoParticlesSendIndices = new int[pseudoParticleTotalSendLength];
    //cuda::copy(h_pseudoParticlesSendIndices, d_pseudoParticles2SendIndices, pseudoParticleTotalSendLength, To::host);
    //int debug_offset = 0;
    //for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
    //    if (proc != proc < subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
    //        for (int i = 0; i < pseudoParticleSendLengths[proc]; i++) {
    //            for (int j = 0; j < pseudoParticleSendLengths[proc]; j++) {
    //                if (i != j) {
    //                    if (h_pseudoParticlesSendIndices[i + debug_offset] == h_pseudoParticlesSendIndices[j + debug_offset] ||
    //                            (particleHandler->h_x[h_pseudoParticlesSendIndices[i + debug_offset]] == particleHandler->h_x[h_pseudoParticlesSendIndices[j + debug_offset]] &&
    //                            particleHandler->h_y[h_pseudoParticlesSendIndices[i + debug_offset]] == particleHandler->h_y[h_pseudoParticlesSendIndices[j + debug_offset]])) {
    //                        Logger(INFO) << "found duplicate regarding proc " << proc << ": index: i: " << i << " = "
    //                                     << h_pseudoParticlesSendIndices[i + debug_offset]
    //                                     << ", j: " << j << " = " << h_pseudoParticlesSendIndices[j + debug_offset] <<
    //                                     " (" <<  particleHandler->h_x[h_pseudoParticlesSendIndices[i + debug_offset]]
    //                                     << ", " << particleHandler->h_x[h_pseudoParticlesSendIndices[j + debug_offset]] << ")";
    //                    }
    //                }
    //            }
    //        }
    //        debug_offset += pseudoParticleSendLengths[proc];
    //    }
    //}
    //delete [] h_pseudoParticlesSendIndices;
    // end: debug

    Logger(INFO) << "pseudoParticleTotalReceiveLength: " << pseudoParticleTotalReceiveLength;
    Logger(INFO) << "pseudoParticleTotalSendLength: " << pseudoParticleTotalSendLength;

    integer treeIndex;
    cuda::copy(&treeIndex, treeHandler->d_index, 1, To::host);

    // WRITING PARTICLES TO SEND TO H5 FILE
    // writing particles
    //int *h_particles2SendIndices = new int[particleTotalSendLength];
    //cuda::copy(h_particles2SendIndices, d_particles2SendIndices, particleTotalSendLength, To::host);
    //std::string filename = "Gravity2SendParticles";
    //particles2file(filename, h_particles2SendIndices, particleTotalSendLength);
    //delete [] h_particles2SendIndices;
    // end: writing particles

    // writing pseudo-particles
    //int *h_pseudoParticles2SendIndices = new int[pseudoParticleTotalSendLength];
    //cuda::copy(h_pseudoParticles2SendIndices, d_pseudoParticles2SendIndices, pseudoParticleTotalSendLength, To::host);
    //std::string filename = "Gravity2SendPseudoParticles";
    //particles2file(filename, h_pseudoParticles2SendIndices, pseudoParticleTotalSendLength);
    //delete [] h_pseudoParticles2SendIndices;
    // end: writing pseudo-particles

    // writing both: particles and pseudo-particles
    //int *h_sendIndices = new int[particleTotalSendLength + pseudoParticleTotalSendLength];
    //cuda::copy(&h_sendIndices[0], d_particles2SendIndices, particleTotalSendLength, To::host);
    //cuda::copy(&h_sendIndices[particleTotalSendLength], d_pseudoParticles2SendIndices, pseudoParticleTotalSendLength, To::host);
    //std::string filename = "Gravity2SendBoth";
    //particles2file(filename, h_sendIndices, particleTotalSendLength + pseudoParticleTotalSendLength);
    //delete [] h_sendIndices;
    // end: writing both: particles and pseudo-particles
    // END: WRITING PARTICLES TO SEND TO H5 FILE

#if DEBUGGING
    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles, treeIndex,
    //                                 treeIndex + pseudoParticleTotalReceiveLength);
    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal,
    //                                 numParticlesLocal + particleTotalReceiveLength);
#endif

    // x-entry pseudo-particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_pseudoParticles2SendIndices, particleHandler->d_x, d_collectedEntries,
                                             pseudoParticleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_x[treeIndex], pseudoParticleSendLengths,
                  pseudoParticleReceiveLengths);
    // x-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_x, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_x[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    //Particle velocity and acceleration:
    // vx-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_vx, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_vx[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // ax-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_ax, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_ax[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
#if DIM > 1
    // y-entry pseudo-particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_pseudoParticles2SendIndices, particleHandler->d_y, d_collectedEntries,
                                             pseudoParticleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_y[treeIndex], pseudoParticleSendLengths,
                  pseudoParticleReceiveLengths);
    // y-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_y, d_collectedEntries,
                                             particleTotalSendLength);

    sendParticles(d_collectedEntries, &particleHandler->d_y[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    //Particle velocity and acceleration:
    // vy-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_vy, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_vy[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // ay-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_ay, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_ay[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
#if DIM == 3
    // z-entry pseudo-particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_pseudoParticles2SendIndices, particleHandler->d_z, d_collectedEntries,
                                             pseudoParticleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_z[treeIndex], pseudoParticleSendLengths,
                  pseudoParticleReceiveLengths);
    // z-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_z, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_z[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    //Particle velocity and acceleration:
    // vz-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_vz, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_vz[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // az-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_az, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_az[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

#endif
#endif

    // mass-entry pseudo-particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_pseudoParticles2SendIndices, particleHandler->d_mass, d_collectedEntries,
                                             pseudoParticleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_mass[treeIndex], pseudoParticleSendLengths,
                  pseudoParticleReceiveLengths);
    // mass-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_mass, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_mass[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    // PSEUDO-PARTICLE level exchange
    sendParticles(d_pseudoParticles2SendLevels, d_pseudoParticles2ReceiveLevels, pseudoParticleSendLengths,
                  pseudoParticleReceiveLengths);

    time = timer.elapsed();
    totalTime += time;
    Logger(TIME) << "parallel_force(): sending particles: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::Gravity::sending, time);

#if DEBUGGING
    //Logger(INFO) << "exchanged particle entry: x";
    //if (subDomainKeyTreeHandler->h_subDomainKeyTree->rank == 0) {
    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles, treeIndex,
    //                             treeIndex + pseudoParticleTotalReceiveLength);
    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal,
    //                             numParticlesLocal + particleTotalReceiveLength);
    //}
#endif

    treeHandler->h_toDeleteLeaf[0] = numParticlesLocal;
    treeHandler->h_toDeleteLeaf[1] = numParticlesLocal + particleTotalReceiveLength;
    cuda::copy(treeHandler->h_toDeleteLeaf, treeHandler->d_toDeleteLeaf, 2, To::device);

    //int debugOffset = 0;
    //for (int proc=0; proc<subDomainKeyTreeHandler->h_numProcesses; proc++) {
    //    gpuErrorcheck(cudaMemset(buffer->d_integerVal, 0, sizeof(integer)));
    //    CudaUtils::Kernel::Launch::findDuplicateEntries(&particleHandler->d_x[treeIndex + debugOffset],
    //                                                    &particleHandler->d_y[treeIndex + debugOffset],
    //                                                    buffer->d_integerVal,
    //                                                    particleReceiveLengths[proc]);
    //    integer duplicates;
    //    gpuErrorcheck(cudaMemcpy(&duplicates, buffer->d_integerVal, sizeof(integer), cudaMemcpyDeviceToHost));
    //    Logger(INFO) << "duplicates: " << duplicates << " between: " << treeIndex + debugOffset << " and " << treeIndex + particleReceiveLengths[proc] + debugOffset;
    //
    //    debugOffset += particleReceiveLengths[proc];
    //}


/*#if DEBUGGING
    gpuErrorcheck(cudaMemset(buffer->d_integerVal, 0, sizeof(integer)));
    CudaUtils::Kernel::Launch::findDuplicateEntries(&particleHandler->d_x[treeHandler->h_toDeleteLeaf[0]],
                                                    &particleHandler->d_y[treeHandler->h_toDeleteLeaf[0]],
                                                    buffer->d_integerVal,
                                                    particleTotalReceiveLength);
    integer duplicates;
    gpuErrorcheck(cudaMemcpy(&duplicates, buffer->d_integerVal, sizeof(integer), cudaMemcpyDeviceToHost));
    Logger(INFO) << "duplicates: " << duplicates << " between: " << treeHandler->h_toDeleteLeaf[0] << " and " << treeHandler->h_toDeleteLeaf[0] + particleTotalReceiveLength;
#endif*/

#if DEBUGGING
    // debugging
    gpuErrorcheck(cudaMemset(buffer->d_integerVal, 0, sizeof(integer)));
    CudaUtils::Kernel::Launch::findDuplicateEntries(&particleHandler->d_x[0],
                                                    &particleHandler->d_y[0],
                                                    buffer->d_integerVal,
                                                    numParticlesLocal + particleTotalReceiveLength);
    integer duplicates;
    gpuErrorcheck(cudaMemcpy(&duplicates, buffer->d_integerVal, sizeof(integer), cudaMemcpyDeviceToHost));
    Logger(INFO) << "duplicates: " << duplicates << " between: " << 0 << " and " << numParticlesLocal + particleTotalReceiveLength;
    // end: debugging
#endif

    treeHandler->h_toDeleteNode[0] = treeIndex;
    treeHandler->h_toDeleteNode[1] = treeIndex + pseudoParticleTotalReceiveLength;
    cuda::copy(treeHandler->h_toDeleteNode, treeHandler->d_toDeleteNode, 2, To::device);

//#if DEBUGGING
    Logger(INFO) << "toDeleteLeaf: " << treeHandler->h_toDeleteLeaf[0] << " : " << treeHandler->h_toDeleteLeaf[1];
    Logger(INFO) << "toDeleteNode: " << treeHandler->h_toDeleteNode[0] << " : " << treeHandler->h_toDeleteNode[1];
//#endif

    time = 0;
    // insert received pseudo-particles per level in order to ensure correct order (avoid race condition)
    for (int level=0; level<MAX_LEVEL; level++) {
        time += Gravity::Kernel::Launch::insertReceivedPseudoParticles(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                                       treeHandler->d_tree, particleHandler->d_particles,
                                                                       d_pseudoParticles2ReceiveLevels, level, numParticles,
                                                                       numParticles);
    }

    //Logger(INFO) << "toDeleteNode: " << treeHandler->h_toDeleteNode[0] << ", " << treeHandler->h_toDeleteNode[1];

    totalTime += time;
    Logger(TIME) << "parallel_gravity: inserting received pseudo-particles: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::Gravity::insertReceivedPseudoParticles, time);

    time = 0;
    //if (treeHandler->h_toDeleteLeaf[0] < treeHandler->h_toDeleteLeaf[1]) {
    time += Gravity::Kernel::Launch::insertReceivedParticles(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                             treeHandler->d_tree, particleHandler->d_particles,
                                                             domainListHandler->d_domainList,
                                                             lowestDomainListHandler->d_domainList,
                                                             numParticles, numParticles);
    //}
    totalTime += time;
    Logger(TIME) << "parallel_gravity: inserting received particles: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::Gravity::insertReceivedParticles, time);

    Logger(INFO) << "Finished inserting received particles!";

    time = 0;
    // TODO: sorting only needed for Gravity::Kernel::Launch::computeForces()
    //time = TreeNS::Kernel::Launch::sort(treeHandler->d_tree, numParticlesLocal, numParticles, true);

    Logger(TIME) << "sorting: " << time << " ms";

    //TreeNS::Kernel::Launch::testTree(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal, numParticles);

    //actual (local) force
    //integer warp = 32;
    //integer stackSize = 64; //128; //64;
    //integer blockSize = 256;
    //time = Gravity::Kernel::Launch::computeForces(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal, numParticles,
    //                                                      blockSize, warp, stackSize, subDomainKeyTreeHandler->d_subDomainKeyTree);
    //time = Gravity::Kernel::Launch::computeForcesUnsorted(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal, numParticles,
    //                                       blockSize, warp, stackSize, subDomainKeyTreeHandler->d_subDomainKeyTree);
    time = Gravity::Kernel::Launch::computeForcesMiluphcuda(treeHandler->d_tree, particleHandler->d_particles,
                                                            numParticles, numParticles,
                                                            subDomainKeyTreeHandler->d_subDomainKeyTree);

    //NOTE: time(computeForces) < time(computeForceMiluphcuda) for kepler disk, but time(computeForces) >> time(computeForceMiluphcuda) for plummer!!!

    totalTime += time;
    Logger(TIME) << "computeForces: " << time << " ms";
    profiler.value2file(ProfilerIds::Time::Gravity::force, time);

    // repairTree
    // necessary? Tree is build for every iteration
    // only necessary if subsequent SPH
    int debug_lowestDomainListIndex;
    cuda::copy(&debug_lowestDomainListIndex, lowestDomainListHandler->d_domainListIndex, 1, To::host);
    Logger(INFO) << "lowest Domain list index: " << debug_lowestDomainListIndex;

    Logger(INFO) << "repairing tree...";
    Gravity::Kernel::Launch::repairTree(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                        particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                        numParticlesLocal, numNodes, curveType);

    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles, treeIndex, treeIndex + pseudoParticleTotalReceiveLength);
    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles, numParticlesLocal, numParticlesLocal + particleTotalReceiveLength);

    delete [] h_relevantDomainListProcess;
    delete [] h_particles2SendCount;
    delete [] h_pseudoParticles2SendCount;

    return totalTime;
}

real Miluphpc::sph() {

    real time = 0;
    time = parallel_sph();

    return time;
}

// IN PRINCIPLE it should be possible to reuse already sent particles from (parallel) gravity
real Miluphpc::parallel_sph() {

    real time;
    real totalTime = 0;

    cuda::set(d_particles2SendIndices, -1, numParticles);
    cuda::set(d_particles2SendCount, 0, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);

    cuda::set(lowestDomainListHandler->d_domainListCounter, 0, 1);

    time = SPH::Kernel::Launch::compTheta(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                          particleHandler->d_particles,
                                          lowestDomainListHandler->d_domainList, curveType);

    totalTime += time;
    Logger(TIME) << "sph: compTheta: " << time << " ms";

    integer relevantIndicesCounter;
    cuda::copy(&relevantIndicesCounter, lowestDomainListHandler->d_domainListCounter, 1, To::host);
    Logger(INFO) << "relevantIndicesCounter: " << relevantIndicesCounter;

    integer particlesOffset = 0;
    integer particlesOffsetBuffer;

    integer *h_relevantDomainListProcess;
    h_relevantDomainListProcess = new integer[relevantIndicesCounter];
    cuda::copy(h_relevantDomainListProcess, lowestDomainListHandler->d_relevantDomainListProcess,
               relevantIndicesCounter, To::host);

    integer *d_markedSendIndices = buffer->d_integerBuffer;
    real *d_collectedEntries = buffer->d_realBuffer;
    integer *h_particles2SendCount = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];


    // determine search radius

    boost::mpi::communicator comm;
    real h_searchRadius;

    /*const unsigned int blockSizeReduction = 256;
    real *d_searchRadii;
    cuda::malloc(d_searchRadii, blockSizeReduction);
    cuda::set(d_searchRadii, (real)0., blockSizeReduction);
    time += CudaUtils::Kernel::Launch::reduceBlockwise<real, blockSizeReduction>(particleHandler->d_sml, d_searchRadii,
                                                                                 numParticlesLocal);
    real *d_intermediateResult;
    cuda::malloc(d_intermediateResult, 1);
    cuda::set(d_intermediateResult, (real)0., 1);
    time += CudaUtils::Kernel::Launch::blockReduction<real, blockSizeReduction>(d_searchRadii, d_intermediateResult);

    cuda::copy(&h_searchRadius, d_intermediateResult, 1, To::host);

    h_searchRadius /= subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses;
    all_reduce(comm, boost::mpi::inplace_t<real*>(&h_searchRadius), 1, std::plus<real>());

    cuda::free(d_searchRadii);
    cuda::free(d_intermediateResult);*/

    real *d_intermediateResult;
    cuda::malloc(d_intermediateResult, 1);

    // testing
    real *d_potentialSearchRadii;
    cuda::malloc(d_potentialSearchRadii, numParticlesLocal);
    SPH::Kernel::Launch::determineSearchRadii(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                              particleHandler->d_particles, domainListHandler->d_domainList,
                                              lowestDomainListHandler->d_domainList, d_potentialSearchRadii,
                                              numParticlesLocal, 0, curveType);

    h_searchRadius = HelperNS::reduceAndGlobalize(d_potentialSearchRadii, d_intermediateResult,
                                                  numParticlesLocal, Reduction::max);

    cuda::free(d_potentialSearchRadii);
    // end:testing

    //h_searchRadius = HelperNS::reduceAndGlobalize(particleHandler->d_sml, d_intermediateResult,
    //                                              numParticlesLocal, Reduction::max);


    cuda::free(d_intermediateResult);

    Logger(INFO) << "search radius: " << h_searchRadius;
    // end: determine search radius

    time = 0;
    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            cuda::set(d_markedSendIndices, -1, numParticles); //numParticlesLocal should be sufficient
            for (int relevantIndex = 0; relevantIndex < relevantIndicesCounter; relevantIndex++) {
                if (h_relevantDomainListProcess[relevantIndex] == proc) {
                    Logger(INFO) << "h_relevantDomainListProcess[" << relevantIndex << "] = "
                                 << h_relevantDomainListProcess[relevantIndex];

                    time += SPH::Kernel::Launch::symbolicForce(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                               treeHandler->d_tree, particleHandler->d_particles,
                                                               lowestDomainListHandler->d_domainList,
                                                               d_markedSendIndices, h_searchRadius, numParticlesLocal, numParticles,
                                                               relevantIndex, curveType);

                }
            }
            time += SPH::Kernel::Launch::collectSendIndices(treeHandler->d_tree, particleHandler->d_particles,
                                                            d_markedSendIndices, &d_particles2SendIndices[particlesOffset],
                                                            &d_particles2SendCount[proc], numParticles, numParticles, curveType);

            cuda::copy(&particlesOffsetBuffer, &d_particles2SendCount[proc], 1, To::host);
            particlesOffset += particlesOffsetBuffer;
        }
    }

    totalTime += time;
    Logger(TIME) << "sph: symbolicForce: " << time << " ms";

    cuda::copy(h_particles2SendCount, d_particles2SendCount, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses, To::host);

    Timer timer;

    integer *particleSendLengths;
    particleSendLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    particleSendLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;
    integer *particleReceiveLengths;
    particleReceiveLengths = new integer[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    particleReceiveLengths[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] = 0;


    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        Logger(INFO) << "sph: h_particles2SendCount[" << proc << "] = " << h_particles2SendCount[proc];
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            particleSendLengths[proc] = h_particles2SendCount[proc];
        }
    }

    mpi::messageLengths(subDomainKeyTreeHandler->h_subDomainKeyTree, particleSendLengths, particleReceiveLengths);

    integer particleTotalReceiveLength = 0;
    integer particleTotalSendLength = 0;
    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            particleTotalReceiveLength += particleReceiveLengths[proc];
            particleTotalSendLength += particleSendLengths[proc];
        }
    }

    // writing particles to send to h5 file
    int *h_particles2SendIndices = new int[particleTotalSendLength];
    cuda::copy(h_particles2SendIndices, d_particles2SendIndices, particleTotalSendLength, To::host);
    std::string filename = "SPH2Send";
    particles2file(filename, h_particles2SendIndices, particleTotalSendLength);
    delete [] h_particles2SendIndices;
    // end: writing particles to send to h5 file

    Logger(INFO) << "sph: particleTotalReceiveLength: " << particleTotalReceiveLength;
    Logger(INFO) << "sph: particleTotalSendLength: " << particleTotalSendLength;

    delete [] h_relevantDomainListProcess;
    delete [] h_particles2SendCount;
    //delete [] h_pseudoParticles2SendCount;

    treeHandler->h_toDeleteLeaf[0] = numParticlesLocal;
    treeHandler->h_toDeleteLeaf[1] = numParticlesLocal + particleTotalReceiveLength;
    cuda::copy(treeHandler->h_toDeleteLeaf, treeHandler->d_toDeleteLeaf, 2, To::device);

    // x-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_x, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_x[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // x-vel-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_vx, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_vx[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // x-acc-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_ax, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_ax[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
#if DIM > 1
    // y-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_y, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_y[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // y-vel-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_vy, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_vy[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // y-acc-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_ay, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_ay[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
#if DIM == 3
    // z-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_z, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_z[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // z-vel-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_vz, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_vz[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // z-acc-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_az, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_az[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
#endif
#endif
    // mass-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_mass, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_mass[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    // sml-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_sml, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_sml[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    // rho-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_rho, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_rho[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    // pressure-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_p, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_p[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    // cs-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_cs, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_cs[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    // cs-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_e, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_e[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    //TODO: all entries...     (material id, ...)

    time = timer.elapsed();
    totalTime += time;
    Logger(TIME) << "sph: sending particles: " << time;

    Logger(INFO) << "checking for nans before assigning particles...";
    //ParticlesNS::Kernel::Launch::check4nans(particleHandler->d_particles, numParticlesLocal);

    cuda::copy(&treeHandler->h_toDeleteNode[0], treeHandler->d_index, 1, To::host);

#if DEBUGGING
    // debug
    gpuErrorcheck(cudaMemset(buffer->d_integerVal, 0, sizeof(integer)));
    CudaUtils::Kernel::Launch::findDuplicateEntries(&particleHandler->d_x[0],
                                                    &particleHandler->d_y[0],
                                                    &particleHandler->d_z[0],
                                                    buffer->d_integerVal,
                                                    numParticlesLocal + particleTotalReceiveLength);
    integer duplicates;
    gpuErrorcheck(cudaMemcpy(&duplicates, buffer->d_integerVal, sizeof(integer), cudaMemcpyDeviceToHost));
    Logger(INFO) << "duplicates: " << duplicates << " between: " << 0 << " and " << numParticlesLocal + particleTotalReceiveLength;
    if (duplicates > 0) {
        MPI_Finalize();
        exit(0);
    }
    //end: debug
#endif

    //TreeNS::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles, numParticles, numNodes);

    //DomainListNS::Kernel::Launch::info(particleHandler->d_particles, domainListHandler->d_domainList);

    //if (treeHandler->h_toDeleteLeaf[1] > treeHandler->h_toDeleteLeaf[0]) {
    time = SPH::Kernel::Launch::insertReceivedParticles(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                        treeHandler->d_tree,
                                                        particleHandler->d_particles, domainListHandler->d_domainList,
                                                        lowestDomainListHandler->d_domainList,
                                                        numParticles, //treeHandler->h_toDeleteLeaf[1],
                                                        numParticles);
    //}

    for (int level=MAX_LEVEL; level>0; --level) {
        SPH::Kernel::Launch::calculateCentersOfMass(treeHandler->d_tree, particleHandler->d_particles, level);
    }

    totalTime += time;
    Logger(TIME) << "sph: inserting received particles: " << time << " ms";

    cuda::copy(&treeHandler->h_toDeleteNode[1], treeHandler->d_index, 1, To::host);
    cuda::copy(treeHandler->h_toDeleteNode, treeHandler->d_toDeleteNode, 2, To::device);


    Logger(INFO) << "treeHandler->h_toDeleteNode[0]: " << treeHandler->h_toDeleteNode[0];
    Logger(INFO) << "treeHandler->h_toDeleteNode[1]: " << treeHandler->h_toDeleteNode[1];


    //time = SPH::Kernel::Launch::fixedRadiusNN(treeHandler->d_tree, particleHandler->d_particles, particleHandler->d_nnl,
    //                                          numParticlesLocal, numParticles, numNodes);

#if VARIABLE_SML
    time = SPH::Kernel::Launch::fixedRadiusNN_variableSML(materialHandler->d_materials, treeHandler->d_tree, particleHandler->d_particles, particleHandler->d_nnl,
                                              numParticlesLocal, numParticles, numNodes);
#endif

    time = SPH::Kernel::Launch::fixedRadiusNN(treeHandler->d_tree, particleHandler->d_particles, particleHandler->d_nnl,
                                              numParticlesLocal, numParticles, numNodes);

    // überprüfen inwiefern sich die sml geändert hat, sml_new <= sml_global_search
        // if sml_new > sml_global_search

    totalTime += time;
    Logger(TIME) << "fixedRadiusNN: " << time << " ms";

    //SPH::Kernel::Launch::info(treeHandler->d_tree, particleHandler->d_particles, helperHandler->d_helper,
    //                          numParticlesLocal, numParticles, numNodes);


    //calculateDensity(::SPH::SPHKernel *kernel, Particles *particles, int *interactions, int numParticles)
    //TODO: where to put calculateSoundSpeed()?

    Logger(INFO) << "checking for nans... numParticlesLocal: " << numParticlesLocal;
    //ParticlesNS::Kernel::Launch::check4nans(particleHandler->d_particles, numParticlesLocal);

    Logger(INFO) << "calculate density";
    SPH::Kernel::Launch::calculateDensity(kernelHandler.kernel, treeHandler->d_tree, particleHandler->d_particles,
                                          particleHandler->d_nnl, numParticlesLocal); //treeHandler->h_toDeleteLeaf[1]);

    Logger(INFO) << "calculate sound speed";
    SPH::Kernel::Launch::calculateSoundSpeed(particleHandler->d_particles, materialHandler->d_materials,
                                             numParticlesLocal); // treeHandler->h_toDeleteLeaf[1]);

    Logger(INFO) << "calculate pressure";
    SPH::Kernel::Launch::calculatePressure(materialHandler->d_materials, particleHandler->d_particles,
                                           numParticlesLocal); // treeHandler->h_toDeleteLeaf[1]);


    Logger(INFO) << "particle exchange";
    // TODO: update rho, cs, p, sml
    // updating necessary particle entries
    // sml-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_sml, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_sml[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    // rho-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_rho, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_rho[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    // pressure-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_p, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_p[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);

    // cs-entry particle exchange
    CudaUtils::Kernel::Launch::collectValues(d_particles2SendIndices, particleHandler->d_cs, d_collectedEntries,
                                             particleTotalSendLength);
    sendParticles(d_collectedEntries, &particleHandler->d_cs[numParticlesLocal], particleSendLengths,
                  particleReceiveLengths);
    // end: updating necessary particle entries

    Logger(INFO) << "internal forces";

    SPH::Kernel::Launch::internalForces(kernelHandler.kernel, materialHandler->d_materials, treeHandler->d_tree,
                                        particleHandler->d_particles, particleHandler->d_nnl, numParticlesLocal);

    Gravity::Kernel::Launch::repairTree(subDomainKeyTreeHandler->d_subDomainKeyTree, treeHandler->d_tree,
                                        particleHandler->d_particles, lowestDomainListHandler->d_domainList,
                                        numParticlesLocal, numNodes, curveType);

    Logger(TIME) << "sph: totalTime: " << totalTime << " ms";
    return totalTime;

}

void Miluphpc::fixedLoadBalancing() {

    Logger(INFO) << "fixedLoadBalancing()";

    keyType rangePerProc = (1UL << (21 * DIM))/(subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses);
    Logger(INFO) << "rangePerProc: " << rangePerProc;
    for (int i=0; i<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        subDomainKeyTreeHandler->h_subDomainKeyTree->range[i] = (keyType)i * rangePerProc;
    }
    subDomainKeyTreeHandler->h_subDomainKeyTree->range[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses] = KEY_MAX;

    //subDomainKeyTreeHandler->h_subDomainKeyTree->range[0] = 0UL;
    //subDomainKeyTreeHandler->h_subDomainKeyTree->range[1] = (4UL << 60) + (4UL << 57);
    //subDomainKeyTreeHandler->h_subDomainKeyTree->range[2] = (4UL << 60) + (2UL << 57);
    //subDomainKeyTreeHandler->h_subDomainKeyTree->range[3] = (6UL << 60) + (1UL << 57);
    //subDomainKeyTreeHandler->h_subDomainKeyTree->range[4] = KEY_MAX;
    // FOR TESTING PURPOSES:
    // 1, 0, 0, 0, ...: 1152921504606846976
    // 2, 0, 0, 0, ...: 2305843009213693952;
    // 3, 0, 0, 0, ...: 3458764513820540928;
    //subDomainKeyTreeHandler->h_subDomainKeyTree->range[0] = 0UL;
    //subDomainKeyTreeHandler->h_subDomainKeyTree->range[1] = 1048576; //2199023255552;//4194304; //  + (4UL << 57); // + (3UL << 54) + (1UL << 42) + (2UL << 18) + (1UL << 3) + (4);
    // // |4 (60)|0 (57)|0 (54)|4 (51)|0 (48)|6 (45)|1 (42)|1 (39)|1 (36)|5 (33)|6 (30)|4 (27)|5 (24)|7 (21)|0 (18)|6 (15)|5 (12)|1 (9)|1 (6)|4 (3)|3 (0)|
    // //subDomainKeyTreeHandler->h_subDomainKeyTree->range[1] = (4UL << 60) + (4UL << 51) + (6UL << 45) + (1UL << 42) + (1UL << 39) + (1UL << 36) + (5UL << 33)
    // //        + (6UL << 30) + (4UL << 27) + (5UL << 24) + (7UL << 21) + (6UL << 15) + (5UL << 12) + (1UL << 9) + (1UL << 6) + (4UL << 3);
    //subDomainKeyTreeHandler->h_subDomainKeyTree->range[2] = KEY_MAX; //9223372036854775808;

    for (int i=0; i<=subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        printf("range[%i] = %lu\n", i, subDomainKeyTreeHandler->h_subDomainKeyTree->range[i]);
    //    Logger(INFO) << "range[" << i << "] = " << subDomainKeyTreeHandler->h_subDomainKeyTree->range[i]:
    }

    subDomainKeyTreeHandler->copy(To::device, true, true);
}

void Miluphpc::dynamicLoadBalancing(int bins) {

    boost::mpi::communicator comm;

    Logger(INFO) << "dynamicLoadBalancing()";

    int *processParticleCounts = new int[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];

    all_gather(comm, &numParticlesLocal, 1, processParticleCounts);

    int totalAmountOfParticles = 0;
    for (int i=0; i<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        //Logger(INFO) << "numParticles on process: " << i << " = " << processParticleCounts[i];
        totalAmountOfParticles += processParticleCounts[i];
    }

    int aimedParticlesPerProcess = totalAmountOfParticles/subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses;
#if DEBUGGING
    Logger(INFO) << "aimedParticlesPerProcess = " << aimedParticlesPerProcess;
#endif

    updateRangeApproximately(aimedParticlesPerProcess, 2000);

    delete [] processParticleCounts;
}

void Miluphpc::updateRangeApproximately(int aimedParticlesPerProcess, int bins) {

    // introduce "bin size" regarding keys
    //  keyHistRanges = [0, 1 * binSize, 2 * binSize, ... ]
    // calculate key of particles on the fly and assign to keyHistRanges
    //  keyHistNumbers = [1023, 50032, ...]
    // take corresponding keyHistRange as new range if (sum(keyHistRange[i]) > aimNumberOfParticles ...
    // communicate new ranges among processes

    boost::mpi::communicator comm;

    helperHandler->reset();

    Gravity::Kernel::Launch::createKeyHistRanges(helperHandler->d_helper, bins);

    Gravity::Kernel::Launch::keyHistCounter(treeHandler->d_tree, particleHandler->d_particles,
                                            subDomainKeyTreeHandler->d_subDomainKeyTree, helperHandler->d_helper,
                                            bins, numParticlesLocal, curveType);

    all_reduce(comm, boost::mpi::inplace_t<integer*>(helperHandler->d_integerBuffer), bins - 1, std::plus<integer>());

    Gravity::Kernel::Launch::calculateNewRange(subDomainKeyTreeHandler->d_subDomainKeyTree, helperHandler->d_helper,
                                               bins, aimedParticlesPerProcess, curveType);
    keyType keyMax = (keyType)KEY_MAX;
    cuda::set(&subDomainKeyTreeHandler->d_range[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses], keyMax, 1);
    subDomainKeyTreeHandler->copy(To::host, true, true);

    //Logger(INFO) << "numProcesses: " << subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses;
    for (int i=0; i<=subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; i++) {
        Logger(INFO) << "range[" << i << "] = " << subDomainKeyTreeHandler->h_subDomainKeyTree->range[i];
    }
}

real Miluphpc::removeParticles() {

    int *d_particles2remove = d_particles2removeBuffer; //&buffer->d_integerBuffer[0];
    int *d_particles2remove_counter = d_particles2removeVal; //buffer->d_integerVal;

    real *d_temp = &buffer->d_realBuffer[0];
    integer *d_tempInt;
    cuda::malloc(d_tempInt, numParticles);

    cuda::set(d_particles2remove_counter, 0, 1);

    auto time = ParticlesNS::Kernel::Launch::mark2remove(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                         treeHandler->d_tree, particleHandler->d_particles,
                                                         d_particles2remove, d_particles2remove_counter,
                                                         numParticlesLocal);

    int h_particles2remove_counter;
    cuda::copy(&h_particles2remove_counter, d_particles2remove_counter, 1, To::host);
    Logger(INFO) << "#particles to be removed: " << h_particles2remove_counter;

    time += HelperNS::sortArray(particleHandler->d_x, d_temp, d_particles2remove, buffer->d_integerBuffer, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_x, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_vx, d_temp, d_particles2remove, buffer->d_integerBuffer, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_vx, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_ax, d_temp, d_particles2remove, buffer->d_integerBuffer, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_ax, d_temp, numParticlesLocal);
#if DIM > 1
    time += HelperNS::sortArray(particleHandler->d_y, d_temp, d_particles2remove, buffer->d_integerBuffer, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_y, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_vy, d_temp, d_particles2remove, buffer->d_integerBuffer, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_vy, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_ay, d_temp, d_particles2remove, buffer->d_integerBuffer, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_ay, d_temp, numParticlesLocal);
#if DIM == 3
    time += HelperNS::sortArray(particleHandler->d_z, d_temp, d_particles2remove, buffer->d_integerBuffer, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_z, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_vz, d_temp, d_particles2remove, buffer->d_integerBuffer, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_vz, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_az, d_temp, d_particles2remove, buffer->d_integerBuffer, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_az, d_temp, numParticlesLocal);
#endif
#endif
    time += HelperNS::sortArray(particleHandler->d_mass, d_temp, d_particles2remove, buffer->d_integerBuffer, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_mass, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_uid, d_tempInt, d_particles2remove, buffer->d_integerBuffer, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_uid, d_tempInt, numParticlesLocal);
#if SPH_SIM
    time += HelperNS::sortArray(particleHandler->d_materialId, d_tempInt, d_particles2remove, buffer->d_integerBuffer, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_materialId, d_tempInt, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_sml, d_temp, d_particles2remove, buffer->d_integerBuffer, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_sml, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_rho, d_temp, d_particles2remove, buffer->d_integerBuffer, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_rho, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_p, d_temp, d_particles2remove, buffer->d_integerBuffer, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_p, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_e, d_temp, d_particles2remove, buffer->d_integerBuffer, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_e, d_temp, numParticlesLocal);
    time += HelperNS::sortArray(particleHandler->d_cs, d_temp, d_particles2remove, buffer->d_integerBuffer, numParticlesLocal);
    time += HelperNS::Kernel::Launch::copyArray(particleHandler->d_cs, d_temp, numParticlesLocal);
#endif


    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_x[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_vx[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_ax[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
#if DIM > 1
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_y[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_vy[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_ay[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
#if DIM == 3
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_z[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_vz[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_az[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
#endif
#endif
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_mass[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_uid[numParticlesLocal-h_particles2remove_counter],
                                                 (integer)0, h_particles2remove_counter);
#if SPH_SIM
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_materialId[numParticlesLocal-h_particles2remove_counter],
                                                 (integer)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_sml[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_rho[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_p[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_e[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
    time += HelperNS::Kernel::Launch::resetArray(&particleHandler->d_cs[numParticlesLocal-h_particles2remove_counter],
                                                 (real)0, h_particles2remove_counter);
#endif

    //TODO: all entries (removing particles)

    cuda::free(d_tempInt);

    numParticlesLocal -= h_particles2remove_counter;
    Logger(INFO) << "removing #" << h_particles2remove_counter << " particles!";

    return time;
}

// used for gravity and sph
template <typename T>
integer Miluphpc::sendParticles(T *sendBuffer, T *receiveBuffer, integer *sendLengths, integer *receiveLengths) {
    boost::mpi::communicator comm;

    std::vector<boost::mpi::request> reqParticles;
    std::vector<boost::mpi::status> statParticles;

    integer reqCounter = 0;
    integer receiveOffset = 0;
    integer sendOffset = 0;

    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            reqParticles.push_back(comm.isend(proc, 17, &sendBuffer[sendOffset], sendLengths[proc]));
            statParticles.push_back(comm.recv(proc, 17, &receiveBuffer[receiveOffset], receiveLengths[proc]));

            receiveOffset += receiveLengths[proc];
            sendOffset += sendLengths[proc];
        }
    }

    boost::mpi::wait_all(reqParticles.begin(), reqParticles.end());

}

// used for assigning particles to corresponding process
template <typename T>
integer Miluphpc::sendParticlesEntry(integer *sendLengths, integer *receiveLengths, T *entry, T *entryBuffer, T *copyBuffer) {

    boost::mpi::communicator comm;

    std::vector<boost::mpi::request> reqParticles;
    std::vector<boost::mpi::status> statParticles;

    integer reqCounter = 0;
    integer receiveOffset = 0;
    integer sendOffset = 0;

    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++) {
        if (proc != subDomainKeyTreeHandler->h_subDomainKeyTree->rank) {
            if (proc == 0) {
                reqParticles.push_back(comm.isend(proc, 17, &entry[0], sendLengths[proc]));
                //Logger(INFO) << "Sending from: " << 0 << " to proc: " << proc;
            }
            else {
                reqParticles.push_back(comm.isend(proc, 17,
                                                  &entry[subDomainKeyTreeHandler->h_procParticleCounter[proc-1] + sendOffset],
                                                  sendLengths[proc]));

                //Logger(INFO) << "Sending from: " << subDomainKeyTreeHandler->h_procParticleCounter[proc-1] + sendOffset << " to proc: " << proc;
            }
            //reqParticles.push_back(comm.isend(proc, 17,&entry[sendOffset], sendLengths[proc]));
            statParticles.push_back(comm.recv(proc, 17, &entryBuffer[0] + receiveOffset,
                                              receiveLengths[proc]));

            //Logger(INFO) << "Receiving at " << receiveOffset << " from proc: " << proc;

            //sendOffset += subDomainKeyTreeHandler->h_procParticleCounter[proc-1]; //sendLengths[proc];
            receiveOffset += receiveLengths[proc];
        }
        sendOffset += subDomainKeyTreeHandler->h_procParticleCounter[proc-1]; //sendLengths[proc];
    }

    boost::mpi::wait_all(reqParticles.begin(), reqParticles.end());

    integer offset = 0;
    for (int i=0; i < subDomainKeyTreeHandler->h_subDomainKeyTree->rank; i++) {
        offset += subDomainKeyTreeHandler->h_procParticleCounter[i];
    }

    if (subDomainKeyTreeHandler->h_subDomainKeyTree->rank != 0) {
        if (offset > 0 && (subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] - offset) > 0) {
            //TODO: following line needed? (probably not)
            //HelperNS::Kernel::Launch::copyArray(&entry[0], &entry[subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] - offset], offset);
            //KernelHandler.copyArray(&entry[0], &entry[h_procCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank] - offset], offset);
        }
        HelperNS::Kernel::Launch::copyArray(&copyBuffer[0], &entry[offset], subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]);
        HelperNS::Kernel::Launch::copyArray(&entry[0], &copyBuffer[0], subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]);
        //KernelHandler.copyArray(&d_tempArray_2[0], &entry[offset], subDomainKeyTreeHandler->d_h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]);
        //KernelHandler.copyArray(&entry[0], &d_tempArray_2[0], subDomainKeyTreeHandler->d_h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]);
        //Logger(INFO) << "moving from offet: " << offset << " length: " << subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank];
    }

    HelperNS::Kernel::Launch::resetArray(&entry[subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]],
                                         (T)0, numParticles-subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]);
    HelperNS::Kernel::Launch::copyArray(&entry[subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]],
                                        entryBuffer, receiveOffset);
     //KernelHandler.resetFloatArray(&entry[h_procCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]], 0, numParticles-h_procCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank]); //resetFloatArrayKernel(float *array, float value, int n)
    //KernelHandler.copyArray(&entry[h_procCounter[h_subDomainHandler->rank]], d_tempArray, receiveOffset);

    //Logger(INFO) << "numParticlesLocal = " << receiveOffset << " + " << subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank];
    return receiveOffset + subDomainKeyTreeHandler->h_procParticleCounter[subDomainKeyTreeHandler->h_subDomainKeyTree->rank];
}

real Miluphpc::particles2file(int step) {

    Timer timer;

    boost::mpi::communicator comm;
    sumParticles = numParticlesLocal;
    all_reduce(comm, boost::mpi::inplace_t<integer*>(&sumParticles), 1, std::plus<integer>());

    std::stringstream stepss;
    stepss << std::setw(6) << std::setfill('0') << step;

    HighFive::File h5file("output/ts" + stepss.str() + ".h5",
                          HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate,
                          HighFive::MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));

    std::vector <size_t> dataSpaceDims(2);
    dataSpaceDims[0] = std::size_t(sumParticles);
    dataSpaceDims[1] = DIM;

    HighFive::DataSet ranges = h5file.createDataSet<keyType>("/hilbertRanges",
                                                             HighFive::DataSpace(subDomainKeyTreeHandler->h_numProcesses + 1));

    keyType *rangeValues;
    rangeValues = new keyType[subDomainKeyTreeHandler->h_numProcesses + 1];

    subDomainKeyTreeHandler->copy(To::host, true, false);
    for (int i=0; i<=subDomainKeyTreeHandler->h_numProcesses; i++) {
        rangeValues[i] = subDomainKeyTreeHandler->h_subDomainKeyTree->range[i];
        Logger(INFO) << "rangeValues[" << i << "] = " << rangeValues[i];
    }

    ranges.write(rangeValues);

    delete [] rangeValues;

    // TODO: add uid (and other entries?)
    HighFive::DataSet pos = h5file.createDataSet<real>("/x", HighFive::DataSpace(dataSpaceDims));
    HighFive::DataSet vel = h5file.createDataSet<real>("/v", HighFive::DataSpace(dataSpaceDims));
    HighFive::DataSet key = h5file.createDataSet<keyType>("/hilbertKey", HighFive::DataSpace(sumParticles));
    HighFive::DataSet h5_mass = h5file.createDataSet<real>("/m", HighFive::DataSpace(sumParticles));
#if SPH_SIM
    HighFive::DataSet h5_rho = h5file.createDataSet<real>("/rho", HighFive::DataSpace(sumParticles));
    HighFive::DataSet h5_p = h5file.createDataSet<real>("/p", HighFive::DataSpace(sumParticles));
    HighFive::DataSet h5_e = h5file.createDataSet<real>("/e", HighFive::DataSpace(sumParticles));
    HighFive::DataSet h5_sml = h5file.createDataSet<real>("/sml", HighFive::DataSpace(sumParticles));
    HighFive::DataSet h5_noi = h5file.createDataSet<integer>("/noi", HighFive::DataSpace(sumParticles));
#endif

    // ----------

    std::vector<std::vector<real>> x, v; // two dimensional vector for 3D vector data
    std::vector<keyType> k; // one dimensional vector holding particle keys
    std::vector<real> mass;
#if SPH_SIM
    std::vector<real> rho, p, e, sml;
    std::vector<integer> noi;
#endif

    particleHandler->copyDistribution(To::host, true, false);
#if SPH_SIM
    particleHandler->copySPH(To::host);
#endif

    keyType *d_keys;
    cuda::malloc(d_keys, numParticlesLocal);

    //TreeNS::Kernel::Launch::getParticleKeys(treeHandler->d_tree, particleHandler->d_particles,
    //                                        d_keys, 21, numParticlesLocal, curveType);
    SubDomainKeyTreeNS::Kernel::Launch::getParticleKeys(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                        treeHandler->d_tree, particleHandler->d_particles,
                                                        d_keys, 21, numParticlesLocal, curveType);
    //SubDomainKeyTreeNS::Kernel::Launch::getParticleKeys(subDomainKeyTreeHandler->d_subDomainKeyTree,
    //                                                    treeHandler->d_tree, particleHandler->d_particles,
    //                                                    d_keys, 21, numParticlesLocal, curveType);


    keyType *h_keys;
    h_keys = new keyType[numParticlesLocal];
    cuda::copy(h_keys, d_keys, numParticlesLocal, To::host);

    integer keyProc;

    for (int i=0; i<numParticlesLocal; i++) {
#if DIM == 1
        x.push_back({particleHandler->h_x[i]});
        v.push_back({particleHandler->h_vx[i]});
#elif DIM == 2
        x.push_back({particleHandler->h_x[i], particleHandler->h_y[i]});
        v.push_back({particleHandler->h_vx[i], particleHandler->h_vy[i]});
#else
        x.push_back({particleHandler->h_x[i], particleHandler->h_y[i], particleHandler->h_z[i]});
        v.push_back({particleHandler->h_vx[i], particleHandler->h_vy[i], particleHandler->h_vz[i]});
#endif
        k.push_back(h_keys[i]);
        mass.push_back(particleHandler->h_mass[i]);
        //Logger(INFO) << "mass[" << i << "] = " << mass[i];
#if SPH_SIM
        rho.push_back(particleHandler->h_rho[i]);
        p.push_back(particleHandler->h_p[i]);
        e.push_back(particleHandler->h_e[i]);
        sml.push_back(particleHandler->h_sml[i]);
        noi.push_back(particleHandler->h_noi[i]);
#endif
    }

    cuda::free(d_keys);
    delete [] h_keys;

    // receive buffer
    int procN[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];

    // send buffer
    int sendProcN[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++){
        sendProcN[proc] = subDomainKeyTreeHandler->h_subDomainKeyTree->rank == proc ? numParticlesLocal : 0;
    }

    all_reduce(comm, sendProcN, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses, procN,
               boost::mpi::maximum<integer>());

    std::size_t nOffset = 0;
    // count total particles on other processes
    for (int proc = 0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->rank; proc++){
        nOffset += procN[proc];
    }
    Logger(DEBUG) << "Offset to write to datasets: " << std::to_string(nOffset);

    // write to associated datasets in h5 file
    // only working when load balancing has been completed and even number of particles
    pos.select({nOffset, 0},
                {std::size_t(numParticlesLocal), std::size_t(DIM)}).write(x);
    vel.select({nOffset, 0},
                {std::size_t(numParticlesLocal), std::size_t(DIM)}).write(v);
    key.select({nOffset}, {std::size_t(numParticlesLocal)}).write(k);
    h5_mass.select({nOffset}, {std::size_t(numParticlesLocal)}).write(mass);
#if SPH_SIM
    h5_rho.select({nOffset}, {std::size_t(numParticlesLocal)}).write(rho);
    h5_p.select({nOffset}, {std::size_t(numParticlesLocal)}).write(p);
    h5_e.select({nOffset}, {std::size_t(numParticlesLocal)}).write(e);
    h5_sml.select({nOffset}, {std::size_t(numParticlesLocal)}).write(sml);
    h5_noi.select({nOffset}, {std::size_t(numParticlesLocal)}).write(noi);
#endif

    return timer.elapsed();

}

real Miluphpc::particles2file(int step, real com[DIM], real t) {

    Timer timer;

    boost::mpi::communicator comm;
    sumParticles = numParticlesLocal;
    all_reduce(comm, boost::mpi::inplace_t<integer*>(&sumParticles), 1, std::plus<integer>());

    std::stringstream stepss;
    stepss << std::setw(6) << std::setfill('0') << step;

    HighFive::File h5file("output/ts" + stepss.str() + ".h5",
                          HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate,
                          HighFive::MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));

    std::vector <size_t> dataSpaceDims(2);
    dataSpaceDims[0] = std::size_t(sumParticles);
    dataSpaceDims[1] = DIM;

    HighFive::DataSet ranges = h5file.createDataSet<keyType>("/hilbertRanges",
                                                             HighFive::DataSpace(subDomainKeyTreeHandler->h_numProcesses + 1));

    keyType *rangeValues;
    rangeValues = new keyType[subDomainKeyTreeHandler->h_numProcesses + 1];

    subDomainKeyTreeHandler->copy(To::host, true, false);
    for (int i=0; i<=subDomainKeyTreeHandler->h_numProcesses; i++) {
        rangeValues[i] = subDomainKeyTreeHandler->h_subDomainKeyTree->range[i];
        Logger(INFO) << "rangeValues[" << i << "] = " << rangeValues[i];
    }

    ranges.write(rangeValues);

    delete [] rangeValues;

    HighFive::DataSet mass = h5file.createDataSet<keyType>("/m", HighFive::DataSpace(sumParticles));
    HighFive::DataSet pos = h5file.createDataSet<real>("/x", HighFive::DataSpace(dataSpaceDims));
    HighFive::DataSet vel = h5file.createDataSet<real>("/v", HighFive::DataSpace(dataSpaceDims));
    HighFive::DataSet key = h5file.createDataSet<keyType>("/hilbertKey", HighFive::DataSpace(sumParticles));
    HighFive::DataSet _com = h5file.createDataSet<real>("/COM", HighFive::DataSpace(DIM));
    HighFive::DataSet tDataSet = h5file.createDataSet<real>("/t", HighFive::DataSpace::From(t));


    // ----------

    std::vector<std::vector<real>> x, v; // two dimensional vector for 3D vector data
    std::vector<keyType> m, k; // one dimensional vector holding particle keys
    std::vector<real> centerOfMass;

    particleHandler->copyDistribution(To::host, true, false);

    keyType *d_keys;
    cuda::malloc(d_keys, numParticlesLocal);

    //TreeNS::Kernel::Launch::getParticleKeys(treeHandler->d_tree, particleHandler->d_particles,
    //                                        d_keys, 21, numParticlesLocal, curveType);
    SubDomainKeyTreeNS::Kernel::Launch::getParticleKeys(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                        treeHandler->d_tree, particleHandler->d_particles,
                                                        d_keys, 21, numParticlesLocal, curveType);
    //SubDomainKeyTreeNS::Kernel::Launch::getParticleKeys(subDomainKeyTreeHandler->d_subDomainKeyTree,
    //                                                    treeHandler->d_tree, particleHandler->d_particles,
    //                                                    d_keys, 21, numParticlesLocal, curveType);


    keyType *h_keys;
    h_keys = new keyType[numParticlesLocal];
    cuda::copy(h_keys, d_keys, numParticlesLocal, To::host);

    integer keyProc;

    centerOfMass.push_back(com[0]);
#if DIM > 1
    centerOfMass.push_back(com[1]);
#if DIM == 3
    centerOfMass.push_back(com[2]);
#endif
#endif

    for (int i=0; i<numParticlesLocal; i++) {
#if DIM == 1
        x.push_back({particleHandler->h_x[i]});
        v.push_back({particleHandler->h_vx[i]});
#elif DIM == 2
        x.push_back({particleHandler->h_x[i], particleHandler->h_y[i]});
        v.push_back({particleHandler->h_vx[i], particleHandler->h_vy[i]});
#else
        x.push_back({particleHandler->h_x[i], particleHandler->h_y[i], particleHandler->h_z[i]});
        v.push_back({particleHandler->h_vx[i], particleHandler->h_vy[i], particleHandler->h_vz[i]});
#endif
        k.push_back(h_keys[i]);
        m.push_back(particleHandler->h_mass[i]);
    }

    cuda::free(d_keys);
    delete [] h_keys;

    // receive buffer
    int procN[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];

    // send buffer
    int sendProcN[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++){
        sendProcN[proc] = subDomainKeyTreeHandler->h_subDomainKeyTree->rank == proc ? numParticlesLocal : 0;
    }

    all_reduce(comm, sendProcN, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses, procN,
               boost::mpi::maximum<integer>());

    std::size_t nOffset = 0;
    // count total particles on other processes
    for (int proc = 0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->rank; proc++){
        nOffset += procN[proc];
    }
    Logger(DEBUG) << "Offset to write to datasets: " << std::to_string(nOffset);

    // write to associated datasets in h5 file
    // only working when load balancing has been completed and even number of particles
    pos.select({nOffset, 0},
               {std::size_t(numParticlesLocal), std::size_t(DIM)}).write(x);
    vel.select({nOffset, 0},
               {std::size_t(numParticlesLocal), std::size_t(DIM)}).write(v);
    key.select({nOffset}, {std::size_t(numParticlesLocal)}).write(k);
    mass.select({nOffset}, {std::size_t(numParticlesLocal)}).write(m);
    _com.select({0}, {std::size_t(DIM)}).write(centerOfMass);
    tDataSet.write(t);

    return timer.elapsed();

}


real Miluphpc::particles2file(const std::string& filename, int *particleIndices, int length) {

    Timer timer;

    boost::mpi::communicator comm;
    int totalLength = length;
    all_reduce(comm, boost::mpi::inplace_t<integer*>(&totalLength), 1, std::plus<integer>());

    std::stringstream file;
    file << "log/" <<  filename.c_str() << ".h5";
    //stepss << std::setw(6) << std::setfill('0') << step;

    HighFive::File h5file(file.str(),
                          HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate,
                          HighFive::MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));

    std::vector <size_t> dataSpaceDims(2);
    dataSpaceDims[0] = std::size_t(totalLength);
    dataSpaceDims[1] = DIM;

    HighFive::DataSet ranges = h5file.createDataSet<keyType>("/hilbertRanges",
                                                             HighFive::DataSpace(subDomainKeyTreeHandler->h_numProcesses + 1));

    keyType *rangeValues;
    rangeValues = new keyType[subDomainKeyTreeHandler->h_numProcesses + 1];

    subDomainKeyTreeHandler->copy(To::host, true, false);
    for (int i=0; i<=subDomainKeyTreeHandler->h_numProcesses; i++) {
        rangeValues[i] = subDomainKeyTreeHandler->h_subDomainKeyTree->range[i];
        Logger(INFO) << "rangeValues[" << i << "] = " << rangeValues[i];
    }

    ranges.write(rangeValues);

    delete [] rangeValues;

    HighFive::DataSet pos = h5file.createDataSet<real>("/x", HighFive::DataSpace(dataSpaceDims));
    //HighFive::DataSet vel = h5file.createDataSet<real>("/v", HighFive::DataSpace(dataSpaceDims));
    HighFive::DataSet key = h5file.createDataSet<keyType>("/hilbertKey", HighFive::DataSpace(totalLength));

    // ----------

    std::vector<std::vector<real>> x, v; // two dimensional vector for 3D vector data
    std::vector<keyType> k; // one dimensional vector holding particle keys

    particleHandler->copyDistribution(To::host, true, false, true);

    keyType *d_keys;
    cuda::malloc(d_keys, numNodes);

    //TreeNS::Kernel::Launch::getParticleKeys(treeHandler->d_tree, particleHandler->d_particles,
    //                                        d_keys, 21, numParticlesLocal, curveType);
    SubDomainKeyTreeNS::Kernel::Launch::getParticleKeys(subDomainKeyTreeHandler->d_subDomainKeyTree,
                                                        treeHandler->d_tree, particleHandler->d_particles,
                                                        d_keys, 21, numNodes, curveType);
    //SubDomainKeyTreeNS::Kernel::Launch::getParticleKeys(subDomainKeyTreeHandler->d_subDomainKeyTree,
    //                                                    treeHandler->d_tree, particleHandler->d_particles,
    //                                                    d_keys, 21, numParticlesLocal, curveType);


    keyType *h_keys;
    h_keys = new keyType[numNodes];
    cuda::copy(h_keys, d_keys, numNodes, To::host);

    integer keyProc;

    for (int i=0; i<length; i++) {
#if DIM == 1
        x.push_back({particleHandler->h_x[particleIndices[i]]});
        //v.push_back({particleHandler->h_vx[particleIndices[i]]});
#elif DIM == 2
        x.push_back({particleHandler->h_x[particleIndices[i]], particleHandler->h_y[particleIndices[i]]});
        //v.push_back({particleHandler->h_vx[i], particleHandler->h_vy[i]});
#else
        x.push_back({particleHandler->h_x[particleIndices[i]], particleHandler->h_y[particleIndices[i]],
                     particleHandler->h_z[particleIndices[i]]});
        //v.push_back({particleHandler->h_vx[particleIndices[i]], particleHandler->h_vy[particleIndices[i]],
        //             particleHandler->h_vz[particleIndices[i]]});
#endif
        k.push_back(h_keys[particleIndices[i]]);
    }

    cuda::free(d_keys);
    delete [] h_keys;

    // receive buffer
    int procN[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];

    // send buffer
    int sendProcN[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
    for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++){
        sendProcN[proc] = subDomainKeyTreeHandler->h_subDomainKeyTree->rank == proc ? length : 0;
    }

    all_reduce(comm, sendProcN, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses, procN,
               boost::mpi::maximum<integer>());

    std::size_t nOffset = 0;
    // count total particles on other processes
    for (int proc = 0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->rank; proc++){
        nOffset += procN[proc];
    }
    Logger(DEBUG) << "Offset to write to datasets: " << std::to_string(nOffset);

    // write to associated datasets in h5 file
    // only working when load balancing has been completed and even number of particles
    pos.select({nOffset, 0},
               {std::size_t(length), std::size_t(DIM)}).write(x);
    //vel.select({nOffset, 0},
    //           {std::size_t(length), std::size_t(DIM)}).write(v);
    key.select({nOffset}, {std::size_t(length)}).write(k);

    return timer.elapsed();

}
