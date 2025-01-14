# Prerequisites/Dependencies

| library         | licence           | usage             | link               |
| --------------- | ----------------- | ----------------- | ------------------ |
| GNU             | GPLv3+            | compiler          | [gnu.org](https://www.gnu.org/home.de.html) |
| OpenMPI         | BSD 3-Clause      | compiler, MPI Implementation | [open-mpi.org](https://www.open-mpi.org/) |
| CUDA            | CUDA Toolkit End User License Agreement | compiler, CUDA Toolkit and API | [developer.nvidia.com](https://developer.nvidia.com/) |
| CUDA cub        | BSD 3-Clause "New" or "Revised" License | device wide parallel primitives | [github.com/NVIDIA/cub](https://github.com/NVIDIA/cub) |
| HDF5            | HDF5 License (BSD-Style) | parallel HDF5 for I/O operations | [hdf5group.org](https://www.hdfgroup.org/solutions/hdf5/) |
| HighFive        | Boost Software License 1.0 | C++ wrapper for parallel HDF5 | [github.com/BlueBrain/HighFive](https://github.com/BlueBrain/HighFive) |
| Boost           | Boost Software License 1.0 | config file parsing, C++ wrapper for MPI | [boost.org](https://www.boost.org/) |
| cxxopts         | MIT license | command line argument parsing | [github.com/jarro2783/cxxopts](https://github.com/jarro2783/cxxopts) |
| libconfig       |  LGPL-2.1 | material config parsing| [github.io/libconfig](http://hyperrealm.github.io/libconfig/) |

* in general there is no need for the usage of the GNU compiler and OpenMPI as MPI implementation, as long as a proper C++ compiler as well as MPI implementation (CUDA-aware) are available and corresponding changes in the Makefile are done

_____

## Installing/Building the dependencies

### MPI implementation

> Multiple MPI implementations exist and can be used to execute milupHPC. However, the code was primarily tested with OpenMPI >= 4.0.

#### OpenMPI


| library         | licence           | usage             | link               |
| --------------- | ----------------- | ----------------- | ------------------ |
| OpenMPI         | BSD 3-Clause      | compiler, MPI Implementation | [open-mpi.org](https://www.open-mpi.org/) |


see also [MPI_Versions.md](MPI_Versions.md)

* if either MPI is not available or MPI is not CUDA-aware
	* e.g. via `ompi_info -a | grep "\-with\-cuda"` 
* build, see e.g.
	* [Building Open MPI](https://www.open-mpi.org/faq/?category=building)
	* [Building CUDA-aware Open MPI](https://www.open-mpi.org/faq/?category=buildcuda)
	* [UCX OpenMPI installation](https://github.com/openucx/ucx/wiki/OpenMPI-and-OpenSHMEM-installation-with-UCX)

	
### Boost

| library         | licence           | usage             | link               |
| --------------- | ----------------- | ----------------- | ------------------ |
| Boost           | Boost Software License 1.0 | config file parsing, C++ wrapper for MPI | [boost.org](https://www.boost.org/) |

* `<boost version>` e.g. `1_78_0`
* `</usr>` corresponds to the storage location

```
$ wget https://boostorg.jfrog.io/artifactory/main/release/1.78.0/source/boost_<boost version>.tar.gz
$ tar zxvf boost_<boost version>.tar.gz
$ cd boost_<boost version>
$ ./bootstrap.sh --with-libraries=all
$ vim project-config.jam
$ >> using mpi ;
$ ./b2
$ ./b2 install --prefix=</usr>
```

### HighFive

| library         | licence           | usage             | link               |
| --------------- | ----------------- | ----------------- | ------------------ |
| HighFive        | Boost Software License 1.0 | C++ wrapper for parallel HDF5 | [github.com/BlueBrain/HighFive](https://github.com/BlueBrain/HighFive) |

* `git clone https://github.com/BlueBrain/HighFive.git --branch v2.4.1`
* To set the version manually `git checkout -b v2.4.1`
```
$ git clone https://github.com/BlueBrain/HighFive.git
$ cd HighFive
$ git checkout -b v2.4.1
```


### CUDA cub

* included in the CUDA Toolkit since CUDA 11.0

| library         | licence           | usage             | link               |
| --------------- | ----------------- | ----------------- | ------------------ |
| CUDA cub        | BSD 3-Clause "New" or "Revised" License | device wide parallel primitives | [github.com/NVIDIA/cub](https://github.com/NVIDIA/cub) |

* `git clone https://github.com/NVIDIA/cub.git --branch v1.8.0`
	* in dependence of used CUDA Toolkit e.g. `git checkout -b v1.8.0`
```
$ git clone https://github.com/NVIDIA/cub.git
$ cd cub
$ git checkout -b v1.8.0
```


### Cxxopts

| library         | licence           | usage             | link               |
| --------------- | ----------------- | ----------------- | ------------------ |
| cxxopts         | MIT license | command line argument parsing | [github.com/jarro2783/cxxopts](https://github.com/jarro2783/cxxopts) |

* `git clone https://github.com/jarro2783/cxxopts.git`


### libconfig

| library         | licence           | usage             | link               |
| --------------- | ----------------- | ----------------- | ------------------ |
| libconfig       |  LGPL-2.1 | material config parsing| [github.io/libconfig](http://hyperrealm.github.io/libconfig/) |

* `<libconfig version>` e.g. `1.7.3`
* `</usr>` corresponds to the storage location

```
$ wget http://hyperrealm.github.io/libconfig/dist/libconfig-<libconfig version>.tar.gz
$ tar zxvf libconfig-<libconfig version>.tar.gz
$ cd libconfig-<libconfig version>
$ ./configure --prefix=</usr>
$ make
$ make install
```


### HDF5

| library         | licence           | usage             | link               |
| --------------- | ----------------- | ----------------- | ------------------ |
| HDF5            | HDF5 License (BSD-Style) | parallel HDF5 for I/O operations | [hdf5group.org](https://www.hdfgroup.org/solutions/hdf5/) |

* refer to [realease_docs](https://github.com/HDFGroup/hdf5/tree/develop/release_docs) and build parallel
* `</usr>` corresponds to the storage location

e.g.:

```
$ wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.2/src/hdf5-1.12.2.tar.gz
$ tar -zxvf hdf5-1.12.0.tar.gz
$ cd hdf5
$ CC=mpicc CXX=mpic++ ./configure --prefix=</usr> --enable-parallel --enable-hl
$ make install
```


_____

### Postprocessing 

For postprocessing purposes several scripts are available. Some of those scripts require the following.

#### Python

* Python: [realpython.com/installing-python](https://realpython.com/installing-python/)
	* and packages: numpy, matplotlib, ...

#### ffmpeg
* `</usr/local/bin/>` corresponds to the storage location
```
$ sudo wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
$ sudo wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz.md5
$ md5sum -c ffmpeg-release-amd64-static.tar.xz.md5
$ sudo tar xvf ffmpeg*.xz
$ cd ffmpeg-*-static
$ sudo ln -s "${PWD}/ffmpeg" </usr/local/bin/>
$ sudo ln -s "${PWD}/ffprobe" </usr/local/bin/>
```
