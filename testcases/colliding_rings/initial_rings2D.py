'created by Anne Vera Jeschke 10th February 2023'
import numpy as np
import matplotlib.pyplot as plt
import h5py

"""
this program creates two 2-D rings (z = 0) around the origin which are then shifted to their final positions
used for colliding rings testcase, see Gray, Monaghan, Swift SPH elastic dynamics, journal of Computer methods
in applied mechanics and engineering
"""
# Dim of Rings
dim = 2

# ring properties: inner and outer radius
r_inner = 3.0
r_outer = 4.0

# shift of the rings from origin on x-axis
shift = 6.0

# particle spacing
delta_p = 0.1
# projected speed
v_p = 0.059

density = 1
mass = delta_p**2 * density

# create initial distribution through creating a 2d grid with dims 2*r_outer x 2*r_outer
# then delete particles which are not on the ring

# calc number of particles within square
N_length = int(2*r_outer/delta_p)
# number of particles in square
N_square = int(N_length**2)

# coordinates of particles in square
r = np.zeros((N_square, dim))

# 2D meshgrid
a = np.mgrid[0:N_length, 0:N_length]

# create square
for i in range(dim):
    k = 0
    for j in range(N_square):
        if j % N_length == 0 and j > 0:
            k += 1
            k = k % N_length
        # print(i, k, j)

        r[j, i] = (a[i, k, j % N_length]-(N_length-1)/2)*delta_p

# count particles in one ring
N = 0
arr = np.zeros(N_square) 
for i in range(N_square):
    radius = np.sqrt(r[i, 0]**2 + r[i, 1]**2)
    if r_outer >= radius >= r_inner:
        N += 1
        arr[i] = 1

# construct two rings with N particles which then are shifted along the x-axis
r_ring = np.zeros((N, dim))  # first ring
r_ring2 = np.zeros((N, dim))  # second ring
v = np.zeros((N, dim))
v2 = np.zeros((N, dim))

m = np.ones(2*N)*mass # 2N because of two rings
rho = np.ones(2*N)*density
materialId = np.zeros(2*N, dtype=np.int8)
#Sxx = np.zeros(2*N)
#Sxy = np.zeros(2*N)

# create ring 1
counter = 0
for i in range(N_square):
    if arr[i] == 1:
        r_ring[counter, 0] = r[i, 0] - shift
        r_ring[counter, 1] = r[i, 1]
        # r_ring[counter, 2] = 0
        v[counter, 0] = v_p
        counter += 1
# create ring 2
counter = 0
for i in range(N_square):
    if arr[i] == 1:
        r_ring2[counter, 0] = r[i, 0] + shift
        r_ring2[counter, 1] = r[i, 1]
        # r_ring2[counter, 2] = 0
        v2[counter, 0] = -v_p
        counter += 1
# put two rings in one array
r_final = np.concatenate((r_ring, r_ring2))
v_final = np.concatenate((v, v2))

h5f = h5py.File("rings_2N{}.h5".format(N), "w")
print("Saving to rings.h5 ...")

# write to hdf5 data set
h5f.create_dataset("x", data=r_final)
h5f.create_dataset("v", data=v_final)
h5f.create_dataset("m", data=m)
h5f.create_dataset("materialId", data=materialId)
h5f.create_dataset("rho", data=rho)
#h5f.create_dataset("Sxx", data=Sxx)
#h5f.create_dataset("Sxy", data=Sxy)

h5f.close()
print("Finished")