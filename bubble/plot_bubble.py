#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 10:05:56 2023

@author: dreardon
"""

from scipy.interpolate import RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import Galactic

c = Galactic(l=253.394*u.degree, b=-41.963*u.degree, distance=0.15679*u.kpc)

x = c.cartesian.x.value * 1000
y = c.cartesian.y.value * 1000
z = c.cartesian.z.value * 1000

# XY
data = np.load('STILISM_XY.npz')

X = data['X']
Y = data['Y']
E = data['E']


plt.figure(figsize=(9, 9))
plt.pcolormesh(X, Y, np.log(E), shading='nearest', cmap='magma')
plt.scatter([x], [y], marker='.', color='r')
plt.scatter([0], [0], marker='.', color='skyblue')
plt.axis('equal')
plt.xlim([-300, 300])
plt.ylim([-300, 300])
plt.xlabel('X (pc)')
plt.ylabel('Y (pc)')
plt.savefig('xy.png')
plt.show()

# import sys
# sys.exit()

# XZ
data = np.load('STILISM_XZ.npz')

X = data['X']
Z = data['Z']
E = data['E']

plt.figure(figsize=(9, 9))
plt.pcolormesh(Z, X, np.log(E), shading='nearest', cmap='magma')
plt.scatter([z], [x], marker='.', color='r')
plt.scatter([0], [0], marker='.', color='skyblue')
plt.axis('equal')
plt.xlim([-300, 300])
plt.ylim([-300, 300])
plt.xlabel('Z (pc)')
plt.ylabel('X (pc)')
plt.savefig('xz.png')
plt.show()

# # YZ
# data = np.load('STILISM_YZ.npz')

# Y = data['Y']
# Z = data['Z']
# E = data['E']

# plt.figure(figsize=(9, 9))
# plt.pcolormesh(Z, Y, E, shading='nearest', cmap='magma')
# plt.scatter([z], [y], marker='.', color='r')
# plt.scatter([0], [0], marker='.', color='skyblue')
# plt.axis('equal')
# plt.xlim([-300, 300])
# plt.ylim([-300, 300])
# plt.xlabel('Z (pc)')
# plt.ylabel('Y (pc)')
# plt.show()


data = np.load('ExtinctionCube/STILISM_cube.npz')
X, Y, Z, E = data['X'], data['Y'], data['Z'], np.log(data['E'])

xind = np.argwhere(np.abs(X) <= 300).squeeze()
yind = np.argwhere(np.abs(Y) <= 300).squeeze()
zind = np.argwhere(np.abs(Z) <= 300).squeeze()

Xnew = X[xind].squeeze()
Ynew = Y[yind].squeeze()
Znew = Z[zind].squeeze()
Enew = E[xind][yind][zind].squeeze()

np.shape(Xnew, Ynew, Enew)

np.savez('compact.npz', X=Xnew, Y=Ynew, Z=Znew, E=Enew)


# Simulating the data since the real data can't be read
np.random.seed(0)  # for reproducibility
# X_sim = np.linspace(-100, 100, 200)
# Y_sim = np.linspace(-100, 100, 200)
# Z_sim = np.linspace(-100, 100, 161)
# E_sim = np.random.rand(200, 200, 161)  # random values for the energy

# Points P1 and P2
P1 = np.array([0, 0, 0])
P2 = np.array([-33.3, -111.7, -104.8])

# Function to find the plane equation given two points


def plane_equation(p1, p2):
    # Vector from p1 to p2
    v = p2 - p1
    # Normal vector to the plane
    n = np.cross(v, np.array([1, 0, 0]))
    # Constant term in the plane equation
    d = -np.dot(n, p1)
    return n, d


# Finding the plane equation
n, d = plane_equation(P1, P2)

# Interpolating the energy data
interpolator = RegularGridInterpolator((X_sim, Y_sim, Z_sim), E_sim)

# Creating a grid for the slice
x = np.linspace(X_sim.min(), X_sim.max(), 1000)
y = np.linspace(Y_sim.min(), Y_sim.max(), 1000)
xx, yy = np.meshgrid(x, y)
zz = (-d - n[0]*xx - n[1]*yy) / n[2]  # z from the plane equation

# Masking out-of-bounds points
mask = (zz >= Z_sim.min()) & (zz <= Z_sim.max())
points = np.array([xx[mask], yy[mask], zz[mask]]).T

# Evaluating the interpolator at the grid points
slice_data = np.full(xx.shape, np.nan)
slice_data[mask] = interpolator(points)

# Plotting
plt.figure(figsize=(10, 8))
plt.pcolormesh(xx, yy, slice_data, shading='auto', cmap='magma')
plt.colorbar(label='Log(Energy)')
plt.scatter(*P1[:2], color='red', label='P1 (0,0,0)')
plt.scatter(*P2[:2], color='green', label='P2 (-33.3, -111.7)')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-300, 300])
plt.ylim([-300, 300])
plt.title('Interpolated 2D Slice of Energy Data')
plt.show()
